import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from joblib import Parallel, delayed
from dadapy import data
import argparse
import json


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def shuffle_tokens(ids):
    N = ids.shape[-1]
    permutation = np.random.permutation(N)
    new_ids = ids.reshape((1, N))
    new_ids = new_ids[0, permutation]
    return new_ids

def parse_arguments():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--login_token", type=str, default=None)
    parser.add_argument("--find_nn_sim", type=str, default="False")
    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args

def convert_to_distances(hs):
    return torch.stack([torch.cdist(item, item).squeeze() for item in hs])

def extract_hidden_states(sequence, model, tokenizer, max_length):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  with torch.no_grad():  # Disable gradient computation  
      inputs = tokenizer(sequence.strip() , add_special_tokens = False, \
                         return_tensors = "pt", max_length = max_length, \
                             truncation=True).to(device)
      outputs = model(**inputs, labels = inputs['input_ids'].clone(), \
                      output_hidden_states=True)
      hidden_states, loss = outputs.hidden_states, outputs.loss
  return {
          "hidden_distances" : convert_to_distances(hidden_states).cpu().detach().numpy(),\
          "loss": loss.to(torch.float32).cpu().detach().numpy()
              }


def compute_ids(full_reps):
    ids = []
    for full_rep in full_reps:
        _, indices = np.unique(full_rep, axis=0, return_index=True)
        rep = full_rep[indices, :][:, indices]
        _data = data.Data(distances=rep, maxk=300)
        ids.append(_data.return_id_scaling_gride(range_max=256))
    return np.array(ids)

if __name__ == "__main__":
    args = parse_arguments()
    login(token=args.login_token)

    path_dict = {
        "Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
        "Mistral-7B": "mistralai/Mistral-7B-v0.1",
        "Pythia-6.9B": "EleutherAI/pythia-6.9b",
        "Pythia-6.9B-Deduped": "EleutherAI/pythia-6.9b-deduped",
        "Pythia-160M-Deduped": "EleutherAI/pythia-160m-deduped",
        "Pythia-410M-Deduped": "EleutherAI/pythia-410m-deduped",
        "Pythia-1.4B-Deduped": "EleutherAI/pythia-1.4b-deduped",
        "Pythia-2.8B-Deduped": "EleutherAI/pythia-2.8b-deduped",
        "Opt-6.7B": "facebook/opt-6.7b",
        "Gpt2": "gpt2",
        "Gpt2-large": "gpt2-large",
        "Gpt2-xl": "gpt2-xl"
    }
    
    if args.model_name not in path_dict:
        raise ValueError(f"{args.model_name} is not supported. Supported models: {list(path_dict.keys())}")
    
    model_path = path_dict[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=f"/projects/0/gusr0688/llama-stuff/models/{args.model_name}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto",
        cache_dir=f"/projects/0/gusr0688/llama-stuff/models/{args.model_name}"
    )
    
    ds = load_dataset("NeelNanda/pile-10k")['train']
    sequences = ds['text']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_length = 1024
    
    output_folder = f"{args.input_dir}/Pile-{args.method.capitalize()}/{args.model_name}"
    os.makedirs(f"{output_folder}/summaries", exist_ok=True)
    
    if args.method == "structured":
        batch_sz = 32
        filtered_indices = np.load('filtered_indices.npy')
        filtered_sequences = [sequences[idx] for idx in filtered_indices]
        ids_output, losses = [], []
        
        for batch_start in tqdm(range(0, len(filtered_indices), batch_sz)):
            batch_sequences = filtered_sequences[batch_start: batch_start + batch_sz]
            intermediate_reps = [
                extract_hidden_states(seq, model, tokenizer, max_length=max_length)
                for seq in batch_sequences
            ]
            hidden_distances = np.array([item["hidden_distances"][1:] for item in intermediate_reps])
            hidden_ids = np.array(Parallel(n_jobs=-1)(delayed(compute_ids)(hs) for hs in hidden_distances))
            ids_output.extend(hidden_ids)
            losses.extend([item["loss"] for item in intermediate_reps])
        
        np.save(f'{output_folder}/summaries/losses.npy', losses)
        np.save(f'{output_folder}/summaries/gride.npy', np.array(ids_output))
    
    elif args.method == "shuffled":
        new_filtered_indices = np.load('subset_indices.npy')
        filtered_sequences = [sequences[idx] for idx in new_filtered_indices]
        all_losses, all_ids = [], []
        
        for test_seq in tqdm(filtered_sequences):
            intermediate_reps, losses = [], []
            for _ in range(20):
                inputs = tokenizer(test_seq.strip(), add_special_tokens=False, return_tensors="pt",
                                   max_length=max_length, truncation=True).to(device)
                new_ids = shuffle_tokens(inputs['input_ids'].squeeze()).to(device)
                outputs = model(input_ids=new_ids, labels=new_ids.clone(), output_hidden_states=True)
                intermediate_reps.append(convert_to_distances(outputs.hidden_states).cpu().numpy())
                losses.append(outputs.loss.to(torch.float32).cpu().numpy())
            
            hidden_distances = np.array([item[1:] for item in intermediate_reps])
            hidden_ids = np.array(Parallel(n_jobs=-1)(delayed(compute_ids)(hs) for hs in hidden_distances))
            all_ids.extend(hidden_ids)
            all_losses.extend(losses)
        
        np.save(f"{output_folder}/summaries/losses_20.npy", all_losses)
        np.save(f"{output_folder}/summaries/gride_20.npy", all_ids)