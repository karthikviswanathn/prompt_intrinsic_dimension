import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
import os
from tuned_lens.nn.lenses import TunedLens
import argparse
import json

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--login_token", type=str, default=None)
    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args

def extract_entropies_from_input_ids(input_ids, model, lens, tokenizer):
    with torch.no_grad():
        input_ids_th = input_ids.clone().detach().to(model.device)
        outputs = model(input_ids_th, output_hidden_states=True)
        stream = list(outputs.hidden_states)
        entropy, hfe, expected_energy = [], [], []
      
        for i, h in enumerate(stream[:-1]):
            logits = lens.forward(h, i).squeeze()
            probs = torch.nn.functional.softmax(logits, dim=-1)
            energies = logits - torch.max(logits, axis=-1, keepdims=True).values
            hfe.append(-torch.log(torch.exp(energies).sum(axis=-1)).mean().cpu().numpy())
            entropy.append(-(torch.sum(probs * torch.log(probs + 1e-12), axis=-1)).mean().cpu().numpy())
            expected_energy.append(torch.sum(probs * (-energies), axis=-1).mean().cpu().numpy())
        
        logits = outputs.logits.squeeze()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        energies = logits - torch.max(logits, axis=-1, keepdims=True).values
        hfe.append(-torch.log(torch.exp(energies).sum(axis=-1)).mean().cpu().numpy())
        entropy.append(-(torch.sum(probs * torch.log(probs), axis=-1)).mean().cpu().numpy())
        expected_energy.append(torch.sum(probs * (-energies), axis=-1).mean().cpu().numpy())
        
        return {"entropy": np.array(entropy), "free_energy": np.array(hfe), "energy": np.array(expected_energy)}

def extract_entropies(sequence, model, lens, tokenizer, max_length):
    input_ids = tokenizer.encode(sequence.strip(), add_special_tokens=False, return_tensors="pt", max_length=max_length, truncation=True)
    return extract_entropies_from_input_ids(input_ids, model, lens, tokenizer)

def save_outputs(tuned_stats, output_folder):
    np.save(f"{output_folder}/entropy.npy", np.array([item["entropy"] for item in tuned_stats]))
    np.save(f"{output_folder}/free_energy.npy", np.array([item["free_energy"] for item in tuned_stats]))
    np.save(f"{output_folder}/energy.npy", np.array([item["energy"] for item in tuned_stats]))

if __name__ == "__main__":
    args = parse_arguments()
    login(token=args.login_token)
    
    path_dict = {
        "Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
        "Mistral-7B": "mistralai/Mistral-7B-v0.1",
        "Pythia-160M-Deduped": "EleutherAI/pythia-160m-deduped",
        "Pythia-410M-Deduped": "EleutherAI/pythia-410m-deduped",
        "Pythia-1.4B-Deduped": "EleutherAI/pythia-1.4b-deduped",
        "Pythia-2.8B-Deduped": "EleutherAI/pythia-2.8b-deduped",
        "Pythia-6.9B-Deduped": "EleutherAI/pythia-6.9b-deduped",
        "Pythia-6.9B": "EleutherAI/pythia-6.9b",
        "Opt-6.7B": "facebook/opt-6.7b",
        "Gpt2": "gpt2",
        "Gpt2-large": "gpt2-large",
        "Gpt2-xl": "gpt2-xl"
    }
    
    if args.model_name not in path_dict:
        raise ValueError(f"{args.model_name} not supported. Supported models: {path_dict.keys()}")
    
    tokenizer = AutoTokenizer.from_pretrained(path_dict[args.model_name])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(path_dict[args.model_name]).to(device)
    tuned_lens = TunedLens.from_model_and_pretrained(model).to(device)
    ds = load_dataset("NeelNanda/pile-10k")['train']
    sequences = ds['text']
    max_length = 1024
    input_dir = args.input_dir
    filtered_indices = np.load('filtered_indices.npy')
    filtered_sequences = [sequences[idx].strip() for idx in filtered_indices]
    output_folder = f"{input_dir}/Pile-Structured/{args.model_name}"
    os.makedirs(output_folder, exist_ok=True)
    tuned_stats = [extract_entropies(seq, model, tuned_lens, tokenizer, max_length) for seq in tqdm(filtered_sequences)]
    save_outputs(tuned_stats, output_folder)
    