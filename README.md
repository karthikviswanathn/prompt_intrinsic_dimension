# The Intrinsic Dimension of Prompts in Internal Representations of Large Language Models
Source code for the paper: 'The Intrinsic Dimension of Prompts in Internal Representations of Large Language Models'

## Description

In this project, we analyze the geometry of tokens in the hidden layers of large language models
using intrinsic dimension (ID). Given a prompt,
this is done in roughly two steps -

1. Extract the internal representations of the tokens -  We use the
[hidden states](https://huggingface.co/docs/transformers/v4.45.2/en/internal/generation_utils#generate-outputs)
variable from the [Transformers](https://huggingface.co/docs/transformers/index) library on Hugging Face. 
Make sure you are logged into HuggingFace Hub and have access to the models.

2. Calculating the intrinsic dimension - For each hidden layer, we consider the point cloud formed
by the token representation at that layer. On this point cloud, we calculate
the intrinsic dimension (ID).
Specifically, we use the
[Generalized Ratio Intrinsic Dimension Estimator (GRIDE)](https://www.nature.com/articles/s41598-022-20991-1)
to estimate the intrinsic dimension implemented using the
[DADApy library](https://github.com/sissa-data-science/DADApy).


## Dataset
Currently we use the prompts from [Pile-10K](https://huggingface.co/datasets/NeelNanda/pile-10k).
We filter only prompts with atleast `1024` tokens according to the tokenization schemes
of all the above models. This results in `2244` prompts after filtering.
The indices of the filtered prompts is stored in `filtered_indices.npy`

## References

- [DADApy](https://github.com/sissa-data-science/DADApy)
- [GRIDE](https://www.nature.com/articles/s41598-022-20991-1)
- [Transformers Library](https://huggingface.co/docs/transformers/index)
