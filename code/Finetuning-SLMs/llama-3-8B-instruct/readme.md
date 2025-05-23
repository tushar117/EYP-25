Finetuning the Llama-3 8B model, please upgrade the transformers package after creating the environment as follows:
```
pip install --upgrade transformers==4.40.0
```

Also Llama3 are gated model you need to request for access then export the HF token before running the finetuning/inference scripts as follows:
```
export HF_TOKEN=<your_key>
```