---
base_model: mdmy/qwen2-vl-audio-projection-sft
library_name: transformers
model_name: e2e_sft_post_audio_projection_sft
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for e2e_sft_post_audio_projection_sft

This model is a fine-tuned version of [mdmy/qwen2-vl-audio-projection-sft](https://huggingface.co/mdmy/qwen2-vl-audio-projection-sft).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="mdmy/e2e_sft_post_audio_projection_sft", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/paypal/qwen-asr-finetuning/runs/qqo0qwdj) 


This model was trained with SFT.

### Framework versions

- TRL: 0.17.0
- Transformers: 4.47.0
- Pytorch: 2.4.1
- Datasets: 4.3.0
- Tokenizers: 0.21.4

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```