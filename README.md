# SMS Spam Detection using Llama 3.2 3B Instruct and QLoRA

An end-to-end LLM fine-tuning project exploring how Parameter-Efficient Fine-Tuning (PEFT) techniques can adapt a pretrained Llama model for SMS spam classification.

The project uses the Kaggle SMS Spam Collection dataset and implements supervised fine-tuning using Quantized Low-Rank Adaptation (QLoRA).

## Project Overview

The objective is to fine-tune a pretrained Llama 3.2 3B Instruct model to classify SMS messages into two categories:

- **Ham** — Legitimate SMS messages
- **Spam** — Unsolicited or spam messages

The project focuses on understanding and implementing the complete LLM fine-tuning pipeline independently, including dataset preparation, instruction formatting, quantization, LoRA configuration, training, inference, and evaluation.

## Dataset

**Dataset:** SMS Spam Collection Dataset

**Source:** Kaggle

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

The dataset contains SMS messages labeled as ham or spam.

| Column | Description |
|---|---|
| `v1` | Message classification label |
| `v2` | SMS message text |
| `Unnamed: 2` | Additional dataset column |
| `Unnamed: 3` | Additional dataset column |
| `Unnamed: 4` | Additional dataset column |

Only the classification label and SMS message text are required for this project.

## Base Model

**Model:** Meta Llama 3.2 3B Instruct

https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

The model will be loaded using 4-bit quantization to reduce GPU memory requirements.

## Fine-Tuning Approach

The project implements the following techniques:

- Supervised Fine-Tuning (SFT)
- Parameter-Efficient Fine-Tuning (PEFT)
- Low-Rank Adaptation (LoRA)
- 4-bit Quantization using BitsAndBytes
- Quantized Low-Rank Adaptation (QLoRA)
- Hugging Face Transformers
- Hugging Face TRL SFTTrainer

## Implementation Roadmap

- [ ] Environment setup and GPU verification
- [ ] Dataset exploration and analysis
- [ ] Data cleaning and preprocessing
- [ ] Train, validation, and test dataset splitting
- [ ] Instruction dataset preparation
- [ ] Base model loading and quantization
- [ ] LoRA configuration
- [ ] QLoRA fine-tuning
- [ ] Model inference
- [ ] Base model vs fine-tuned model evaluation
- [ ] Final performance analysis
- [ ] Model adapter saving and loading

## Evaluation

The fine-tuned model will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Special attention will be given to spam detection recall and false-negative classification.

## Repository Structure

```text
sms-spam-llama-qlora/
│
├── notebooks/
│   └── sms_spam_qlora_finetuning.ipynb
│
├── data/
│   └── README.md
│
├── outputs/
│   └── evaluation_results/
│
├── src/
│   └── inference.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Learning Objectives

This project aims to develop practical understanding of:

- Transformer-based language models
- Tokenization and model inputs
- Instruction formatting
- Supervised Fine-Tuning
- 4-bit quantization
- LoRA adapters and trainable parameters
- QLoRA training configuration
- Model inference and evaluation
- GPU memory optimization

## Development Philosophy

This project is implemented incrementally with a learning-first approach.

The implementation is developed manually, with AI coding assistants used primarily for explanations, code reviews, debugging, and targeted code suggestions rather than complete automated implementations.

## License

This project is released under the MIT License.

## Acknowledgements

- Kaggle SMS Spam Collection Dataset
- Meta AI — Llama 3.2
- Hugging Face Transformers
- Hugging Face PEFT
- Hugging Face TRL
- BitsAndBytes
