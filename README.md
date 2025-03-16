# Fine-Tuning GPT-2 for Summarization with Prompt Tuning, LoRA, and Traditional Fine-Tuning


## Introduction

This project focuses on fine-tuning the GPT-2 small model for text summarization using three different fine-tuning methods:

- Prompt Tuning: Uses soft prompts to guide the model without updating its core parameters.

- LoRA (Low-Rank Adaptation): Introduces low-rank matrices to fine-tune model parameters efficiently.

- Traditional Fine-Tuning (Last Layers Only): Updates only the last few layers of GPT-2 while keeping the rest frozen.

## Project Objectives

The main objectives of this project are:

- Implement and compare the three fine-tuning methods.

- Evaluate their performance on the CNN/Daily Mail summarization dataset.

- Compare evaluation loss, ROUGE scores, and efficiency (training time, memory usage, etc.).

## Setup

### Prerequisites

Ensure you have the following installed:

`pip install torch transformers datasets peft`

### Loading GPT-2

We use the Hugging Face transformers library to load the GPT-2 model and tokenizer.

## Implementation Details

### Prompt Tuning

- Uses an additional soft prompt embedding layer.

- The soft prompt embeddings are updated during training, but GPT-2 remains frozen.

- Implementation follows: "The Power of Scale for Parameter-Efficient Prompt Tuning".

### LoRA (Low-Rank Adaptation)

- Introduces trainable low-rank matrices into GPT-2 without updating the full model.

- LoRA allows adapting the model efficiently with minimal parameters.

- Implemented using PEFT (Parameter-Efficient Fine-Tuning) library.

### Traditional Fine-Tuning

- Freezes all layers except the last few layers of GPT-2.

- Fine-tunes only the language modeling head.

- Requires fewer additional parameters than full fine-tuning but more than LoRA.

## Dataset

- Dataset: CNN/Daily Mail summarization dataset.

- Preprocessing: Tokenization using GPT-2 tokenizer.

- Train/Validation/Test Split: 21k/6k/3k samples.

## Training Details

- Batch Size: Variable (depends on available GPU memory, with gradient accumulation used if needed).

- Epochs: Up to 10 (early stopping based on validation loss).

- Gradient Clipping: Applied to prevent exploding gradients (clip_norm=1.0).

- Optimization: AdamW optimizer with a learning rate scheduler.

## Evaluation Metrics

The models are evaluated based on:

- Evaluation Loss

- ROUGE Score (ROUGE-1, ROUGE-2, ROUGE-L)


## Model links

- Drive link for soft-prompt model: https://drive.google.com/file/d/1KQMGQAt867Gj3icbRd5ajACnbIzWSkPG/view?usp=sharing

- Drive link for lora fine tunned model: https://drive.google.com/file/d/1_Ixp702mxQztmsU8_NniFk84Vqy205Ei/view?usp=sharing

- Drive link for traditional fine tunned model: https://drive.google.com/file/d/1Om8kuaHVWm98B7v_fwWDrD55xDOE-vbK/view?usp=sharing

- Drive link for prompt_tunning notebook: https://drive.google.com/file/d/1rrl4Xmwhph0RqQns_i-HwcGk0dA2974N/view?usp=drive_link

- Drive link for lora fine tunning notebook: https://drive.google.com/file/d/19rySi0HDtCbSACkEaW-2vD1ttemOuDam/view?usp=drive_link

- Drive link for traditional fine tunning notebook: https://drive.google.com/file/d/19rySi0HDtCbSACkEaW-2vD1ttemOuDam/view?usp=drive_link
