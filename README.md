## Mini GPT Implementation 
## Educational Transformer Decoder Model

This project implements a Mini GPT (Generative Pre-trained Transformer) model from scratch using PyTorch.

It demonstrates:

Masked Self-Attention

Multi-Head Attention

Transformer Decoder Blocks

GELU activation

Layer Normalization

Dropout

Positional Embeddings

Autoregressive Text Generation

Attention Visualization

This is a simplified educational version of GPT-style models.

## Objective

To understand how GPT-style decoder-only Transformers work internally by implementing:

Token embeddings

Positional embeddings

Masked self-attention

Feed-forward layers

Residual connections

Attention weight visualization

## Model Architecture
Input Tokens
    
     ↓

Token Embedding
  
     +

Positional Embedding
   
     ↓

[ GPT Block × 2 ]
   
     ↓

LayerNorm
  
     ↓

Linear Projection
  
     ↓

Next Token Prediction

## Key Components

🔹 1. Masked Self-Attention

Prevents the model from seeing future tokens.

mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

This ensures:

Token_t can only attend to tokens ≤ t

🔹 2. Multi-Head Attention

Allows the model to focus on multiple relationships simultaneously.

Hyperparameters:

Embedding Dimension: 32
Number of Heads: 4

🔹 3. Feed-Forward Network

Uses:

Linear layer

GELU activation (used in real GPT)

Dropout

Linear projection

🔹 4. Residual Connections
x = x + Attention
x = x + FeedForward

Improves gradient flow and stability.

🔹 5. Layer Normalization

Applied before attention and feed-forward layers.

🔹 6. Positional Embeddings

Since transformers have no recurrence, positional embeddings provide order information.

## Dataset

Small toy dataset:

deep learning is powerful
deep learning is fun
transformers are powerful
attention is all you need
deep models learn representations
transformers use self attention

Vocabulary is built dynamically from dataset.

## Hyperparameters
Parameter	Value
Embedding Dimension	32
Number of Heads	4
Number of Layers	2
Dropout	0.1
Learning Rate	0.001
Epochs	300

## Training

Loss Function:

CrossEntropyLoss

Optimizer:

Adam

Training is autoregressive:

Input:  word1 word2 word3
Target: word2 word3 word4

## Text Generation

Example:

print(generate("deep"))

Output example:

deep learning is powerful

The model predicts tokens one by one using:

torch.argmax(logits[:, -1, :])

## Attention Visualization

The project includes attention heatmap visualization using:

Matplotlib

Seaborn

plot_attention(attention_weights, layer=0)

This shows:

Which words attend to which words

Internal reasoning of the transformer

## Installation
pip install torch matplotlib seaborn

▶️ How to Run

Open Python or Google Colab

Paste the full script

Run training

Generate text

Visualize attention weights

## Concepts Demonstrated

Transformer architecture

Decoder-only model (GPT style)

Self-attention mechanism

Multi-head attention

Masked attention

Residual connections

GELU activation

Positional encoding

Autoregressive language modeling

Attention heatmap interpretation

## Learning Outcomes

After completing this project, you will understand:

How GPT models work internally

Why masking is required

How attention weights are computed

How transformers generate text

Difference between RNN and Transformer

## Possible Improvements

Add larger dataset

Add sampling instead of argmax

Add temperature sampling

Add top-k / nucleus sampling

Increase model depth

Add batching

Train on GPU

Add perplexity evaluation

Compare with LSTM

## Academic Relevance

Suitable for:

NLP Coursework

Transformer Architecture Lab

Mini GPT Demonstration

Deep Learning Projects

AI Viva Preparation

Resume Portfolio Project
