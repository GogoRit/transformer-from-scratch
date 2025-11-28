# Transformer from Scratch

Implementation of the Transformer architecture from the paper "Attention is All You Need" (Vaswani et al., 2017).

## Overview

Building a neural machine translation system from scratch using PyTorch. The implementation includes the core transformer components: multi-head attention, positional encoding, encoder-decoder architecture, and feed-forward networks.

## Current Status

- Transformer model implementation (encoder-decoder architecture)
- Training pipeline with validation and BLEU scoring
- English-Italian translation model training
- Greedy decoding for inference

## End Goal

Deploy a multi-language translation service supporting multiple language pairs (en-it, en-fr, it-en, etc.) with:
- Beam search decoding
- REST API deployment
- Model optimization and quantization
- Production-ready inference pipeline

## Project Structure

- `model.py` - Transformer architecture implementation
- `train.py` - Training script with validation
- `dataset.py` - Bilingual dataset and data processing
- `config.py` - Model and training configuration
- `tokenizer-*.json` - Trained tokenizers for source/target languages

## Training

```bash
python train.py
```

Training progress is logged to TensorBoard in `runs/` directory.

## Dependencies

See `requirements.txt` for full list. Main dependencies:
- PyTorch
- HuggingFace datasets and tokenizers
- TensorBoard for logging
- SacreBLEU for evaluation

