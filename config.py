import torch
from pathlib import Path

def get_config():
    return {
        "batch_size": 32,  # Reduced from 64 to avoid CUDA OOM while still faster than 8
        "seq_len": 400,  # Filter out sequences longer than this during training
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel-{lang_src}-{lang_tgt}-{epoch:02d}.pt",
        "tokenizer_file": "tokenizer-{lang}.json",
        "experiment_name": "runs/tmodel-{lang_src}-{lang_tgt}",
        "h": 8,
        "N": 6,
        "dropout": 0.1,
        "lr": 1e-4,  # Consider increasing slightly if you push batch_size much higher (e.g. 128+)
        "epochs": 3,  # Fewer epochs for quicker test runs; increase later if needed
        "max_samples": 50000,  # Limit total examples for quick tests; set to None to use full dataset
        "vocab_size": 37000,
        "preload": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = model_basename.format(lang_src=config['lang_src'], lang_tgt=config['lang_tgt'], epoch=epoch)
    return str(Path('.') / model_folder / model_filename)