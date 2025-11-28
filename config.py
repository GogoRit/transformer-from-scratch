import torch
from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "seq_len": 350,
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
        "lr": 1e-4,
        "epochs": 50,
        "vocab_size": 37000,
        "preload": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = model_basename.format(lang_src=config['lang_src'], lang_tgt=config['lang_tgt'], epoch=epoch)
    return str(Path('.') / model_folder / model_filename)