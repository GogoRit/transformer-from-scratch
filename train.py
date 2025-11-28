import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, casual_mask
from model import Transformer, build_transformer

from tqdm import tqdm
from config import get_config, get_weights_file_path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import warnings

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang] # return the sentence in the given language

def get_or_build_tokenizer(config, ds, lang):
    # get the tokenizer path
    tokenizer_path = Path(config['tokenizer_file'].format(lang=lang))
    if not Path.exists(tokenizer_path):
        # build the tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], vocab_size=config['vocab_size'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        # save the tokenizer
        tokenizer.save(str(tokenizer_path))
    else:
        # load the tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # load the dataset
    ds_raw = load_dataset('Helsinki-NLP/opus-100', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    # get the tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # keep 10% for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # create a custom dataset
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    # Calculate max lengths from the raw dataset (before processing)
    for item in train_ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Maximum length of source text: {max_len_src}")
    print(f"Maximum length of target text: {max_len_tgt}")


    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model'],
        h=config['h'],
        N=config['N'],
        dropout=config['dropout']
    )

    return model

def train_model(config):
    device = config['device']
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Initialize TensorBoard writer
    writer = SummaryWriter(config['experiment_name'].format(lang_src=config['lang_src'], lang_tgt=config['lang_tgt']))

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    for epoch in range(initial_epoch, config['epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch + 1}/{config['epochs']}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            label = batch['label'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            # forward pass
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, vocab_size)

            # compute the loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix(loss=f"{loss.item():6.3f}")

            # log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()
            
            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        # save the model
        model_filename = get_weights_file_path(config, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

        print(f"Model saved to {model_filename}")
    
    writer.close()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)