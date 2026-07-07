import os
import re
import sys
import urllib.request 
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizers import SimpleTokenizerV2
from dataset import create_dataloader_v1

import urllib.request 

import numpy as np 

import json

from functools import partial



URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)
FILE_PATH = "the-verdict.txt"

if not os.path.exists(FILE_PATH):
    urllib.request.urlretrieve(URL, FILE_PATH)

import torch.nn as nn 

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import zipfile
import os
from pathlib import Path

import pandas as pd

from gpt_download import download_and_load_gpt2
# GPTModel and load_weights_into_gpt are defined locally below

SPAM_DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
    

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
    
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values 
        return context_vec
    

class MultiHeadAttentionWrapper(nn.Module): 
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
    
    def forward(self, x):return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False): 
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads #1
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) #2         
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x): 
        b, num_tokens, d_in = x.shape 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size     
    "context_length": 256, # Context length     
    "emb_dim": 768, # Embedding dimension     
    "n_heads": 12, # Number of attention heads     
    "n_layers": 12, # Number of layers     
    "drop_rate": 0.1, # Dropout rate     
    "qkv_bias": False # Query-Key-Value bias 
    }

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
              )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, in_idx): 
        batch_size, seq_len = in_idx.shape 
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
        
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x):
        return x 

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x 
    

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift 


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),)
    def forward(self, x): 
        return self.layers(x) 
    

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), 
                          GELU()), 
                          nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()), 
                          nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()), 
                          nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()), 
                          nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())]) 
    def forward(self, x): 
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else: 
                x = layer_output
        return x
    

def print_gradients(model, x):
    output = model(x) 
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")



class TransformerBlock(nn.Module):
    def __init__(self, cfg): 
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], 
            d_out=cfg["emb_dim"], 
            context_length=cfg["context_length"], 
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"], 
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x 
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x 
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) 
        self.final_norm = LayerNorm(cfg["emb_dim"]) 
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape 
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits 
    
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        if max_length is None:
            self.max_length = self.longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        
        self.encoded_texts = [encoded + [pad_token_id] * (self.max_length - len(encoded)) for encoded in self.encoded_texts]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long))
    
    def __len__(self):
        return len(self.data)
    
    def longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if len(encoded_text) > max_length:
                max_length = encoded_length
        return max(len(encoded) for encoded in self.encoded_texts)


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
    

def generate_text_simple(model, idx, 
                         max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def text_to_token_ids(text, tokenizer): 
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# def calc_loss_batch(input_batch, target_batch, model, device): 
#     input_batch = input_batch.to(device)
#     target_batch = target_batch.to(device)
#     logits = model(input_batch) 
#     loss = torch.nn.functional.cross_entropy(
#         logits.flatten(0, 1), target_batch.flatten()
#         )
#     return loss

# def calc_loss_loader(data_loader, model, device, num_batches=None): 
#     total_loss = 0. 
#     if len(data_loader) == 0:
#         return float("nan") 
#     elif num_batches is None: 
#         num_batches = len(data_loader)
#     else: 
#         num_batches = min(num_batches, len(data_loader))
#     for i, (input_batch, target_batch) in enumerate(data_loader): 
#         if i < num_batches: 
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             total_loss += loss.item()
#         else: 
#             break 
#     return total_loss / num_batches

def calc_loss_batch(input_batch, target_batch, model, device, classification=False):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    if classification:
        loss = torch.nn.functional.cross_entropy(logits[:, -1, :], target_batch)
    else:
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None, classification=False):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device, classification=classification)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0,0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def evaluate_model(model, train_loader, val_loader, device, eval_iter, classification=False): 
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter, classification=classification
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter, classification=classification
        )
    model.train() 
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context): 
    model.eval() 
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad(): 
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
            )
    decoded_text = token_ids_to_text(token_ids, tokenizer) 
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model, 
                       train_loader, 
                       val_loader, 
                       optimizer, 
                       device, 
                       num_epochs, 
                       eval_freq, 
                       eval_iter, 
                       start_context, 
                       tokenizer): 
                       
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1 
    for epoch in range(num_epochs):
        model.train() 
        for input_batch, target_batch in train_loader: 
            optimizer.zero_grad() 
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1 
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      ) 
        generate_and_print_sample(model, tokenizer, device, start_context) 
    return train_losses, val_losses, track_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3)) 
    ax1.plot(epochs_seen, train_losses, label="Training loss") 
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) 
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device, classification=True)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter, classification=True
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                          )
                    
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen
    
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    if max_length is None:
        max_length = supported_context_length
    max_length = min(max_length, supported_context_length)

    input_ids = input_ids[:max_length]

    input_ids += [pad_token_id] * (max_length - len(input_ids)) 

    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )   
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry['input'] else ""
    )
    return instruction_text + input_text

def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item) +1 for item in batch)
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item) +1 for item in batch)
    inputs_lst, targets_lst = [],[]

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) +1 for item in batch)
    inputs_lst, targets_lst = [],[]

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def main():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of characters:", len(raw_text))
    print(raw_text[:99])

    # --- SimpleTokenizerV2 with custom vocab ---
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)
    vocab = {token: integer for integer, token in enumerate(all_words)}

    tokenizer = SimpleTokenizerV2(vocab)
    text = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

    # Extend vocab with special tokens
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    print(len(vocab))

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    tokenizer = SimpleTokenizerV2(vocab)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))

    # --- tiktoken GPT-2 tokenizer ---
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Akwirw ier."
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)
    print(tokenizer.decode(integers))

    # --- Context window / next-token prediction ---
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))
    enc_sample = enc_text[50:]
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size + 1]
    print(f"x: {x}")
    print(f"y: {y}")

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

    # --- DataLoader experiments ---
    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)
    data_iter = iter(dataloader)
    print(next(data_iter))
    print(next(data_iter))

    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)

    # --- Embedding layers ---
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(6, 3)
    print(embedding_layer.weight)
    print(embedding_layer(torch.tensor([3])))

    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)
    pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print(pos_embeddings.shape)

    inputs = torch.tensor([[0.43, 0.15, 0.89], 
                            [0.55, 0.87, 0.66], 
                            [0.57, 0.85, 0.64], 
                            [0.22, 0.58, 0.33],
                            [0.77, 0.25, 0.10], 
                            [0.05, 0.80, 0.55]]
    )

    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0]) 
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query) 
    print(attn_scores_2)
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum() 
    print("Attention weights:", attn_weights_2_tmp) 
    print("Sum:", attn_weights_2_tmp.sum())

    def softmax_naive(x): 
        return torch.exp(x) / torch.exp(x).sum(dim=0) 
    attn_weights_2_naive = softmax_naive(attn_scores_2) 
    print("Attention weights:", attn_weights_2_naive) 
    print("Sum:", attn_weights_2_naive.sum())

    attn_weights_2 = torch.softmax(attn_scores_2, dim=0) 
    print("Attention weights:", attn_weights_2) 
    print("Sum:", attn_weights_2.sum())

    query = inputs[1] 
    context_vec_2 = torch.zeros(query.shape) 
    for i,x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i]*x_i 
    print(context_vec_2)

    attn_scores = torch.empty(6, 6) 
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j) 
    print(attn_scores)

    attn_scores = inputs @ inputs.T 
    print(attn_scores)

    attn_weights = torch.softmax(attn_scores, dim=-1) 
    print(attn_weights)

    row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]) 
    print("Row 2 sum:", row_2_sum) 
    print("All row sums:", attn_weights.sum(dim=-1))

    all_context_vecs = attn_weights @ inputs 
    print(all_context_vecs)
    print("Previous 2nd context vector:", context_vec_2)

    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2
    torch.manual_seed(123) 
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) 
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) 
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    query_2 = x_2 @ W_query 
    key_2 = x_2 @ W_key 
    value_2 = x_2 @ W_value
    print(query_2)
    keys = inputs @ W_key 
    values = inputs @ W_value 
    print("keys.shape:", keys.shape) 
    print("values.shape:", values.shape)
    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2) 
    print(attn_score_22) 
    attn_scores_2 = query_2 @ keys.T
    print(attn_scores_2)

    d_k = keys.shape[-1] 
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) 
    print(attn_weights_2)

    context_vec_2 = attn_weights_2 @ values 
    print(context_vec_2)

    torch.manual_seed(123) 
    sa_v1 = SelfAttention_v1(d_in, d_out) 
    print(sa_v1(inputs))

    torch.manual_seed(789) 
    sa_v2 = SelfAttention_v2(d_in, d_out) 
    print(sa_v2(inputs))

    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs) 
    attn_scores = queries @ keys.T 
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) 
    print(attn_weights)

    context_length = attn_scores.shape[0] 
    mask_simple = torch.tril(torch.ones(context_length, context_length)) 
    print(mask_simple)

    masked_simple = attn_weights*mask_simple 
    print(masked_simple)

    row_sums = masked_simple.sum(dim=-1, keepdim=True) 
    masked_simple_norm = masked_simple / row_sums 
    print(masked_simple_norm)

    row_sums = masked_simple.sum(dim=-1, keepdim=True) 
    masked_simple_norm = masked_simple / row_sums 
    print(masked_simple_norm)


    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked)

    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1) 
    print(attn_weights)

    torch.manual_seed(123) 
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)
    print(dropout(example))

    torch.manual_seed(123) 
    print(dropout(attn_weights))

    torch.manual_seed(123) 
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)
    print(dropout(example))

    torch.manual_seed(123) 
    print(dropout(attn_weights)) 

    batch = torch.stack((inputs, inputs), dim=0) 
    print(batch.shape)

    torch.manual_seed(123) 
    context_length = batch.shape[1] 
    ca = CausalAttention(d_in, d_out, context_length, 0.0) 
    context_vecs = ca(batch) 
    print("context_vecs.shape:", context_vecs.shape)

    torch.manual_seed(123) 
    context_length = batch.shape[1] # This is the number of tokens 
    d_in, d_out = 3, 1
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2 ) 
    context_vecs = mha(batch) 
    print(context_vecs) 
    print("context_vecs.shape:", context_vecs.shape)

    a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],[0.8993, 0.0390, 0.9268, 0.7388], [0.7179, 0.7058, 0.9156, 0.4340]], [[0.0772, 0.3565, 0.1479, 0.5331], [0.4066, 0.2318, 0.4545, 0.9737], [0.4606, 0.5159, 0.4220, 0.5786]]]])
    print(a @ a.transpose(2, 3))

    first_head = a[0, 0, :, :] 
    first_res = first_head @ first_head.T 
    print("First head:\n", first_res) 
    
    second_head = a[0, 1, :, :] 
    second_res = second_head @ second_head.T 
    print("\nSecond head:\n", second_res)

    torch.manual_seed(123) 
    batch_size, context_length, d_in = batch.shape 
    d_out = 2 
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2) 
    context_vecs = mha(batch) 
    print(context_vecs) 
    print("context_vecs.shape:", context_vecs.shape)

    tokenizer = tiktoken.get_encoding("gpt2") 
    batch = [] 
    txt1 = "Every effort moves you" 
    txt2 = "Every day holds a" 
    batch.append(torch.tensor(tokenizer.encode(txt1))) 
    batch.append(torch.tensor(tokenizer.encode(txt2))) 
    batch = torch.stack(batch, dim=0) 
    print(batch)


    torch.manual_seed(123) 
    model = DummyGPTModel(GPT_CONFIG_124M) 
    logits = model(batch) 
    print("Output shape:", logits.shape) 
    print(logits)


    torch.manual_seed(123) 
    batch_example = torch.randn(2, 5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU()) 
    out = layer(batch_example) 
    print(out)

    mean = out.mean(dim=-1, keepdim=True) 
    var = out.var(dim=-1, keepdim=True)
    print("Mean:\n", mean) 
    print("Variance:\n", var)

    out_norm = (out - mean) / torch.sqrt(var) 
    mean = out_norm.mean(dim=-1, keepdim=True) 
    var = out_norm.var(dim=-1, keepdim=True) 
    print("Normalized layer outputs:\n", out_norm) 
    print("Mean:\n", mean) 
    print("Variance:\n", var)

    torch.set_printoptions(sci_mode=False) 
    print("Mean:\n", mean) 
    print("Variance:\n", var)

    ln = LayerNorm(emb_dim=6) 
    out_ln = ln(out) 
    mean = out_ln.mean(dim=-1, keepdim=True) 
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean) 
    print("Variance:\n", var)

    gelu, relu = GELU(), nn.ReLU() 
    
    # x = torch.linspace(-3, 3, 100)
    # y_gelu, y_relu = gelu(x), relu(x) 
    # plt.figure(figsize=(8, 3)) 
    # for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    #     plt.subplot(1, 2, i)
    #     plt.plot(x, y) 
    #     plt.title(f"{label} activation function")
    #     plt.xlabel("x")
    #     plt.ylabel(f"{label}(x)")
    #     plt.grid(True) 
    #     plt.tight_layout() 
    #     plt.show()


    ffn = FeedForward(GPT_CONFIG_124M) 
    x = torch.rand(2, 3, 768)
    out = ffn(x) 
    print(out.shape)

    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]]) 
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork( layer_sizes, use_shortcut=False )

    print_gradients(model_without_shortcut, sample_input)


    torch.manual_seed(123) 
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True ) 
    print_gradients(model_with_shortcut, sample_input)


    torch.manual_seed(123) 
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M) 
    output = block(x) 
    print("Input shape:", x.shape) 
    print("Output shape:", output.shape)

    torch.manual_seed(123) 
    model = GPTModel(GPT_CONFIG_124M) 
    out = model(batch) 
    print("Input batch:\n", batch) 
    print("\nOutput shape:", out.shape) 
    print(out)

    total_params = sum(p.numel() for p in model.parameters()) 
    print(f"Total number of parameters: {total_params:,}")

    print("Token embedding layer shape:", model.tok_emb.weight.shape) 
    print("Output layer shape:", model.out_head.weight.shape)
    total_params_gpt2 = (
        total_params - sum(p.numel()
        for p in model.out_head.parameters()) ) 
    print(f"Number of trainable parameters "f"considering weight tying: {total_params_gpt2:,}" )

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

    start_context = "Hello, I am sometimes" 
    encoded = tokenizer.encode(start_context) 
    print("encoded:", encoded) 
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=6, 
        context_size=GPT_CONFIG_124M["context_length"] ) 
    print("Output:", out) 
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist()) 
    print(decoded_text)


    torch.manual_seed(123) 
    model = GPTModel(GPT_CONFIG_124M) 
    model.eval()

    start_context = "Every effort moves you" 
    tokenizer = tiktoken.get_encoding("gpt2") 
    token_ids = generate_text_simple(model=model, idx=text_to_token_ids(start_context, tokenizer), max_new_tokens=10, context_size=GPT_CONFIG_124M["context_length"] ) 
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    inputs = torch.tensor([[16833, 3626, 6100],# ["every effort moves",
                           [40, 1107, 588]]) #  "I really like"]
    
    targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
                            [1107, 588, 11311]]) #  " really like chocolate"]
    
    with torch.no_grad():
        logits = model(inputs) 
    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)

    token_ids = torch.argmax(probas, dim=-1, keepdim=True) 
    print("Token IDs:\n", token_ids)

    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}") 
    print(f"Outputs batch 1:" f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")


    text_idx = 0 
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]] 
    print("Text 1:", target_probas_1) 
    
    text_idx = 1 
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]] 
    print("Text 2:", target_probas_2)

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2))) 
    print(log_probas)

    avg_log_probas = torch.mean(log_probas) 
    print(avg_log_probas)

    neg_avg_log_probas = avg_log_probas * -1 
    print(neg_avg_log_probas)

    print("Logits shape:", logits.shape) 
    print("Targets shape:", targets.shape)

    logits_flat = logits.flatten(0, 1) 
    targets_flat = targets.flatten() 
    print("Flattened logits:", logits_flat.shape) 
    print("Flattened targets:", targets_flat.shape)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat) 
    print(loss)

    file_path = "the-verdict.txt" 
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    total_characters = len(text_data) 
    total_tokens = len(tokenizer.encode(text_data)) 
    print("Characters:", total_characters) 
    print("Tokens:", total_tokens) 

    train_ratio = 0.90 
    split_idx = int(train_ratio * len(text_data)) 
    train_data = text_data[:split_idx] 
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data, 
        batch_size=2, 
        max_length=GPT_CONFIG_124M["context_length"], 
        stride=GPT_CONFIG_124M["context_length"], 
        drop_last=True, 
        shuffle=True, 
        num_workers=0 
    ) 
    val_loader = create_dataloader_v1( 
        val_data, 
        batch_size=2, 
        max_length=GPT_CONFIG_124M["context_length"], 
        stride=GPT_CONFIG_124M["context_length"], 
        drop_last=False, 
        shuffle=False, 
        num_workers=0 
    )

    print("Train loader:") 
    for x, y in train_loader: 
        print(x.shape, y.shape) 
        
    print("\nValidation loader:") 
    for x, y in val_loader: 
        print(x.shape, y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device) 
    print("Training loss:", train_loss) 
    print("Validation loss:", val_loss)


    # TRAINING THE MODEL
    # torch.manual_seed(123) 
    # model = GPTModel(GPT_CONFIG_124M) 
    # model.to(device) 
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1 ) 
    # num_epochs = 10 
    # train_losses, val_losses, tokens_seen = train_model_simple(model, 
    #                                                            train_loader, 
    #                                                            val_loader, 
    #                                                            optimizer, 
    #                                                            device, 
    #                                                            num_epochs=num_epochs, 
    #                                                            eval_freq=5, 
    #                                                            eval_iter=5, 
    #                                                            start_context="Every effort moves you", 
    #                                                            tokenizer=tokenizer)
    

    # epochs_tensor = torch.linspace(0, num_epochs, len(train_losses)) 
    # plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


    # transferring the model back from the GPU to the CPU since inference with a relatively small model does not require a GPU.
    model.to("cpu") 
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2") 
    token_ids = generate_text_simple( model=model, idx=text_to_token_ids("Every effort moves you", tokenizer), max_new_tokens=25, context_size=GPT_CONFIG_124M["context_length"] ) 
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    vocab = {"closer": 0, "every": 1, "effort": 2, "forward": 3, "inches": 4, "moves": 5, "pizza": 6, "toward": 7, "you": 8, } 
    inverse_vocab = {v: k for k, v in vocab.items()}

    next_token_logits = torch.tensor( [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79] )

    probas = torch.softmax(next_token_logits, dim=0) 
    next_token_id = torch.argmax(probas).item() 
    print(inverse_vocab[next_token_id])

    torch.manual_seed(123) 
    next_token_id = torch.multinomial(probas, num_samples=1).item() 
    print(inverse_vocab[next_token_id])

    def print_sampled_tokens(probas): 
        torch.manual_seed(123) 
        sample = [torch.multinomial(probas, num_samples=1).item() 
                  for i in range(1_000)]
        sampled_ids = torch.bincount(torch.tensor(sample))
        for i, freq in enumerate(sampled_ids): 
            print(f"{freq} x {inverse_vocab[i]}") 
            
    print_sampled_tokens(probas) 

    def softmax_with_temperature(logits, temperature): 
        scaled_logits = logits / temperature 
        return torch.softmax(scaled_logits, dim=0)
    

    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print("Top Logits:", top_logits)
    print("Top Positions:", top_pos)

    new_logits = torch.where(
        condition = next_token_logits < top_logits[-1],
        input = torch.tensor(float('-inf')),
        other = next_token_logits
    )

    print(new_logits)
    topk_probas = torch.softmax(new_logits, dim=0)
    print(topk_probas)

    def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:,-1,:]
            if top_k is not None:
                top_logits,_ = torch.topk(logits, top_k)
                min_val = top_logits[:,-1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits,dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            if idx_next == eos_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    torch.manual_seed(123) 
    token_ids = generate(model=model, idx=text_to_token_ids("Every effort moves you", tokenizer), max_new_tokens=15, context_size=GPT_CONFIG_124M["context_length"], top_k=25, temperature=1.4 ) 
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    torch.save(model.state_dict(), "model.pth")

    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),    
    },
    "model_and_optimizer.pth"
    )

    checkpoint = torch.load("model_and_optimizer.pth", map_location=device) 
    model = GPTModel(GPT_CONFIG_124M) 
    model.load_state_dict(checkpoint["model_state_dict"]) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1) 
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 
    model.train();

    gpt_download_url = (
        "https://raw.githubusercontent.com/rasbt/" 
        "LLMs-from-scratch/main/ch05/" 
        "01_main-chapter-code/gpt_download.py" 
        ) 
    filename = gpt_download_url.split('/')[-1] 
    if not os.path.exists(filename):
        urllib.request.urlretrieve(gpt_download_url, filename)
    from gpt_download import download_and_load_gpt2

    settings, params = download_and_load_gpt2(
        model_size = "124M", models_dir = "gpt2"
    )

    print('Settings:', settings)
    print('Parameter dictionary keys:', params.keys())

    print(params['wte'])
    print('Token embedding weight tensor dimensions:', params['wte'].shape)

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}, 
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, 
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20}, 
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}, 
        }
    model_name = "gpt2-small (124M)" 
    NEW_CONFIG = GPT_CONFIG_124M.copy() 
    NEW_CONFIG.update(model_configs[model_name])

    NEW_CONFIG.update({"context_length": 1024}) 

    NEW_CONFIG.update({"qkv_bias": True}) 

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    def assign(left, right):
        if left.shape != right.shape: 
            raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                             "Right: {right.shape}") 
        return torch.nn.Parameter(torch.tensor(right))
    
    def load_weights_into_gpt(gpt, params):
        gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
        gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

        for b in range(len(params["blocks"])):
            block = params["blocks"][b]
            q_w, k_w, v_w = np.split(block["attn"]["c_attn"]["w"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                block["attn"]["c_attn"]["b"], 3, axis=-1) 
            gpt.trf_blocks[b].att.W_query.bias = assign( 
                gpt.trf_blocks[b].att.W_query.bias, q_b) 
            gpt.trf_blocks[b].att.W_key.bias = assign(
                gpt.trf_blocks[b].att.W_key.bias, k_b) 
            gpt.trf_blocks[b].att.W_value.bias = assign(
                gpt.trf_blocks[b].att.W_value.bias, v_b)

            gpt.trf_blocks[b].att.out_proj.weight = assign(
                gpt.trf_blocks[b].att.out_proj.weight, 
                block["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = assign(
                gpt.trf_blocks[b].att.out_proj.bias, 
                block["attn"]["c_proj"]["b"])

            gpt.trf_blocks[b].ff.layers[0].weight = assign(
                gpt.trf_blocks[b].ff.layers[0].weight, block["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = assign(
                gpt.trf_blocks[b].ff.layers[0].bias, block["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = assign(
                gpt.trf_blocks[b].ff.layers[2].weight, block["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = assign(
                gpt.trf_blocks[b].ff.layers[2].bias, block["mlp"]["c_proj"]["b"])

            gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, block["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, block["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, block["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, block["ln_2"]["b"])

        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

    load_weights_into_gpt(gpt, params)
    gpt.to(device)

    torch.manual_seed(123)
    token_ids = generate(
        model=gpt, 
        idx=text_to_token_ids("Every effort moves you", tokenizer), 
        max_new_tokens=25, context_size=NEW_CONFIG["context_length"], 
        top_k=50, temperature=1.5 
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    def download_and_unzip_spam_data(
            url, zip_path, extracted_path, data_file_path):
        if data_file_path.exists():
            print(f"Data file already exists at {data_file_path}. Skipping download and extraction.")
            return
        with urllib.request.urlopen(url) as response:
            with open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_path)

        original_file_path = Path(extracted_path) / "SMSSpamCollection"
        os.rename(original_file_path, data_file_path)
        print(f"File downloaded and saved as {data_file_path}")
    
    download_and_unzip_spam_data(SPAM_DATA_URL, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(data_file_path, sep='\t', header=None, names=['Label', 'Text'])
    df

    print(df["Label"].value_counts())

    def create_balanced_dataset(df):
        num_spam = df[df["Label"] == "spam"].shape[0]
        ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
        balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
        return balanced_df
    
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    def random_split(df, train_frac, validation_frac):
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        train_end = int(len(df) * train_frac)
        validation_end = train_end + int(len(df) * validation_frac)

        train_df = df[:train_end]
        validation_df = df[train_end:validation_end]
        test_df = df[validation_end:]

        return train_df, validation_df, test_df
    
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    train_dataset = SpamDataset(csv_file="train.csv", tokenizer=tokenizer, max_length=None)

    print(train_dataset.max_length)

    val_dataset = SpamDataset(csv_file="validation.csv", tokenizer=tokenizer, max_length=train_dataset.max_length)
    test_dataset = SpamDataset(csv_file="test.csv", tokenizer=tokenizer, max_length=train_dataset.max_length)

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(dataset = val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)

    for input_batch, target_batch in train_loader:
        print("Input batch dimensions:", input_batch.shape)
        print("Target batch dimensions:", target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}, 
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split()[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    text_1 = "Every effort moves you"
    token_ids = generate_text_simple(
        model=model, 
        idx=text_to_token_ids(text_1, tokenizer), 
        max_new_tokens=15, 
        context_size=BASE_CONFIG["context_length"], 
    )
    print(token_ids_to_text(token_ids, tokenizer))

    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        "You are a winner you have been specially"
        "selected to receive $1000 cash or a $2000 award."
    )
    token_ids = generate_text_simple(
        model=model, 
        idx=text_to_token_ids(text_2, tokenizer), 
        max_new_tokens=23, 
        context_size=BASE_CONFIG["context_length"], 
    )
    print(token_ids_to_text(token_ids, tokenizer))


    # Printing the model archtiecture
    # GPTModel(
    #     (tok_emb) = Embedding(50257, 768),
    #     (pos_emb) = Embedding(1024, 768),
    #     (drop_emb) = Dropout(p=0.0, inplace=False)

    # )

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Input Shape:", inputs.shape)

    with torch.no_grad():
        outputs = model(inputs)
    print("Outputs:\n", outputs)
    print("Outputs shape:", outputs.shape)

    print("Last output token:", outputs[:, -1, :])

    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    label = torch.argmax(probas)
    print("Class label:", label.item())

    logits = outputs[:, -1, :]
    label = torch.argmax(logits)
    print("Class label from logits:", label.item())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    print(f"Train accuracy (10 batches): {train_accuracy*100:.2f}%")
    print(f"Validation accuracy (10 batches): {val_accuracy*100:.2f}%")
    print(f"Test accuracy (10 batches): {test_accuracy*100:.2f}%")


    
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5, classification=True)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5, classification=True)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5, classification=True)
    print(f"Train loss (5 batches): {train_loss:.3f}")
    print(f"Validation loss (5 batches): {val_loss:.3f}")
    print(f"Test loss (5 batches): {test_loss:.3f}")

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    train_losses, val_losses, train_accs, val_accs, examples_seen = \
        train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=50, eval_iter=5)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)

    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    text_1 = ("You are a winner you have been specially "
              "selected to receive $1000 cash or a $2000 award.")
    print(classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length))

    text_2 = (
        "Hey, just wanted to check if we're still on "
        "for dinner tonight? Let me know!"
    )

    print(classify_review(text_2, model, tokenizer, device, max_length=train_dataset.max_length))

    torch.save(model.state_dict(), "review_classifier.pth")

    model_state_dict = torch.load("review_classifier.pth", map_location=device)
    model.load_state_dict(model_state_dict)

    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(file_path, url)
    print("Number of Entries:", len(data))

    print("Example entry:\n", data[0])

    print("Another example entry:\n", data[999])

    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print(model_input + desired_response)

    model_input = format_input(data[999])
    desired_response = f"\n\n### Response:\n{data[999]['output']}"
    print(model_input + desired_response)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    tokenizer  = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    inputs_1 = [0,1,2,3,4]
    inputs_2 = [5,6]
    inputs_3 = [7,8,9]
    batch = (inputs_1, inputs_2, inputs_3)
    print(custom_collate_draft_1(batch))

    inputs, targets = custom_collate_draft_2(batch)
    print(inputs)
    print(targets)


    inputs, targets = custom_collate_fn(batch)
    print(inputs)
    print(targets)

    logits_1 = torch.tensor([[-1.0, 1.0], [-0.5, 1.5]])
    targets_1 = torch.tensor([0,1])
    loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    print(loss_1)

    logits_2 = torch.tensor([[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]])
    targets_2 = torch.tensor([0,1,1])
    loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
    print(loss_2)

    targets_3 = torch.tensor([0,1,-100])
    loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
    print(loss_3)
    print("loss_1 == loss_3:", loss_1 == loss_3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): # uncomment these two lines to use the GPU on Apple Silicon chip
        device = torch.device("mps")
    print("Device:", device)

    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=customized_collate_fn,
        drop_last=True
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=customized_collate_fn,
        drop_last=False
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=customized_collate_fn,
        drop_last=False
    )

    print("Train loader:")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)

    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    settings, params = download_and_load_gpt2(
        model_size = model_size, models_dir = "gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval();

    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    print(input_text)

    token_ids = generate(
        model=model, 
        idx=text_to_token_ids(input_text, tokenizer), 
        max_new_tokens=35, 
        context_size=BASE_CONFIG["context_length"], 
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = generated_text[len(input_text):].strip()
    print(response_text)

    model.to(device)
    torch.manual_seed(123)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5, classification=False)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5, classification=False)

    print(f"Train loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


if __name__ == "__main__":
    main()
