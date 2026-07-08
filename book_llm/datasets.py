import json
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


SPAM_DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        if max_length is None:
            self.max_length = self.longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded + [pad_token_id] * (self.max_length - len(encoded))
            for encoded in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def longest_encoded_length(self):
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


def download_and_load_json(file_path, url):
    file_path = Path(file_path)
    if not file_path.exists():
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        file_path.write_text(text_data, encoding="utf-8")
    return json.loads(file_path.read_text(encoding="utf-8"))


def download_and_unzip_spam_data(
    url=SPAM_DATA_URL,
    zip_path="sms_spam_collection.zip",
    extracted_path="sms_spam_collection",
    data_file_path=None,
):
    if data_file_path is None:
        data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    data_file_path = Path(data_file_path)

    if data_file_path.exists():
        print(f"Data file already exists at {data_file_path}. Skipping download and extraction.")
        return data_file_path

    zip_path = Path(zip_path)
    extracted_path = Path(extracted_path)
    with urllib.request.urlopen(url) as response:
        zip_path.write_bytes(response.read())
    print(f"File downloaded and saved as {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    return data_file_path


def prepare_spam_csv_splits(
    data_file_path,
    train_csv="train.csv",
    validation_csv="validation.csv",
    test_csv="test.csv",
    seed=123,
    train_frac=0.7,
    validation_frac=0.1,
):
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

    spam_subset = df[df["Label"] == 1]
    ham_subset = df[df["Label"] == 0].sample(len(spam_subset), random_state=seed)
    balanced_df = pd.concat([ham_subset, spam_subset])
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_end = int(len(balanced_df) * train_frac)
    validation_end = train_end + int(len(balanced_df) * validation_frac)
    train_df = balanced_df[:train_end]
    validation_df = balanced_df[train_end:validation_end]
    test_df = balanced_df[validation_end:]

    train_df.to_csv(train_csv, index=None)
    validation_df.to_csv(validation_csv, index=None)
    test_df.to_csv(test_csv, index=None)
    return train_df, validation_df, test_df


def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs_lst.append(torch.tensor(padded[:-1]))

    return torch.stack(inputs_lst).to(device)


def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs_lst.append(torch.tensor(padded[:-1]))
        targets_lst.append(torch.tensor(padded[1:]))

    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu",
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
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

    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)
