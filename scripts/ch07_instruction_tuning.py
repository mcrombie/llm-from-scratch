import argparse
import json
import time
from functools import partial
from pathlib import Path

import _bootstrap  # noqa: F401
import tiktoken
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from book_llm.datasets import (
    InstructionDataset,
    custom_collate_fn,
    download_and_load_json,
    format_input,
)
from book_llm.generation import generate, text_to_token_ids, token_ids_to_text
from book_llm.gpt2_weights import load_pretrained_gpt2
from book_llm.plotting import plot_losses
from book_llm.training import calc_loss_loader, train_model_simple


INSTRUCTION_DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)


def load_instruction_splits(path, url=INSTRUCTION_DATA_URL):
    data = download_and_load_json(path, url)
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    return train_data, val_data, test_data


def build_loaders(train_data, val_data, test_data, tokenizer, device, batch_size):
    collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024,
    )
    train_loader = DataLoader(
        dataset=InstructionDataset(train_data, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=InstructionDataset(val_data, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=InstructionDataset(test_data, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def generate_responses(model, tokenizer, test_data, cfg, device, output_path):
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=cfg["context_length"],
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        test_data[i]["model_response"] = response_text

    Path(output_path).write_text(json.dumps(test_data, indent=4), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Chapter 7 instruction tuning runner.")
    parser.add_argument("--data-file", default="instruction-data.json")
    parser.add_argument("--model", default="gpt2-medium (355M)")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--generate-responses", action="store_true")
    parser.add_argument("--responses-file", default="instruction-data-with-response.json")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    tokenizer = tiktoken.get_encoding("gpt2")
    train_data, val_data, test_data = load_instruction_splits(args.data_file)
    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print("Example input:\n", format_input(train_data[0]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print("Device:", device)

    train_loader, val_loader, test_loader = build_loaders(
        train_data,
        val_data,
        test_data,
        tokenizer,
        device,
        args.batch_size,
    )
    for inputs, targets in train_loader:
        print("Train batch:", inputs.shape, targets.shape)
        break
    if args.data_only:
        print(f"Validation loader batches: {len(val_loader)}")
        print(f"Test loader batches: {len(test_loader)}")
        print("Data-only check complete.")
        return

    model, cfg, _settings = load_pretrained_gpt2(args.model)
    model.to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")

    input_text = format_input(val_data[0])
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=35,
        context_size=cfg["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(generated_text[len(input_text):].strip())

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, args.eval_batches)
        val_loss = calc_loss_loader(val_loader, model, device, args.eval_batches)
    print(f"Train loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")

    if args.epochs > 0:
        start_time = time.time()
        torch.manual_seed(123)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
        train_losses, val_losses, tokens_seen = train_model_simple(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=args.epochs,
            eval_freq=5,
            eval_iter=args.eval_batches,
            start_context=format_input(val_data[0]),
            tokenizer=tokenizer,
        )
        print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")
        epochs_tensor = torch.linspace(0, args.epochs, len(train_losses))
        plot_losses(
            epochs_tensor,
            tokens_seen,
            train_losses,
            val_losses,
            output_path="artifacts/plots/instruction-loss-plot.pdf",
        )

        checkpoint = args.checkpoint or f"artifacts/checkpoints/{args.model.replace(' ', '').replace('(', '').replace(')', '')}-sft.pth"
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        print(f"Saved model as {checkpoint}")

    if args.generate_responses:
        generate_responses(model, tokenizer, test_data, cfg, device, args.responses_file)
        print(f"Wrote responses to {args.responses_file}")

    print(f"Test loader batches: {len(test_loader)}")


if __name__ == "__main__":
    main()
