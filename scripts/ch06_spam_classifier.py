import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import tiktoken
import torch
from torch.utils.data import DataLoader

from book_llm.datasets import (
    SPAM_DATA_URL,
    SpamDataset,
    download_and_unzip_spam_data,
    prepare_spam_csv_splits,
)
from book_llm.generation import text_to_token_ids, token_ids_to_text
from book_llm.gpt2_weights import load_pretrained_gpt2
from book_llm.plotting import plot_values
from book_llm.training import (
    calc_accuracy_loader,
    calc_loss_loader,
    classify_review,
    train_classifier_simple,
)


def prepare_dataloaders(tokenizer, batch_size, num_workers, rebuild_splits=False):
    data_file_path = download_and_unzip_spam_data(
        SPAM_DATA_URL,
        "sms_spam_collection.zip",
        "sms_spam_collection",
        Path("sms_spam_collection") / "SMSSpamCollection.tsv",
    )
    split_paths = [Path("train.csv"), Path("validation.csv"), Path("test.csv")]
    if rebuild_splits or not all(path.exists() for path in split_paths):
        prepare_spam_csv_splits(data_file_path)

    train_dataset = SpamDataset("train.csv", tokenizer=tokenizer, max_length=None)
    val_dataset = SpamDataset(
        "validation.csv",
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,
    )
    test_dataset = SpamDataset(
        "test.csv",
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_dataset, train_loader, val_loader, test_loader


def configure_classifier(model, emb_dim):
    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(in_features=emb_dim, out_features=2)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    return model


def main():
    parser = argparse.ArgumentParser(description="Chapter 6 spam classifier runner.")
    parser.add_argument("--model", default="gpt2-small (124M)")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--rebuild-splits", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--save-model", default="artifacts/checkpoints/review_classifier.pth")
    args = parser.parse_args()

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset, train_loader, val_loader, test_loader = prepare_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        num_workers=0,
        rebuild_splits=args.rebuild_splits,
    )
    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")
    if args.data_only:
        print("Data-only check complete.")
        return

    model, cfg, _settings = load_pretrained_gpt2(args.model)
    model = configure_classifier(model, cfg["emb_dim"])

    sample_ids = tokenizer.encode("Do you have time")
    sample_input = torch.tensor(sample_ids).unsqueeze(0)
    with torch.no_grad():
        outputs = model(sample_input)
    print("Sample output shape:", outputs.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.manual_seed(123)

    print(
        f"Train accuracy ({args.eval_batches} batches): "
        f"{calc_accuracy_loader(train_loader, model, device, num_batches=args.eval_batches) * 100:.2f}%"
    )
    print(
        f"Validation accuracy ({args.eval_batches} batches): "
        f"{calc_accuracy_loader(val_loader, model, device, num_batches=args.eval_batches) * 100:.2f}%"
    )
    print(
        f"Test accuracy ({args.eval_batches} batches): "
        f"{calc_accuracy_loader(test_loader, model, device, num_batches=args.eval_batches) * 100:.2f}%"
    )
    print(
        f"Train loss ({args.eval_batches} batches): "
        f"{calc_loss_loader(train_loader, model, device, num_batches=args.eval_batches, classification=True):.3f}"
    )

    if args.epochs > 0:
        start_time = time.time()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
        train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=args.epochs,
            eval_freq=50,
            eval_iter=args.eval_batches,
        )
        print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

        epochs_tensor = torch.linspace(0, args.epochs, len(train_losses))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
        plot_values(
            epochs_tensor,
            examples_seen_tensor,
            train_losses,
            val_losses,
            output_path="artifacts/plots/loss-plot.pdf",
        )

        epochs_tensor = torch.linspace(0, args.epochs, len(train_accs))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
        plot_values(
            epochs_tensor,
            examples_seen_tensor,
            train_accs,
            val_accs,
            label="accuracy",
            output_path="artifacts/plots/accuracy-plot.pdf",
        )

        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"Saved classifier to {args.save_model}")

    text_1 = "You are a winner you have been specially selected to receive $1000 cash."
    text_2 = "Hey, just wanted to check if we're still on for dinner tonight?"
    print("Spam sample:", classify_review(text_1, model, tokenizer, device, train_dataset.max_length))
    print("Ham sample:", classify_review(text_2, model, tokenizer, device, train_dataset.max_length))

    token_ids = text_to_token_ids("Is this thing on?", tokenizer)
    print("Tokenizer smoke:", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
