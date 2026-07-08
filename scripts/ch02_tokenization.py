import argparse
import re
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
import tiktoken

from book_llm import SimpleTokenizerV2


VERDICT_URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)


def ensure_verdict_text(path):
    path = Path(path)
    if not path.exists():
        urllib.request.urlretrieve(VERDICT_URL, path)
    return path


def build_book_vocab(raw_text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    return {token: integer for integer, token in enumerate(all_tokens)}


def main():
    parser = argparse.ArgumentParser(description="Chapter 2 tokenization demo.")
    parser.add_argument("--text-file", default="the-verdict.txt")
    args = parser.parse_args()

    text_path = ensure_verdict_text(args.text_file)
    raw_text = text_path.read_text(encoding="utf-8")
    print("Total number of characters:", len(raw_text))

    tokenizer = SimpleTokenizerV2(build_book_vocab(raw_text))
    sample = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
    ids = tokenizer.encode(sample)
    print("SimpleTokenizerV2 ids:", ids)
    print("SimpleTokenizerV2 decoded:", tokenizer.decode(ids))

    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    gpt2_ids = gpt2_tokenizer.encode("Akwirw ier.")
    print("GPT-2 ids:", gpt2_ids)
    print("GPT-2 decoded:", gpt2_tokenizer.decode(gpt2_ids))


if __name__ == "__main__":
    main()
