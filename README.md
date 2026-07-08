Learning to Build a Large Language Model from Scratch

This repo follows Sebastian Raschka's book examples, but the reusable code has
been split out of the original notebook-style walkthrough into an importable
`book_llm` package.

The purpose of this repo is not to become a production LLM framework. It is a
learning project: code from the book was written out, run locally, debugged,
and then reorganized into smaller pieces so each stage can be inspected and
rerun.

The companion blog post should use this repo as evidence of that learning
process: tokenization, tensor shapes, attention, training loops, fine-tuning,
evaluation, and the engineering work needed to make the examples repeatable.

## Layout

- `book_llm/` contains reusable model, attention, dataset, training,
  generation, plotting, GPT-2 weight-loading, and Ollama evaluation helpers.
- `scripts/` contains runnable stage scripts for the book exercises.
- `tests/` contains focused smoke/unit tests for the refactored helpers.
- `walkthroughs/full_book_walkthrough.py` preserves the original linear
  full-book script for reference.
- `blog-assets/` contains plots retained for the blog post.
- `BLOG_NOTES.md` captures the post thesis, evidence, and likely structure.

The known-good local runtime is:

```powershell
.\.venv-tf\Scripts\python.exe
```

## Attribution

This project follows Sebastian Raschka's *Build a Large Language Model (From
Scratch)*. The code and notes here are a personal implementation, debugging,
and refactoring record rather than a substitute for the book.

## Useful Commands

Run the standard verification pass:

```powershell
.\.venv-tf\Scripts\python.exe scripts\verify.py
```

Run the focused tests:

```powershell
.\.venv-tf\Scripts\python.exe -m unittest discover -s tests
```

Run the lightweight refactor smoke:

```powershell
.\.venv-tf\Scripts\python.exe scripts\smoke_refactor.py
```

Run chapter 2 tokenization:

```powershell
.\.venv-tf\Scripts\python.exe scripts\ch02_tokenization.py
```

Inspect the chapter 6 spam-classifier data pipeline:

```powershell
.\.venv-tf\Scripts\python.exe scripts\ch06_spam_classifier.py --data-only
```

Train the spam classifier:

```powershell
.\.venv-tf\Scripts\python.exe scripts\ch06_spam_classifier.py --epochs 5
```

Inspect the chapter 7 instruction-tuning data pipeline:

```powershell
.\.venv-tf\Scripts\python.exe scripts\ch07_instruction_tuning.py --data-only
```

Score generated instruction responses with Ollama:

```powershell
.\.venv-tf\Scripts\python.exe scripts\ch07_ollama_eval.py --model llama3
```
