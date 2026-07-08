import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PYTHON_FILES = [
    "dataset.py",
    "tokenizers.py",
    "book_llm/__init__.py",
    "book_llm/attention.py",
    "book_llm/datasets.py",
    "book_llm/generation.py",
    "book_llm/gpt2_weights.py",
    "book_llm/model.py",
    "book_llm/ollama_eval.py",
    "book_llm/plotting.py",
    "book_llm/tokenization.py",
    "book_llm/training.py",
    "scripts/_bootstrap.py",
    "scripts/ch02_tokenization.py",
    "scripts/ch06_spam_classifier.py",
    "scripts/ch07_instruction_tuning.py",
    "scripts/ch07_ollama_eval.py",
    "scripts/smoke_refactor.py",
    "scripts/verify.py",
    "walkthroughs/full_book_walkthrough.py",
]


def run_step(name, command):
    print(f"\n== {name} ==", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, stderr=subprocess.STDOUT)


def main():
    python = sys.executable
    run_step("py_compile", [python, "-m", "py_compile", *PYTHON_FILES])
    run_step("unit tests", [python, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run_step("refactor smoke", [python, "scripts/smoke_refactor.py"])
    run_step(
        "chapter 6 data pipeline",
        [python, "scripts/ch06_spam_classifier.py", "--data-only"],
    )
    run_step(
        "chapter 7 data pipeline",
        [python, "scripts/ch07_instruction_tuning.py", "--data-only"],
    )
    print("\nVerification passed.", flush=True)


if __name__ == "__main__":
    main()
