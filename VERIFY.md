# Verification

Use the project venv:

```powershell
.\.venv-tf\Scripts\python.exe scripts\verify.py
```

The verifier runs:

- `py_compile` on the package, compatibility shims, scripts, and walkthrough
- the focused `unittest` suite
- `scripts/smoke_refactor.py`
- `scripts/ch06_spam_classifier.py --data-only`
- `scripts/ch07_instruction_tuning.py --data-only`

The verifier intentionally does not train models or call Ollama. Those paths
are slower and should be run deliberately when needed:

```powershell
.\.venv-tf\Scripts\python.exe scripts\ch06_spam_classifier.py --epochs 5
.\.venv-tf\Scripts\python.exe scripts\ch07_instruction_tuning.py --epochs 2
.\.venv-tf\Scripts\python.exe scripts\ch07_ollama_eval.py --model llama3
```

