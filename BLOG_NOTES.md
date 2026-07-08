# Blog Notes

Working title:

- What I Learned Building an LLM From Scratch
- The LLM Was Mostly Tensor Shapes
- Making the Magic Mechanical

## Thesis

Copying the code was not the lesson. Making the code run, debugging it, and
then refactoring it into a project was the lesson.

Building the model from book code made LLMs feel less like a single mysterious
object and more like a stack of concrete choices: tokenization, tensor shapes,
attention masks, batch dimensions, optimizer state, checkpoints, file paths,
and evaluation loops.

## What Became Real

### Tokens Before Intelligence

Text becomes integer IDs almost immediately. The model does not see prose in
the way I do. It sees token IDs, positions, batches, and tensors. The early
tokenizer work was a useful demystifier because it showed how quickly language
turns into accounting.

Useful artifact:

- `scripts/ch02_tokenization.py`

### Attention Clicked Through Shapes

The attention formulas made more sense once the code had to respect dimensions:
batch, sequence length, embedding dimension, heads, and head dimension. Shape
errors were not distractions from the lesson. They were the lesson.

Useful artifact:

- `book_llm/attention.py`
- `book_llm/model.py`
- `tests/test_model.py`

### Fine-Tuning Was Reframing

The spam classifier made it clear that using a GPT-style model for a new task
can be a matter of changing the target, the head, the data, and which parts of
the model are allowed to move.

Useful artifacts:

- `scripts/ch06_spam_classifier.py`
- `blog-assets/loss-plot.pdf`
- `blog-assets/accuracy-plot.pdf`

### Debugging Was Part Of Understanding

The `classify_review()` tensor-rank bug was a good example. The model expected
a 2D tensor shaped `(batch_size, seq_len)`, but an extra batch dimension caused
the forward pass to fail. Fixing it turned an abstract shape convention into
something I could feel in the code.

Useful artifacts:

- `book_llm/training.py`
- `tests/test_training.py`

### Evaluation Is Not A Checkbox

The instruction-tuning chapter and Ollama scoring step exposed a practical
truth: evaluating generated text is itself a modeling problem. The workflow can
produce scores, but the scores still need judgment.

Useful artifacts:

- `scripts/ch07_instruction_tuning.py`
- `scripts/ch07_ollama_eval.py`

### Refactoring Marked A Change In Understanding

The project began as one long script copied while following the book. After the
first full runs, the code was split into:

- `book_llm/` for reusable model code
- `scripts/` for runnable learning stages
- `tests/` for focused behavior checks
- `walkthroughs/full_book_walkthrough.py` for the original linear reference

That reorganization is part of the story. It shows the difference between code
that teaches linearly and code that can be rerun, inspected, and trusted.

## Blog Assets

- `blog-assets/loss-plot.pdf`
- `blog-assets/accuracy-plot.pdf`

## Commands Worth Mentioning

```powershell
.\.venv-tf\Scripts\python.exe scripts\verify.py
.\.venv-tf\Scripts\python.exe scripts\ch02_tokenization.py
.\.venv-tf\Scripts\python.exe scripts\ch06_spam_classifier.py --data-only
.\.venv-tf\Scripts\python.exe scripts\ch07_instruction_tuning.py --data-only
```

## Attribution

This project follows Sebastian Raschka's book *Build a Large Language Model
(From Scratch)*. The blog post should be framed as a reflection on implementing,
debugging, running, and reorganizing the book code, not as a replacement for
the book.

