import torch

import _bootstrap  # noqa: F401
from book_llm.generation import generate
from book_llm.model import GPTModel
from book_llm.training import classify_review


class TinyTokenizer:
    def encode(self, text):
        return [1, 2, 3]


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = torch.nn.Embedding(4, 2)
        self.seen_shape = None

    def forward(self, input_tensor):
        self.seen_shape = tuple(input_tensor.shape)
        logits = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], 2)
        logits[:, -1, 1] = 1.0
        return logits


def main():
    tiny_cfg = {
        "vocab_size": 16,
        "context_length": 8,
        "emb_dim": 8,
        "n_heads": 2,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(tiny_cfg)
    out = model(torch.tensor([[1, 2, 3]]))
    print("GPTModel output shape:", tuple(out.shape))

    generated = generate(
        model,
        torch.tensor([[1, 2]]),
        max_new_tokens=2,
        context_size=tiny_cfg["context_length"],
    )
    print("Generated shape:", tuple(generated.shape))

    classifier = TinyClassifier()
    label = classify_review("hello", classifier, TinyTokenizer(), torch.device("cpu"), max_length=10)
    print("Classification label:", label)
    print("Classification tensor shape:", classifier.seen_shape)


if __name__ == "__main__":
    main()
