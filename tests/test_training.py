import unittest

import torch

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


class TrainingHelperTests(unittest.TestCase):
    def test_classify_review_clamps_length_and_uses_single_batch_dim(self):
        model = TinyClassifier()

        label = classify_review(
            "hello",
            model,
            TinyTokenizer(),
            torch.device("cpu"),
            max_length=10,
        )

        self.assertEqual(label, "spam")
        self.assertEqual(model.seen_shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
