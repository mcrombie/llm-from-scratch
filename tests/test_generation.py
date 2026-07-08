import unittest

import torch

from book_llm.generation import generate


class EosModel(torch.nn.Module):
    def __init__(self, vocab_size=5, eos_id=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_id = eos_id

    def forward(self, idx):
        logits = torch.zeros(idx.shape[0], idx.shape[1], self.vocab_size)
        logits[:, -1, self.eos_id] = 10.0
        return logits


class GenerationTests(unittest.TestCase):
    def test_generate_stops_before_appending_eos(self):
        model = EosModel()
        start = torch.tensor([[1, 2]])

        result = generate(
            model,
            start,
            max_new_tokens=5,
            context_size=8,
            eos_id=4,
        )

        self.assertTrue(torch.equal(result, start))

    def test_generate_appends_argmax_token_without_eos(self):
        model = EosModel()
        start = torch.tensor([[1, 2]])

        result = generate(model, start, max_new_tokens=2, context_size=8)

        self.assertEqual(tuple(result.shape), (1, 4))
        self.assertEqual(result[0, -1].item(), 4)


if __name__ == "__main__":
    unittest.main()
