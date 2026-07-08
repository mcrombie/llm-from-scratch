import unittest

import torch

from book_llm.datasets import custom_collate_fn, format_input


class DatasetHelperTests(unittest.TestCase):
    def test_format_input_includes_optional_input(self):
        entry = {
            "instruction": "Translate",
            "input": "hello",
            "output": "hola",
        }

        result = format_input(entry)

        self.assertIn("### Instruction:\nTranslate", result)
        self.assertIn("### Input:\nhello", result)
        self.assertNotIn("hola", result)

    def test_format_input_omits_empty_input_section(self):
        entry = {
            "instruction": "Summarize",
            "input": "",
            "output": "summary",
        }

        self.assertNotIn("### Input", format_input(entry))

    def test_custom_collate_masks_repeated_padding_targets(self):
        inputs, targets = custom_collate_fn(
            [[1, 2], [3]],
            pad_token_id=0,
            ignore_index=-100,
            device="cpu",
        )

        self.assertTrue(torch.equal(inputs, torch.tensor([[1, 2], [3, 0]])))
        self.assertTrue(torch.equal(targets, torch.tensor([[2, 0], [0, -100]])))

    def test_custom_collate_truncates_allowed_max_length(self):
        inputs, targets = custom_collate_fn(
            [[1, 2, 3], [4, 5]],
            pad_token_id=0,
            allowed_max_length=2,
            device="cpu",
        )

        self.assertEqual(tuple(inputs.shape), (2, 2))
        self.assertEqual(tuple(targets.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
