import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from book_llm.ollama_eval import generate_model_scores, require_ollama_running


def main():
    parser = argparse.ArgumentParser(description="Score instruction-tuned outputs with Ollama.")
    parser.add_argument("--responses-file", default="instruction-data-with-response.json")
    parser.add_argument("--json-key", default="model_response")
    parser.add_argument("--model", default="llama3")
    args = parser.parse_args()

    require_ollama_running()
    data = json.loads(Path(args.responses_file).read_text(encoding="utf-8"))
    scores = generate_model_scores(data, json_key=args.json_key, model=args.model)
    print(f"Number of scores: {len(scores)} of {len(data)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}")


if __name__ == "__main__":
    main()
