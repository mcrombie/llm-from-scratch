import json
import urllib.request

import psutil
from tqdm import tqdm

from .datasets import format_input


def check_if_running(process_name):
    process_name = process_name.lower()
    for proc in psutil.process_iter(["name"]):
        name = (proc.info.get("name") or "").lower()
        if process_name in name:
            return True
    return False


def require_ollama_running():
    if not check_if_running("ollama"):
        raise RuntimeError(
            "The ollama server is not running. Please start the server before running this script."
        )


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": 123, "temperature": 0, "num_ctx": 2048},
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data


def score_prompt(entry, json_key):
    return (
        f"Given the input `{format_input(entry)}`"
        f"and correct output `{entry['output']}`,"
        f"score the model response `{entry[json_key]}`"
        f"on a scale from 0 to 100, where 100 is the best score."
        f"Respond with the integer number only."
    )


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        score = query_model(score_prompt(entry, json_key), model=model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score '{score}' to int for entry: {entry}")
    return scores
