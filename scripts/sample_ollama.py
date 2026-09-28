"""Ask the local Ollama model a few fixed prompts and write the answers to ``results/samples.md``.

Usage: ``uv run python scripts/sample_ollama.py minimind-jt`` (Ollama on 127.0.0.1:11434).
"""

import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

OLLAMA = "http://127.0.0.1:11434/api/chat"
OUT = Path(__file__).resolve().parents[1] / "results" / "samples.md"
PROMPTS = [
    "你好，请介绍一下你自己。",
    "为什么天空是蓝色的？",
    "请用三句话解释什么是机器学习。",
    "Hello! What can you do?",
    "Bonjour, qui es-tu ?",
]
OPTIONS = {"seed": 42, "temperature": 0.7, "num_predict": 160}


def ask(model: str, prompt: str) -> dict:
    """Send one chat turn to Ollama.

    Parameters
    ----------
    model : str
        Ollama model name.
    prompt : str
        User message.

    Returns
    -------
    dict
        Ollama's JSON reply (``message.content``, ``eval_count``, ``eval_duration`` in ns).
    """
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": OPTIONS}
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def main(model: str) -> None:
    lines = [f"# Samples from `{model}`", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} with Ollama, options `{json.dumps(OPTIONS)}`.", ""]
    for prompt in PROMPTS:
        r = ask(model, prompt)
        rate = r["eval_count"] / (r["eval_duration"] / 1e9)
        lines += [f"## {prompt}", "", r["message"]["content"].strip(), "",
                  f"*{r['eval_count']} tokens, {rate:.0f} tokens/s*", ""]
    OUT.write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "minimind-jt")
