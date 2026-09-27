"""Export MiniMind weights (.pth) to a Qwen3 transformers folder, then import it into Ollama.

Upstream ``scripts/convert_model.py`` has no CLI (hard-coded paths, module-level ``lm_config``);
this wrapper calls its ``convert_torch2transformers`` with explicit paths, without touching the
submodule. Ollama imports Qwen3 safetensors directly, so no llama.cpp step is needed.

Usage: ``uv run python scripts/export_ollama.py out/full_sft_768.pth minimind-jt``
"""

import argparse
import importlib.util
import logging
import os
import subprocess
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
UPSTREAM = LAB / "upstream"
log = logging.getLogger("export_ollama")

# ChatML, as in upstream model/tokenizer_config.json (bos <|im_start|>, eos <|im_end|>), without tools.
MODELFILE = '''FROM {hf_dir}
TEMPLATE """{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{- range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
"""
PARAMETER stop <|im_end|>
PARAMETER temperature 0.7
'''


def to_transformers(weights: Path, hf_dir: Path) -> None:
    """Convert a dense MiniMind-3 checkpoint to a Qwen3 transformers folder (fp16 safetensors).

    Parameters
    ----------
    weights : Path
        Upstream ``*.pth`` state dict (e.g. ``full_sft_768.pth``).
    hf_dir : Path
        Output folder, created by ``save_pretrained``.
    """
    spec = importlib.util.spec_from_file_location("convert_model", UPSTREAM / "scripts" / "convert_model.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    # Same config as upstream's __main__ block (dense 768 x 8).
    conv.lm_config = conv.MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)
    src, dst = str(weights.resolve()), str(hf_dir.resolve())  # before chdir
    cwd = os.getcwd()
    os.chdir(UPSTREAM / "scripts")  # upstream loads the tokenizer from ../model/
    try:
        conv.convert_torch2transformers(src, dst)
    finally:
        os.chdir(cwd)


def ollama_create(name: str, hf_dir: Path) -> None:
    """Write ``hf_dir/Modelfile`` and register the model in the local Ollama.

    Parameters
    ----------
    name : str
        Ollama model name.
    hf_dir : Path
        Folder produced by :func:`to_transformers`.
    """
    modelfile = hf_dir / "Modelfile"
    modelfile.write_text(MODELFILE.format(hf_dir=hf_dir.resolve()))
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights", type=Path)
    parser.add_argument("name")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    hf_dir = LAB / "results" / "hf" / args.name
    to_transformers(args.weights, hf_dir)
    log.info("transformers folder: %s", hf_dir)
    ollama_create(args.name, hf_dir)
    log.info("ollama model: %s", args.name)


if __name__ == "__main__":
    main()
