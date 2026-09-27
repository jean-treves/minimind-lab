"""Run upstream ``train_pretrain.py`` on Apple MPS with float16 autocast, without editing the submodule.

Upstream picks ``nullcontext()`` for every non-CUDA device (``--device mps`` trains in fp32) and a
CUDA-only ``GradScaler``. This runner swaps those two lines in memory, failing loudly if upstream
changed them, then executes the script as ``__main__``.

Usage: ``DEVICE=mps ENTRY=$PWD/scripts/mps_patch.py zsh scripts/cpu_smoke.sh --dtype float16``
"""

import os
import sys
from pathlib import Path

TRAINER = Path(__file__).resolve().parents[1] / "upstream" / "trainer"
SCRIPT = TRAINER / "train_pretrain.py"
PATCHES = {
    'autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)':
        'autocast_ctx = torch.autocast(device_type="mps", dtype=dtype) if args.device == "mps" '
        'else (nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype))',
    "scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))":
        "scaler = torch.amp.GradScaler('mps' if args.device == 'mps' else 'cuda', "
        "enabled=(args.dtype == 'float16'))",
}


def patched_source(src: str) -> str:
    """Apply :data:`PATCHES` to the upstream source.

    Parameters
    ----------
    src : str
        Content of ``train_pretrain.py``.

    Returns
    -------
    str
        Patched source.

    Raises
    ------
    RuntimeError
        If a patched line does not appear exactly once (upstream changed).
    """
    for old, new in PATCHES.items():
        if src.count(old) != 1:
            raise RuntimeError(f"upstream changed, expected this line once: {old}")
        src = src.replace(old, new)
    return src


def main() -> None:
    code = compile(patched_source(SCRIPT.read_text(encoding="utf-8")), str(SCRIPT), "exec")
    os.chdir(TRAINER)  # upstream resolves ../model from the cwd
    sys.argv = [str(SCRIPT), *sys.argv[1:]]
    exec(code, {"__name__": "__main__", "__file__": str(SCRIPT)})


if __name__ == "__main__":
    main()
