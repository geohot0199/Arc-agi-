#!/usr/bin/env python3
"""Build the Kaggle submission notebook for ATLAS.

Follows the exact pattern of the official ARC-AGI-3-Kaggle-Starter
(build_notebook.py), which itself mirrors Kaggle's official sample:

  Cell 1: install arc-agi + python-dotenv offline from the competition wheels.
  Cell 2: (rerun only) bring up the local vLLM OpenAI server for the model,
          degrading gracefully to reflex-only if vLLM is unavailable.
  Cell 3: write the agent to /tmp/my_agent.py (NOT /kaggle/working — it must
          not appear as an output file next to submission.parquet).
  Cell 4: competition-rerun cell: wait for the gateway sidecar, copy the
          ARC-AGI-3-Agents framework into /kaggle/working, register MyAgent
          as 'myagent', point .env at the gateway (plus ATLAS_* flags), run
          `python main.py --agent myagent`. The gateway records every action
          and emits submission.parquet.
  Cell 5: commit-mode cell: dummy submission.parquet so Save & Run All passes.

Usage:  python scripts/build_notebook.py  ->  notebooks/submission.ipynb
        (also refreshes notebooks/kernel-metadata.json for `kaggle kernels push`)
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# CONFIG — edit these two lines only
# ---------------------------------------------------------------------------
ACCELERATOR = "rtx6000"  # cpu | t4 | p100 | rtx6000
MODEL_DIR = (
    "/kaggle/input/models/keithtyser/"
    "qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"

_ACCELERATORS = {
    "cpu": {"name": "none", "gpu": False},
    "t4": {"name": "nvidiaTeslaT4", "gpu": True},
    "p100": {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000", "gpu": True},
}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


INSTALL_CELL = (
    "!pip install --no-index --find-links \\\n"
    "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
    "    arc-agi python-dotenv"
)

# Rerun-only: serve the model locally with vLLM. Every failure path ends with
# ATLAS_LLM=0 so the agent still plays (reflex mode) and the submission is
# still produced. Phase-0 guarantee: the notebook never dies here.
SERVE_CELL = dedent(
    f"""\
    import glob, os, subprocess, sys, time, urllib.request

    MODEL_DIR = {MODEL_DIR!r}
    RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
    os.environ['ATLAS_LLM'] = '0'

    if RERUN and os.path.isdir(MODEL_DIR):
        have_vllm = False
        try:
            import vllm  # noqa: F401
            have_vllm = True
        except Exception:
            pass
        if not have_vllm:
            # Try offline install from any wheels shipped in attached inputs.
            roots = set()
            for pat in ('/kaggle/input/**/vllm*.whl', '/kaggle/input/**/*vllm*'):
                for p in glob.glob(pat, recursive=True):
                    roots.add(os.path.dirname(p))
            for r in sorted(roots):
                os.system(f'pip install --no-index --find-links "{{r}}" vllm >> /kaggle/working/pip.log 2>&1')
                try:
                    import vllm  # noqa: F401
                    have_vllm = True
                    break
                except Exception:
                    continue
        print('vllm available:', have_vllm)
        if have_vllm:
            # The CUDA driver libs are not on the default search path in the
            # competition image; vLLM fails to load without this.
            os.environ['LIBRARY_PATH'] = (
                '/usr/local/nvidia/lib64:' + os.environ.get('LIBRARY_PATH', ''))
            log = open('/kaggle/working/vllm.log', 'w')
            cmd = [
                sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
                '--model', MODEL_DIR,
                '--served-model-name', 'atlas',
                '--host', '127.0.0.1', '--port', '8000',
                '--dtype', 'bfloat16',
                '--quantization', 'modelopt_fp4',
                '--max-model-len', '32768',
                '--max-num-seqs', '8',
                '--gpu-memory-utilization', '0.90',
                '--kv-cache-dtype', 'auto',
                '--enable-chunked-prefill',
                '--no-enable-prefix-caching',
                '--tool-call-parser', 'qwen3_coder',
                '--reasoning-parser', 'qwen3',
            ]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            print('vLLM starting (pid', proc.pid, ') — waiting for /v1/models ...')
            deadline = time.time() + 1500  # 25 min model-load budget
            ok = False
            while time.time() < deadline:
                try:
                    urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5)
                    ok = True
                    break
                except Exception:
                    time.sleep(10)
            os.environ['ATLAS_LLM'] = '1' if ok else '0'
            if not ok:
                print('vLLM did not come up in time; ATLAS_LLM=0 (reflex mode)')
        else:
            print('vLLM not installable offline; ATLAS_LLM=0 (reflex mode)')
    else:
        print('commit mode or model missing — skipping model serve (ATLAS_LLM=0)')
    print('ATLAS_LLM =', os.environ['ATLAS_LLM'])
    """
)

RUN_CELL = dedent(
    """\
    import os

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Wait for the gateway sidecar to be ready.
        !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
              --retry-max-time 600 http://gateway:8001/api/games

        # Copy the framework into a writable location.
        !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
               /kaggle/working/ARC-AGI-3-Agents

        # Drop our agent in as a framework template.
        !cp /tmp/my_agent.py \\
            /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

        # Register MyAgent (rewrite __init__.py: the upstream one eagerly
        # imports optional deps we do not ship).
        INIT_SRC = "\\n".join([
            "from typing import Type",
            "from dotenv import load_dotenv",
            "from .agent import Agent, Playback",
            "from .swarm import Swarm",
            "from .templates.random_agent import Random",
            "from .templates.my_agent import MyAgent",
            "",
            "load_dotenv()",
            "",
            "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {",
            "    'random': Random,",
            "    'myagent': MyAgent,",
            "}",
        ])
        with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
            f.write(INIT_SRC)

        # Point the framework at the gateway sidecar + pass ATLAS flags.
        llm = os.getenv('ATLAS_LLM', '0')
        ENV_LINES = [
            'SCHEME=http',
            'HOST=gateway',
            'PORT=8001',
            'ARC_API_KEY=test-key-123',
            'ARC_BASE_URL=http://gateway:8001/',
            'OPERATION_MODE=online',
            'ENVIRONMENTS_DIR=',
            'RECORDINGS_DIR=/kaggle/working/server_recording',
            f'ATLAS_LLM={llm}',
            'ATLAS_LLM_URL=http://127.0.0.1:8000/v1',
            'ATLAS_MAX_ACTIONS=360',
            'ATLAS_LEVEL_BUDGET=110',
            'ATLAS_LEVEL_HARD_CAP=220',
            # Per-game wall clock.  The Swarm plays every game in its own
            # thread, so this is concurrent, not additive; 2400s keeps the
            # whole run well inside the 32400s rerun budget with the gateway's
            # own teardown to spare.
            'ATLAS_DEADLINE_S=2400',
            'ATLAS_CAL_FILE=/kaggle/working/atlas_calibration.jsonl',
        ]
        with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
            f.write("\\n".join(ENV_LINES) + "\\n")

        # Run it. The gateway records every action and emits submission.parquet.
        !cd /kaggle/working/ARC-AGI-3-Agents && \\
            MPLBACKEND=agg \\
            python main.py --agent myagent
    """
)

DUMMY_CELL = dedent(
    """\
    import os
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Save-and-run-all (commit) mode: emit a dummy submission so the
        # commit succeeds. The real submission.parquet is produced by the
        # gateway during the competition rerun.
        import pandas as pd
        submission = pd.DataFrame(
            data=[['1_0', '1', True, 1]],
            columns=['row_id', 'game_id', 'end_of_game', 'score'])
        submission.to_parquet('/kaggle/working/submission.parquet', index=False)
        submission.head()
    """
)


OFFLINE_CELL_PATH = ROOT / "notebooks" / "offline_eval_cell.py"


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(f"Unknown ACCELERATOR={ACCELERATOR!r}; pick from {sorted(_ACCELERATORS)}")
    accel = _ACCELERATORS[ACCELERATOR]

    agent_body = AGENT_SRC.read_text()
    if not OFFLINE_CELL_PATH.exists():
        raise SystemExit(f"Could not find {OFFLINE_CELL_PATH}")
    offline_body = OFFLINE_CELL_PATH.read_text()

    return {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ATLAS — ARC Prize 2026 (ARC-AGI-3)\n\n"
                "Auto-generated from `agent/my_agent.py` via `scripts/build_notebook.py`. "
                "Edit the agent source, rebuild, push. Do not edit cells by hand.\n\n"
                f"- Accelerator: `{ACCELERATOR}`\n"
                f"- Model: `{MODEL_DIR}`\n"
                "- Internet: **disabled** (required)\n"
                "- Cell 5 plays the 25 public games from `environment_files` "
                "in an interactive session and prints RHAE per game."
            ),
            code_cell(INSTALL_CELL),
            code_cell(SERVE_CELL),
            code_cell("%%writefile /tmp/my_agent.py\n" + agent_body),
            code_cell(RUN_CELL),
            code_cell(offline_body),
            code_cell(DUMMY_CELL),
        ],
    }


def write_metadata() -> None:
    # For `kaggle kernels push -p notebooks/`. Update slug to your account.
    meta = {
        "id": "REPLACE_WITH_YOUR_KAGGLE_USERNAME/arc-atlas-p0",
        "title": "arc-atlas-p0",
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": _ACCELERATORS[ACCELERATOR]["gpu"],
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [
            "keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4"
        ],
        "kernel_sources": [],
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2))


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    write_metadata()
    print(f"[build_notebook] wrote {NOTEBOOK_PATH.relative_to(ROOT)} (accelerator: {ACCELERATOR})")
    print(f"[build_notebook] wrote {METADATA_PATH.relative_to(ROOT)} — edit 'id' with your username before pushing")


if __name__ == "__main__":
    main()
