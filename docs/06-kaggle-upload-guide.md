# How to Run This Repo as a Kaggle Submission (ARC Prize 2026)

Answers: "how do I implement this repo in a Kaggle notebook" and "how do I
upload it" — with internet OFF, per the competition rules.

---

## 0. What you need once (5 minutes, online, on your own machine)

1. A Kaggle account that has **accepted the competition rules** at
   `kaggle.com/competitions/arc-prize-2026-arc-agi-3` (Rules tab → I accept).
   Without this the Submit button stays gray.
2. (CLI route only) A Kaggle API token: `kaggle.com → Settings → Create New
   Token`. Save the file; you'll point `KAGGLE_API_TOKEN` at it.

## 1. How the submission actually works (read this first)

Kaggle runs your notebook **twice**:

| Phase | Trigger | What runs | Internet |
|---|---|---|---|
| **A — Save & Run All (commit)** | You click *Save Version* | All cells top-to-bottom. No games are served; our notebook writes a dummy `submission.parquet` so the commit passes. | off |
| **B — Competition rerun** | You click *Submit to Competition* on the committed version | Same notebook, now with `KAGGLE_IS_COMPETITION_RERUN=1`. Kaggle starts a **gateway sidecar** (`http://gateway:8001`) that hosts the 110 hidden games, records every action your agent takes, and **generates `submission.parquet` for you**. | off |

Key consequences:

- **You never download the private games.** The gateway serves them; our agent
  discovers them via the framework's `main.py`.
- **Internet is off in both phases** — so everything must come from *attached
  inputs*: the competition dataset (wheels, framework, public games) and
  attached Kaggle Models (our LLM). `pip install` uses `--no-index
  --find-links` against the offline wheels.
- The agent file is written to **`/tmp/my_agent.py`**, not `/kaggle/working`,
  so it doesn't appear in output files and confuse the Submit dialog.

## 2. The notebook's 6 cells (what each does)

1. **Install** — `pip install arc-agi python-dotenv` from the competition's
   offline wheel directory.
2. **Serve model (rerun only)** — starts vLLM locally on the GPU for
   `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1`,
   waits up to 25 min for it to come up. If vLLM or the model is unavailable,
   it sets `ATLAS_LLM=0` and the agent plays in **reflex mode** — the
   submission still gets made. This cell can never kill the notebook.
3. **Write agent** — `%%writefile /tmp/my_agent.py` with the full ATLAS agent.
4. **Run (rerun only)** — waits for the gateway, copies the ARC-AGI-3-Agents
   framework into `/kaggle/working`, installs our agent as
   `agents/templates/my_agent.py`, registers it as `myagent`, writes `.env`
   pointing at the gateway (+ ATLAS flags), runs `python main.py --agent
   myagent`. The gateway emits `submission.parquet`.
5. **Commit-mode fallback** — writes a dummy `submission.parquet` in phase A.

## 3. Route A — upload via the Kaggle UI (no CLI; recommended first time)

1. Build the notebook locally:
   ```bash
   python3 scripts/build_notebook.py     # writes notebooks/submission.ipynb
   ```
2. In Kaggle: competition page → **Code** tab → **New Notebook**
   (or open any existing notebook of yours).
3. **File → Import Notebook** → upload `notebooks/submission.ipynb`.
4. Attach inputs (right panel → **Add Input**):
   - **Competition data**: search `arc-prize-2026-arc-agi-3` → add. (This is
     what provides `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/…`.)
   - **Model**: Models → search `qwen3-8-27b-fp8-repacked-v1` (owner
     `foysalemonshanto`) → add the `pytorch/hf-fp8` variant. Verify the path
     in the notebook's model cell matches what appears under
     `/kaggle/input/models/...`.
5. **Settings** (right panel):
   - Accelerator → **GPU RTX 6000** (this competition only; burns quota
     faster — use GPU T4 while iterating, RTX 6000 for real runs).
   - Internet → **OFF** (required).
6. **Save Version** → *Save & Run All (Commit)*. Wait for it to complete
   (phase A; fast because serving and gameplay are skipped).
7. On the version page, click **Submit to Competition**, pick
   **`submission.parquet`** from the output file list. Phase B then runs on
   Kaggle's side (this is the scored 9-hour run) and your leaderboard entry
   appears when it finishes.

> Quota: 5 submissions/day. RTX 6000 hours are limited — prefer T4 while
> debugging plumbing, switch to RTX 6000 when the LLM actually matters.

## 4. Route B — push via the Kaggle CLI (for fast iteration)

1. `pip install kaggle` on your machine.
2. Edit `notebooks/kernel-metadata.json`:
   - `"id": "YOUR_KAGGLE_USERNAME/arc-atlas-p0"` ← replace.
   - `competition_sources` already lists the competition; `model_sources`
     already lists the model; `enable_internet` is false; `enable_gpu` true.
3. Export your token: `export KAGGLE_API_TOKEN=$(cat /path/kaggle.json)`
   (or place `kaggle.json` at `~/.kaggle/kaggle.json`).
4. Push and watch:
   ```bash
   kaggle kernels push -p notebooks/
   kaggle kernels status YOUR_KAGGLE_USERNAME/arc-atlas-p0
   kaggle kernels output YOUR_KAGGLE_USERNAME/arc-atlas-p0 -p ./out
   ```
5. Submit the finished version from the notebook's page (step 7 above) or:
   ```bash
   kaggle competitions submit arc-prize-2026-arc-agi-3 \
       -f ./out/submission.parquet -b -m "atlas-p0"
   ```
   (UI submission of the rerun version is the canonical path; `-b` chooses
   the best-scoring submitted version for final scoring.)

## 5. Verifying before you burn a submission

- **Local smoke test (no Kaggle, no GPU):**
  ```bash
  python3 -m unittest discover -s tests -v      # 15 tests, pure logic
  ```
- **Local real-engine test (needs internet once to cache games):** install
  Python 3.12 + `pip install arc-agi`, then run the agent against the 25
  public games offline (the ARC-AGI-3-Kaggle-Starter `make play-local`
  flow). Everything the agent needs (`environment_files/`) ships in the
  competition dataset.
- **Commit-phase check:** phase A must complete < 9 h with internet off —
  ours does (installs + file writes + dummy parquet).

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Submit button gray | Rules not accepted; or last commit failed — read its log |
| `Could not find provided output file` | Wrong file selected — pick `submission.parquet`, not `my_agent.py` (we write the agent to /tmp precisely to avoid this) |
| Rerun log shows `curl: (7)` then success | Normal — it's waiting for the gateway sidecar |
| Score 0 | Rerun crashed mid-game or agent took zero actions; check the phase-B log for `[atlas]` lines; the calibration log `atlas_calibration.jsonl` and recordings land in output |
| vLLM never comes up | Cell 2 falls back to `ATLAS_LLM=0` (reflex mode) by design — submit anyway to validate plumbing, then fix serving (check `vllm.log` in outputs) |
| Notebook dies on install | Wheel path changed upstream — confirm `arc_agi_3_wheels` exists in the attached competition data |

## 7. Rules that bite (checklist)

- Notebook ≤ 9 h in **both** phases; internet **disabled**.
- External data/models allowed only if publicly available on Kaggle — ours are.
- RTX 6000 **only** for this competition (Kaggle moderation warning).
- Prize eligibility: notebook + code must be public under an open license
  (CC0/MIT-0) by the milestone/final dates.
