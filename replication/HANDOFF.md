# EReLELA FullMazeS5 replication — handoff notes

Written for a Claude Code session (or a human) picking up the replication of the EReLELA
paper results on MiniWorld FullMazeS5 with PPO/ETD. State as of 2026-10-24.

## Goal

Replicate the original EReLELA FullMazeS5 runs to within ±10% of their final return. The
original W&B runs are in entity `near3213`, project `EReLELA-FullMaze-S5`:

| Seed | W&B run | Final raw return (~5M steps) | Smoothed-20 return at ~1.1M steps |
|---|---|---|---|
| 10 | `h7evmv4x` | ~0.90 | 0.247 |
| 20 | `otuhph0r` | ~0.76 | 0.163 |

Both were run from ETD commit `085c7b3`. The launch config is `train_erelela_MAZES5.sh`
(RG period 512k, 4 RG dataloader workers, 16 envs, 5M steps).

## Ground rules from the project owner

- Don't modify the paper or the owner's working checkouts or branches unless they ask.
  The replication was done in a separate git worktree, and all fixes live on this branch.
- Ask before deleting anything (W&B runs, logs) and before running sudo commands.

## What this branch contains (based on `e9c33cb`)

`e9c33cb` is two commits behind `a6c392e`, the tip of `develop-fullmazes5-sb3v110`. Those two
commits only add `mp.set_start_method('spawn')` to `train.py` and two unrelated launch scripts.
The spawn start method made every RG DataLoader worker copy the dataset (~2GB per worker), so
it was left out. Commits on top:

1. **CUDA init guard** (`ppo_etd/train.py`). Only the launched process calls `th.cuda.init()`.
   SubprocVecEnv and DataLoader workers re-import `train.py` as `__mp_main__`, so without the
   guard each of the 16 env workers creates its own CUDA context (~150MB RAM + 72MB VRAM each).
2. **Off-policy buffer fix** (`ppo_etd/algo/ppo_rollout.py`). This is the main memory fix.
   `PPORolloutBuffer` defaults to a 1e6-frame off-policy replay (`aux_observations` +
   `aux_future_observations`, uint8 3x64x64, **24.6GB**). The arrays are allocated lazily with
   `np.zeros` and filled at ~24KB per env step, so RSS climbs linearly and plateaus near 1M
   steps. Only TDD with `--offpolicy_data=1` reads this buffer; EReLELA never does. The fix
   sizes the buffer to one rollout (`n_steps*n_envs` = 8192 frames) when `offpolicy_data` is 0.
   The original runs carry the same 25GB (34GB RSS at 8h on a 128GB machine). Results are
   unaffected, except that one `np.random.permutation(offpolicy_buffer_size)` call in
   `PPORolloutBuffer.get()` now draws from a different global-numpy RNG stream.
3. **Optional memory diagnostics** (`ppo_etd/diag_mem.py` plus a hook in `train.py`). This is
   off unless `ERELELA_MEMDIAG=<logfile>` is set. Each period
   (`ERELELA_MEMDIAG_PERIOD`, default 300s) it writes a census of CPU tensors and objects.
   With `ERELELA_TRACEMALLOC=1` it also writes tracemalloc growth diffs, which slows the run.
4. **`replication/`**: launcher, memory loggers and this file (see below).
5. **W&B outage guard** (`ppo_etd/utils/wandb_utils.py`, called from `configs.py`).
   ReferentialGym logs `wandb.Table`s during RG phases. Each Table is logged as an artifact, and
   `Run._assert_can_log_artifact` contacts the server first. A Wi-Fi drop there raised
   `AuthenticationError: Unable to connect to server to verify API token.` and killed run
   7lckdovg at 4.11M steps. The guard skips that best-effort check on `wandb.errors.CommError`.

Contents of `replication/`:
   - `run_erelela_MAZES5.sh` is `train_erelela_MAZES5.sh` with three changes: no ipdb, the
     interpreter set through `$PYTHON`, and the RG dataloader worker count as argument 4
     (default 1).
   - `memlog.sh` logs system memory once a minute.
   - `proc_sampler.sh` logs main and env-worker RSS/PSS/swap.
   - This file.

## Environment used (Ubuntu 20.04, 28GB RAM, 1 GPU, 8 cores)

Python 3.8.10, torch 2.4.1+cu121, stable-baselines3 1.1.0, gym 0.24.0, numpy 1.22.3, wandb 0.22.2.
Editable installs, all from the `Near32/EReLELA` superproject (`main` @ `74df92d`) and its
submodules:

| Package | Repo | Commit |
|---|---|---|
| regym | Near32/Regym (`develop`) | `0578393` |
| ReferentialGym | Near32/ReferentialGym (`develop-regym-integration`) | `0a1c740` |
| Archi | Near32/Archi | `28964de` |
| gym_miniworld | liuqh16/Miniworld (ETD submodule `ppo_etd/env/gym_miniworld`) | `bc15e7c` |
| minigrid | Near32/Minigrid | `c130142` |

Regym records different SHAs for its ReferentialGym and Archi submodules; the ones above are
the SHAs actually checked out and used. `ppo_etd` itself is taken from the launch directory,
because the script sets `PYTHONPATH=./`.

## How to launch on another machine

```bash
git clone https://github.com/Near32/ETD && cd ETD
git checkout replication-fullmazes5-memfix
git submodule update --init ppo_etd/env/gym_miniworld      # bc15e7c
# install regym/ReferentialGym/Archi/gym_miniworld at the commits above into a venv
WANDB_MODE=offline WANDB_DIR=$PWD/wandb_offline \
PYTHON=/path/to/venv/bin/python \
  setsid nohup replication/run_erelela_MAZES5.sh 0 20 replication-memfix-seed20 1 > run_seed20.log 2>&1 &
replication/memlog.sh replication-memfix-seed20 > memlog_seed20.log 2>&1 &
```

- `xvfb-run` must be installed. The script wraps training in a virtual display for MiniWorld rendering.
- **W&B always runs online.** `configs.py` passes `wandb.Settings(mode="shared")`, which overrides
  `WANDB_MODE=offline`, so runs stream live to the project and no `wandb sync` is needed. Earlier
  runs here were launched with `WANDB_MODE=offline` plus hourly syncs; the syncs were redundant.
  `WANDB_DIR` still sets where the local run folder goes. Log in with `wandb login` before
  launching. Since the outage guard, short network drops don't stop training.
- **W&B volume:** regym's `wandb_log` writes several rows per env step (~400MB per 500k steps).
- **Dataloader workers:** with the fix, a 4-worker run needs a bit more RAM in RG phases.
  It should fit comfortably on a machine with ≥48GB. 4 workers is the original setting.

## Expected resource profile (with fixes, 1 dataloader worker)

- **Main process:** ~7.5GB RSS, flat. Before the fix it grew to ~30GB+ and swap-thrashed a 28GB box.
- **Env workers:** 16 × ~490MB PSS (~7.8GB total).
- **RG DataLoader worker:** up to ~4GB PSS during RG phases.
- **PPO phases:** ~75 steps/s on a GTX-1070-class laptop.
- **RG phases:** about 1.5h each; the first starts at ~16k steps (when the 8192-capacity RG
  storage is full), then every 512k steps. 5M steps takes ~30h.

## Run history on the original machine

All runs below are seed 10 and were launched from the worktree `REPLICATION/ETD-e9c33cb`
(this branch). The offline W&B dirs are under `REPLICATION/wandb_offline/wandb/`.

| W&B id | Code | Outcome |
|---|---|---|
| m0tru5l0 | a6c392e | OOM (spawn DataLoader workers + Firefox) |
| gyldcpek | a6c392e | killed |
| rg6vlrdp | e9c33cb + CUDA guard, 4 DL workers | killed; RG phase hit the RAM limit |
| gcpbd61u | e9c33cb + CUDA guard, 1 DL worker | stopped at 1.12M steps; swap-thrashed down to ~6 steps/s from the off-policy buffer; smoothed return 0.058 at 1.12M |
| 7lckdovg | CUDA guard + off-policy fix, 1 DL worker | crashed at 4.11M steps (2026-10-24 03:10) from the W&B `AuthenticationError` above; memory flat throughout. Return curve below |
| (next) | all fixes incl. W&B outage guard, 1 DL worker | seed 10 relaunched on 2026-10-24 |

Return (`rollout/ep_info_rew_mean`, mean over the 210k steps before each mark):

| Steps | 7lckdovg | orig. seed 10 (h7evmv4x) | orig. seed 20 (otuhph0r) |
|---|---|---|---|
| 1.0M | 0.007 | 0.136 | 0.090 |
| 2.0M | 0.050 | 0.234 | 0.160 |
| 2.5M | 0.085 | 0.687 | 0.325 |
| 3.0M | 0.176 | 0.643 | 0.376 |
| 3.5M | 0.288 | 0.848 | 0.437 |
| 4.0M | 0.577 | 0.604 | 0.365 |
| 4.1M | 0.613 | 0.681 | 0.382 |
| 5.0M | — | 0.824 | 0.735 |

7lckdovg took off ~1–1.5M steps later than seed 10, then matched it within ~10% by 4.0–4.1M.
The two original seeds already differ ~2x mid-training.

To fetch the original curves: `wandb.Api().run('near3213/EReLELA-FullMaze-S5/h7evmv4x').history(samples=2000,
keys=['time/total_timesteps','rollout/ep_info_rew_mean'])`. `scan_history` is too slow for these runs.

Because W&B was in fact online, all of these runs are probably on W&B already. The owner considers the dead runs disposable
but has not yet approved deleting them.

## Open questions / next steps

1. **Is the late take-off systematic?** 7lckdovg lagged both original seeds until ~3.5M steps, then
   caught up with seed 10. The relaunched seed 10 and other seeds will show whether that is noise.
   No run here has reached 5M steps yet, which is where the ±10% target is assessed.
2. **Seed 20** (`run_erelela_MAZES5.sh 0 20 ...`) has not been run yet. Run it, ideally on a
   machine with more RAM, and also with 4 DL workers to match the original exactly.
3. **Possible remaining differences from the original runs:**
   - regym, ReferentialGym and gym_miniworld commits: the originals' exact versions are unknown.
     The W&B run metadata or config of `h7evmv4x` may record them.
   - 1 vs 4 RG dataloader workers.
   - The RNG stream changed by the off-policy fix.
   - The spawn-removal (`e9c33cb` vs `a6c392e`/`085c7b3`). Check `git diff 085c7b3 e9c33cb`.
4. **Remaining minor memory growth:** about +0.5GB of main RSS per RG phase, probably
   allocator fragmentation; `distractor_sampling_likelihoods` allocates two
   `np.ones((N, N))` arrays (~1.6GB) per dataset rebuild. This is harmless at 5M steps
   (~10 phases).
