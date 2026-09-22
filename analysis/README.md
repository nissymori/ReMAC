# Analysis: from runs to the figures and tables

Everything that turns training runs into the paper's figures and tables. Run all of it
from the **repository root**.

## The data

| file | what it is |
|---|---|
| `data/brax-remax-ac-report.pkl.gz` | The frame every Brax figure is drawn from: one row per (run, seed, step), long format. Read with `pandas.read_pickle`. |
| `data/diagnostics.pkl` | The diagnostic runs (SGD actor, state coverage); see `experiments/README.md`. |

The first frame was originally fetched from wandb by `fetch_wandb_data_with_seeds()` in
`plot.py`, which `plot.py` still calls when `--cache` points at a file that does not
exist. It keeps the 20 columns the figures and tables read, out of the 51 the fetch
produced. A seed is `(run_id, seed_id)`: `seed_id` is the vmap index inside a process,
so a configuration run as 1 process x 10 vmapped seeds and one run as 5 x 2 both give
n = 10.

## The scripts

| script | produces |
|---|---|
| `plot.sh` | Regenerates every Brax figure from `data/` into `fig/`. |
| `plot.py` | The main learning-curve figures. Subcommands: `2x4` (the eight main tasks), `2x3` (the original six), `lr-sweep`, `lr-sweep-grid`, `b-ablation`, `damping`. |
| `make_eps_sweeps.py` | The per-epsilon M sweeps, Figs. 7-14. |
| `make_diagnostic_figs.py` | Fig. 18, and the figure form of the diagnostic tables. |
| `make_humanoid_figs.py` | HumanoidStandup on its own, in the same visual language. |
| `make_tables.py` | The LaTeX bodies of Tabs. 4, 6, 7, 8, 9 -- printed from the data rather than transcribed. |
| `parse_brax_logs.py` | Turns stdout run logs into rows of the frame's schema, and merges them into it. Needed because the runs added for the camera-ready version were logged to files, not wandb. |
| `parse_sigma_logs.py` | Tab. 7's scale-gradient rows, from `brax/logs/sigma_m8/runs/`. |
| `pick_reacher_lr.py`, `pick_pusher_lr.py` | Apply the paper's learning-rate selection rule (App. C.1) to a tuning sweep's logs. |

## Reproducing the figures

```bash
bash analysis/plot.sh         # every Brax figure, into fig/
python analysis/make_tables.py
```

or a single figure, e.g. Fig. 3 (return on the eight main tasks):

```bash
python analysis/plot.py 2x4 \
    --project brax-remax-ac-report --cache data/brax-remax-ac-report.pkl.gz \
    --out-dir fig --metrics eval/return \
    --ms 1,2,4,8 --epsilons 1e-8 --bs-per-m 1:8,2:8,4:8,8:16 \
    --include-sac --baselines SAC,PPO
```

`--bs-per-m 1:8,2:8,4:8,8:16` is the paper's design: $B = 8$ action samples per state,
raised to $B = 16$ at the largest retry budget $M = 8$, where the estimator is noisiest.

## Adding a new run to the data

New runs are logged to files (see `experiments/lib_runlog.sh`); fold them into the frame with

```bash
python analysis/parse_brax_logs.py \
    --base data/brax-remax-ac-report.pkl.gz \
    --out  data/brax-remax-ac-camera-ready.pkl.gz \
    --logs 'brax/logs/pusher/*.log' 'brax/logs/humanoid_eps/*.log'
```

It prints the number of seeds per configuration, so a missing or half-finished run shows
up as a cell with fewer than 10 rather than silently narrowing an error bar.
