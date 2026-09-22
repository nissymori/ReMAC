# Experiments

Everything that runs an experiment, kept out of `brax/` so that the training code
stands on its own.  Two kinds of thing live here: the sweeps that produced the reported
numbers, and the appendix diagnostics, each of which probes a mechanism the analysis
predicts or a stronger notion of exploration than policy stochasticity.

Every script `cd`s into `brax/` itself, so it can be run from anywhere.

| script | what it is |
|---|---|
| `sh/tune.sh <ENV> <GPU> "<M:LR> ..."` | the learning-rate sweep of App. C.1 |
| `sh/final.sh <ENV> <GPU> "<TAG> ..."` | the reported runs at eps = 1e-8 |
| `sh/eps.sh <ENV> <GPU> "<M:EPS> ..."` | the epsilon sweep |
| `sh/make_queue.sh <PHASE>` | writes out the full matrix for a phase |
| `sh/diag_*.sh` | the appendix diagnostics, below |
| `lib_runlog.sh` | what differs per task: `lr_for`, `b_for_m`, `seeds_for` |
| `queue_worker.sh` | one worker per GPU against a shared queue file |

The three phases take the task as an argument rather than existing once per task: the
procedure is the same everywhere, and the three numbers that are not -- the tuned
learning rate, the action-sample batch size, and how the ten seeds are split across
processes -- are looked up in `lib_runlog.sh`.

Everything here is run from the **repository root**.

| script | what it answers | where it lands in the paper |
|---|---|---|
| `experiments/adam_denominator.py` | Does Adam's *denominator* cause the effect, as opposed to its momentum or bias correction? Re-runs the toy problem with an EMA-only optimizer, i.e. Adam with the denominator deleted. | Tab. 1, Fig. 6 |
| `experiments/sh/diag_scale_gradient.sh` | Does the policy scale really increase with $M$ during training? Logs the scale gradient and the scale itself, with $B=16$ at every $M$. | Tab. 7 |
| `experiments/sh/diag_sgd_actor.sh` | Does ReMAC depend on Adam? Re-runs it with a plain-SGD actor, everything else held fixed. | Tab. 8 |
| `experiments/sh/diag_state_coverage.sh` | Does ReMAC visit more of the state space than SAC, not just act more stochastically? | Tab. 9 |
| `experiments/knn_state_entropy.py` | The $k$-nearest-neighbour estimator of the visited-state entropy used by the coverage experiment. | Tab. 9 |

## Running them

```bash
# The scale-gradient sweep (Tab. 7): one GPU per job list.
bash experiments/sh/diag_scale_gradient.sh --gpu 0 --jobs "walker2d:1 walker2d:2"

# The SGD-actor arm (Tab. 8).  Its Adam control is the ReMAC arm of the coverage
# project below -- same env, lr, M, B, seeds and step budget, differing only in the
# actor optimizer -- so do not re-run Adam here.
bash experiments/sh/diag_sgd_actor.sh --gpu 0 --m 4

# Visited-state coverage (Tab. 9).
bash experiments/sh/diag_state_coverage.sh --gpu 0 --m 4
```

`remac_sigma_m8.sh` writes one log per process under `brax/logs/sigma_m8/runs/` and its
metrics are read back with `python analysis/parse_sigma_logs.py`; the other two were run
with wandb and their results are shipped as `data/diagnostics.pkl`, which is
self-describing:

```python
import pickle
d = pickle.load(open("data/diagnostics.pkl", "rb"))
print(d["schema"])   # the pickle documents itself
d["summary"]         # the reported numbers
d["tidy"]            # one row per seed
d["runs"]            # raw learning curves
```

## Reading the numbers

```bash
python analysis/make_tables.py        # Tabs. 4, 6, 7, 8, 9 straight from the data
python analysis/parse_sigma_logs.py   # Tab. 7's scale-gradient rows from the run logs
python analysis/make_diagnostic_figs.py   # the figure form of the same runs
```

## Seeds

A configuration is 10 seeds. On an 11 GB card that is 5 processes of 2 vmapped seeds
(`seed_id` 2..6) for most tasks, and 10 processes of 1 for HumanoidStandup. `seed_id`
0 and 1 are reserved for hyperparameter tuning and never appear in a reported number,
so that tuning does not leak into the results. A unique seed is `(seed_id, vmap index)`.
