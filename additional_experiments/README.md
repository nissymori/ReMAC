# Additional experiments

The experiments that were added during review, kept separate from the main sweeps in
`brax/sh/` because they are diagnostics rather than part of the headline comparison:
each one probes a mechanism the analysis predicts, or a stronger notion of exploration
than policy stochasticity.

Everything here is run from the **repository root**.

| script | what it answers | where it lands in the paper |
|---|---|---|
| `additional_experiments/adam_denominator.py` | Does Adam's *denominator* cause the effect, as opposed to its momentum or bias correction? Re-runs the toy problem with an EMA-only optimizer, i.e. Adam with the denominator deleted. | Tab. 1, Fig. 6 |
| `additional_experiments/sh/remac_sigma_m8.sh` | Does the policy scale really increase with $M$ during training? Logs the scale gradient and the scale itself, with $B=16$ at every $M$. | Tab. 7 |
| `additional_experiments/sh/remac_sgd.sh` | Does ReMAC depend on Adam? Re-runs it with a plain-SGD actor, everything else held fixed. | Tab. 8 |
| `additional_experiments/sh/remac_state_coverage.sh` | Does ReMAC visit more of the state space than SAC, not just act more stochastically? | Tab. 9 |
| `additional_experiments/knn_state_entropy.py` | The $k$-nearest-neighbour estimator of the visited-state entropy used by the coverage experiment. | Tab. 9 |
| `additional_experiments/make_rebuttal_figs.py` | Draws the four figures these experiments produce. | Fig. 18 (and three figures not used in the paper) |

## Running them

```bash
# The scale-gradient sweep (Tab. 7): one GPU per job list.
bash additional_experiments/sh/remac_sigma_m8.sh --gpu 0 --jobs "walker2d:1 walker2d:2"

# The SGD-actor arm (Tab. 8).  Its Adam control is the ReMAC arm of the coverage
# project below -- same env, lr, M, B, seeds and step budget, differing only in the
# actor optimizer -- so do not re-run Adam here.
bash additional_experiments/sh/remac_sgd.sh --gpu 0 --m 4

# Visited-state coverage (Tab. 9).
bash additional_experiments/sh/remac_state_coverage.sh --gpu 0 --m 4
```

`remac_sigma_m8.sh` writes one log per process under `brax/logs/sigma_m8/runs/` and its
metrics are read back with `python analysis/parse_sigma_logs.py`; the other two were run
with wandb and their results are shipped as `data/remac_rebuttal_data.pkl`, which is
self-describing:

```python
import pickle
d = pickle.load(open("data/remac_rebuttal_data.pkl", "rb"))
print(d["schema"])   # the pickle documents itself
d["summary"]         # the reported numbers
d["tidy"]            # one row per seed
d["runs"]            # raw learning curves
```

## Reading the numbers

```bash
python analysis/make_tables.py        # Tabs. 4, 6, 7, 8, 9 straight from the data
python analysis/parse_sigma_logs.py   # Tab. 7's scale-gradient rows from the run logs
python additional_experiments/make_rebuttal_figs.py
```

## Seeds

A configuration is 10 seeds. On an 11 GB card that is 5 processes of 2 vmapped seeds
(`seed_id` 2..6) for most tasks, and 10 processes of 1 for HumanoidStandup. `seed_id`
0 and 1 are reserved for hyperparameter tuning and never appear in a reported number,
so that tuning does not leak into the results. A unique seed is `(seed_id, vmap index)`.
