# The Official Code for "Retry Policy Gradients for Continuous Action Spaces"
[![arXiv](https://img.shields.io/badge/arXiv-2606.05888-b31b1b.svg)](https://arxiv.org/abs/2606.05888)



# Install dependencies
```bash
pip install -r requirements.txt
```

## Repository layout

```
brax/          ReMAC, SAC, PPO, TD3 on Brax, and the environment configs
toy/           the fixed-state toy problem of Sec. 3
experiments/   every run script, and the modules only they need (own README)
analysis/      everything that turns runs into the figures and tables (own README)
data/          the data the figures are drawn from
```

Every command below is run from the repository root unless it says otherwise.

# Reproduce the results

### Toy experiments

#### Figure 1 (vector field)
```bash
python toy/quad_vector_field.py --m 1 --alpha 0.0 --normalize  # M=1
python toy/quad_vector_field.py --m 1 --alpha 0.5 --normalize  # M=1 with entropy bonus
python toy/quad_vector_field.py --m 4 --alpha 0.0 --normalize  # M=4
python toy/quad_vector_field.py --m 8 --alpha 0.0 --normalize  # M=8
```

#### Figure 2 (distance to the optimal policy)
```bash
python toy/quad_opt_update_norm.py --m 1 --distance-only
python toy/quad_opt_update_norm.py --m 2 --distance-only
python toy/quad_opt_update_norm.py --m 4 --distance-only
python toy/quad_opt_update_norm.py --m 8 --distance-only
```

#### Figure 7 (optimization path)
```bash
python toy/quad_opt_update_norm.py --m 1 --optimize-only
python toy/quad_opt_update_norm.py --m 2 --optimize-only
python toy/quad_opt_update_norm.py --m 4 --optimize-only
python toy/quad_opt_update_norm.py --m 8 --optimize-only
```

### For ReMAC
The configurations are in `brax/configs/brax/`. The main evaluation is eight Brax
tasks: Ant, HalfCheetah, Hopper, Reacher, Swimmer, Walker2d, HumanoidStandup and
Pusher.

#### Three phases, any task

Every task goes through the same three phases, so there are three scripts rather than
one per task. Each takes the task as its first argument and `cd`s into `brax/` itself,
so it can be launched from anywhere.

```bash
bash experiments/sh/tune.sh  <ENV> <GPU> "<M:LR> ..."    # the learning-rate sweep
bash experiments/sh/final.sh <ENV> <GPU> "<TAG> ..."     # the reported runs, eps = 1e-8
bash experiments/sh/eps.sh   <ENV> <GPU> "<M:EPS> ..."   # the epsilon sweep
```

`TAG` is `m1`, `m2`, `m4`, `m8`, `sac`, `ppo` or `td3`. `final.sh` and `eps.sh` also take
`--lr` and `--bs`, which is how the lr-versus-epsilon probes of Figs. 15 and 16 and the
`B` ablation of Fig. 17 are run.

What differs between tasks is not the procedure but three numbers -- the tuned learning
rate, the action-sample batch size, and how the ten seeds are split across processes --
and those live in `experiments/lib_runlog.sh` as `lr_for`, `b_for_m` and `seeds_for`.
The split is per task because the seeds are vmapped and an 11 GB card fits ten of them
for Pusher but only one for HumanoidStandup.

`make_queue.sh` writes out the full matrix for a phase, so the set of runs behind a
figure is one command away:

```bash
bash experiments/sh/make_queue.sh final > brax/logs/queue/jobs.txt   # Figs. 3, 4
bash experiments/sh/make_queue.sh eps  >> brax/logs/queue/jobs.txt   # Figs. 7-14
bash experiments/sh/make_queue.sh lr   >> brax/logs/queue/jobs.txt   # Figs. 15, 16
bash experiments/sh/make_queue.sh bs   >> brax/logs/queue/jobs.txt   # Fig. 17
bash experiments/sh/make_queue.sh tune >> brax/logs/queue/jobs.txt   # Tab. 2
```

then one worker per GPU. One process per GPU on purpose: two co-tenants on the same
card measured about 13% slower in total throughput on this hardware, not faster.

```bash
for g in 0 1 2 3; do bash experiments/queue_worker.sh $g brax/logs/queue/jobs.txt & done
```

`analysis/pick_reacher_lr.py` and `analysis/pick_pusher_lr.py` apply the selection rule
of App. C.1 to the tuning logs, so the chosen learning rate is read off the data rather
than by eye. The selected values are in Tab. 2 and are what `lr_for` and each config's
`remax_ac:` block carry.

The appendix diagnostics are the `diag_*` scripts; see `experiments/README.md`.

### Figures and tables

```bash
bash analysis/plot.sh           # every Brax figure, into fig/
python analysis/make_tables.py  # the LaTeX table bodies
```

These read `data/`, so they work without wandb access. See `analysis/README.md`.

### The appendix diagnostics
The scale gradient, the SGD actor, visited-state coverage, the sparse-reward Reacher and
the Adam-denominator ablation live in `experiments/`, with their own README.

## Related Work and Extensions
- [Emergence of Exploration in Policy gradient reinforcement learning via Retrying](https://arxiv.org/abs/2606.00151) (ICML 2026)
  - Proposal of ReMax in discrete action and PPO variant, **Re**Max **PPO** (**RePPO**) [code](https://github.com/nissymori/remax-rl)

- [OrderGrad: Optimizing Beyond the Mean with Order-Statistic Policy Gradient Estimation](https://arxiv.org/abs/2606.06096)
  - Generalization of ReMax PR gradient to **Any** order statistics. [library](https://github.com/paavo5/ordergrad)


## Cite us
```bibtex
@article{nishimori2026retry,
  title={Retry Policy Gradients in Continuous Action Spaces},
  author={Soichiro Nishimori and Paavo Parmas},
  year={2026},
  journal={arXiv preprint arXiv:2606.05888},
}
```

