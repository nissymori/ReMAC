# The Official Code for "Retry Policy Gradients for Continuous Action Spaces"
[![arXiv](https://img.shields.io/badge/arXiv-2606.05888-b31b1b.svg)](https://arxiv.org/abs/2606.05888)



# Install dependencies
```bash
pip install -r requirements.txt
```

## Repository layout

```
brax/                    ReMAC, SAC, PPO, TD3 on Brax + the launch scripts (brax/sh/)
analysis/                everything that turns runs into the figures and tables
additional_experiments/  the diagnostics added during review (own README)
data/                    the data the figures are drawn from
toy/                     the fixed-state toy problem of Sec. 3
```

`analysis/README.md` and `additional_experiments/README.md` cover their own
directories. Every command below is run from the repository root unless it says
otherwise.

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

#### Tuning the learning rate
Sweep `lr` over {1e-4, 2e-4, 3e-4, 5e-4, 1e-3} with 3 seeds at eps = 1e-8, and take the
value that does well across every M (App. C.1). At `brax/`:

```bash
./sh/tune/tune_remax.sh        # the six original tasks
bash sh/reacher_tune.sh 0 "1:0.0001"          # Reacher, re-tuned after the fix below
bash sh/pusher_tune.sh  0 "1:0.0001"          # Pusher
```

`python analysis/pick_reacher_lr.py` and `python analysis/pick_pusher_lr.py` apply the
selection rule to the sweep's logs, so the chosen value is read off the data rather than
by eye. The selected learning rates are in Tab. 2 of the paper and are the defaults in
each config's `remax_ac:` block.

#### Running the experiments
At `brax/`:

```bash
./sh/remac.sh   # ReMAC on the six original tasks and HumanoidStandup, at every eps
./sh/sac.sh     # SAC
./sh/ppo.sh     # PPO
./sh/td3.sh     # TD3

# Reacher and Pusher have their own phase scripts (see the note below)
bash sh/reacher_final.sh dense m4 0 3e-4
bash sh/reacher_eps.sh   0 3e-4 "4:1e-2"
PUSHER_LR=$(cat ../data/PUSHER_LR) bash sh/pusher_final.sh 0 "m1 m2 m4 m8 sac ppo"
PUSHER_LR=$(cat ../data/PUSHER_LR) bash sh/pusher_eps.sh   0 "4:1e-2"

# HumanoidStandup's eps = 1e-2 arm, which the original sweep did not cover
bash sh/humanoid_eps.sh 0 1e-2 "4:2"
```

Many short jobs spread over several GPUs are easiest to run through the queue: put one
command per line in a file and start one worker per GPU. One process per GPU -- two
co-tenants on the same card are slower in total, not faster.

```bash
for g in 0 1 2 3; do bash sh/queue_worker.sh $g logs/queue/jobs.txt & done
```

#### A note on Reacher, and on Pusher
`configs/brax/reacher.yaml` used to set `env_params.episode_length` for `sac:` and
`ppo:` but not for `remax_ac:`, so ReMAC silently trained on Brax's default 1000-step
episodes while the baselines used 50 -- a 20x longer episode, which made the return
comparison on Reacher invalid. The config now sets it for every algorithm, and Reacher
was re-tuned and re-run (`sh/reacher_{tune,final,eps}.sh`); those are the runs the paper
reports. `configs/brax/pusher.yaml` had the same omission and was fixed the same way
before Pusher was run at all.

### Figures and tables

```bash
bash brax/sh/plot.sh            # every Brax figure, into fig/
python analysis/make_tables.py  # the LaTeX table bodies
```

These read `data/`, so they work without wandb access. See `analysis/README.md`.

### Additional experiments
The diagnostics added during review -- the scale gradient, the SGD actor, visited-state
coverage and the Adam-denominator ablation -- live in `additional_experiments/`, with
their own README.

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

