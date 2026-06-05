# The Official Code for "Retry Policy Gradients for Continuous Action Spaces"
[![arXiv](https://img.shields.io/badge/arXiv-2606.05888-b31b1b.svg)](https://arxiv.org/abs/2606.05888)



# Install dependencies
```bash
pip install -r requirements.txt
```

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
The configurations for ReMAC are in the `brax/configs/brax/` directory.

#### Tuning the learning rate
At `brax/`

```bash
./sh/tune/tune_remax.sh
```

#### Running the experiments
At `brax/`

```bash
./sh/remac.sh  # for ReMAC
./sh/sac.sh  # for SAC
./sh/ppo.sh  # for PPO
```

## See Also
- [ReMax PPO](https://github.com/nissymori/remax-rl): Proposal of ReMax in discrete action. [code](https://github.com/nissymori/remax-rl), [paper](https://arxiv.org/abs/2606.00151v1).
- [OrderGrad](https://github.com/paavo5/ordergrad): Generalization of ReMax to **Any** order statistics. [library](https://github.com/paavo5/ordergrad), [paper](https://arxiv.org/abs/2606.06096)


## Cite us
```bibtex
@article{nishimori2026retry,
  title={Retry Policy Gradients in Continuous Action Spaces},
  author={Soichiro Nishimori and Paavo Parmas},
  year={2026},
  journal={arXiv preprint arXiv:2606.05888},
}
```

