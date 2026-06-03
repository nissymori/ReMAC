# The Official Implementation of "Retry Policy Gradients for Continuous Action Spaces"

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

