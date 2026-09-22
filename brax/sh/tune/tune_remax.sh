total_timesteps=3000000
eval_freq=100000
num_seeds=3
# if m=8, remax_num_samples=16, otherwise remax_num_samples=8
if [ "$m" -eq 8 ]; then
  remax_num_samples=16
else
  remax_num_samples=8
fi

wandb_project="tune-remax-ac"

# For M=1, there is no additional damping effect in addition to the normal RL objective, which I think the reason to omit the actor epsilon other than 1e-8.

for m in 1 2 4 8; do
    for lr in 0.0001 0.0002 0.0003 0.0005 0.001; do
        python train.py --config configs/brax/hopper.yaml --wandb-project=$wandb_project --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --set learning_rate=$lr
        python train.py --config configs/brax/ant.yaml --wandb-project=$wandb_project --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --set learning_rate=$lr
        python train.py --config configs/brax/walker2d.yaml --wandb-project=$wandb_project --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --set learning_rate=$lr
        python train.py --config configs/brax/halfcheetah.yaml --wandb-project=$wandb_project --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --set learning_rate=$lr
        python train.py --config configs/brax/reacher.yaml --wandb-project=$wandb_project --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --set learning_rate=$lr
        python train.py --config configs/brax/swimmer.yaml --wandb-project=$wandb_project --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --set learning_rate=$lr
    done
done

