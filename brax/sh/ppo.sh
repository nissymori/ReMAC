export CUDA_VISIBLE_DEVICES=0
total_timesteps=3000000
eval_freq=30000
num_seeds=10
seed=1

wandb_project="brax-remax-ac-report"



# ppo
python train.py --config configs/brax/hopper.yaml --algorithm ppo --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/ant.yaml --algorithm ppo --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/reacher.yaml --algorithm ppo --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/swimmer.yaml --algorithm ppo --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/halfcheetah.yaml --algorithm ppo --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/walker2d.yaml --algorithm ppo --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --wandb-project=$wandb_project --seed_id=$seed
