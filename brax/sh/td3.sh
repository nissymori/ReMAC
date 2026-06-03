export CUDA_VISIBLE_DEVICES=1
total_timesteps=3000000
eval_freq=30000
num_seeds=10
seed=1

wandb_project="brax-remax-ac-report"



# td3
python train.py --config configs/brax/hopper.yaml --algorithm td3 --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/swimmer.yaml --algorithm td3 --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/halfcheetah.yaml --algorithm td3 --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/walker2d.yaml --algorithm td3 --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --wandb-project=$wandb_project --seed_id=$seed