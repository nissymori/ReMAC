total_timesteps=3000000
eval_freq=300000
num_seeds=10
seed=1

wandb_project="brax-remax-ac-report-speedbench"


python train.py --config configs/brax/halfcheetah.yaml --algorithm sac --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb-project=$wandb_project --seed_id=$seed --wandb
for m in 4; do
    for actor_epsilon in 1e-8; do
        python train.py --config configs/brax/halfcheetah.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=8 --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/halfcheetah.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=16 --wandb-project=$wandb_project --seed_id=$seed
    done
done


