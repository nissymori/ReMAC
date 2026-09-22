# Run from anywhere: every path below is relative to brax/.
cd "$(dirname "$0")/../../brax"

gpu=$1
m=$2

export CUDA_VISIBLE_DEVICES=$gpu
total_timesteps=3000000
eval_freq=30000
num_seeds=2
seed=1


wandb_project="brax-remax-ac-report"

for seed_id in 2 3 4 5 6; do
    if [ "$m" -eq 8 ]; then
    remax_num_samples=16
    else
    remax_num_samples=8
    fi
    for actor_epsilon in 0.1 1.0; do
        python train.py --config configs/brax/humanoidstandup.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed_id
    done
done