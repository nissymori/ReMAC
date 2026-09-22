#!/usr/bin/env bash
# ReMAC on the main tasks, at every Adam epsilon, plus the learning-rate probes.
#
# NOTE on Reacher and Pusher.  The Reacher lines below are superseded: remax_ac
# silently trained on 1000-step episodes there until configs/brax/reacher.yaml was
# fixed, so Reacher was re-tuned and re-run with sh/reacher_{tune,final,eps}.sh, and
# those are the runs the paper reports.  Pusher was added for the camera-ready
# version and has its own phase scripts, sh/pusher_{tune,final,eps,sigma_sgd}.sh.
#
# Run from anywhere: every path below is relative to brax/.
cd "$(dirname "$0")/../../brax"

export CUDA_VISIBLE_DEVICES=0
total_timesteps=3000000
eval_freq=30000
num_seeds=10
seed=1

wandb_project="brax-remax-ac-report"

for m in 1 2 4 8; do
    if [ "$m" -eq 8 ]; then
      remax_num_samples=16
    else
      remax_num_samples=8
    fi
    for actor_epsilon in 1e-8 1 1e-1 1e-2; do
        python train.py --config configs/brax/hopper.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/ant.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/reacher.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/swimmer.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/halfcheetah.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/walker2d.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
    done
done

### Additional for humanoidstandup
for m in 1 2 4 8; do
    if [ "$m" -eq 8 ]; then
      remax_num_samples=16
    else
      remax_num_samples=8
    fi
    for actor_epsilon in 1e-8 1 1e-1; do
        python train.py --config configs/brax/humanoidstandup.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
    done
done

for m in 1 2 4 8; do
    if [ "$m" -eq 8 ]; then
      remax_num_samples=16
    else
      remax_num_samples=8
    fi
    for actor_epsilon in 0.1 0.0001; do
        for lr in 0.0001 0.0003 0.0005 0.001; do
          python train.py --config configs/brax/halfcheetah.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --set learning_rate=$lr --seed_id=$seed
          python train.py --config configs/brax/swimmer.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --set learning_rate=$lr --seed_id=$seed
          python train.py --config configs/brax/walker2d.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --set learning_rate=$lr --seed_id=$seed
        done
    done
done


for m in 1 2 4 8; do
    if [ "$m" -eq 8 ]; then
      remax_num_samples=16
    else
      remax_num_samples=8
    fi
    for actor_epsilon in 1e-8; do
        for lr in 0.00005 0.00003 0.00001; do
          python train.py --config configs/brax/halfcheetah.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --set learning_rate=$lr --seed_id=$seed --set actor_epsilon=$actor_epsilon
          python train.py --config configs/brax/swimmer.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --set learning_rate=$lr --seed_id=$seed --set actor_epsilon=$actor_epsilon
          python train.py --config configs/brax/walker2d.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --set learning_rate=$lr --seed_id=$seed --set actor_epsilon=$actor_epsilon
        done
    done
done

for m in 1 2 4; do
    remax_num_samples=16
    for actor_epsilon in 1e-8; do
        python train.py --config configs/brax/hopper.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/ant.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/reacher.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/swimmer.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/halfcheetah.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/walker2d.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
    done
done

for m in 8; do
    remax_num_samples=8
    for actor_epsilon in 1e-8; do
        python train.py --config configs/brax/hopper.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/ant.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/reacher.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/swimmer.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/halfcheetah.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
        python train.py --config configs/brax/walker2d.yaml --algorithm remax_ac --num-seeds=$num_seeds --set remax_m=$m  --set actor_epsilon=$actor_epsilon --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set remax_num_samples=$remax_num_samples --wandb-project=$wandb_project --seed_id=$seed
    done
done