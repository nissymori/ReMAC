#!/usr/bin/env bash
set -euo pipefail

# defaults
GPU="0"
m="2"

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --gpu=*)
      GPU="${1#*=}"
      shift 1
      ;;
    --m)
      m="$2"
      shift 2
      ;;
    --m=*)
      m="${1#*=}"
      shift 1
      ;;
    -h|--help)
      echo "Usage: $0 [--gpu N] [--m M]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--gpu N] [--m M]" >&2
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
total_timesteps=3000000
eval_freq=30000
num_seeds=10
seed=1

wandb_project="remac-report"
python train.py --config configs/brax/hopper.yaml --algorithm sac --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/ant.yaml --algorithm sac --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/walker2d.yaml --algorithm sac --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/halfcheetah.yaml --algorithm sac --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/reacher.yaml --algorithm sac --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set wandb-project=$wandb_project --seed_id=$seed
python train.py --config configs/brax/swimmer.yaml --algorithm sac --num-seeds=$num_seeds --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq --wandb --set wandb-project=$wandb_project --seed_id=$seed
