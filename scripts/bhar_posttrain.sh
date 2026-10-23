#!/bin/bash
# Run BHAR's own posttraining (policy, finetune, evaluation) on PAMAP2 + TinyHAR,
# with the args of experiments/posttraining_scripts/*_time_context-pamap2.sh and the
# cached pretrained classifiers. Use the artifact's own venv.
#
# usage:  bash scripts/bhar_posttrain.sh {conservative|opportunistic} SEED {cached|scratch}
#   cached:  reuse BHAR's learned policy and finetuned classifier (evaluation only)
#   scratch: learn the policy (pattern search) and finetune the classifier again
# results: external/BHAR_codes_artifact/saved_data/results/pamap2/tinyhar/<method>/results_seed<SEED>.pickle
set -e
policy=$1; seed=$2; source=$3
cd "$(dirname "$0")/../external/BHAR_codes_artifact/experiments"

policy_args=()
if [ "$policy" = conservative ]; then
  policy_args=(--single_sensor_checkpoint_prefix single_sensor_classifier_window_8_acc
               --policy_batch_size 2 --policy_lr 200 200 --policy_epochs 31 --policy_val_every_epochs 5
               --policy_param_init_vals 0. 0. --policy_param_min_vals -1000 -1000
               --policy_param_max_vals 1000 1000 --policy_source "$source")
fi

python train_har_policy.py \
  --multisensor_checkpoint_prefix multisensor_classifier_window_8_acc \
  --logging_prefix "$policy"-asynchronous_multisensor_time_context \
  --policy "$policy" \
  --model_type asynchronous_multisensor_time_context \
  --architecture tinyhar \
  --dataset pamap2 \
  --seed "$seed" \
  --subjects 1 2 3 4 5 6 7 8 \
  --sensors acc \
  --body_parts hand chest ankle \
  --activities 0 1 2 3 4 5 6 7 8 9 10 11 \
  --val_frac 0.2 \
  --window_size 8 \
  --harvesting_sensor_window_size 8 \
  --leakage 6.6e-6 \
  --policy_frequency 5 \
  --init_energy 0 \
  --sampling_frequency 25 \
  --max_energy 200e-6 \
  --finetune_batch_size 32 \
  --finetune_lr 3e-4 \
  --finetune_epochs 10 \
  --batch_sampling shuffle \
  --pretrain_source cached \
  --finetune_source "$source" \
  "${policy_args[@]}"
