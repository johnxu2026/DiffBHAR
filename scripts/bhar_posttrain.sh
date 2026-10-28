#!/bin/bash
# Run BHAR's own posttraining (policy, finetune, evaluation) with TinyHAR,
# with the args of experiments/posttraining_scripts/*_time_context-<DATASET>.sh and the
# cached pretrained classifiers. Use the artifact's own venv.
#
# usage:  bash scripts/bhar_posttrain.sh {conservative|opportunistic} SEED {cached|scratch} [TAG] [DATASET]
#   DATASET: pamap2 (default), dsads, rwhar or opportunity; TAG may be "" to give a DATASET
#   env POLICY_EPOCHS overrides BHAR's pattern-search epochs (budget control)
#   cached:  reuse BHAR's learned policy and finetuned classifier (evaluation only)
#   scratch: learn the policy (pattern search) and finetune the classifier again
#   TAG:     run under <TAG>/<method>; BHAR then looks for the policy in
#            saved_data/checkpoints/<DATASET>/conservative-asynchronous_single_sensor/
#            (e.g. TAG=e2e after scripts/bhar_e2e.py wrote its theta there)
# results: external/BHAR_codes_artifact/saved_data/results/<DATASET>/tinyhar/[TAG/]<method>/results_seed<SEED>.pickle
set -e
policy=$1; seed=$2; source=$3; tag=${4:+$4/}; dataset=${5:-pamap2}
case $dataset in
  pamap2)      data=(--subjects 1 2 3 4 5 6 7 8 --body_parts hand chest ankle
                     --activities 0 1 2 3 4 5 6 7 8 9 10 11 --val_frac 0.2 --batch_sampling shuffle)
               lr=200; ep=31; ft=(32 3e-4) ;;
  dsads)       data=(--subjects 1 2 3 4 5 6 7 8 --body_parts torso right_arm left_arm right_leg left_leg
                     --activities $(seq 0 18) --val_frac 0.1 --batch_sampling weighted)
               lr=500; ep=31; ft=(32 3e-4) ;;
  rwhar)       data=(--subjects 1 4 5 7 9 10 11 12 13 14 15 --body_parts chest forearm head shin thigh upperarm waist
                     --activities 0 1 2 3 4 5 6 7 --val_frac 0.25 --batch_sampling weighted)
               lr=100; ep=26; ft=(32 3e-4) ;;
  opportunity) data=(--subjects 1 2 3 4 --body_parts BACK RUA RLA LUA LLA L-SHOE R-SHOE
                     --activities 0 1 2 3 4 --val_frac 0.1 --batch_sampling weighted)
               lr=500; ep=31; ft=(16 1e-3) ;;
  *) echo "unknown dataset $dataset"; exit 1 ;;
esac
ep=${POLICY_EPOCHS:-$ep}   # e.g. POLICY_EPOCHS=512: same number of 100 s segments as bhar_e2e.py (16 x 64 / 2)
cd "$(dirname "$0")/../external/BHAR_codes_artifact/experiments"

policy_args=()
if [ "$policy" = conservative ]; then
  policy_args=(--single_sensor_checkpoint_prefix single_sensor_classifier_window_8_acc
               --policy_batch_size 2 --policy_lr $lr $lr --policy_epochs $ep --policy_val_every_epochs 5
               --policy_param_init_vals 0. 0. --policy_param_min_vals -1000 -1000
               --policy_param_max_vals 1000 1000 --policy_source "$source")
fi

python ../../../scripts/bhar_hook.py train_har_policy.py \
  --multisensor_checkpoint_prefix multisensor_classifier_window_8_acc \
  --logging_prefix "$tag$policy"-asynchronous_multisensor_time_context \
  --policy "$policy" \
  --model_type asynchronous_multisensor_time_context \
  --architecture tinyhar \
  --dataset "$dataset" \
  --seed "$seed" \
  --sensors acc \
  --window_size 8 \
  --harvesting_sensor_window_size 8 \
  --leakage 6.6e-6 \
  --policy_frequency 5 \
  --init_energy 0 \
  --sampling_frequency 25 \
  --max_energy 200e-6 \
  --finetune_batch_size "${ft[0]}" \
  --finetune_lr "${ft[1]}" \
  --finetune_epochs 10 \
  --pretrain_source cached \
  --finetune_source "$source" \
  "${data[@]}" "${policy_args[@]}"
