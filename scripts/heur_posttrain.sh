#!/bin/bash
# Heuristic policy of [42] (BHAR Fig. 12: send after harvesting alpha*E_th or after 10 s, alpha = 1)
# with BHAR's TinyHAR time-context classifier: finetune + evaluate with the artifact's own
# train_har_policy_heur.py, same settings as scripts/bhar_posttrain.sh.
#
# usage:  bash scripts/heur_posttrain.sh SEED DATASET
# results: external/BHAR_codes_artifact/saved_data/results/<DATASET>/tinyhar/heuristic-asynchronous_multisensor_time_context/results_seed<SEED>.pickle
set -e
seed=$1; dataset=${2:-pamap2}
case $dataset in
  pamap2)      data=(--subjects 1 2 3 4 5 6 7 8 --body_parts hand chest ankle
                     --activities 0 1 2 3 4 5 6 7 8 9 10 11 --val_frac 0.2 --batch_sampling shuffle); ft=(32 3e-4) ;;
  dsads)       data=(--subjects 1 2 3 4 5 6 7 8 --body_parts torso right_arm left_arm right_leg left_leg
                     --activities $(seq 0 18) --val_frac 0.1 --batch_sampling weighted); ft=(32 3e-4) ;;
  rwhar)       data=(--subjects 1 4 5 7 9 10 11 12 13 14 15 --body_parts chest forearm head shin thigh upperarm waist
                     --activities 0 1 2 3 4 5 6 7 --val_frac 0.25 --batch_sampling weighted); ft=(32 3e-4) ;;
  opportunity) data=(--subjects 1 2 3 4 --body_parts BACK RUA RLA LUA LLA L-SHOE R-SHOE
                     --activities 0 1 2 3 4 --val_frac 0.1 --batch_sampling weighted); ft=(16 1e-3) ;;
  *) echo "unknown dataset $dataset"; exit 1 ;;
esac
ART="$(cd "$(dirname "$0")/../external/BHAR_codes_artifact" && pwd)"
# train_har_policy_heur.py reads the pretrained classifier from saved_data; point it at the cached one
pre=checkpoints/$dataset/tinyhar/multisensor_classifier_window_8_acc
if [ ! -e "$ART/saved_data/$pre" ]; then
  mkdir -p "$(dirname "$ART/saved_data/$pre")"
  ln -s "$ART/saved_data_cached/$pre" "$ART/saved_data/$pre"
fi
cd "$ART/experiments"
python train_har_policy_heur.py \
  --multisensor_checkpoint_prefix multisensor_classifier_window_8_acc \
  --logging_prefix heuristic-asynchronous_multisensor_time_context \
  --policy heuristic --heur_alpha 1 \
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
  --classifier_source scratch \
  "${data[@]}"
