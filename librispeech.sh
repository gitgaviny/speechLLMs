#!/bin/bash

set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
root_dir="$SCRIPT_DIR"

cd "$root_dir"
echo "++++ Current path: ${root_dir}"
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# General configuration
for arg in "$@"; do
  case $arg in
    stage=*)                stage="${arg#*=}" ;;
    stop_stage=*)           stop_stage="${arg#*=}" ;;
    epoch=*)                epoch="${arg#*=}" ;;
    corpus=*)               corpus="${arg#*=}" ;;
    pretrained_model=*)     pretrained_model="${arg#*=}" ;;
    encoder=*)              encoder="${arg#*=}" ;;
    decoder=*)              decoder="${arg#*=}" ;;
    encoder_freeze=*)       encoder_freeze="${arg#*=}" ;;
    decoder_freeze=*)       decoder_freeze="${arg#*=}" ;;
    adapter_only_decoder=*) adapter_only_decoder="${arg#*=}" ;;
    instruct=*)             instruct="${arg#*=}" ;;
    talker_ctc=*)           talker_ctc="${arg#*=}" ;;
    output_dir=*)           output_dir="${arg#*=}" ;;
    partial_encoder_unfreeze=*)      partial_encoder_unfreeze="${arg#*=}" ;;
    partial_decoder_unfreeze=*)       partial_decoder_unfreeze="${arg#*=}" ;;
    partial_others_unfreeze=*)       partial_others_unfreeze="${arg#*=}" ;;
    eval_steps=*)           eval_steps="${arg#*=}" ;;
    virtual_env=*)          virtual_env="${arg#*=}" ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

echo "++++ stage=$stage"
echo "++++ stop_stage=$stop_stage"
echo "++++ epoch=$epoch"
echo "++++ corpus=$corpus"
echo "++++ pretrained_model=${pretrained_model:-}"
echo "++++ encoder=$encoder"
echo "++++ decoder=$decoder"
echo "++++ encoder_freeze=$encoder_freeze"
echo "++++ decoder_freeze=$decoder_freeze"
echo "++++ adapter_only_decoder=$adapter_only_decoder"
echo "++++ instruct=$instruct"
echo "++++ talker_ctc=$talker_ctc"
echo "++++ partial_encoder_unfreeze=$partial_encoder_unfreeze"
echo "++++ partial_decoder_unfreeze=$partial_decoder_unfreeze"
echo "++++ partial_others_unfreeze=$partial_others_unfreeze"
echo "++++ eval_steps=$eval_steps"
echo "++++ virtual_env=$virtual_env"

output_dir=${output_dir}/${encoder}-${decoder}
if [ "${encoder_freeze}" = "true" ]; then
    output_dir="${output_dir}-encoder_freeze"
else
    output_dir="${output_dir}-encoder_unfreeze"
fi
if [ "${decoder_freeze}" = "true" ]; then
    output_dir="${output_dir}-decoder_freeze"
else
    output_dir="${output_dir}-decoder_unfreeze"
fi
# if [ "${adapter_only_decoder}" = "true" ]; then
#     output_dir="${output_dir}-adater_decoder"
# else
#     output_dir="${output_dir}-adater_encoder_decoder"
# fi
output_dir=${output_dir}-${corpus}-${epoch}
echo "++++ output_dir=$output_dir"

extra_args=""
if [[ "$instruct" == "true" ]]; then
  extra_args+=" --instruct"
fi


PY_BIN="$virtual_env/bin/python"

echo "== Node: $(hostname)"
echo "== Before fix: show current venv python status"
ls -l "$PY_BIN" || echo "ls failed (maybe broken symlink)"
echo "Resolved -> $(readlink -f "$PY_BIN" || echo NA)"
echo "System python on this node:"
command -v python3 || true
python3 -V || true
echo "Available /usr/bin/python* on this node:"
ls -l /usr/bin/python* 2>/dev/null || true

# --- Choose a working system Python on this compute node ---
# Prefer Python 3.10 to match your original venv; fallback to 3.11 or 3.8 if needed.
TARGET_PY="$(command -v python3.10 || true)"
if [[ -z "${TARGET_PY}" ]]; then
  # If 'python3' is 3.10 on this node, use it
  if python3 -V 2>/dev/null | grep -qE '^Python 3\.10\.'; then
    TARGET_PY="$(command -v python3)"
  fi
fi
# Last resort: allow 3.11 or 3.8 (may require reinstalling wheels later)
if [[ -z "${TARGET_PY}" ]]; then
  TARGET_PY="$(command -v python3.11 || true)"
fi
if [[ -z "${TARGET_PY}" ]]; then
  TARGET_PY="$(command -v python3.8 || true)"
fi

if [[ -z "${TARGET_PY}" ]]; then
  echo "ERROR: No suitable system python3 found on this node." >&2
  exit 1
fi
echo "Using system Python: $TARGET_PY ($("$TARGET_PY" -V 2>&1))"

# --- Repair venv python symlinks if broken and align with node's interpreter ---
# Reason: your venv/bin/python pointed to /usr/bin/python3.10, which doesn’t exist on some nodes.
if [[ ! -x "$PY_BIN" ]] || { [[ -L "$PY_BIN" ]] && [[ ! -e "$(readlink -f "$PY_BIN")" ]]; }; then
  echo ">>> Fixing venv python links to: $TARGET_PY"
  ln -sf "$TARGET_PY" "$virtual_env/bin/python"
  ln -sf "$TARGET_PY" "$virtual_env/bin/python3"
  # If the chosen interpreter is 3.10, also create python3.10 name for tools that expect it
  if "$TARGET_PY" -V 2>&1 | grep -qE '^Python 3\.10\.'; then
    ln -sf "$TARGET_PY" "$virtual_env/bin/python3.10"
  fi

  # Upgrade venv layout in-place so all scripts' shebangs point to the new interpreter
  "$TARGET_PY" -m venv --upgrade "$virtual_env"
fi


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1



# 1. Data preparing
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    "$PY_BIN" utils/generate_dataset.py \
        --base_data_path /lustre/users/gao/SLM_datasets/librispeech \
	--suffix _clean \
	--wav_scp_name wav.scp \
	--output_dir datasets/librispeech
fi

# 2. Create the pre-trained AED from pre-trained speech encoder and LLMs
model_ids=${encoder}-${decoder}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  if [ "${instruct}" = "true" ]; then
    save_dir="dump_instruct/${model_ids}"
  else
    save_dir="dump/${model_ids}"
  fi

  "$PY_BIN" utils/create_from_pretrained.py \
    --encoder_id microsoft/wavlm-large \
    --decoder_base /lustre/share/downloaded/models/meta-llama \
    --llm_id ${decoder} \
    --save_dir "${save_dir}" \
    --talker_ctc \
    --check_generate \
    $extra_args
fi

# 3. Training
NUM_GPUS=$("$PY_BIN" -c 'import torch; print(torch.cuda.device_count())')
echo "Detected $NUM_GPUS GPUs"

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  if [[ -n "${pretrained_model:-}" ]]; then
    model_path="${pretrained_model}"
  else
    if [ "${instruct}" = "true" ]; then
      model_path="dump_instruct/${model_ids}"
    else
      model_path="dump/${model_ids}"
    fi
  fi

  "$PY_BIN" -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS finetune.py \
    --dataset_name="datasets/${corpus}" \
    --model_name_or_path="${model_path}" \
    --train_split_name="train" \
    --eval_split_name="validation" \
    --adapter_only_decoder=${adapter_only_decoder} \
    --output_dir=${output_dir} \
    --metric_for_best_model="eval_loss" \
    --greater_is_better=false \
    --preprocessing_num_workers="16" \
    --audio_column_name="audio" \
    --text_column_name="text" \
    --overwrite_output_dir false \
    --num_train_epochs=${epoch} \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --gradient_accumulation_steps="1" \
    --learning_rate="3e-5" \
    --warmup_steps="400" \
    --evaluation_strategy="steps" \
    --save_steps="1600" \
    --eval_steps=${eval_steps} \
    --logging_steps="10" \
    --save_total_limit="5" \
    --freeze_feature_encoder true \
    --freeze_encoder ${encoder_freeze} \
    --freeze_decoder ${decoder_freeze} \
    --partial_encoder_unfreeze="${partial_encoder_unfreeze}" \
    --partial_decoder_unfreeze="${partial_decoder_unfreeze}" \
    --partial_others_unfreeze="${partial_others_unfreeze}" \
    --gradient_checkpointing \
    --fp16 \
    --group_by_length \
    --predict_with_generate \
    --do_train true \
    --do_eval true \
    --do_lower_case

  "$PY_BIN" utils/merge_adapter.py ${output_dir}
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  _set="validation test"

  for subset in $_set; do
    dataset_name="data/${corpus}/${subset}"

    "$PY_BIN" -m torch.distributed.launch \
      --nproc_per_node 1 inference.py \
      --dataset_name="datasets/${corpus}/${subset}" \
      --model_name_or_path="${output_dir}" \
      --adapter_only_decoder="${adapter_only_decoder}" \
      --output_dir="${output_dir}" \
      --metric_for_best_model="eval_loss" \
      --greater_is_better=false \
      --preprocessing_num_workers="16" \
      --audio_column_name="audio" \
      --text_column_name="text" \
      --num_train_epochs="${epoch}" \
      --per_device_train_batch_size="16" \
      --per_device_eval_batch_size="16" \
      --gradient_accumulation_steps="1" \
      --learning_rate="3e-5" \
      --warmup_steps="400" \
      --evaluation_strategy="steps" \
      --save_steps="1600" \
      --eval_steps="${eval_steps}" \
      --logging_steps="10" \
      --save_total_limit="5" \
      --freeze_feature_encoder=true \
      --freeze_encoder="${encoder_freeze}" \
      --freeze_decoder="${decoder_freeze}" \
      --partial_encoder_unfreeze="${partial_encoder_unfreeze}" \
      --partial_decoder_unfreeze="${partial_decoder_unfreeze}" \
      --partial_others_unfreeze="${partial_others_unfreeze}" \
      --gradient_checkpointing \
      --fp16 \
      --group_by_length \
      --predict_with_generate \
      --do_train=false \
      --do_eval=true \
      --do_lower_case

    "$PY_BIN" utils/compute-wer.py \
      --char=1 \
      --v=1 \
      "${output_dir}/${subset}_label.text" "${output_dir}/${subset}_decod.text" \
      > "${output_dir}/${subset}_decod.wer"

    "$PY_BIN" utils/compute-accuracy.py \
      "${output_dir}/${subset}_label_emotion.text" "${output_dir}/${subset}_decod_emotion.text" \
      > "${output_dir}/${subset}_decod.accuracy"
  done

  for subset in $_set; do
    echo "==> ${output_dir} ${subset}"
    tail -n 5 "${output_dir}/${subset}_decod.wer" || true
    cat "${output_dir}/${subset}_decod.accuracy" || true
  done
fi



