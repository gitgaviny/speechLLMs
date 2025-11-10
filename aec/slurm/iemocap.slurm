#!/bin/bash
#SBATCH --job-name=slm
#SBATCH --partition=002-partition-debug
#SBATCH --gpus=8
#SBATCH --container-image=/lustre/container_cache/slam-llm/slam-llm-latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre
#SBATCH --exclusive


export HF_HOME="/lustre/teams/speech/hf_cache/"
export WORKING_DIR=/lustre/users/gao/speechLLMs/aec/slurm

cd $WORKING_DIR

stage=3
stop_stage=4
epoch=50
corpus=iemocap_aec
# pretrained_model=/lustre/users/gao/speechLLMs/exp/wavlm-Llama-3.2-1B-encoder_unfreeze-decoder_freeze-librispeech-5
encoder=wavlm
decoder=Llama-3.2-1B-Instruct
fused_model=true

encoder_freeze=false
decoder_freeze=true
adapter_only_decoder=true
instruct=true
talker_ctc=true
eval_steps=400
virtual_env=/lustre/users/gao/envs/slmser/venv

if [ "$fused_model" = true ]; then
  slm_dir=dump_ff
  output_dir=exp_ff3
fi
if [ "$fused_model" = false ]; then
  slm_dir=dump
  output_dir=exp
fi

partial_encoder_unfreeze=""
partial_decoder_unfreeze="lm_head,embed_tokens,embed_positions,layernorm_embedding"
partial_others_unfreeze="enc_to_dec_proj"

args=(
  stage=$stage
  stop_stage=$stop_stage
  epoch=$epoch
  corpus=$corpus
  # pretrained_model=$pretrained_model   
  encoder=$encoder
  decoder=$decoder
  fused_model=$fused_model
  encoder_freeze=$encoder_freeze
  decoder_freeze=$decoder_freeze
  partial_encoder_unfreeze=$partial_encoder_unfreeze
  partial_decoder_unfreeze=$partial_decoder_unfreeze
  partial_others_unfreeze=$partial_others_unfreeze
  adapter_only_decoder=$adapter_only_decoder
  instruct=$instruct
  talker_ctc=$talker_ctc
  slm_dir=${slm_dir}
  output_dir=${output_dir}
  eval_steps=${eval_steps}
  virtual_env=${virtual_env}
)

bash ../iemocap.sh "${args[@]}"

