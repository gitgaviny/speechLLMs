#!/bin/bash
#SBATCH --job-name=slmser_continue
#SBATCH --partition=002-partition-debug
#SBATCH --gpus=8
#SBATCH --container-image=/lustre/container_cache/slam-llm/slam-llm-latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre
#SBATCH --exclusive

cd /lustre/users/gao/speechLLMs/slurm


stage=3
stop_stage=3
epoch=80
corpus=iemocap
pretrained_model=/lustre/users/gao/speechLLMs/exp/wavlm-Llama-3.2-1B-encoder_unfreeze-decoder_freeze-librispeech-5
encoder=wavlm
decoder=Llama-3.2-1B

encoder_freeze=false
decoder_freeze=true
adapter_only_decoder=true
instruct=false
talker_ctc=true
eval_steps=1600
virtual_env=/lustre/users/gao/envs/slmser/venv

output_dir=exp_iemocap_from_librispeech

partial_encoder_unfreeze=""
partial_decoder_unfreeze="lm_head,embed_tokens,embed_positions,layernorm_embedding"
partial_others_unfreeze="enc_to_dec_proj"

bash ../iemocap_from_librispeech.sh \
	stage=$stage \
	stop_stage=$stop_stage \
	epoch=$epoch \
    corpus=$corpus \
	pretrained_model=$pretrained_model \
	encoder=$encoder \
	decoder=$decoder \
	encoder_freeze=$encoder_freeze \
	decoder_freeze=$decoder_freeze \
	partial_encoder_unfreeze=$partial_encoder_unfreeze \
	partial_decoder_unfreeze=$partial_decoder_unfreeze \
	partial_others_unfreeze=$partial_others_unfreeze \
	adapter_only_decoder=$adapter_only_decoder \
	instruct=$instruct \
	talker_ctc=$talker_ctc \
	output_dir=${output_dir} \
	eval_steps=${eval_steps} \
	virtual_env=${virtual_env}

