#!/bin/bash
#SBATCH --job-name=slmser
#SBATCH --partition=002-partition-debug
#SBATCH --gpus=8
#SBATCH --container-image=/lustre/container_cache/slam-llm/slam-llm-latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre
#SBATCH --exclusive

cd /lustre/users/gao/speechLLMs/instruct_slurm



stage=4
stop_stage=4
epoch=50
corpus=iemocap
encoder=wavlm
decoder=Llama-3.2-1B

encoder_freeze=false
decoder_freeze=true
adapter_only_decoder=true
instruct=true
talker_ctc=true
eval_steps=1600
virtual_env=/lustre/users/gao/envs/slmser/venv

output_dir=exp_instruct

partial_encoder_unfreeze=""
partial_decoder_unfreeze="lm_head,embed_tokens,embed_positions,layernorm_embedding"
partial_others_unfreeze="enc_to_dec_proj"

bash ../iemocap_instruct.sh \
	stage=$stage \
	stop_stage=$stop_stage \
	epoch=$epoch \
    corpus=$corpus \
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

