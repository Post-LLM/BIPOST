set -x

read -r -d '' training_commands <<EOF
bipost.cli.train_bi_objective \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --lora_alpha 16 \
   --lora_rank 16 \
   --target_module q_proj v_proj \

   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --learning_rate 1e-5 \

   --obj_1 KD \
   --dataset_1 Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --max_samples_1 6400 \

   --ref_model meta-llama/Meta-Llama-3-8B-Instruct \
   
   --obj_2 SFT \
   --dataset_2 Open-Orca/OpenOrca \
   --input_key_2 question \
   --output_key_2 response \
   --max_samples_2 6400 \
   
   --save_path ./checkpoint/llama3-8b-kd-sft \
   --max_epochs 2 \
   --zero_stage 3 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --bf16 \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing 

EOF
    # --use_wandb [WANDB_TOKENS]

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
