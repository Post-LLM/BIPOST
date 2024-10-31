set -x

read -r -d '' training_commands <<EOF
bipost.cli.train_bi_objective \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --lora_alpha 16 \
   --lora_rank 16 \
   --target_module q_proj v_proj \
   --lambd 0.5 \

   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --learning_rate 1e-5 \

   
   --obj_1 DPO \
   --dataset_1 Dahoas/rm-hh-rlhf \
   --pref_prompt_key prompt \
   --pref_chosen_key chosen \
   --pref_rejected_key rejected \
   --max_samples_1 9600 \

   --ref_model meta-llama/Meta-Llama-3-8B-Instruct \
   
   --obj_2 SFT \
   --dataset_2 Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --max_samples_2 9600 \
   
   --save_path ./checkpoint/llama3-8b-dpo-sft \
   --max_epochs 3 \
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
