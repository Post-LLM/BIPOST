set -x

read -r -d '' training_commands <<EOF
bipost.cli.train_bi_objective \
   --pretrain EleutherAI/pythia-1b \
   --lora_alpha  16 \
   --lora_rank 16  \
   --target_module query_key_value \

   --train_batch_size 16 \
   --micro_train_batch_size 1 \
   --learning_rate 1e-5 \

   --obj_1 KD\
   --dataset_1  Open-Orca/OpenOrca\
   --input_key question \
   --output_key response\
   --max_samples_1  160\

   --ref_model  yongzx/pythia-1b-sft-hh\
   
   --obj_2 SFT \
   --dataset_2 garage-bAInd/Open-Platypus \
   --input_key_2 instruction \
   --output_key_2 output \
   --max_samples_2 160\
   
   --save_path ./checkpoint/test_model \
   --max_epochs  1 \
   --zero_stage  2 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --bf16 \
   --load_checkpoint \
   --gradient_checkpointing 
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --num_gpus 2 --module $training_commands
fi
