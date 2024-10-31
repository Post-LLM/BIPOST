import argparse
import math
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime

from transformers.trainer import get_scheduler

from bipost.datasets import RewardDataset, SFTDataset
from bipost.models import Actor
from bipost.trainer import BiObjTrainer
from bipost.utils import blending_datasets, get_strategy, get_tokenizer

def get_datalaoders(strategy, obj_index, tokenizer):
    args = strategy.args
    obj_type = getattr(args, f"obj_{obj_index}")
    if obj_type == "SFT" or obj_type=="KD":
        train_data, eval_data = blending_datasets(
            getattr(args, f"dataset_{obj_index}"),
            getattr(args, f"dataset_probs_{obj_index}"),
            strategy,
            args.seed,
            max_count=getattr(args, f"max_samples_{obj_index}"),
            train_split=getattr(args, f"train_split_{obj_index}"),
            eval_split=getattr(args, f"eval_split_{obj_index}"),
        )
        train_data = train_data.select(range(min(getattr(args, f"max_samples_{obj_index}"), len(train_data))))
        eval_data = eval_data.select(range(min(getattr(args, f"max_samples_{obj_index}"), len(eval_data))))
        train_dataset = SFTDataset(
            train_data,
            tokenizer,
            getattr(args, f"max_len_{obj_index}"),
            strategy,
            input_template=getattr(args, f"input_template_{obj_index}"),
            obj_index=obj_index
        )
        eval_dataset = SFTDataset(
            eval_data,
            tokenizer,
            getattr(args, f"max_len_{obj_index}"),
            strategy,
            input_template=getattr(args, f"input_template_{obj_index}"),
            obj_index=obj_index
        )
        train_dataloader = strategy.setup_dataloader(
            train_dataset,
            # getattr(args, f"micro_train_batch_size_{obj_index}"),
            getattr(args, "micro_train_batch_size"),
            True,
            True,
            train_dataset.collate_fn,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            # getattr(args, f"micro_train_batch_size_{obj_index}"),
            getattr(args, "micro_train_batch_size"),
            True,
            False,
            eval_dataset.collate_fn,
        )

    if obj_type == "DPO":
        train_data, eval_data = blending_datasets(
            getattr(args, f"dataset_{obj_index}"),
            getattr(args, f"dataset_probs_{obj_index}"),
            strategy,
            args.seed,
            max_count=getattr(args, f"max_samples_{obj_index}"),
            stopping_strategy="all_exhausted",
            train_split=getattr(args, f"train_split_{obj_index}"),
            eval_split=getattr(args, f"eval_split_{obj_index}"),
        )
        train_data = train_data.select(range(min(getattr(args, f"max_samples_{obj_index}"), len(train_data))))
        eval_data = eval_data.select(range(min(getattr(args, f"max_samples_{obj_index}"), len(eval_data))))
        train_dataset = RewardDataset(
            train_data, 
            tokenizer, 
            getattr(args, f"max_len_{obj_index}"), 
            strategy, 
            input_template=getattr(args, f"input_template_{obj_index}"), 
            is_dpo=True, 
            obj_index=obj_index
        )
        eval_dataset = RewardDataset(
            eval_data, 
            tokenizer, 
            getattr(args, f"max_len_{obj_index}"), 
            strategy, 
            input_template=getattr(args, f"input_template_{obj_index}"), 
            is_dpo=True, 
            obj_index=obj_index
        )

        train_dataloader = strategy.setup_dataloader(
            train_dataset,
            # getattr(args, f"micro_train_batch_size_{obj_index}"),
            getattr(args, "micro_train_batch_size"),
            True,
            True,
            train_dataset.collate_fn,
        )

        eval_dataloader = strategy.setup_dataloader(
            eval_dataset, 
            # getattr(args, f"micro_train_batch_size_{obj_index}"), 
            getattr(args, "micro_train_batch_size"),
            True, False, eval_dataset.collate_fn
        )

    return train_dataloader, eval_dataloader

def train(args):
    # process the referece pareto values needed for stopping criteria of MOO methods
    if args.ref_pareto_1=="":
        ref_pareto_front = [[0.0, 0.0]]
    else:
        ref_pareto_1 = [float(_) for _ in args.ref_pareto_1.split()]
        ref_pareto_2 = [float(_) for _ in args.ref_pareto_2.split()]

        assert len(ref_pareto_1)== len(ref_pareto_2)

        ref_pareto_front = []
        for _1, _2 in zip(ref_pareto_1, ref_pareto_2):
            ref_pareto_front.append([_1, _2])


    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )

    if args.ref_model:
        ref_model = Actor(
            args.ref_model,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
        )
        if args.ref_offload:
            ref_model._offload = True
        if args.ref_model_2:
            ref_model_2 = Actor(
                args.ref_model_2,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
        )
            if args.ref_offload_2:
                ref_model_2._offload = True
        else:
            ref_model_2=None
    else:
        ref_model=None
        ref_model_2=None

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

 
    train_dataloader_1, eval_dataloader_1 = get_datalaoders(strategy, 1, tokenizer)
    train_dataloader_2, eval_dataloader_2 = get_datalaoders(strategy, 2, tokenizer)

    # scheduler #todo:enable non-constant scheduler
    # num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    # max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    # scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optim,
    #     num_warmup_steps=math.ceil(max_steps * 0.03),
    #     num_training_steps=max_steps,
    #     scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    # )
    scheduler = get_scheduler(
        "constant",
        optim,
    )

    # strategy prepare
    if args.ref_model and args.ref_model_2:
        ((model, optim, scheduler), ref_model, ref_model_2) = strategy.prepare((model, optim, scheduler), ref_model, ref_model_2)
    elif args.ref_model:
        ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)
    else:
        model, optim, scheduler = strategy.prepare((model, optim, scheduler))

    if args.load_checkpoint:
        # ! Only printing, but not actually loading?
        strategy.print("Load checkpoint: ", args.save_path)
        # strategy.load_checkpoint(args.save_path + '/rm_model.pt')

    os.makedirs(args.save_path, exist_ok=True)

    trainer = BiObjTrainer(
            model=model,
            strategy=strategy,
            tokenizer=tokenizer,
            optim=optim,
            train_dataloader_1=train_dataloader_1,
            train_dataloader_2=train_dataloader_2,
            eval_dataloader_1=eval_dataloader_1,
            eval_dataloader_2=eval_dataloader_2,
            scheduler=scheduler,
            lambd=args.lambd,
            ref_pareto_front=ref_pareto_front,
            ref_model=ref_model,
            ref_model_2=ref_model_2,
        )

    trainer.fit(args)
    # # save model checkpoint after fitting on only rank0 #todo: no saving?
    # strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Logging and saving
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_biobj")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB

    ## Efficient training args
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size") #todo: diff batch size and micro batch size?
    parser.add_argument("--micro_train_batch_size", type=int, default=1, help="Global training batch size") #todo: diff batch size and micro batch size?
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)

    ## Model
    parser.add_argument("--pretrain", type=str, default=None, help="remote or local initial model path")
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    ## General optimization args
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--lambd", type=float, default=0.5, help="coefficient for mixing ojective 1 and objective 2")
    parser.add_argument("--eps", type=float, default=1e-4, help="stopping threshold tolerance")
    parser.add_argument("--lr_scheduler", type=str, default="constant") #cosine_with_min_lr
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="initial learning rate")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay coefficient")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    # Objective 1
    parser.add_argument("--obj_1", type=str, default="SFT", help="objective 1 type")
    parser.add_argument("--aux_loss_coef_1", type=float, default=0, help="MoE balancing loss coefficient for objective 1")
    parser.add_argument("--obj_opt_1", type=float, default=0.0, help="optimum value of objective 1")
    parser.add_argument("--ref_pareto_1", type=str, default="", help="set of reference pareto front values for objective 1")
    # Objective 2
    parser.add_argument("--obj_2", type=str, default="SFT", help="objective 2 type")
    parser.add_argument("--max_len_2", type=int, default=2048, help="Max tokens for the samples of dataset 2")
    parser.add_argument("--aux_loss_coef_2", type=float, default=0, help="MoE balancing loss coefficient for objective 2")
    parser.add_argument("--obj_opt_2", type=float, default=0.0, help="optimum value of objective 2")
    parser.add_argument("--ref_pareto_2", type=str, default="", help="set of reference pareto front values for objective 2")


    ## Dataset args
    # Dataset 1
    parser.add_argument("--dataset_1", type=str, default=None)
    parser.add_argument("--dataset_probs_1", type=str, default="1.0", help="sampling probs for dataset 1")
    parser.add_argument("--train_split_1", type=str, default="train", help="train split of datset 1")
    parser.add_argument("--eval_split_1", type=str, default="test", help="test split of the dataset 1")
    parser.add_argument("--input_template_1", type=str, default="User: {}\nAssistant: ")
    parser.add_argument("--max_samples_1", type=int, default=1e8, help="Max number of samples for dataset 1")
    parser.add_argument("--max_len_1", type=int, default=2048, help="Max tokens for the samples of dataset 1")
    parser.add_argument(
        "--apply_chat_template_1", action="store_true", default=False, help="Use HF tokenizer chat template for dataset 1" 
    ) # under development, recommend use input_template instead
    parser.add_argument("--tokenizer_chat_template_1", type=str, default=None)
    # Dataset 2
    parser.add_argument("--dataset_2", type=str, default=None)
    parser.add_argument("--dataset_probs_2", type=str, default="1.0", help="sampling probs for dataset 2")
    parser.add_argument("--train_split_2", type=str, default="train", help="train split of datset 2")
    parser.add_argument("--eval_split_2", type=str, default="test", help="test split of the dataset 2")
    parser.add_argument("--input_template_2", type=str, default="User: {}\nAssistant: ")
    parser.add_argument("--max_samples_2", type=int, default=1e8, help="Max number of samples for dataset 2")
    parser.add_argument(
        "--apply_chat_template_2", action="store_true", default=False, help="Use HF tokenizer chat template for dataset 2"
    ) # under development, recommend use input_template instead
    parser.add_argument("--tokenizer_chat_template_2", type=str, default=None)

    # (input,target output) data format
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="output", help="JSON dataset key")
    # when both datasets are (input,target output) format, these will be used for dataset 2
    parser.add_argument("--input_key_2", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key_2", type=str, default="output", help="JSON dataset key")

    # preference dataset format (prompt,preferred,dispreferred)
    parser.add_argument("--pref_prompt_key", type=str, default="prompt")
    parser.add_argument("--pref_chosen_key", type=str, default="chosen")
    parser.add_argument("--pref_rejected_key", type=str, default="rejected")
     # when both datasets are preference dataset format (prompt,preferred,dispreferred), these will be used for dataset 2
    parser.add_argument("--pref_prompt_key_2", type=str, default="prompt")
    parser.add_argument("--pref_chosen_key_2", type=str, default="chosen")
    parser.add_argument("--pref_rejected_key_2", type=str, default="rejected")

    ## Obj unique args
    # DPO and KD
    parser.add_argument("--ref_model", type=str, default=None) # teacher model in KD loss, reference model in DPO loss
    # when obj1 is dpo or kd, obj2 is dpo or kd, ref_model_2 will be used for the obj_2
    parser.add_argument("--ref_model_2", type=str, default=None)

    # DPO args
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--dpo_ipo", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--dpo_label_smoothing", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    parser.add_argument(
        "--dpo_nll_loss_coef", type=float, default=0, help="Regularization with NLL loss, see LLama 3.1 tech report."
    )  
    # when obj_1 and obj_2 both dpo, these will be used for the obj_2
    parser.add_argument("--dpo_beta_2", type=float, default=0.1)
    parser.add_argument("--dpo_ipo_2", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--dpo_label_smoothing_2", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf\
    parser.add_argument(
        "--dpo_nll_loss_coef_2", type=float, default=0, help="Regularization with NLL loss, see LLama 3.1 tech report."
    )  

    ## remote logging
    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="bipost_alright")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="exp_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    args = parser.parse_args()

    if args.obj_1 in ["DPO","KTO"] or args.obj_2 in ["DPO","KTO"]:
        if args.ref_model is None or args.ref_model == "":
            args.ref_model = args.pretrain

    train(args)
