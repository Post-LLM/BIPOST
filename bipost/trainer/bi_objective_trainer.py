import math
import os
from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm


from bipost.models import DPOLoss, GPTLMLoss, KDLoss
import time

# for recording gpu memory used
import subprocess as sp
import numpy as np

from torch.nn import functional as F
from flash_attn.utils.distributed import all_gather


class BiObjTrainer(ABC):
    """
        Trainer for optimizing two objectives in an alternating manner
    """

    def __init__(
        self,
        model,
        strategy,
        tokenizer,
        optim: Optimizer,
        train_dataloader_1,
        train_dataloader_2,
        eval_dataloader_1,
        eval_dataloader_2,
        scheduler,
        lambd=0.5,
        ref_pareto_front=[[0.0, 0.0]],
        ref_model=None,
        ref_model_2=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = strategy.args.max_epochs
        self.max_norm = strategy.args.max_norm
        self.model = model
        self.ref_model = ref_model
        self.ref_model_2 = ref_model_2
        self.train_dataloader_1 = train_dataloader_1
        self.train_dataloader_2 = train_dataloader_2
        self.eval_dataloader_1 = eval_dataloader_1
        self.eval_dataloader_2 = eval_dataloader_2
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        
        # define loss functions
        self.loss_fn_1 = self.get_loss_fn(obj_index=1)
        self.loss_fn_2 = self.get_loss_fn(obj_index=2)

        self.lambd=lambd
        self.ref_pareto_front = ref_pareto_front

        # Mixtral 8*7b
        self.aux_loss_1 = self.args.aux_loss_coef_1 > 1e-8
        self.aux_loss_2 = self.args.aux_loss_coef_2 > 1e-8

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        self._tensorboard = None
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)


    def fit(self, args):

        # get the maximum train_loader length,
        if self.train_dataloader_1.__len__() >= self.train_dataloader_2.__len__():
            train_loader_len = self.train_dataloader_1.__len__()
        else:
            train_loader_len = self.train_dataloader_2.__len__()
        
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = train_loader_len  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        # For blending objectives
        prob_mass = [self.lambd, 1-self.lambd]

        # flag to check whether sop learning criteria had met
        OPTIM_ACHIEVED = False

        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader_1.sampler, DistributedSampler): # important todo: 1) shuffle logic of dataloader, maybe not shuffle here, only shuffle when dataloader is depleted
                self.train_dataloader_1.sampler.set_epoch(epoch)

            if isinstance(self.train_dataloader_2.sampler, DistributedSampler):
                self.train_dataloader_2.sampler.set_epoch(epoch)
            # Setup iterable train_dataloader
            iter_train_dataloader_1 = iter(self.train_dataloader_1)
            iter_train_dataloader_2 = iter(self.train_dataloader_2)
            step_bar = tqdm(
                range(train_loader_len),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model:
                self.ref_model.eval()
            if self.ref_model_2:
                self.ref_model_2.eval()
            acc_mean = 0
            loss_mean = 0

            # Train SFT and DPO in alternating manner, so implement a train_dataloader agnostic loop
            # assuming both train_dataloaders have the same length
            for step in range(train_loader_len): # important todo: 1) shuffle logic of dataloader, here the dataloader is not shuffled when depleted 2) alternating after each micro step or update step? 3) better definition of epoch

                # Choose the objective to be updated
                # syncironize the objective index across processes  
                obj_index = np.random.choice([1, 2], p=prob_mass)
                
                # clear memory cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if obj_index==1:
                    loss_fn = self.loss_fn_1
                    try:
                        data = next(iter_train_dataloader_1)
                    except StopIteration:
                        iter_train_dataloader_1 = iter(self.train_dataloader_1) 
                        data = next(iter_train_dataloader_1)      

                if obj_index==2:
                    loss_fn = self.loss_fn_2 
                    try:
                        data = next(iter_train_dataloader_2)
                    except StopIteration:
                        iter_train_dataloader_2 = iter(self.train_dataloader_2)
                        data = next(iter_train_dataloader_2)       
                
                loss, logs_dict = self.calc_loss(loss_fn, data, obj_index)

            #    # DEBUG
            #     if self.strategy.is_rank_0():
            #         self._log_gpu_memory_usage('before backward')
            #         self._log_gpu_memory_from_nvdia_smi('before backward')
            #     t = time.time()
                self.strategy.backward(loss, self.model, self.optimizer)
                # # DEBUG
                # if self.strategy.is_rank_0():
                #     self._log_time_elapsed('backward', time.time()-t)
                # # DEBUG
                # if self.strategy.is_rank_0():
                #     self._log_gpu_memory_usage('after backward')
                #     self._log_gpu_memory_from_nvdia_smi('after backward')
                # t = time.time()
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                # # DEBUG
                # if self.strategy.is_rank_0():
                #     self._log_time_elapsed('optimizer_step', time.time()-t)
                    
                # logs/checkpoints/evaluate #todo: output both log dicts of the objs 
                eval_loss_1, eval_loss_2 = self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, obj_index)

                if eval_loss_1 is not None and eval_loss_2 is not None:
                    if eval_loss_1-self.args.obj_opt_1 <= self.args.eps or eval_loss_2-self.args.obj_opt_2 <= self.args.eps:
                        OPTIM_ACHIEVED = True
                    
                    else:
                        for ref_loss_pair in self.ref_pareto_front:
                            if eval_loss_1 - ref_loss_pair[0] <= self.args.eps and eval_loss_2 - ref_loss_pair[1] <= self.args.eps:
                                OPTIM_ACHIEVED = True
                                break

                step_bar.update()
                global_step += 1

                if OPTIM_ACHIEVED:
                    break

            epoch_bar.update()

            if OPTIM_ACHIEVED:
                break

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        elif self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def get_loss_fn(self, obj_index:int):
        obj_type = getattr(self.args, f"obj_{obj_index}")
        # when both objectives are from the same objective type
        if self.args.obj_1==self.args.obj_2 and obj_index==2:
            arg_index='_2'
        else:
            arg_index=''

        if obj_type == "SFT":
            loss_fn = GPTLMLoss()
        elif obj_type == "DPO":
            loss_fn = DPOLoss(
                getattr(self.args, f"dpo_beta{arg_index}"), 
                getattr(self.args, f"dpo_label_smoothing{arg_index}"), 
                getattr(self.args, f"dpo_ipo{arg_index}")
                )
        elif obj_type == "KD":
            loss_fn = KDLoss()
        else:
            raise NotImplementedError

        return loss_fn
    
    def calc_loss(self, loss_fn:nn.Module, data, obj_index:int):
        obj_type = getattr(self.args, f"obj_{obj_index}")
        if self.args.obj_1==self.args.obj_2 and obj_index==2:
            arg_index='_2'
        else:
            arg_index=''
        if self.args.obj_1 in ["KD","DPO"] and self.args.obj_2 in ["KD","DPO"] and obj_index==2:
            ref_model = self.ref_model_2
        else:
            ref_model = self.ref_model
        
        is_aux_loss = getattr(self.args, f"aux_loss_coef_{obj_index}") > 1e-8

        if obj_type=="SFT":
            prompts_id_lens, inputs, attention_masks, _ = data
            inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
            attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

            output = self.model(
                inputs, attention_mask=attention_mask, return_output=True,
            )

            # loss function
            labels = torch.where(
                attention_mask.bool(),
                inputs,
                loss_fn.IGNORE_INDEX,
            )
            # mixtral
            if is_aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0

            for label, source_len in zip(labels, prompts_id_lens):
                label[:source_len] = loss_fn.IGNORE_INDEX

            gpt_loss = loss_fn(output.logits, labels)
            loss = gpt_loss + aux_loss * getattr(self.args, f"aux_loss_coef_{obj_index}")

            logs_dict = {"sft loss": gpt_loss.item()}
            if is_aux_loss:
                logs_dict["aux loss"] = aux_loss.item()

        elif obj_type=="DPO":
            is_nll_loss = getattr(self.args, f"dpo_nll_loss_coef{arg_index}") > 1e-8
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

            chosen_logps, rejected_logps, aux_loss, nll_loss = self.concatenated_forward(
                self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
            )
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                    self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )

            # loss function
            preference_loss, chosen_reward, reject_reward = loss_fn(
                chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            # mixtral
            if not is_aux_loss:
                aux_loss = 0
            # nll loss
            if not is_nll_loss:
                nll_loss = 0

            loss = preference_loss + aux_loss * getattr(self.args, f"aux_loss_coef_{obj_index}") + nll_loss * getattr(self.args, f"dpo_nll_loss_coef{arg_index}")

            acc = (chosen_reward > reject_reward).float().mean().item()
            # dpo logs
            logs_dict = {
                "preference loss": preference_loss.item(),
                "acc": acc,
                "chosen_reward": chosen_reward.mean().item(),
                "reject_reward": reject_reward.mean().item(),
                "lr": self.scheduler.get_last_lr()[0],
            }
            if is_nll_loss:
                logs_dict["nll_loss"] = nll_loss.item()  
        elif obj_type=="KD":
            prompts_id_len, inputs, attention_masks, _ = data
            inputs = inputs.squeeze(1).to(torch.cuda.current_device())
            attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
            output = self.model(inputs, attention_mask=attention_mask, return_output=True)

            labels = torch.where(
                attention_mask.bool(),
                inputs,
                loss_fn.IGNORE_INDEX,
            )

            for label, source_len in zip(labels, prompts_id_len):
                label[:source_len] = loss_fn.IGNORE_INDEX

            with torch.no_grad():
                teacher_logits = ref_model(inputs, attention_mask=attention_mask, return_output=True)[
                    "logits"
                ]
            loss = loss_fn(output.logits, teacher_logits, labels)

            logs_dict = {
                "distill loss": loss.item(),
                "lr": self.scheduler.get_last_lr()[0],
            }
        else:
            raise NotImplementedError
        
        return loss, logs_dict


    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._get_batch_logps(
            all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=False
        )
        chosen_logps = all_logps_sum[: chosen_ids.shape[0]]
        rejected_logps = all_logps_sum[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: chosen_ids.shape[0]].mean()

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks, prompt_id_lens * 2

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert average_log_prob == False
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[loss_masks == False] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        return logprobs_sums, logprobs_means

    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens):
        output = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._packed_get_batch_logps(
            all_logits,
            packed_input_ids,
            packed_attention_masks,
            prompt_id_lens * 2,
            packed_seq_lens,
            average_log_prob=False,
        )
        chosen_logps = all_logps_sum[: len(packed_seq_lens) // 2]
        rejected_logps = all_logps_sum[len(packed_seq_lens) // 2 :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: len(packed_seq_lens) // 2].mean()

    def _packed_get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        packed_seq_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        assert average_log_prob == False

        if self.strategy.ring_attn_group is None:
            assert logits.shape[:-1] == labels.shape
            labels = labels[:, 1:]
            logits = logits[:, :-1, :]
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        else:
            rank = self.strategy.ring_attn_rank
            total_seq_len = labels.numel()
            local_seq_len = total_seq_len // self.strategy.ring_attn_size
            local_slice = slice(rank * local_seq_len + 1, (rank + 1) * local_seq_len + 1)
            local_label = labels[:, local_slice]
            if rank == self.strategy.ring_attn_size - 1:
                # add a dummy label to the last logit
                local_label = F.pad(local_label, (0, 1), value=0)
            local_per_token_logps = torch.gather(
                logits.log_softmax(-1), dim=2, index=local_label.unsqueeze(2)
            ).squeeze(2)
            # we may not need to all_gather the entire tensor, but it's easier to implement.
            # use the flash_attn all_gather so that the all_gather has correct backward.
            per_token_logps = all_gather(local_per_token_logps, self.strategy.ring_attn_group).reshape((1, -1))
            per_token_logps = per_token_logps[:, :-1]

        loss_masks = attention_mask.clone().bool()

        index = 0
        for i, seq_len in enumerate(packed_seq_lens):
            loss_masks[0, index : index + prompt_id_lens[i]] = False
            index = index + seq_len

        loss_masks = loss_masks[:, 1:]

        logprobs_sums = []
        logprobs_means = []
        index = 0
        for i, seq_len in enumerate(packed_seq_lens):
            seq = per_token_logps[0, index : index + seq_len - 1]
            mask = loss_masks[0, index : index + seq_len - 1]
            logprobs_sums.append((seq * mask).sum())
            logprobs_means.append((seq * mask).sum() / mask.sum())
            index = index + seq_len

        return torch.stack(logprobs_sums), torch.stack(logprobs_means)


    # logs/checkpoints/evaluate 
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, obj_index=1):
        # logs
        if global_step % args.logging_steps == 0: # todo: fix this logging logic
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            if self.strategy.is_rank_0() and global_step % self.strategy.accumulated_gradient == 0: # todo: fix this logging logic
                if self._wandb is not None:
                    logs = {f"train_{obj_index}/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs_dict.items():
                        self._tensorboard.add_scalar(f"train_{obj_index}/{k}", v, global_step)

        # eval
        eval_loss_1 = None
        eval_loss_2 = None
        if global_step % args.eval_steps == 0 or global_step==1: #todo: make evaluation block neater
            if self.args.obj_1=="SFT":
                eval_loss_1 = self.sft_evaluate(self.eval_dataloader_1, self.loss_fn_1, obj_index, global_step)
            elif self.args.obj_1=="DPO":
                eval_loss_1 = self.dpo_evaluate(self.eval_dataloader_1, self.loss_fn_1, obj_index, global_step)
            elif self.args.obj_1=="KD":
                eval_loss_1 = self.kd_evaluate(self.eval_dataloader_1, self.loss_fn_1, obj_index, global_step)
            if self.args.obj_2=="SFT":
                eval_loss_2 = self.sft_evaluate(self.eval_dataloader_2, self.loss_fn_2, obj_index, global_step)
            elif self.args.obj_2=="DPO":
                eval_loss_2 = self.dpo_evaluate(self.eval_dataloader_2, self.loss_fn_2, obj_index, global_step)
            elif self.args.obj_2=="KD":
                eval_loss_2 = self.kd_evaluate(self.eval_dataloader_2, self.loss_fn_2, obj_index, global_step)
        # save ckpt
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)
        
        return eval_loss_1, eval_loss_2

    def dpo_evaluate(self, eval_dataloader, loss_fn, obj_index, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            acc_sum = 0
            loss_sum = 0
            times = 0
            for data in eval_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_logps, rejected_logps, aux_loss, _ = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )
                with torch.no_grad():
                    reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                        self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )

                loss, chosen_reward, reject_reward = loss_fn(
                    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                )
                acc_sum += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                times += 1
                step_bar.update()

            logs = {
                "eval dpo loss": loss_sum / times,
                "eval dpo acc": acc_sum / times,
            }
            logs = self.strategy.all_reduce(logs)
            step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {f"eval_{obj_index}/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None: 
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval_{obj_index}/{k}", v, steps)
        self.model.train()  # reset model state

        # return logs[f"eval_{obj_index}/dpo_loss"] if self._wandb is not None and self.strategy.is_rank_0() else logs["dpo_loss"]

    def sft_evaluate(self, eval_dataloader, loss_fn, obj_index, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts_id_lens, inputs, attention_masks, infos in eval_dataloader:
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                output = self.model(
                    inputs, attention_mask=attention_mask, return_output=True,
                )

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    loss_fn.IGNORE_INDEX,
                )

                for label, source_len in zip(labels, prompts_id_lens):
                    label[:source_len] = loss_fn.IGNORE_INDEX

                loss = loss_fn(output.logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval sft loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {f"eval_{obj_index}/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None and self.strategy.is_rank_0(): 
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval_{obj_index}/{k}", v, steps)
        self.model.train()  # reset model state

        # return logs[f"eval_{obj_index}/sft_loss"] if self._wandb is not None and self.strategy.is_rank_0() else logs["sft_loss"]
    
    def kd_evaluate(self, eval_dataloader, loss_fn, obj_index, steps=0):
        if self.args.obj_1 in ["KD","DPO"] and self.args.obj_2 in ["KD","DPO"] and obj_index==2:
            ref_model = self.ref_model_2
        else:
            ref_model = self.ref_model
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts_id_len, inputs, attention_masks, _ in eval_dataloader:
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                output = self.model(inputs, attention_mask=attention_mask, return_output=True)

                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    loss_fn.IGNORE_INDEX,
                )

                for label, source_len in zip(labels, prompts_id_len):
                    label[:source_len] = loss_fn.IGNORE_INDEX

                teacher_logits = ref_model(inputs, attention_mask=attention_mask, return_output=True)["logits"]
                loss = loss_fn(output.logits, teacher_logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval distill loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {f"eval_{obj_index}/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None: #todo: add support for tensorboard
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval_{obj_index}/{k}", v, steps)

        self.model.train()  # reset model state

    # DEBUG #todo: add tensorboard support
    def _log_gpu_memory_usage(self, checkpoint):
        max_allocated = torch.cuda.max_memory_allocated()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()

        self._wandb.log({
            f"GPU Memory Allocated @ {checkpoint} (MB)": allocated / (1024 ** 2),
            f"Max. GPU Memory Allocated @ {checkpoint} (MB)": max_allocated / (1024 ** 2),
            # f"GPU Memory Reserved @ {checkpoint} (MB)": reserved / (1024 ** 2)
        })

    # DEBUG
    def _log_time_elapsed(self, checkpoint, t):

        self._wandb.log({
            f"Time @ {checkpoint} (s)": t,
            # f"GPU Memory Reserved @ {checkpoint} (MB)": reserved / (1024 ** 2)
        })

    # DEBUG
    def _log_gpu_memory_from_nvdia_smi(self, checkpoint, gpus=[0, 1]):
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
        try:
            memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
        except sp.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        memory_use_values = np.array([int(x.split()[0]) for i, x in enumerate(memory_use_info)])[gpus]

        self._wandb.log({
            f"NVIDIA-SMI Max. Memory-Usage @ {checkpoint} (MB)": np.max(memory_use_values),
            f"NVIDIA-SMI Avg. Memory-Usage @ {checkpoint} (MB)": np.mean(memory_use_values)
        })
        
        for i, mem in enumerate(memory_use_values):
            self._wandb.log({
                f"NVIDIA-SMI GPU {i} Memory-Usage @ {checkpoint} (MB)": mem,
            })