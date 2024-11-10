import math
from abc import ABC

import torch
from torch import nn
from torch.optim import Optimizer
from bipost.utils.distributed_sampler import DistributedSampler
from tqdm import tqdm
from transformers.trainer import get_scheduler
from bipost.models import GPTLMLoss
from bipost.models import batch_GPTLMLoss


class SelectorTrainer(ABC):
    """
        data selector trainer.

    """

    def __init__(
        self,
        model,
        ref_model,
        ref_constant,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        new_dataloader,
        p,
        p_opt: Optimizer,
        p_scheduler, 
        scheduler,
        max_norm: float = 1,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.new_dataloader = new_dataloader
        self.scheduler = scheduler
        self.model = model
        self.ref_model = ref_model
        self.ref_constant = ref_constant
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.ul_weight = self.args.upperlevel_weight
        self.ul_weight_decay = self.args.upperlevel_weight_decay

        self.loss_fn = GPTLMLoss()
        self.batch_loss_fn = batch_GPTLMLoss()
        
        # data selection policy param
        self.p = p
        self.p_opt = p_opt
        self.p_scheduler = p_scheduler

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef_2 > 1e-8

        # wandb setting
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

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)
                self.new_dataloader.sampler.set_epoch(epoch)
            if epoch>=1:
                self.ul_weight -= self.ul_weight_decay
                print("\n UL weight now", self.ul_weight, end="\n")
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            self.p.train()
            loss_mean = 0
            gpt_loss_mean=0
            for (prompts_id_len, inputs, attention_masks, _), \
                (ide, new_prompts_id_len, new_input, new_attention_masks, _) \
                    in zip(self.train_dataloader,self.new_dataloader):
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                
                new_inputs = new_input.squeeze(1).to(torch.cuda.current_device())
                new_attention_mask = new_attention_masks.squeeze(1).to(torch.cuda.current_device())
                new_output = self.model(new_inputs, attention_mask=new_attention_mask, return_output=True)
                with torch.no_grad():
                    batch_weights = self.p()[ide]
                

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                new_labels = torch.where(
                    new_attention_mask.bool(),
                    new_inputs,
                    self.batch_loss_fn.IGNORE_INDEX,
                )
                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0

                for label, source_len, new_label, new_source_len in zip(labels, prompts_id_len, new_labels, new_prompts_id_len):
                    label[:source_len] = self.loss_fn.IGNORE_INDEX
                    new_label[:new_source_len] = self.batch_loss_fn.IGNORE_INDEX

                gpt_loss = self.batch_loss_fn(output.logits, labels, sequence_reduce="mean").mean(0)
                batch_gpt_loss = self.batch_loss_fn(new_output.logits, new_labels, sequence_reduce="mean")
                
                weighted_gpt_loss = (batch_weights * batch_gpt_loss).mean(0)
                loss = self.ul_weight * gpt_loss + (1-self.ul_weight) * weighted_gpt_loss + aux_loss * self.args.aux_loss_coef_2
                
                self.strategy.backward(loss, self.model, self.optimizer) 
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                selector_loss = (self.p()[ide]*batch_gpt_loss.detach()).mean()
                self.strategy.backward(selector_loss, self.p, self.p_opt)
                
                self.strategy.optimizer_step(self.p_opt, self.p, self.p_scheduler)

                gpt_loss_mean = gpt_loss_mean * 0.95 + 0.05 * gpt_loss.item()
                loss_mean = loss_mean * 0.95 + 0.05 * loss.item()
                logs_dict = {"upper_loss": gpt_loss.item(), "upper_loss_mean": gpt_loss_mean, "loss_mean":loss_mean}
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()

                # logs/checkpoints/evaluation
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            if self.strategy.is_rank_0():
                p_name = "./checkpoint/"+args.selector_name+"_"+ args.selector_activation \
                    +"_ep"+str(epoch+1)+".pt"
                torch.save(self.p.logits, p_name)
                print(self.p.logits)
            epoch_bar.update()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)

    def evaluate(self, eval_dataloader, steps=0):
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
                logits = self.model(inputs, attention_mask=attention_mask, return_output=True)["logits"]

                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                for label, source_len in zip(labels, prompts_id_len):
                    label[:source_len] = self.loss_fn.IGNORE_INDEX
                loss = self.batch_loss_fn(logits, labels).mean(0)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state