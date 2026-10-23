import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from data import SEED, load_and_tokenize_gsm, load_and_tokenize_med

MAX_SEQ_LENGTH = 1024
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 6
LEARNING_RATE = 2e-5
LR_SCHEDULER_TYPE = "constant"
WARMUP_RATIO = 0.0

EAFT_ALPHA = 1.0
EAFT_TOP_K = 20


class DFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        V = shift_logits.shape[-1]
        token_loss = loss_fct(shift_logits.view(-1, V), shift_labels.view(-1))

        token_prob = (
            torch.softmax(shift_logits, dim=-1)
            .view(-1, V)
            .gather(1, shift_labels.view(-1, 1).clamp(min=0))
            .squeeze(-1)
            .detach()
        )
        dft_loss = token_loss * token_prob

        active_mask = (shift_labels.view(-1) != -100).float()
        loss = (dft_loss * active_mask).sum() / active_mask.sum().clamp(min=1)
        return (loss, outputs) if return_outputs else loss


def eaft_loss_func(outputs, labels, num_items_in_batch=None, alpha: float = EAFT_ALPHA):
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())

    topk_logits = torch.topk(shift_logits, k=EAFT_TOP_K, dim=-1).values
    topk_probs = F.softmax(topk_logits, dim=-1)
    entropy = -torch.sum(topk_probs * torch.log(topk_probs + 1e-9), dim=-1)
    norm_entropy = (entropy / math.log(EAFT_TOP_K)).detach()

    eaft_loss = (norm_entropy * alpha) * ce_loss
    mask = (shift_labels != -100).float()

    if num_items_in_batch is not None:
        return (eaft_loss * mask).sum() / num_items_in_batch
    return (eaft_loss * mask).sum() / mask.sum().clamp(min=1e-5)


class EAFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = eaft_loss_func(outputs, labels, num_items_in_batch)
        return (loss, outputs) if return_outputs else loss


TRAINERS = {"sft": Trainer, "dft": DFTTrainer, "eaft": EAFTTrainer}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="sft", choices=list(TRAINERS))
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset", required=True, choices=["gsm", "med"])
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.dataset == "gsm":
        train_dataset = load_and_tokenize_gsm(tokenizer, MAX_SEQ_LENGTH)
    else:
        train_dataset = load_and_tokenize_med(
            tokenizer, MAX_SEQ_LENGTH, per_class_rng=(args.method == "dft")
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=output_dir + "-checkpoints",
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        optim="adamw_torch",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=20,
        eval_strategy="no",
        save_strategy="no",
        fp16=False,
        bf16=True,
        seed=SEED,
        report_to="none",
    )

    trainer = TRAINERS[args.method](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
