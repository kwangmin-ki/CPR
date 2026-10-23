import argparse
import os
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from data import (
    MED_CTX_MAX_TOKENS,
    MED_FEWSHOT_PREFIX,
    MED_SAMPLES_PER_CLASS,
    SEED,
    clean_gsm8k_answer,
    get_choices_and_letter,
    pubmedqa_context,
)
from router import HierarchicalRouterNetwork, get_hidden_size

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROUTER_BATCH_SIZE = 16
ROUTER_LEARNING_RATE = 1e-4
ROUTER_EPOCHS = 6
ROUTER_WARMUP_RATIO = 0.1

MAX_SEQ_LENGTH = 1024


def data_path(output_dir: str, tag: str, dataset: str) -> str:
    return os.path.join(output_dir, f"router_training_data_{tag}_{dataset}.pt")


def router_save_path(output_dir: str, tag: str, dataset: str) -> str:
    return os.path.join(output_dir, f"router_model_{tag}_{dataset}.pt")


def load_backbone_models(base_model: str, expert_model: str):
    tok_base = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    mdl_base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True,
    ).eval()

    mdl_expert = AutoModelForCausalLM.from_pretrained(
        expert_model, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True,
    ).eval()
    return mdl_base, tok_base, mdl_expert


def label_sequence(
    base_logits: torch.Tensor,
    expert_logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> List[int]:
    labels: List[int] = []
    for t in range(input_ids.size(1) - 1):
        if t < prompt_len - 1:
            labels.append(-100)
            continue
        target = input_ids[0, t + 1].item()
        if base_logits[t].argmax().item() == target:
            labels.append(0)
        elif expert_logits[t].argmax().item() == target:
            labels.append(1)
        else:
            labels.append(-100)
    return labels


def question_start(tokenizer, prompt: str) -> int:
    last_q_idx = prompt.rfind("\n\nQuestion: ")
    return len(tokenizer(prompt[:last_q_idx])["input_ids"]) if last_q_idx != -1 else 0


def generate_gsm8k_data(mdl_base, tok_base, mdl_expert, out_path: str, limit: Optional[int] = None):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    processed = []
    for sample in tqdm(ds, desc="GSM8K labeling"):
        answer = clean_gsm8k_answer(sample["answer"])
        prompt_text = f"Question: {sample['question']}\nAnswer: Let's think step by step."
        full_text = f"Question: {sample['question']}\nAnswer: Let's think step by step. {answer}{tok_base.eos_token}"

        input_ids = tok_base(full_text, return_tensors="pt").to(DEVICE)["input_ids"]
        prompt_len = len(tok_base(prompt_text)["input_ids"])

        with torch.no_grad():
            base_logits = mdl_base(input_ids).logits[0]
            expert_logits = mdl_expert(input_ids).logits[0]

        labels = label_sequence(base_logits, expert_logits, input_ids, prompt_len)
        if any(l != -100 for l in labels):
            processed.append({
                "input_ids": input_ids[0].cpu().tolist(),
                "labels": labels,
                "prompt_len": prompt_len,
                "q_start": question_start(tok_base, prompt_text),
            })

    torch.save(processed, out_path)


def generate_med_data(mdl_base, tok_base, mdl_expert, out_path: str, limit: Optional[int] = None):
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")

    rng = random.Random(SEED)
    yes_processed: List[dict] = []
    no_processed: List[dict] = []

    for sample in tqdm(ds, desc="PubMedQA labeling"):
        if len(yes_processed) >= MED_SAMPLES_PER_CLASS and len(no_processed) >= MED_SAMPLES_PER_CLASS:
            break

        decision = str(sample["final_decision"]).lower()
        if decision not in {"yes", "no"}:
            continue
        if decision == "yes" and len(yes_processed) >= MED_SAMPLES_PER_CLASS:
            continue
        if decision == "no" and len(no_processed) >= MED_SAMPLES_PER_CLASS:
            continue

        ctx_str = pubmedqa_context(sample["context"])
        ctx_tok_len = len(tok_base(ctx_str, truncation=False, padding=False)["input_ids"])
        if ctx_tok_len >= MED_CTX_MAX_TOKENS:
            continue

        question_part = f"{ctx_str} {sample['question']}".strip() if ctx_str else sample["question"]
        choices_str, decision_letter = get_choices_and_letter(decision, rng)

        prompt_text = (
            MED_FEWSHOT_PREFIX
            + f"Question: {question_part}\n"
            f"{choices_str}\n"
            "Answer: Let's think step by step."
        )
        full_text = prompt_text + f" {sample['long_answer']} The answer is ({decision_letter}).{tok_base.eos_token}"

        full_ids = tok_base(full_text, truncation=False, padding=False)["input_ids"]
        if len(full_ids) > MAX_SEQ_LENGTH:
            continue

        input_ids = torch.tensor([full_ids], dtype=torch.long).to(DEVICE)
        prompt_len = len(tok_base(prompt_text, truncation=False, padding=False)["input_ids"])

        with torch.no_grad():
            base_logits = mdl_base(input_ids).logits[0]
            expert_logits = mdl_expert(input_ids).logits[0]

        labels = label_sequence(base_logits, expert_logits, input_ids, prompt_len)
        if any(l != -100 for l in labels):
            entry = {
                "input_ids": input_ids[0].cpu().tolist(),
                "labels": labels,
                "prompt_len": prompt_len,
                "q_start": question_start(tok_base, prompt_text),
            }
            (yes_processed if decision == "yes" else no_processed).append(entry)

    processed = yes_processed + no_processed
    if limit:
        processed = processed[:limit]

    torch.save(processed, out_path)


class RouterDataset(Dataset):
    def __init__(self, path: str):
        self.data: List[dict] = torch.load(path, map_location="cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.float),
            "prompt_len": item["prompt_len"],
            "q_start": item.get("q_start", 0),
        }


def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    prompt_lens = torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long)
    q_starts = torch.tensor([b["q_start"] for b in batch], dtype=torch.long)
    return (
        nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0),
        nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100.0),
        prompt_lens,
        q_starts,
    )


def calc_pos_weight(dataset: RouterDataset) -> torch.Tensor:
    total_pos = total_neg = 0
    for item in dataset.data:
        lbl = torch.tensor(item["labels"])
        valid = lbl[lbl != -100]
        total_pos += (valid == 1).sum().item()
        total_neg += (valid == 0).sum().item()

    if total_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor((total_neg / total_pos) ** 0.5)


def forward_router_step(base_mdl, router, batch_ids, batch_lbl, batch_plen, batch_qstart):
    batch_ids = batch_ids.to(DEVICE)
    batch_lbl = batch_lbl.to(DEVICE)

    with torch.no_grad():
        last_hs = base_mdl(batch_ids, output_hidden_states=True).hidden_states[-1]

    pooled = torch.stack([
        last_hs[i, batch_qstart[i].item() : batch_plen[i].item(), :].mean(dim=0)
        for i in range(last_hs.size(0))
    ]).to(DEVICE)

    router_logit = router(last_hs[:, :-1, :], pooled).squeeze(-1)
    return router_logit, batch_lbl


def train_router(base_model: str, labels_path: str):
    base_mdl = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True,
    ).eval()

    hidden_size = get_hidden_size(base_mdl)
    router = HierarchicalRouterNetwork(input_dim=hidden_size).to(device=DEVICE, dtype=torch.bfloat16)
    router.train()
    optimizer = optim.AdamW(router.parameters(), lr=ROUTER_LEARNING_RATE)

    ds = RouterDataset(labels_path)
    if len(ds) == 0:
        raise RuntimeError("Empty label file")
    dl = DataLoader(ds, batch_size=ROUTER_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    total_steps = len(dl) * ROUTER_EPOCHS
    warmup_steps = int(total_steps * ROUTER_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    pw = calc_pos_weight(ds).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)

    for epoch in range(ROUTER_EPOCHS):
        pbar = tqdm(dl, desc=f"Epoch {epoch + 1}/{ROUTER_EPOCHS}")

        for batch_ids, batch_lbl, batch_plen, batch_qstart in pbar:
            router_logit, batch_lbl = forward_router_step(
                base_mdl, router, batch_ids, batch_lbl, batch_plen, batch_qstart
            )
            mask = (batch_lbl != -100).float()
            masked_loss = (criterion(router_logit, batch_lbl) * mask).sum() / (mask.sum() + 1e-8)

            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=f"{masked_loss.item():.4f}")

    return router


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--expert_model", required=True)
    p.add_argument("--dataset", required=True, choices=["gsm", "med"])
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--tag", default=None)
    p.add_argument("--skip_datagen", action="store_true")
    p.add_argument("--skip_training", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    tag = args.tag or os.path.basename(os.path.normpath(args.base_model))
    os.makedirs(args.output_dir, exist_ok=True)
    labels_path = data_path(args.output_dir, tag, args.dataset)
    router_path = router_save_path(args.output_dir, tag, args.dataset)

    if not args.skip_datagen:
        mdl_base, tok_base, mdl_expert = load_backbone_models(args.base_model, args.expert_model)
        generate_fn = generate_gsm8k_data if args.dataset == "gsm" else generate_med_data
        generate_fn(mdl_base, tok_base, mdl_expert, labels_path, args.limit)
        del mdl_base, tok_base, mdl_expert
        torch.cuda.empty_cache()
    elif not os.path.exists(labels_path):
        raise FileNotFoundError(labels_path)

    if not args.skip_training:
        router = train_router(args.base_model, labels_path)
        torch.save(router.state_dict(), router_path)


if __name__ == "__main__":
    main()
