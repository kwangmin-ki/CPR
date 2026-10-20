"""Training-data formatting for GSM8K (math) and PubMedQA (medical).

Prompt formats (identical to the evaluation templates):
  GSM8K    : "Question: {q}\nAnswer: Let's think step by step. {rationale} The answer is {a}.{eos}"
  PubMedQA : 2-shot MMLU demonstrations + "Question: {context} {q}\n(A) .. (B) .. (C) ..\n"
             "Answer: Let's think step by step. {long_answer} The answer is ({letter}).{eos}"
             The yes/no/maybe option order is shuffled per sample; `maybe` samples are
             dropped and yes/no are balanced (4,000 each).
"""

import random
import re

from datasets import concatenate_datasets, load_dataset

SEED = 42

MED_CTX_MAX_TOKENS = 512
MED_SAMPLES_PER_CLASS = 4000

MED_FEWSHOT_PREFIX = """\
Question: Glucose is transported into the muscle cell:
(A) via protein transporters called GLUT4. (B) only in the presence of insulin. (C) via hexokinase. (D) via monocarbylic acid transporters.
Answer: Let's think step by step. Glucose (also known as the blood sugar) is the main sugar found in the human body. It is transported into the muscle cell via diffusion through protein transporters called GLUT4. The answer is (A).

Question: In a genetic test of a newborn, a rare genetic disorder is found that has X-linked recessive transmission. Which of the following statements is likely true regarding the pedigree of this disorder?
(A) All descendants on the maternal side will have the disorder. (B) Females will be approximately twice as affected as males in this family. (C) All daughters of an affected male will be affected. (D) There will be equal distribution of males and females affected.
Answer: Let's think step by step. Let's recall first that females have two X chromosomes, while males have one X and one Y chromosome. This is an important fact we need to know before answering this question. Because a male can only pass his only one X chromosome to a daughter, if he is affected by this rare genetic disorder, then we know for sure that he will pass this rare genetic disorder to all his future-born daughters. The answer is (C).

"""


def clean_gsm8k_answer(answer: str) -> str:
    """Drop <<calc>> annotations and rewrite '#### X' as 'The answer is X'."""
    answer = re.sub(r"<<[^>]*>>", "", answer)
    answer = re.sub(r"####\s*", "The answer is ", answer)
    return answer.strip()


def get_choices_and_letter(decision: str, rng: random.Random) -> tuple[str, str]:
    """Randomly permute (yes, no, maybe) to avoid a positional shortcut."""
    labels = ["yes", "no", "maybe"]
    rng.shuffle(labels)
    choices_str = f"(A) {labels[0]} (B) {labels[1]} (C) {labels[2]}"
    letter = ["A", "B", "C"][labels.index(decision)]
    return choices_str, letter


def pubmedqa_context(ctx) -> str:
    ctx_texts = ctx.get("contexts", []) if isinstance(ctx, dict) else []
    return " ".join(ctx_texts).strip() if ctx_texts else ""


def _tokenize_one(tokenizer, prompt: str, full_text: str, max_seq_length: int):
    """Tokenize one sample with prompt tokens masked; return None if too long."""
    full_enc = tokenizer(full_text, truncation=False, padding=False)
    if len(full_enc["input_ids"]) > max_seq_length:
        return None

    prompt_len = len(tokenizer(prompt, truncation=False, padding=False)["input_ids"])
    input_ids = full_enc["input_ids"]
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": full_enc["attention_mask"],
    }


def load_and_tokenize_gsm(tokenizer, max_seq_length: int):
    print(">>> Loading GSM8K train split...")
    gsm8k_raw = load_dataset("openai/gsm8k", "main", split="train")

    def tokenize_gsm8k(examples):
        all_input_ids, all_labels, all_attn_mask = [], [], []
        for q, a in zip(examples["question"], examples["answer"]):
            a = clean_gsm8k_answer(a)
            prompt = f"Question: {q}\nAnswer: Let's think step by step."
            full = f"Question: {q}\nAnswer: Let's think step by step. {a}{tokenizer.eos_token}"
            result = _tokenize_one(tokenizer, prompt, full, max_seq_length)
            if result is None:
                continue
            all_input_ids.append(result["input_ids"])
            all_labels.append(result["labels"])
            all_attn_mask.append(result["attention_mask"])
        return {"input_ids": all_input_ids, "labels": all_labels, "attention_mask": all_attn_mask}

    gsm8k_ds = gsm8k_raw.map(
        tokenize_gsm8k, batched=True,
        remove_columns=gsm8k_raw.column_names, desc="Tokenizing GSM8K",
    )
    excluded = len(gsm8k_raw) - len(gsm8k_ds)
    print(f">>> GSM8K: {len(gsm8k_ds)} samples (excluded {excluded} > {max_seq_length} tokens)")
    return gsm8k_ds.shuffle(seed=SEED)


def load_and_tokenize_med(tokenizer, max_seq_length: int, per_class_rng: bool = False):
    """PubMedQA (pqa_artificial, train), balanced yes/no with shuffled options.

    per_class_rng=False: one RNG shared across the yes and no splits (SFT, EAFT).
    per_class_rng=True : separate RNGs seeded SEED / SEED+1 per split (DFT).
    Caching is disabled so the option shuffling never depends on a cached RNG state.
    """
    print(">>> Loading PubMedQA (pqa_artificial) train split...")
    med_raw = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    valid_decisions = {"yes", "no"}

    def make_tokenize_fn(rng):
        def tokenize_med(examples):
            all_input_ids, all_labels, all_attn_mask = [], [], []
            for q, ctx, long_ans, decision in zip(
                examples["question"],
                examples["context"],
                examples["long_answer"],
                examples["final_decision"],
            ):
                decision = str(decision).lower()
                if decision not in valid_decisions:
                    continue

                ctx_str = pubmedqa_context(ctx)
                ctx_tok_len = len(tokenizer(ctx_str, truncation=False, padding=False)["input_ids"])
                if ctx_tok_len >= MED_CTX_MAX_TOKENS:
                    continue

                question_part = f"{ctx_str} {q}".strip() if ctx_str else q
                choices_str, decision_letter = get_choices_and_letter(decision, rng)

                prompt = (
                    MED_FEWSHOT_PREFIX
                    + f"Question: {question_part}\n"
                    f"{choices_str}\n"
                    "Answer: Let's think step by step."
                )
                full = prompt + f" {long_ans} The answer is ({decision_letter}).{tokenizer.eos_token}"

                result = _tokenize_one(tokenizer, prompt, full, max_seq_length)
                if result is None:
                    continue
                all_input_ids.append(result["input_ids"])
                all_labels.append(result["labels"])
                all_attn_mask.append(result["attention_mask"])
            return {"input_ids": all_input_ids, "labels": all_labels, "attention_mask": all_attn_mask}
        return tokenize_med

    if per_class_rng:
        yes_fn = make_tokenize_fn(random.Random(SEED))
        no_fn = make_tokenize_fn(random.Random(SEED + 1))
    else:
        yes_fn = no_fn = make_tokenize_fn(random.Random(SEED))

    yes_raw = med_raw.filter(lambda x: str(x["final_decision"]).lower() == "yes")
    no_raw = med_raw.filter(lambda x: str(x["final_decision"]).lower() == "no")
    print(f">>> PubMedQA raw: yes={len(yes_raw)}, no={len(no_raw)} (maybe excluded)")

    yes_ds = yes_raw.map(
        yes_fn, batched=True, load_from_cache_file=False,
        remove_columns=yes_raw.column_names, desc="Tokenizing yes",
    )
    no_ds = no_raw.map(
        no_fn, batched=True, load_from_cache_file=False,
        remove_columns=no_raw.column_names, desc="Tokenizing no",
    )

    yes_ds = yes_ds.select(range(min(MED_SAMPLES_PER_CLASS, len(yes_ds))))
    no_ds = no_ds.select(range(min(MED_SAMPLES_PER_CLASS, len(no_ds))))
    print(f">>> PubMedQA balanced: yes={len(yes_ds)}, no={len(no_ds)}, total={len(yes_ds) + len(no_ds)}")

    return concatenate_datasets([yes_ds, no_ds]).shuffle(seed=SEED)
