# CPR for LLMs: Critical-Point Routing against Catastrophic Forgetting in Domain Adaptation

## Installation

```bash
conda create -n cpr python=3.11 -y && conda activate cpr
pip install -r requirements.txt
```

## Repository structure

```
model_wrappers.py      # inference: CPR and baselines
router.py              # hierarchical router
data.py                # GSM8K and PubMedQA training data
train_expert.py        # expert fine-tuning: SFT, DFT, EAFT
train_router.py        # critical-point labeling + router training
scripts/               # training and evaluation scripts
lm-evaluation-harness/ # lm-eval with the CPR adapter (lm_eval/models/critical_wrapper.py)
```

## Training

```bash
# 1) Domain expert (--method sft | dft | eaft, --dataset gsm | med)
python train_expert.py --method sft --model_path google/gemma-3-4b-pt \
    --dataset gsm --output_dir checkpoints/gemma3-4b-gsm-sft

# 2) Critical-point labels + router
python train_router.py --base_model google/gemma-3-4b-pt \
    --expert_model checkpoints/gemma3-4b-gsm-sft --dataset gsm --tag gemma3-4b
```

## Evaluation

```bash
lm_eval --model hf-wrapper \
  --model_args wrapper=cpr,base_model_path=google/gemma-3-4b-pt,expert_model_path=checkpoints/gemma3-4b-gsm-sft,router_model_path=checkpoints/router_model_gemma3-4b_gsm.pt \
  --tasks lm-evaluation-harness/lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml \
  --num_fewshot 0 --batch_size 1 --output_path results/cpr
```

```bash
WRAPPER=cpr DOMAIN=math \
MODEL_ARGS="base_model_path=...,expert_model_path=...,router_model_path=..." \
bash scripts/eval.sh
```

| Domain  | Benchmarks (task config) | Shots |
|---------|--------------------------|-------|
| Math    | GSM8K (`gsm8k/gsm8k-cot-zeroshot.yaml`), ASDiv (`asdiv/asdiv-cot-manual.yaml`), SVAMP (`manual_svamp/svamp-cot-manual.yaml`) | 0 |
| Medical | PubMedQA (`pubmedqa/pubmedqa-cot-manual.yaml`), MedQA (`medqa/medqa_cot.yaml`), CareQA (`careqa/careqa_en_cot.yaml`) | 2 |
| General | MMLU (`mmlu_flan_cot_fewshot`), CommonsenseQA (`commonsense_qa/cot.yaml`), ARC-C (`arc/flan_cot/arc_challenge_cot.yaml`) | 2 |

### Methods

| `wrapper=` | Method |
|------------|--------|
| `base` | Base model |
| `expert` | SFT expert |
| `eaft` | EAFT |
| `lfu` | LfU |
| `ensemble` | Ensemble |
| `cd` | Contrastive Decoding |
| `switch` | Switch Generation (`bunsenfeng/PFA_switcher_1`) |
| `routellm_causal` | RouteLLM (`routellm/causal_llm_gpt4_augmented`) |
| `cpr` | CPR |
