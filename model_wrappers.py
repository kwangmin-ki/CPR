"""Inference wrappers for CPR and the baselines.

Every wrapper exposes the same interface, used by the lm-eval adapter
(lm-evaluation-harness/lm_eval/models/critical_wrapper.py):

    text  = wrapper.generate(prompt)
    texts = wrapper.generate_batch(prompts)

All methods use greedy decoding.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from router import HierarchicalRouterNetwork, get_hidden_size

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_device_map(device: str) -> str:
    """Shard across GPUs with accelerate when more than one GPU is visible."""
    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        return "auto"
    return device


def _load_tokenizer(path: str, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(path: str, device_map, **kwargs):
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map=device_map, **kwargs
    ).eval()


def _eos_ids(tokenizer) -> set:
    eos = tokenizer.eos_token_id
    return set(eos if isinstance(eos, (list, tuple)) else [eos])


class BaseModelWrapper(ABC):
    stop_strings: List[str] = []

    def _truncate_at_stop(self, text: str) -> str:
        for stop in self.stop_strings:
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx]
        return text

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]


# ---------------------------------------------------------------------------
# Single-model baselines
# ---------------------------------------------------------------------------

class BaseWrapper(BaseModelWrapper):
    """A single model with batched greedy decoding (base model)."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 512,
        device: str = DEVICE,
        stop_strings: Optional[List[str]] = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.stop_strings = stop_strings or []

        print(f"[{self.__class__.__name__}] Loading model from {model_path}")
        self.tokenizer = _load_tokenizer(model_path, padding_side="left")
        self.model = _load_model(model_path, _resolve_device_map(device), trust_remote_code=True)
        self.device = self.model.device

    def _eos_token_id(self):
        return self.tokenizer.eos_token_id

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self._eos_token_id(),
                do_sample=False,
                use_cache=True,
            )

        texts = self.tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
        return [self._truncate_at_stop(t) for t in texts]


class ExpertWrapper(BaseWrapper):
    """The SFT expert alone."""


class LFUWrapper(BaseWrapper):
    """A model fine-tuned with LfU (Nam et al., 2026)."""


class EAFTWrapper(BaseWrapper):
    """A model fine-tuned with EAFT (eager attention, EOS from generation_config)."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 512,
        device: str = DEVICE,
        stop_strings: Optional[List[str]] = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.stop_strings = stop_strings or []

        print(f"[EAFTWrapper] Loading model from {model_path}")
        self.tokenizer = _load_tokenizer(model_path, padding_side="left")
        self.model = _load_model(
            model_path, device, trust_remote_code=True, attn_implementation="eager"
        )

    def _eos_token_id(self):
        return self.model.generation_config.eos_token_id


# ---------------------------------------------------------------------------
# Multi-model baselines
# ---------------------------------------------------------------------------

class EnsembleWrapper(BaseModelWrapper):
    """Uniform average of the next-token distributions of all models at every step."""

    def __init__(
        self,
        model_paths: List[str],
        max_new_tokens: int = 512,
        device: str = DEVICE,
        stop_strings: Optional[List[str]] = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.stop_strings = stop_strings or []

        self.models = []
        self.tokenizers = []
        print(f"[EnsembleWrapper] Loading {len(model_paths)} models...")
        for path in model_paths:
            print(f"  {path}")
            self.tokenizers.append(_load_tokenizer(path, padding_side="left"))
            self.models.append(_load_model(path, device))

        # All models are assumed to share the same vocabulary.
        self.base_tokenizer = self.tokenizers[0]

    def generate(self, prompt: str) -> str:
        input_ids = self.base_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids.clone()
        eos_ids = _eos_ids(self.base_tokenizer)

        past_key_values = [None] * len(self.models)
        next_input_ids = input_ids

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                all_probs = []
                for i, model in enumerate(self.models):
                    outputs = model(
                        input_ids=next_input_ids,
                        past_key_values=past_key_values[i],
                        use_cache=True,
                    )
                    past_key_values[i] = outputs.past_key_values
                    all_probs.append(torch.softmax(outputs.logits[:, -1, :], dim=-1))

                avg_probs = torch.mean(torch.stack(all_probs), dim=0)
                next_token = torch.argmax(avg_probs, dim=-1).unsqueeze(0)
                if next_token.item() in eos_ids:
                    break

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                next_input_ids = next_token

        text = self.base_tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        return self._truncate_at_stop(text)


class ContrastiveDecodingWrapper(BaseModelWrapper):
    """Contrastive Decoding (Li et al., 2023) with the expert as expert and the base as amateur.

    score(w) = log p_expert(w) - alpha * log p_base(w), restricted to the plausible set
    V_head = {w : p_base(w) >= beta * max p_base}.
    """

    CD_ALPHA = 0.1
    VHEAD_BETA = 0.1

    def __init__(
        self,
        base_model_path: str,
        expert_model_path: str,
        max_new_tokens: int = 512,
        device: str = DEVICE,
        stop_strings: Optional[List[str]] = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.stop_strings = stop_strings or []
        self.cd_alpha = self.CD_ALPHA
        self.vhead_beta = self.VHEAD_BETA

        print("[ContrastiveDecodingWrapper] Loading base (amateur) model...")
        self.tokenizer = _load_tokenizer(base_model_path, trust_remote_code=True, padding_side="left")
        self.base_model = _load_model(base_model_path, device, trust_remote_code=True)

        print("[ContrastiveDecodingWrapper] Loading expert model...")
        self.expert_model = _load_model(expert_model_path, device, trust_remote_code=True)


    def generate(self, prompt: str) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids.clone()
        eos_ids = _eos_ids(self.tokenizer)

        with torch.no_grad():
            base_kv = self.base_model(input_ids, use_cache=True).past_key_values
            expert_kv = self.expert_model(input_ids, use_cache=True).past_key_values

            for _ in range(self.max_new_tokens):
                base_out = self.base_model(generated_ids[:, -1:], past_key_values=base_kv, use_cache=True)
                base_kv = base_out.past_key_values
                base_logits = base_out.logits[0, -1].float()

                exp_out = self.expert_model(generated_ids[:, -1:], past_key_values=expert_kv, use_cache=True)
                expert_kv = exp_out.past_key_values
                expert_logits = exp_out.logits[0, -1].float()

                base_probs = torch.softmax(base_logits, dim=-1)
                vhead_mask = base_probs >= self.vhead_beta * base_probs.max()

                base_logp = torch.log_softmax(base_logits, dim=-1)
                expert_logp = torch.log_softmax(expert_logits, dim=-1)
                cd_scores = expert_logp - self.cd_alpha * base_logp
                cd_scores[~vhead_mask] = float("-inf")

                next_token = torch.argmax(cd_scores).unsqueeze(0).unsqueeze(0)
                if next_token.item() in eos_ids:
                    break
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

        text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return self._truncate_at_stop(text)


class SwitchGenerationWrapper(BaseModelWrapper):
    """Switch Generation (Feng et al., 2026): a switcher LM picks a model for every patch.

    The first and last patches always use candidate 1 (the expert). If
    objective_flag is set, "The answer is" is appended before the last patch.
    """

    def __init__(
        self,
        candidate_model_paths: List[str],
        switcher_model_path: str,
        max_total_tokens: int = 512,
        patch_size: int = 50,
        objective_flag: bool = True,
        stop_strings: Optional[List[str]] = None,
    ):
        self.max_total_tokens = max_total_tokens
        self.stop_strings = stop_strings or []
        self.patch_size = patch_size
        self.objective_flag = objective_flag
        self.rounds = max_total_tokens // patch_size

        self.models = []
        self.tokenizers = []
        print(f"[SwitchGenerationWrapper] Loading {len(candidate_model_paths)} candidate models...")
        for i, path in enumerate(candidate_model_paths):
            print(f"  Candidate [{i}] {path}")
            tokenizer = _load_tokenizer(path)
            tokenizer.padding_side = "left"
            self.tokenizers.append(tokenizer)
            self.models.append(_load_model(path, "auto"))

        print(f"[SwitchGenerationWrapper] Loading switcher model from {switcher_model_path}")
        self.switcher_tokenizer = _load_tokenizer(switcher_model_path)
        self.switcher_tokenizer.padding_side = "left"
        self.switcher_model = _load_model(switcher_model_path, "auto")


    def _batch_generate(self, model, tokenizer, prompts: List[str], max_new: int):
        """Greedy batch generation; also returns whether each sequence hit EOS."""
        if not prompts:
            return [], []
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        is_done = [(tokenizer.eos_token_id in seq.tolist()) for seq in gen_tokens]
        return texts, is_done

    def _make_switcher_prompt(self, query: str, segments: List[str], seg_model_ids: List[int]) -> str:
        prompt = query
        for seg, mid in zip(segments, seg_model_ids):
            prompt += f" <model {mid} begins> {seg} <model {mid} ends>"
        prompt += (
            f" Which model should generate the next segment? "
            f"Please respond with a number from 0 to {len(self.models) - 1}. "
            f"The answer is model "
        )
        return prompt

    def _parse_switcher_output(self, out: str) -> int:
        for j in range(len(self.models)):
            if str(j) in out:
                return j
        return 1

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: List[str]) -> List[str]:
        n = len(prompts)
        logs = [
            {"query": p, "segments": [], "seg_model_ids": [], "sequence": "", "done": False}
            for p in prompts
        ]

        for round_id in range(self.rounds):
            if all(log["done"] for log in logs):
                break

            if round_id == 0 or round_id == self.rounds - 1:
                which_model = [1] * n
            else:
                switcher_prompts = [
                    self._make_switcher_prompt(log["query"], log["segments"], log["seg_model_ids"])
                    for log in logs
                ]
                switcher_outs, _ = self._batch_generate(
                    self.switcher_model, self.switcher_tokenizer, switcher_prompts, max_new=10
                )
                which_model = [self._parse_switcher_output(out) for out in switcher_outs]

            model_to_prompts = {i: [] for i in range(len(self.models))}
            model_to_indices = {i: [] for i in range(len(self.models))}
            for i, log in enumerate(logs):
                if log["done"]:
                    continue
                if round_id == self.rounds - 1 and self.objective_flag:
                    if log["segments"]:
                        log["segments"][-1] += " The answer is"
                    log["sequence"] += " The answer is"

                mid = which_model[i]
                model_to_prompts[mid].append(log["query"] + log["sequence"])
                model_to_indices[mid].append(i)

            for mid in range(len(self.models)):
                if not model_to_prompts[mid]:
                    continue
                texts, is_done_list = self._batch_generate(
                    self.models[mid], self.tokenizers[mid], model_to_prompts[mid], self.patch_size
                )
                for orig_idx, text, done in zip(model_to_indices[mid], texts, is_done_list):
                    logs[orig_idx]["segments"].append(text)
                    logs[orig_idx]["seg_model_ids"].append(mid)
                    logs[orig_idx]["sequence"] += text
                    if done:
                        logs[orig_idx]["done"] = True

        return [self._truncate_at_stop(log["sequence"].strip()) for log in logs]


class RouteLLMCausalWrapper(BaseModelWrapper):
    """RouteLLM causal-LLM classifier (Ong et al., 2025): query-level routing.

    The router scores the query with the [[1]] / [[3]] / [[5]] label tokens
    (expert wins / tie / base suffices). The base model is used when
    softmax([[1]], [[3]], [[5]])[2] >= router_threshold, otherwise the expert.
    """

    SCORE_TOKENS = ["[[1]]", "[[3]]", "[[5]]"]
    CAUSAL_INSTRUCTION = (
        "Given the following query, predict which model will produce a better "
        "response. Output [[1]] if the expert model wins, [[3]] if both models "
        "perform similarly, or [[5]] if the base model is sufficient."
    )

    def __init__(
        self,
        base_model_path: str,
        expert_model_path: str,
        router_model_path: str,
        max_new_tokens: int = 512,
        router_threshold: float = 0.5,
        device: str = DEVICE,
        stop_strings: Optional[List[str]] = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.router_threshold = router_threshold
        self.device = device
        self.stop_strings = stop_strings or []

        print("[RouteLLMCausalWrapper] Loading base model...")
        self.tokenizer_base = _load_tokenizer(base_model_path, trust_remote_code=True, padding_side="left")
        self.base_model = _load_model(base_model_path, device, trust_remote_code=True)

        print("[RouteLLMCausalWrapper] Loading expert model...")
        self.tokenizer_expert = _load_tokenizer(expert_model_path, trust_remote_code=True, padding_side="left")
        self.expert_model = _load_model(expert_model_path, device, trust_remote_code=True)

        print(f"[RouteLLMCausalWrapper] Loading router from {router_model_path}")
        self.tokenizer_router = _load_tokenizer(router_model_path, trust_remote_code=True)
        self.router_model = _load_model(router_model_path, device, trust_remote_code=True)

        self._score_token_ids = []
        missing_tokens = []
        for tok in self.SCORE_TOKENS:
            ids = self.tokenizer_router.encode(tok, add_special_tokens=False)
            if len(ids) == 1:
                self._score_token_ids.append(ids[0])
            else:
                vocab = self.tokenizer_router.get_vocab()
                if tok in vocab:
                    self._score_token_ids.append(vocab[tok])
                else:
                    missing_tokens.append(tok)

        if missing_tokens:
            print(
                f"[RouteLLMCausalWrapper] WARNING: score tokens {missing_tokens} are not single tokens "
                f"in the router tokenizer. Adding them as special tokens (fallback)."
            )
            self.tokenizer_router.add_special_tokens({"additional_special_tokens": missing_tokens})
            self.router_model.resize_token_embeddings(len(self.tokenizer_router))
            self._score_token_ids = []
            for tok in self.SCORE_TOKENS:
                ids = self.tokenizer_router.encode(tok, add_special_tokens=False)
                if len(ids) == 1:
                    self._score_token_ids.append(ids[0])
                else:
                    self._score_token_ids.append(self.tokenizer_router.get_vocab()[tok])

        assert len(self._score_token_ids) == 3, f"Expected 3 score token IDs, got {self._score_token_ids}"
        print(f"[RouteLLMCausalWrapper] Score token IDs: {dict(zip(self.SCORE_TOKENS, self._score_token_ids))}")


    def _route(self, prompt: str) -> bool:
        """Return True if the query should go to the expert."""
        formatted = f"{self.CAUSAL_INSTRUCTION}\n\nQuery: {prompt}\nAnswer: "
        input_ids = self.tokenizer_router.encode(
            formatted, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.router_model(input_ids).logits[:, -1, :]

        scores = torch.softmax(logits[0, self._score_token_ids], dim=-1)
        win_base_prob = scores[2].item()
        return win_base_prob < self.router_threshold

    def generate(self, prompt: str) -> str:
        if self._route(prompt):
            model, tokenizer = self.expert_model, self.tokenizer_expert
        else:
            model, tokenizer = self.base_model, self.tokenizer_base

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        return self._truncate_at_stop(text)


# ---------------------------------------------------------------------------
# CPR
# ---------------------------------------------------------------------------

class CPRWrapper(BaseModelWrapper):
    """CPR: token-level routing between the base model and its domain expert.

    At every step t:
      1. The base model always runs; its last hidden state h_t feeds the router.
      2. p_t = sigmoid(router(h_t, c)), where c is the mean base hidden state
         over the last question span.
      3. m_t = alpha * m_{t-1} + (1 - alpha) * p_t
      4. Threshold-gated dispatch:
           m_t <  tau_low             -> base logits only (expert not run)
           tau_low <= m_t <= tau_high -> (1 - m_t) * p_base + m_t * p_expert
           m_t >  tau_high            -> expert logits only
    When the expert is (re)activated, its KV cache is rebuilt over the full
    sequence in a single batched forward pass.
    """

    def __init__(
        self,
        base_model_path: str,
        expert_model_path: str,
        router_model_path: str,
        max_new_tokens: int = 512,
        momentum_alpha: float = 0.5,
        tau_low: float = 0.35,
        tau_high: float = 0.65,
        device: str = DEVICE,
        stop_strings: Optional[List[str]] = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.momentum_alpha = momentum_alpha
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.device = device
        self.stop_strings = stop_strings or []

        print("[CPR] Loading base model...")
        self.tokenizer_base = _load_tokenizer(base_model_path, trust_remote_code=True, padding_side="left")
        self.base_model = _load_model(base_model_path, device, trust_remote_code=True)

        print("[CPR] Loading expert model...")
        self.expert_model = _load_model(expert_model_path, device, trust_remote_code=True)

        print(f"[CPR] Loading router from {router_model_path}")
        self.router = HierarchicalRouterNetwork(input_dim=get_hidden_size(self.base_model)).to(
            device=device, dtype=torch.bfloat16
        )
        if not os.path.exists(router_model_path):
            raise FileNotFoundError(f"Router weights not found: {router_model_path}")
        self.router.load_state_dict(torch.load(router_model_path, map_location=device))
        self.router.eval()

    def _pooled_question_context(self, prompt: str, hidden_states: torch.Tensor) -> torch.Tensor:
        """Mean hidden state over the last question (few-shot demonstrations excluded)."""
        last_q_idx = prompt.rfind("\n\nQuestion: ")
        q_start = len(self.tokenizer_base.encode(prompt[:last_q_idx])) if last_q_idx != -1 else 0
        return hidden_states[:, q_start:, :].mean(dim=1)

    def generate(self, prompt: str) -> str:
        input_ids = self.tokenizer_base.encode(prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids.clone()
        eos_token_id = self.tokenizer_base.eos_token_id

        with torch.no_grad():
            prompt_out = self.base_model(input_ids, output_hidden_states=True, use_cache=True)
            pooled_context = self._pooled_question_context(prompt, prompt_out.hidden_states[-1])
            base_kv = prompt_out.past_key_values
            expert_kv = None
            prev_need_expert = False

        momentum = 0.0

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                base_out = self.base_model(
                    generated_ids[:, -1:],
                    past_key_values=base_kv,
                    output_hidden_states=True,
                    use_cache=True,
                )
                base_kv = base_out.past_key_values
                base_logits = base_out.logits[:, -1, :]

                last_h = base_out.hidden_states[-1][:, -1, :]
                p_t = torch.sigmoid(self.router(last_h, pooled_context)).item()
                momentum = self.momentum_alpha * momentum + (1 - self.momentum_alpha) * p_t

                use_soft = self.tau_low <= momentum <= self.tau_high
                use_expert = momentum > self.tau_high
                need_expert = use_soft or use_expert

                expert_logits = None
                if need_expert:
                    if expert_kv is None or not prev_need_expert:
                        # KV-cache catch-up over all positions skipped by the expert.
                        exp_out = self.expert_model(generated_ids, use_cache=True)
                    else:
                        exp_out = self.expert_model(
                            generated_ids[:, -1:], past_key_values=expert_kv, use_cache=True
                        )
                    expert_kv = exp_out.past_key_values
                    expert_logits = exp_out.logits[:, -1, :]
                elif prev_need_expert:
                    expert_kv = None
                prev_need_expert = need_expert

                if use_expert:
                    next_token = torch.argmax(expert_logits, dim=-1).unsqueeze(0)
                elif use_soft:
                    w = momentum
                    base_probs = torch.softmax(base_logits, dim=-1)
                    expert_probs = torch.softmax(expert_logits, dim=-1)
                    final_probs = (1.0 - w) * base_probs + w * expert_probs
                    next_token = torch.argmax(final_probs, dim=-1).unsqueeze(0)
                else:
                    next_token = torch.argmax(base_logits, dim=-1).unsqueeze(0)

                if next_token.item() == eos_token_id:
                    break
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

        text = self.tokenizer_base.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return self._truncate_at_stop(text)

