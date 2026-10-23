from __future__ import annotations

import importlib
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    _add_special_kwargs,
    chunks,
    handle_stop_sequences,
    has_bos_prefix,
    normalize_gen_kwargs,
    postprocess_generated_text,
)


eval_logger = logging.getLogger(__name__)


def _load_wrapper_module():
    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / "model_wrappers.py").exists():
        raise FileNotFoundError(f"Could not locate model_wrappers.py in {repo_root}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module("model_wrappers")


def _as_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _safe_log_probs(probs: torch.Tensor) -> torch.Tensor:
    return torch.log(probs.clamp_min(1e-12))


@register_model("hf-wrapper", "critical-wrapper", "critical-hf")
class CriticalWrapperLM(TemplateLM):
    backend = "causal"
    _DEFAULT_MAX_LENGTH = 2048
    _SINGLE_MODEL_WRAPPERS = {
        "base": "BaseWrapper",
        "expert": "ExpertWrapper",
        "eaft": "EAFTWrapper",
        "lfu": "LFUWrapper",
    }
    _SUPPORTED_WRAPPERS = set(_SINGLE_MODEL_WRAPPERS) | {"cpr", "ensemble", "switch", "routellm_causal", "cd"}

    def __init__(
        self,
        wrapper: str,
        pretrained: str | None = None,
        model_path: str | None = None,
        base_model_path: str | None = None,
        expert_model_path: str | None = None,
        router_model_path: str | None = None,
        switcher_model_path: str | None = None,
        model_paths: list[str] | None = None,
        candidate_model_paths: list[str] | None = None,
        device: str | None = "cuda",
        batch_size: int | str | None = 1,
        max_length: int | None = None,
        max_new_tokens: int = 256,
        stop_strings: str | list[str] | None = None,
        truncation: bool | None = False,
        add_bos_token: bool | None = None,
        router_threshold: float = 0.5,
        momentum_alpha: float = 0.5,
        tau_low: float = 0.35,
        tau_high: float = 0.65,
        patch_size: int = 50,
        objective_flag: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        wrapper = wrapper.strip()
        if wrapper not in self._SUPPORTED_WRAPPERS:
            raise ValueError(
                f"Unsupported wrapper '{wrapper}'. Supported: {sorted(self._SUPPORTED_WRAPPERS)}"
            )

        self.wrapper_name = wrapper
        self.default_max_new_tokens = int(max_new_tokens)
        self.default_stop_strings = _as_list(stop_strings)
        self.truncation = bool(truncation)
        self.add_bos_token = add_bos_token
        self._max_length = max_length
        self._batch_size = 1 if batch_size in (None, "auto") else int(batch_size)
        if batch_size == "auto":
            eval_logger.warning(
                "`batch_size=auto` is not supported for hf-wrapper. Falling back to batch_size=1."
            )

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and torch.cuda.is_available():
            self._device = torch.device(device)
            if self._device.index is not None:
                torch.cuda.set_device(self._device)
            wrapper_device = "cuda"
        else:
            self._device = torch.device("cpu")
            wrapper_device = "cpu"

        self.wrapper_impl = self._build_wrapper(
            wrappers=_load_wrapper_module(),
            wrapper=wrapper,
            single_path=model_path or pretrained,
            base_model_path=base_model_path,
            expert_model_path=expert_model_path,
            router_model_path=router_model_path,
            switcher_model_path=switcher_model_path,
            model_paths=model_paths,
            candidate_model_paths=candidate_model_paths,
            device=wrapper_device,
            max_new_tokens=self.default_max_new_tokens,
            stop_strings=self.default_stop_strings,
            router_threshold=router_threshold,
            momentum_alpha=momentum_alpha,
            tau_low=tau_low,
            tau_high=tau_high,
            patch_size=patch_size,
            objective_flag=objective_flag,
        )

        self.tokenizer = self._resolve_tokenizer()
        self._tokenizer_name = getattr(
            self.tokenizer, "name_or_path", self.tokenizer.__class__.__name__
        ).replace("/", "__")
        self._objective_flag = bool(objective_flag)

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self) -> int:
        if getattr(self.tokenizer, "bos_token_id", None) is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        if self._max_length is not None:
            return self._max_length

        config = None
        wrapper = self.wrapper_impl
        if hasattr(wrapper, "model"):
            config = wrapper.model.config
        elif hasattr(wrapper, "base_model"):
            config = wrapper.base_model.config
        elif hasattr(wrapper, "models") and wrapper.models:
            config = wrapper.models[0].config

        if config is not None:
            for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
                if hasattr(config, attr):
                    return getattr(config, attr)
            if hasattr(config, "text_config"):
                for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
                    if hasattr(config.text_config, attr):
                        return getattr(config.text_config, attr)

        tokenizer_max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max_length, int) and tokenizer_max_length < 10**12:
            return tokenizer_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return self.default_max_new_tokens

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    def _build_wrapper(
        self,
        *,
        wrappers,
        wrapper: str,
        single_path: str | None,
        base_model_path: str | None,
        expert_model_path: str | None,
        router_model_path: str | None,
        switcher_model_path: str | None,
        model_paths: list[str] | None,
        candidate_model_paths: list[str] | None,
        device: str,
        max_new_tokens: int,
        stop_strings: list[str],
        router_threshold: float,
        momentum_alpha: float,
        tau_low: float,
        tau_high: float,
        patch_size: int,
        objective_flag: bool,
    ):
        common = dict(max_new_tokens=max_new_tokens, device=device, stop_strings=stop_strings)

        def require(**kwargs):
            missing = [f"`{k}`" for k, v in kwargs.items() if not v]
            if missing:
                raise ValueError(f"{', '.join(missing)} required for wrapper={wrapper}")

        if wrapper in self._SINGLE_MODEL_WRAPPERS:
            require(pretrained=single_path)
            cls = getattr(wrappers, self._SINGLE_MODEL_WRAPPERS[wrapper])
            return cls(model_path=single_path, **common)

        if wrapper == "ensemble":
            paths = [p for p in (model_paths or [base_model_path, expert_model_path]) if p]
            if len(paths) < 2:
                raise ValueError(
                    "`model_paths` or both `base_model_path` and `expert_model_path` are required for wrapper=ensemble"
                )
            return wrappers.EnsembleWrapper(model_paths=paths, **common)

        if wrapper == "switch":
            paths = [p for p in (candidate_model_paths or [base_model_path, expert_model_path]) if p]
            if len(paths) < 2 or not switcher_model_path:
                raise ValueError(
                    "`candidate_model_paths` (or base/expert paths) and `switcher_model_path` are required for wrapper=switch"
                )
            return wrappers.SwitchGenerationWrapper(
                candidate_model_paths=paths,
                switcher_model_path=switcher_model_path,
                max_total_tokens=max_new_tokens,
                patch_size=patch_size,
                objective_flag=objective_flag,
                stop_strings=stop_strings,
            )

        if wrapper == "cpr":
            require(
                base_model_path=base_model_path,
                expert_model_path=expert_model_path,
                router_model_path=router_model_path,
            )
            return wrappers.CPRWrapper(
                base_model_path=base_model_path,
                expert_model_path=expert_model_path,
                router_model_path=router_model_path,
                momentum_alpha=momentum_alpha,
                tau_low=tau_low,
                tau_high=tau_high,
                **common,
            )

        if wrapper == "routellm_causal":
            require(
                base_model_path=base_model_path,
                expert_model_path=expert_model_path,
                router_model_path=router_model_path,
            )
            return wrappers.RouteLLMCausalWrapper(
                base_model_path=base_model_path,
                expert_model_path=expert_model_path,
                router_model_path=router_model_path,
                router_threshold=router_threshold,
                **common,
            )

        if wrapper == "cd":
            require(base_model_path=base_model_path, expert_model_path=expert_model_path)
            return wrappers.ContrastiveDecodingWrapper(
                base_model_path=base_model_path,
                expert_model_path=expert_model_path,
                **common,
            )

        raise AssertionError(f"Unhandled wrapper type: {wrapper}")

    def _resolve_tokenizer(self):
        wrapper = self.wrapper_impl
        if hasattr(wrapper, "tokenizer"):
            return wrapper.tokenizer
        if hasattr(wrapper, "base_tokenizer"):
            return wrapper.base_tokenizer
        if hasattr(wrapper, "tokenizer_base"):
            return wrapper.tokenizer_base
        if hasattr(wrapper, "tokenizers") and wrapper.tokenizers:
            return wrapper.tokenizers[0]
        raise ValueError(f"Could not determine tokenizer for wrapper={self.wrapper_name}")

    def _set_generation_config(self, max_new_tokens: int, stop_strings: list[str]) -> None:
        wrapper = self.wrapper_impl
        if hasattr(wrapper, "max_new_tokens"):
            wrapper.max_new_tokens = int(max_new_tokens)
        if hasattr(wrapper, "max_total_tokens"):
            wrapper.max_total_tokens = int(max_new_tokens)
            if hasattr(wrapper, "patch_size") and hasattr(wrapper, "rounds"):
                wrapper.rounds = max(1, math.ceil(max_new_tokens / wrapper.patch_size))
        wrapper.stop_strings = list(stop_strings)

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **kwargs,
    ) -> list[int]:
        special_tokens_kwargs = _add_special_kwargs(add_special_tokens, self.add_bos_token)
        if add_special_tokens is None and has_bos_prefix(
            string, getattr(self.tokenizer, "bos_token", None)
        ):
            special_tokens_kwargs["add_special_tokens"] = False
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs, **kwargs)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens, skip_special_tokens: bool = True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _prepare_window(
        self,
        context_enc: list[int],
        continuation_enc: list[int],
    ) -> tuple[list[int], int, int]:
        full_tokens = context_enc + continuation_enc
        window = full_tokens[-(self.max_length + 1) :]
        contlen = min(len(continuation_enc), max(0, len(window) - 1))
        context_len = len(window) - contlen
        if context_len <= 0:
            raise ValueError("Context window collapsed while preparing wrapper scoring")
        return window, context_len, contlen

    def _score_single_model(
        self,
        model,
        context_enc: list[int],
        continuation_enc: list[int],
    ) -> tuple[float, bool]:
        window, context_len, contlen = self._prepare_window(context_enc, continuation_enc)
        inp = torch.tensor(window[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = model(inp).logits[:, context_len - 1 : context_len - 1 + contlen, :]
        target = torch.tensor(window[-contlen:], dtype=torch.long, device=self.device).unsqueeze(0)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        selected = torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)
        greedy = (log_probs.argmax(dim=-1) == target).all().item()
        return float(selected.sum().item()), bool(greedy)

    def _score_ensemble(
        self,
        context_enc: list[int],
        continuation_enc: list[int],
    ) -> tuple[float, bool]:
        window, context_len, contlen = self._prepare_window(context_enc, continuation_enc)
        inp = torch.tensor(window[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        probs = []
        with torch.no_grad():
            for model in self.wrapper_impl.models:
                logits = model(inp).logits[:, context_len - 1 : context_len - 1 + contlen, :]
                probs.append(torch.softmax(logits.float(), dim=-1))
        avg_probs = torch.mean(torch.stack(probs, dim=0), dim=0)
        target = torch.tensor(window[-contlen:], dtype=torch.long, device=self.device).unsqueeze(0)
        log_probs = _safe_log_probs(avg_probs)
        selected = torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)
        greedy = (avg_probs.argmax(dim=-1) == target).all().item()
        return float(selected.sum().item()), bool(greedy)

    def _select_switch_model(self, query: str, segments: list[str], seg_model_ids: list[int]) -> int:
        wrapper = self.wrapper_impl
        prompt = wrapper._make_switcher_prompt(query, segments, seg_model_ids)
        outs, _ = wrapper._batch_generate(
            wrapper.switcher_model, wrapper.switcher_tokenizer, [prompt], max_new=10
        )
        return wrapper._parse_switcher_output(outs[0] if outs else "")

    def _score_switch(
        self,
        context_text: str,
        context_enc: list[int],
        continuation_enc: list[int],
    ) -> tuple[float, bool]:
        wrapper = self.wrapper_impl
        if self._objective_flag:
            eval_logger.warning(
                "Switch wrapper loglikelihood ignores `objective_flag=True` prompt injection. "
                "Generation still uses the original wrapper behavior."
            )

        window, context_len, contlen = self._prepare_window(context_enc, continuation_enc)
        inp = torch.tensor(window[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        targets = torch.tensor(window[-contlen:], dtype=torch.long, device=self.device)

        all_log_probs = []
        all_argmax = []
        with torch.no_grad():
            for model in wrapper.models:
                logits = model(inp).logits[:, context_len - 1 : context_len - 1 + contlen, :]
                log_probs = F.log_softmax(logits.float(), dim=-1)[0]
                all_log_probs.append(log_probs)
                all_argmax.append(log_probs.argmax(dim=-1))

        effective_continuation = window[-contlen:]
        token_chunks = [
            effective_continuation[idx : idx + wrapper.patch_size]
            for idx in range(0, len(effective_continuation), wrapper.patch_size)
        ]
        segment_texts = [
            wrapper.tokenizers[0].decode(chunk, skip_special_tokens=True) for chunk in token_chunks
        ]

        chosen_models = []
        seen_segments: list[str] = []
        seen_model_ids: list[int] = []
        for round_idx in range(len(token_chunks)):
            if round_idx == 0 or round_idx == len(token_chunks) - 1:
                selected = 1
            else:
                selected = self._select_switch_model(context_text, seen_segments, seen_model_ids)
            chosen_models.append(selected)
            seen_segments.append(segment_texts[round_idx])
            seen_model_ids.append(selected)

        total_logprob = 0.0
        greedy = True
        token_offset = 0
        for patch_idx, patch_tokens in enumerate(token_chunks):
            selected = chosen_models[patch_idx]
            patch_len = len(patch_tokens)
            patch_targets = targets[token_offset : token_offset + patch_len]
            patch_log_probs = all_log_probs[selected][token_offset : token_offset + patch_len]
            patch_argmax = all_argmax[selected][token_offset : token_offset + patch_len]
            total_logprob += float(
                torch.gather(patch_log_probs, 1, patch_targets.unsqueeze(-1)).sum().item()
            )
            greedy = greedy and bool((patch_argmax == patch_targets).all().item())
            token_offset += patch_len

        return total_logprob, greedy

    def _score_cd(
        self,
        context_enc: list[int],
        continuation_enc: list[int],
    ) -> tuple[float, bool]:
        wrapper = self.wrapper_impl
        window, context_len, contlen = self._prepare_window(context_enc, continuation_enc)
        inp = torch.tensor(window[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        targets = torch.tensor(window[-contlen:], dtype=torch.long, device=self.device)

        with torch.no_grad():
            base_logp_all = torch.log_softmax(wrapper.base_model(inp).logits[0].float(), dim=-1)
            expert_logp_all = torch.log_softmax(wrapper.expert_model(inp).logits[0].float(), dim=-1)

        total_logprob = 0.0
        greedy = True
        for rel_idx in range(contlen):
            pos = context_len - 1 + rel_idx
            target = targets[rel_idx]
            cd_scores = expert_logp_all[pos] - wrapper.cd_alpha * base_logp_all[pos]
            total_logprob += float(cd_scores[target].item())
            greedy = greedy and int(cd_scores.argmax().item()) == int(target.item())

        return total_logprob, greedy

    def _score_request(
        self,
        request_str: tuple[str, str] | None,
        context_enc: list[int],
        continuation_enc: list[int],
    ) -> tuple[float, bool]:
        context_text = request_str[0] if request_str is not None else ""
        if self.wrapper_name in self._SINGLE_MODEL_WRAPPERS:
            return self._score_single_model(self.wrapper_impl.model, context_enc, continuation_enc)
        if self.wrapper_name == "ensemble":
            return self._score_ensemble(context_enc, continuation_enc)
        if self.wrapper_name == "switch":
            return self._score_switch(context_text, context_enc, continuation_enc)
        if self.wrapper_name == "routellm_causal":
            use_expert = self.wrapper_impl._route(context_text)
            model = self.wrapper_impl.expert_model if use_expert else self.wrapper_impl.base_model
            return self._score_single_model(model, context_enc, continuation_enc)
        if self.wrapper_name == "cd":
            return self._score_cd(context_enc, continuation_enc)
        raise NotImplementedError(
            f"Log-likelihood scoring is not supported for wrapper={self.wrapper_name}; "
            "use generative (generate_until) tasks."
        )

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        **_: Any,
    ) -> list[tuple[float, bool]]:
        res = []
        for request_str, context_enc, continuation_enc in tqdm(
            requests,
            total=len(requests),
            desc="Running wrapper loglikelihood requests",
            disable=disable_tqdm,
        ):
            answer = self._score_request(request_str, context_enc, continuation_enc)
            res.append(answer)
            self.cache_hook.add_partial("loglikelihood", request_str, answer)
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        results = []
        for (string,) in tqdm(
            [req.args for req in requests],
            total=len(requests),
            desc="Running wrapper rolling loglikelihood requests",
            disable=disable_tqdm,
        ):
            rolling_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            wrapped_windows = [(None, ctx, cont) for ctx, cont in rolling_windows]
            nlls = self._loglikelihood_tokens(wrapped_windows, disable_tqdm=True)
            total = float(sum(x[0] for x in nlls))
            results.append(total)
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), total)
        return results

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        grouped: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
        for idx, (context, gen_kwargs) in enumerate(req.args for req in requests):
            kwargs = dict(normalize_gen_kwargs(gen_kwargs, self.max_gen_toks))
            key = repr(sorted(kwargs.items()))
            grouped.setdefault(key, []).append((idx, context, kwargs))

        results = [""] * len(requests)
        progress = tqdm(
            total=len(requests),
            desc="Running wrapper generate_until requests",
            disable=disable_tqdm,
        )

        for group in grouped.values():
            kwargs = dict(group[0][2])
            until = handle_stop_sequences(kwargs.pop("until", None), eos=None)
            max_gen_toks = int(kwargs.pop("max_gen_toks"))
            do_sample = bool(kwargs.pop("do_sample", False))
            if do_sample:
                eval_logger.warning(
                    "hf-wrapper only supports greedy generation. Ignoring do_sample=True."
                )
            if kwargs:
                eval_logger.warning(f"Unused generation kwargs for hf-wrapper: {sorted(kwargs.keys())}")

            self._set_generation_config(
                max_new_tokens=max_gen_toks,
                stop_strings=until or self.default_stop_strings,
            )

            for batch_items in chunks(group, n=self.batch_size):
                contexts = [item[1] for item in batch_items]
                batch_outputs = self.wrapper_impl.generate_batch(contexts)

                for (idx, context, gen_kwargs), output in zip(batch_items, batch_outputs, strict=True):
                    text = postprocess_generated_text(output, until, think_end_token=None)
                    results[idx] = text
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)

                progress.update(len(batch_outputs))

        progress.close()
        return results

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
