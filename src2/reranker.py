import numpy as np
import torch
from tqdm import tqdm

from .retrievers import resolve_model


class _CrossEncoderPredictor:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_length: int,
        instruction: str | None = None,
        prompt_name: str = "query",
    ):
        from sentence_transformers import CrossEncoder

        kwargs = {
            "max_length": max_length,
            "automodel_args": {"torch_dtype": torch.float16},
        }
        if "minicpm" in model_name.lower():
            kwargs["trust_remote_code"] = True
        if instruction:
            kwargs["prompts"] = {prompt_name: instruction}
            kwargs["default_prompt_name"] = prompt_name

        self.model = CrossEncoder(model_name, **kwargs)
        self.model.model.to(device)
        if "minicpm" in model_name.lower():
            self.model.tokenizer.padding_side = "right"

    def predict(self, pairs: list[tuple[str, str]], batch_size: int, show_progress_bar: bool) -> np.ndarray:
        return self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )


class _GemmaRerankerPredictor:
    INSTRUCTION = (
        "Given a query A and a passage B, determine whether the passage contains an "
        "answer to the query by providing a prediction of either 'Yes' or 'No'."
    )

    def __init__(self, model_name: str, device: torch.device, max_length: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.max_length = max_length
        self.yes_token_id = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        self._prompt_ids = self.tokenizer(self.INSTRUCTION, add_special_tokens=False)["input_ids"]
        self._sep_ids = self.tokenizer("\n", add_special_tokens=False)["input_ids"]

    def _build_one(self, query: str, passage: str) -> dict:
        body_max = self.max_length
        q_ids = self.tokenizer(
            f"A: {query}",
            add_special_tokens=False,
            max_length=body_max * 3 // 4,
            truncation=True,
        )["input_ids"]
        p_ids = self.tokenizer(
            f"B: {passage}",
            add_special_tokens=False,
            max_length=body_max,
            truncation=True,
        )["input_ids"]
        item = self.tokenizer.prepare_for_model(
            [self.tokenizer.bos_token_id] + q_ids,
            self._sep_ids + p_ids,
            truncation="only_second",
            max_length=body_max,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        item["input_ids"] = item["input_ids"] + self._sep_ids + self._prompt_ids
        item["attention_mask"] = [1] * len(item["input_ids"])
        return item

    def _pad_batch(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        padded_len = ((max_len + 7) // 8) * 8
        input_ids = torch.full(
            (len(batch), padded_len),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (len(batch), padded_len),
            dtype=torch.long,
            device=self.device,
        )
        for row, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            input_ids[row, -seq_len:] = torch.tensor(item["input_ids"], dtype=torch.long, device=self.device)
            attention_mask[row, -seq_len:] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def predict(self, pairs: list[tuple[str, str]], batch_size: int, show_progress_bar: bool) -> np.ndarray:
        items = [self._build_one(q, p) for q, p in pairs]
        order = np.argsort([len(item["input_ids"]) for item in items])
        scores = np.empty(len(pairs), dtype=np.float32)
        ranges = list(range(0, len(pairs), batch_size))
        iterator = tqdm(ranges, desc="Rerank") if show_progress_bar else ranges

        with torch.inference_mode():
            for start in iterator:
                idx = order[start:start + batch_size]
                batch = [items[i] for i in idx]
                inputs = self._pad_batch(batch)
                logits = self.model(**inputs).logits[:, -1, self.yes_token_id]
                scores[idx] = logits.float().cpu().numpy()
        return scores


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int,
        instruction: str | None = None,
        prompt_name: str = "query",
        query_prefix: str = "",
    ):
        print(f"Loading reranker: {model_name} on {device}")
        resolved = resolve_model(model_name)
        torch_device = torch.device(device)
        is_llm_reranker = "gemma" in model_name.lower()
        self.query_prefix = query_prefix
        if is_llm_reranker:
            self._predictor = _GemmaRerankerPredictor(resolved, torch_device, max_length)
        else:
            self._predictor = _CrossEncoderPredictor(
                resolved,
                torch_device,
                max_length,
                instruction=instruction,
                prompt_name=prompt_name,
            )

    def rerank_batch(
        self,
        queries: list[str],
        candidates_batch: list[list[str]],
        *,
        top_k: int,
        batch_size: int,
    ) -> list[np.ndarray]:
        if self.query_prefix:
            queries = [f"{self.query_prefix}{query}" for query in queries]
        n_candidates = len(candidates_batch[0])
        all_pairs = [
            (query, doc)
            for query, candidates in zip(queries, candidates_batch)
            for doc in candidates
        ]
        all_scores = self._predictor.predict(
            all_pairs,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        results = []
        for i in range(len(queries)):
            scores = all_scores[i * n_candidates:(i + 1) * n_candidates]
            results.append(np.argsort(-scores)[:top_k])
        return results
