"""Frozen recall-model evaluation shared by selector RL and final evaluation."""

from __future__ import annotations

import re
from contextlib import contextmanager

import torch

from memory_rl.modeling import adapter_disabled


STOP = set(
    "the a an of to in on at for and or but with without is are was were be been "
    "this that these those it its her his their your my our".split()
)


def normalize_answer(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def grade_answer(output: str, gold: str) -> bool:
    """The exact lightweight grader used by experiments/downstream.py."""
    out, target = normalize_answer(output), normalize_answer(gold)
    if target and target in out:
        return True
    words = [w for w in target.split() if len(w) >= 3 and w not in STOP]
    return any(w in out for w in words) if words else target in out


def recall_prompt(concepts: list[str], question: str) -> str:
    memory = ", ".join(concepts) if concepts else "(nothing)"
    return (
        "Earlier you read a passage. The only things you remember from it are: "
        f"{memory}.\nUsing only what you remember, answer concisely.\n"
        f"Question: {question}\nAnswer:"
    )


def full_context_prompt(context: str, question: str) -> str:
    return f"{context}\nAnswer concisely.\nQuestion: {question}\nAnswer:"


def no_memory_prompt(question: str) -> str:
    return f"Answer this question concisely.\nQuestion: {question}\nAnswer:"


def render_chat(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


@contextmanager
def _generation_mode(model):
    was_training = model.training
    old_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if old_cache is not None:
        model.config.use_cache = True
    try:
        with adapter_disabled(model):
            yield
    finally:
        if old_cache is not None:
            model.config.use_cache = old_cache
        model.train(was_training)


class FrozenRecall:
    """Greedy, adapter-disabled recall with a prompt-result cache."""

    def __init__(self, model, tokenizer, device: str, max_new_tokens: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.cache: dict[str, str] = {}

    @torch.no_grad()
    def generate(self, prompts: list[str]) -> list[str]:
        answers: list[str | None] = [self.cache.get(p) for p in prompts]
        missing = [i for i, value in enumerate(answers) if value is None]
        if not missing:
            return [str(x) for x in answers]
        rendered = [render_chat(self.tokenizer, prompts[i]) for i in missing]
        old_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            batch = self.tokenizer(rendered, padding=True, return_tensors="pt")
        finally:
            self.tokenizer.padding_side = old_padding
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with _generation_mode(self.model):
            output = self.model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            )
        prompt_len = batch["input_ids"].shape[1]
        decoded = self.tokenizer.batch_decode(
            output[:, prompt_len:], skip_special_tokens=True
        )
        for index, value in zip(missing, decoded):
            answer = value.strip()
            answers[index] = answer
            self.cache[prompts[index]] = answer
        return [str(x) for x in answers]

    def evaluate_sets(self, episode, sets: list[list[int]]) -> list[dict]:
        concepts = [
            [episode.candidates[i].concept for i in sorted(set(indices))]
            for indices in sets
        ]
        prompts = [recall_prompt(memory, episode.probe_question) for memory in concepts]
        answers = self.generate(prompts)
        return [
            {
                "selected_concepts": memory,
                "answer": answer,
                "correct": grade_answer(answer, episode.answer),
            }
            for memory, answer in zip(concepts, answers)
        ]

    def references(self, episode, budget: int) -> dict:
        positives = [
            i for i, candidate in enumerate(episode.candidates)
            if candidate.label == "load_bearing"
        ]
        negatives = [i for i in range(len(episode.candidates)) if i not in positives]
        oracle = (positives + negatives)[:budget]
        prompts = [
            recall_prompt(
                [episode.candidates[i].concept for i in oracle],
                episode.probe_question,
            ),
            full_context_prompt(episode.context, episode.probe_question),
            no_memory_prompt(episode.probe_question),
        ]
        answers = self.generate(prompts)
        return {
            "oracle_set": oracle,
            "oracle_answer": answers[0],
            "oracle_correct": grade_answer(answers[0], episode.answer),
            "full_context_answer": answers[1],
            "full_context_correct": grade_answer(answers[1], episode.answer),
            "no_memory_answer": answers[2],
            "no_memory_correct": grade_answer(answers[2], episode.answer),
        }
