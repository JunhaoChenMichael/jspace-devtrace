"""Pure-PyTorch objectives for workspace-guided memory admission.

The admission policy has two related action spaces:

* Stage A uses a constrained binary action (``0 = No``, ``1 = Yes``).
* Stages B/C choose an unordered, exact-budget subset.  Sampling is performed
  with Gumbel top-k, whose induced ordered distribution is Plackett--Luce.  For
  the small candidate sets used by this project, an unordered set probability
  can be computed exactly by summing over the ``k!`` latent orders.

This module deliberately contains no model, tokenizer, TRL, or reward-evaluator
dependencies.  Rewards and advantages are detached by the caller/trainer; the
probability and KL functions remain differentiable with respect to policy
scores.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import torch
from torch import Tensor


def _floating_tensor(values: Tensor | Sequence[float]) -> Tensor:
    """Return ``values`` as a floating tensor without moving an existing tensor."""
    if isinstance(values, Tensor):
        if values.is_floating_point():
            return values
        return values.to(dtype=torch.get_default_dtype())
    return torch.as_tensor(values, dtype=torch.get_default_dtype())


def _check_temperature(temperature: float) -> None:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature}")


def percentile_ranks(values: Tensor | Sequence[float]) -> Tensor:
    """Return tie-aware within-episode percentile ranks in ``[0, 1]``.

    Ranks are ascending (the largest workspace score receives ``1``) and ties
    receive their midrank.  A singleton or an all-tied episode receives ``0.5``.
    The quadratic implementation is intentional: candidate sets in this project
    contain only a handful of items, and the direct definition is easy to audit.
    """
    scores = _floating_tensor(values)
    if scores.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {tuple(scores.shape)}")
    if scores.numel() == 0:
        raise ValueError("values must not be empty")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("values must all be finite")
    if scores.numel() == 1:
        return torch.full_like(scores, 0.5)

    # Zero-based midrank: number strictly below plus half the other tied items.
    below = (scores[:, None] > scores[None, :]).sum(dim=1).to(scores.dtype)
    tied = (scores[:, None] == scores[None, :]).sum(dim=1).to(scores.dtype)
    midrank = below + 0.5 * (tied - 1.0)
    return midrank / (scores.numel() - 1)


def workspace_action_reward(
    actions: Tensor | Sequence[int],
    percentiles: Tensor | Sequence[float],
    *,
    signed: bool = True,
) -> Tensor:
    """Workspace agreement reward for Stage-A Yes/No actions.

    ``actions`` use ``0 = No`` and ``1 = Yes``.  With ``signed=True`` this is

    ``(2 * action - 1) * (2 * percentile - 1)``

    and lies in ``[-1, 1]``.  The unsigned affine equivalent is
    ``action * percentile + (1-action) * (1-percentile)`` in ``[0, 1]``.
    Inputs follow normal PyTorch broadcasting rules.
    """
    action_device = actions.device if isinstance(actions, Tensor) else None
    ranks = _floating_tensor(percentiles)
    if action_device is not None:
        ranks = ranks.to(action_device)
    acts = torch.as_tensor(actions, device=ranks.device)
    if not bool(((acts == 0) | (acts == 1)).all()):
        raise ValueError("actions must contain only 0 (No) or 1 (Yes)")
    if not bool(torch.isfinite(ranks).all()) or not bool(((ranks >= 0) & (ranks <= 1)).all()):
        raise ValueError("percentiles must be finite and lie in [0, 1]")
    acts = acts.to(dtype=ranks.dtype)
    reward = (2.0 * acts - 1.0) * (2.0 * ranks - 1.0)
    return reward if signed else 0.5 * (reward + 1.0)


def _validated_selection(selected: Tensor | Sequence[int], n: int, device: torch.device) -> Tensor:
    indices = torch.as_tensor(selected, dtype=torch.long, device=device)
    if indices.ndim == 0:
        indices = indices.unsqueeze(0)
    if indices.ndim < 1 or indices.shape[-1] == 0:
        raise ValueError("selected must contain at least one candidate index")
    if bool(((indices < 0) | (indices >= n)).any()):
        raise ValueError(f"selected indices must lie in [0, {n})")

    flat = indices.reshape(-1, indices.shape[-1])
    if flat.shape[1] > 1:
        ordered = flat.sort(dim=-1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            raise ValueError("selected indices must be unique within each set")
    return indices


def workspace_set_reward(
    percentiles: Tensor | Sequence[float],
    selected: Tensor | Sequence[int],
    *,
    contrastive: bool = False,
) -> Tensor:
    """Workspace utility of one or more exact-budget selected sets.

    ``selected`` has shape ``[..., k]`` and indexes a one-dimensional vector of
    tie-aware workspace percentiles.  The default reward is the selected mean in
    ``[0, 1]``.  The optional contrastive reward is

    ``0.5 * (1 + mean(selected) - mean(unselected))``.

    At fixed ``n`` and ``k`` the contrastive form is only an affine rescaling of
    the selected mean; it cannot change the optimal subset.  It is unavailable
    for ``k == n`` because the complement is empty.
    """
    ranks = _floating_tensor(percentiles)
    if ranks.ndim != 1 or ranks.numel() == 0:
        raise ValueError("percentiles must be a non-empty one-dimensional tensor")
    if not bool(torch.isfinite(ranks).all()) or not bool(((ranks >= 0) & (ranks <= 1)).all()):
        raise ValueError("percentiles must be finite and lie in [0, 1]")
    indices = _validated_selection(selected, ranks.numel(), ranks.device)
    selected_values = ranks[indices]
    selected_mean = selected_values.mean(dim=-1)
    if not contrastive:
        return selected_mean

    k = indices.shape[-1]
    n = ranks.numel()
    if k == n:
        raise ValueError("contrastive workspace reward requires at least one unselected item")
    selected_sum = selected_values.sum(dim=-1)
    unselected_mean = (ranks.sum() - selected_sum) / (n - k)
    return 0.5 * (1.0 + selected_mean - unselected_mean)


def hybrid_reward(
    qa_reward: Tensor | Sequence[float],
    workspace_reward: Tensor | Sequence[float],
    *,
    lambda_qa: float = 1.0,
    lambda_workspace: float = 1.0,
    normalize_weights: bool = True,
) -> Tensor:
    """Combine detached QA and workspace rewards for Stage C.

    Both coefficients must be non-negative and at least one must be positive.
    Normalizing by their sum keeps the hybrid in ``[0, 1]`` when its components
    are in ``[0, 1]`` and makes a separately-added KL coefficient comparable
    across coefficient ratios.
    """
    if lambda_qa < 0 or lambda_workspace < 0:
        raise ValueError("reward coefficients must be non-negative")
    weight_sum = lambda_qa + lambda_workspace
    if weight_sum <= 0:
        raise ValueError("at least one reward coefficient must be positive")

    qa = _floating_tensor(qa_reward)
    workspace = _floating_tensor(workspace_reward).to(device=qa.device, dtype=qa.dtype)
    combined = lambda_qa * qa + lambda_workspace * workspace
    return combined / weight_sum if normalize_weights else combined


def normalize_advantages(
    rewards: Tensor,
    *,
    mode: str = "auto",
    group_dim: int = -1,
    min_std: float = 1e-6,
    detach: bool = True,
) -> Tensor:
    """Compute group-relative advantages without mixing different prompts.

    Modes:

    * ``center``: subtract the group mean but retain reward magnitude.  This is
      useful for Stage A because per-pair z-scoring erases ``abs(2*r - 1)``.
    * ``zscore``: population-standardize each non-flat group.  Exactly or nearly
      flat groups return zero rather than NaN or amplified numerical noise.
    * ``auto``: z-score groups with ``std >= min_std`` and only center smaller
      groups.  Thus tiny floating-point differences are not magnified.

    KL must not be subtracted before this function: a prompt-level exact KL is a
    groupwise constant and would cancel.  Add KL as a separate loss term.
    """
    if not isinstance(rewards, Tensor) or not rewards.is_floating_point():
        rewards = _floating_tensor(rewards)
    if rewards.ndim == 0:
        raise ValueError("rewards must have a group dimension")
    if mode not in {"auto", "center", "zscore"}:
        raise ValueError(f"unknown advantage normalization mode: {mode}")
    if not math.isfinite(min_std) or min_std <= 0:
        raise ValueError("min_std must be finite and > 0")
    if rewards.shape[group_dim] < 2:
        raise ValueError("each reward group must contain at least two samples")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must all be finite")

    centered = rewards - rewards.mean(dim=group_dim, keepdim=True)
    if mode == "center":
        result = centered
    else:
        std = centered.square().mean(dim=group_dim, keepdim=True).sqrt()
        standardized = centered / std.clamp_min(min_std)
        if mode == "zscore":
            result = torch.where(std >= min_std, standardized, torch.zeros_like(centered))
        else:
            result = torch.where(std >= min_std, standardized, centered)
    return result.detach() if detach else result


def categorical_kl(policy_log_probs: Tensor, reference_log_probs: Tensor) -> Tensor:
    """Exact ``KL(policy || reference)`` for categorical log-probabilities.

    The final dimension is treated as the action dimension; all leading batch
    dimensions are preserved.  Inputs are re-normalized for numerical safety.
    """
    if policy_log_probs.shape != reference_log_probs.shape:
        raise ValueError("policy and reference log-probabilities must have identical shapes")
    if policy_log_probs.ndim == 0 or policy_log_probs.shape[-1] < 1:
        raise ValueError("log-probabilities must have a non-empty action dimension")
    policy = torch.log_softmax(policy_log_probs, dim=-1)
    reference = torch.log_softmax(
        reference_log_probs.to(device=policy.device, dtype=policy.dtype), dim=-1
    )
    probs = policy.exp()
    return (probs * (policy - reference)).sum(dim=-1)


def sample_gumbel_topk(
    scores: Tensor | Sequence[float],
    k: int,
    *,
    num_samples: int = 1,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample canonical unordered exact-``k`` subsets with Gumbel top-k.

    Scores are unnormalized log weights.  The returned tensor has shape
    ``[num_samples, k]``; every row is sorted by candidate id, not by its latent
    Plackett--Luce draw order.  Sorting makes the action explicitly set-valued.
    """
    logits = _floating_tensor(scores)
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("scores must be a non-empty one-dimensional tensor")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("scores must all be finite")
    if not isinstance(k, int) or k < 1 or k > logits.numel():
        raise ValueError(f"k must be an integer in [1, {logits.numel()}]")
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    _check_temperature(temperature)

    work = logits / temperature
    uniform = torch.rand(
        (num_samples, logits.numel()),
        dtype=work.dtype,
        device=work.device,
        generator=generator,
    )
    finfo = torch.finfo(work.dtype)
    uniform = uniform.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
    gumbel = -torch.log(-torch.log(uniform))
    latent_order = torch.topk(work.unsqueeze(0) + gumbel, k=k, dim=-1, sorted=True).indices
    return latent_order.sort(dim=-1).values


def _validate_scores_and_k(scores: Tensor | Sequence[float], k: int) -> Tensor:
    logits = _floating_tensor(scores)
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("scores must be a non-empty one-dimensional tensor")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("scores must all be finite")
    if not isinstance(k, int) or k < 1 or k > logits.numel():
        raise ValueError(f"k must be an integer in [1, {logits.numel()}]")
    return logits


def _ordered_logprob(scores: Tensor, order: Sequence[int], temperature: float) -> Tensor:
    """Log-probability of one ordered Plackett--Luce draw (internal helper)."""
    scaled = scores / temperature
    available = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
    terms: list[Tensor] = []
    for index in order:
        denominator = torch.logsumexp(scaled.masked_fill(~available, -torch.inf), dim=0)
        terms.append(scaled[index] - denominator)
        available[index] = False
    return torch.stack(terms).sum()


def set_logprob(
    scores: Tensor | Sequence[float],
    selected: Tensor | Sequence[int],
    *,
    temperature: float = 1.0,
    max_permutations: int = 100_000,
) -> Tensor:
    """Exact log-probability of an unordered Gumbel top-k subset.

    The probability is the sum of the Plackett--Luce probabilities of every
    latent ordering of the selected candidates.  It is invariant to the input
    order of ``selected`` and differentiable with respect to ``scores``.
    """
    _check_temperature(temperature)
    logits = _floating_tensor(scores)
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("scores must be a non-empty one-dimensional tensor")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("scores must all be finite")
    indices = _validated_selection(selected, logits.numel(), logits.device)
    if indices.ndim != 1:
        raise ValueError("set_logprob accepts exactly one selected set")
    k = indices.numel()
    if math.factorial(k) > max_permutations:
        raise ValueError(f"selected set requires more than {max_permutations} permutations")

    canonical = indices.sort().values.tolist()
    ordered = [
        _ordered_logprob(logits, permutation, temperature)
        for permutation in itertools.permutations(canonical)
    ]
    return torch.logsumexp(torch.stack(ordered), dim=0)


def _all_set_logprobs(scores: Tensor, k: int, temperature: float) -> tuple[Tensor, Tensor]:
    combinations = list(itertools.combinations(range(scores.numel()), k))
    sets = torch.tensor(combinations, dtype=torch.long, device=scores.device)
    log_probs = torch.stack(
        [set_logprob(scores, subset, temperature=temperature) for subset in combinations]
    )
    return sets, log_probs


def exact_set_kl(
    policy_scores: Tensor | Sequence[float],
    reference_scores: Tensor | Sequence[float],
    k: int,
    *,
    temperature: float = 1.0,
    normalize_by_k: bool = False,
    max_sets: int = 100_000,
) -> Tensor:
    """Exact KL between two unordered Gumbel top-k set policies.

    This enumerates ``C(n, k)`` subsets and is intended for the project's small
    candidate sets (currently ``n <= 7``, ``k <= 3``).  The reference scores are
    treated as ordinary input; callers should detach/freeze them.  Set
    log-probabilities are explicitly renormalized to remove round-off error.
    """
    _check_temperature(temperature)
    policy = _validate_scores_and_k(policy_scores, k)
    reference = _floating_tensor(reference_scores).to(device=policy.device, dtype=policy.dtype)
    if reference.shape != policy.shape:
        raise ValueError("policy and reference scores must have identical one-dimensional shapes")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("reference scores must all be finite")
    if math.comb(policy.numel(), k) > max_sets:
        raise ValueError(f"selection space contains more than {max_sets} sets")

    _, policy_lp = _all_set_logprobs(policy, k, temperature)
    _, reference_lp = _all_set_logprobs(reference, k, temperature)
    policy_lp = policy_lp - torch.logsumexp(policy_lp, dim=0)
    reference_lp = reference_lp - torch.logsumexp(reference_lp, dim=0)
    kl = (policy_lp.exp() * (policy_lp - reference_lp)).sum()
    return kl / k if normalize_by_k else kl


def exact_set_entropy(
    scores: Tensor | Sequence[float],
    k: int,
    *,
    temperature: float = 1.0,
    normalize: bool = False,
    max_sets: int = 100_000,
) -> Tensor:
    """Entropy of the exact unordered Gumbel top-k set distribution.

    Normalized entropy is divided by ``log(C(n, k))`` and therefore lies in
    ``[0, 1]`` for the selector regimes used by this project.
    """

    _check_temperature(temperature)
    policy = _validate_scores_and_k(scores, k)
    set_count = math.comb(policy.numel(), k)
    if set_count > max_sets:
        raise ValueError(f"selection space contains more than {max_sets} sets")
    _, log_probs = _all_set_logprobs(policy, k, temperature)
    log_probs = log_probs - torch.logsumexp(log_probs, dim=0)
    entropy = -(log_probs.exp() * log_probs).sum()
    if not normalize:
        return entropy
    if set_count == 1:
        return entropy * 0.0
    return entropy / math.log(set_count)


def grpo_clipped_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float = 0.2,
    kl: Tensor | None = None,
    beta: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """PPO/GRPO clipped surrogate loss with an independent exact-KL term.

    ``current_log_probs`` and ``old_log_probs`` are log-probabilities of the
    sampled binary action or unordered set.  ``old_log_probs`` and ``advantages``
    are detached internally.  If supplied, ``kl`` is added *after* the policy
    surrogate; it must not be folded into rewards before group normalization.
    """
    if current_log_probs.shape != old_log_probs.shape or current_log_probs.shape != advantages.shape:
        raise ValueError("current, old, and advantage tensors must have identical shapes")
    if not math.isfinite(clip_epsilon) or clip_epsilon < 0 or clip_epsilon >= 1:
        raise ValueError("clip_epsilon must lie in [0, 1)")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"unknown reduction: {reduction}")

    log_ratio = current_log_probs - old_log_probs.detach()
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    detached_advantages = advantages.detach()
    surrogate = torch.minimum(ratio * detached_advantages, clipped_ratio * detached_advantages)

    if reduction == "mean":
        loss = -surrogate.mean()
    elif reduction == "sum":
        loss = -surrogate.sum()
    else:
        loss = -surrogate

    if kl is not None:
        penalty = kl.mean() if reduction == "mean" else kl.sum() if reduction == "sum" else kl
        loss = loss + beta * penalty
    elif beta != 0:
        raise ValueError("a non-zero beta requires an explicit KL tensor")
    return loss


__all__ = [
    "categorical_kl",
    "exact_set_entropy",
    "exact_set_kl",
    "grpo_clipped_loss",
    "hybrid_reward",
    "normalize_advantages",
    "percentile_ranks",
    "sample_gumbel_topk",
    "set_logprob",
    "workspace_action_reward",
    "workspace_set_reward",
]
