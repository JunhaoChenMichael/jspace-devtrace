import itertools
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_rl.objectives import (
    categorical_kl,
    exact_set_entropy,
    exact_set_kl,
    grpo_clipped_loss,
    hybrid_reward,
    normalize_advantages,
    percentile_ranks,
    sample_gumbel_topk,
    set_logprob,
    workspace_action_reward,
    workspace_set_reward,
)


def test_percentile_ranks_are_tie_aware_and_bounded():
    ranks = percentile_ranks(torch.tensor([10.0, 20.0, 20.0, 40.0]))
    torch.testing.assert_close(ranks, torch.tensor([0.0, 0.5, 0.5, 1.0]))
    torch.testing.assert_close(percentile_ranks([3.0, 3.0, 3.0]), torch.full((3,), 0.5))
    torch.testing.assert_close(percentile_ranks([7.0]), torch.tensor([0.5]))


def test_percentile_ranks_reject_empty_nonfinite_and_nonvector_inputs():
    with pytest.raises(ValueError):
        percentile_ranks([])
    with pytest.raises(ValueError):
        percentile_ranks([0.0, float("nan")])
    with pytest.raises(ValueError):
        percentile_ranks(torch.ones(2, 2))


def test_workspace_action_reward_uses_zero_no_one_yes():
    ranks = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    yes = workspace_action_reward(torch.ones(5, dtype=torch.long), ranks)
    no = workspace_action_reward(torch.zeros(5, dtype=torch.long), ranks)
    torch.testing.assert_close(yes, torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]))
    torch.testing.assert_close(no, -yes)

    unsigned = workspace_action_reward([1, 0, 1], [0.8, 0.8, 0.5], signed=False)
    torch.testing.assert_close(unsigned, torch.tensor([0.8, 0.2, 0.5]))
    with pytest.raises(ValueError):
        workspace_action_reward([2], [0.5])


def test_workspace_set_reward_supports_batches_and_is_order_invariant():
    ranks = torch.tensor([0.0, 0.25, 0.75, 1.0])
    selected = torch.tensor([[0, 3], [2, 3]])
    reward = workspace_set_reward(ranks, selected)
    torch.testing.assert_close(reward, torch.tensor([0.5, 0.875]))
    torch.testing.assert_close(
        workspace_set_reward(ranks, [3, 0]), workspace_set_reward(ranks, [0, 3])
    )
    with pytest.raises(ValueError):
        workspace_set_reward(ranks, [1, 1])


def test_contrastive_workspace_reward_is_affine_at_fixed_budget():
    ranks = torch.tensor([0.0, 0.2, 0.7, 0.8, 1.0], dtype=torch.float64)
    sets = list(itertools.combinations(range(5), 2))
    selected_mean = torch.stack([workspace_set_reward(ranks, subset) for subset in sets])
    contrast = torch.stack(
        [workspace_set_reward(ranks, subset, contrastive=True) for subset in sets]
    )

    # 0.5 * (1 + n/(n-k)*selected_mean - total/(n-k)).
    expected = 0.5 * (1.0 + 5.0 / 3.0 * selected_mean - ranks.sum() / 3.0)
    torch.testing.assert_close(contrast, expected)
    assert torch.equal(selected_mean.argsort(), contrast.argsort())
    with pytest.raises(ValueError):
        workspace_set_reward(ranks, range(5), contrastive=True)


def test_hybrid_reward_has_clean_endpoint_semantics():
    qa = torch.tensor([0.0, 1.0])
    workspace = torch.tensor([0.8, 0.2])
    torch.testing.assert_close(
        hybrid_reward(qa, workspace, lambda_qa=1.0, lambda_workspace=0.0), qa
    )
    torch.testing.assert_close(
        hybrid_reward(qa, workspace, lambda_qa=0.0, lambda_workspace=1.0), workspace
    )
    torch.testing.assert_close(
        hybrid_reward(qa, workspace, lambda_qa=1.0, lambda_workspace=0.5),
        (qa + 0.5 * workspace) / 1.5,
    )
    with pytest.raises(ValueError):
        hybrid_reward(qa, workspace, lambda_qa=0.0, lambda_workspace=0.0)


def test_advantage_modes_normalize_only_within_each_group():
    rewards = torch.tensor([[-1.0, 1.0, -1.0, 1.0], [3.0, 3.0, 3.0, 3.0]])
    centered = normalize_advantages(rewards, mode="center")
    zscored = normalize_advantages(rewards, mode="zscore")
    torch.testing.assert_close(centered[0], rewards[0])
    torch.testing.assert_close(centered[1], torch.zeros(4))
    torch.testing.assert_close(zscored[0], rewards[0])
    torch.testing.assert_close(zscored[1], torch.zeros(4))
    torch.testing.assert_close(zscored.mean(dim=-1), torch.zeros(2))
    assert not zscored.requires_grad


def test_auto_advantage_does_not_amplify_near_flat_groups():
    rewards = torch.tensor([[1.0, 1.001, 0.999, 1.0]], requires_grad=True)
    auto = normalize_advantages(rewards, mode="auto", min_std=0.1)
    centered = normalize_advantages(rewards, mode="center", min_std=0.1)
    torch.testing.assert_close(auto, centered)


def test_groupwise_constants_cancel_so_kl_must_be_a_separate_loss():
    rewards = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    exact_prompt_kl = 12.5
    without_kl = normalize_advantages(rewards, mode="zscore")
    incorrectly_folded_kl = normalize_advantages(rewards - exact_prompt_kl, mode="zscore")
    torch.testing.assert_close(without_kl, incorrectly_folded_kl)


def test_categorical_kl_matches_manual_value_and_has_correct_direction():
    policy = torch.tensor([[0.8, 0.2]], dtype=torch.float64).log()
    reference = torch.tensor([[0.5, 0.5]], dtype=torch.float64).log()
    expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
    torch.testing.assert_close(
        categorical_kl(policy, reference), torch.tensor([expected], dtype=torch.float64)
    )
    torch.testing.assert_close(categorical_kl(policy, policy), torch.zeros(1, dtype=torch.float64))


def test_gumbel_topk_returns_reproducible_canonical_exact_budget_sets():
    scores = torch.tensor([0.2, -0.3, 1.1, 0.0])
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)
    sample_a = sample_gumbel_topk(scores, 2, num_samples=256, generator=generator_a)
    sample_b = sample_gumbel_topk(scores, 2, num_samples=256, generator=generator_b)
    assert torch.equal(sample_a, sample_b)
    assert sample_a.shape == (256, 2)
    assert bool((sample_a[:, 0] < sample_a[:, 1]).all())
    assert int(sample_a.min()) >= 0 and int(sample_a.max()) < scores.numel()
    with pytest.raises(ValueError):
        sample_gumbel_topk(scores, 0)
    with pytest.raises(ValueError):
        sample_gumbel_topk(scores, 5)


def test_set_logprob_matches_manual_plackett_luce_sum_and_ignores_order():
    scores = torch.tensor([0.3, -0.4, 1.0], dtype=torch.float64, requires_grad=True)
    actual = set_logprob(scores, [0, 2])
    reversed_input = set_logprob(scores, [2, 0])
    torch.testing.assert_close(actual, reversed_input)

    weights = scores.detach().exp()
    total = weights.sum()
    expected_probability = (
        weights[0] / total * weights[2] / (total - weights[0])
        + weights[2] / total * weights[0] / (total - weights[2])
    )
    torch.testing.assert_close(actual.exp(), expected_probability)
    actual.backward()
    assert bool(torch.isfinite(scores.grad).all())
    # Adding a constant to every score cannot alter the set distribution.
    torch.testing.assert_close(set_logprob(scores.detach() + 19.0, [0, 2]), actual.detach())


@pytest.mark.parametrize("k", [1, 2, 3])
def test_all_unordered_set_probabilities_sum_to_one(k):
    scores = torch.tensor([0.2, -0.1, 0.8], dtype=torch.float64)
    probabilities = torch.stack(
        [set_logprob(scores, subset).exp() for subset in itertools.combinations(range(3), k)]
    )
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_gumbel_topk_empirical_set_distribution_matches_exact_logprob():
    scores = torch.tensor([0.3, -0.4, 1.0], dtype=torch.float64)
    subsets = list(itertools.combinations(range(3), 2))
    expected = {subset: float(set_logprob(scores, subset).exp()) for subset in subsets}
    draws = sample_gumbel_topk(
        scores, 2, num_samples=30_000, generator=torch.Generator().manual_seed(9)
    )
    observed = {subset: 0 for subset in subsets}
    for row in draws.tolist():
        observed[tuple(row)] += 1
    for subset in subsets:
        frequency = observed[subset] / len(draws)
        assert abs(frequency - expected[subset]) < 0.015


def test_exact_set_kl_is_zero_at_reference_positive_otherwise_and_differentiable():
    reference = torch.tensor([0.0, 0.2, -0.5, 0.7], dtype=torch.float64)
    policy = torch.tensor([1.0, -0.4, 0.1, 0.0], dtype=torch.float64, requires_grad=True)
    same = exact_set_kl(reference, reference, 2)
    different = exact_set_kl(policy, reference, 2)
    torch.testing.assert_close(same, torch.tensor(0.0, dtype=torch.float64), atol=1e-12, rtol=0)
    assert float(different) > 0
    different.backward()
    assert bool(torch.isfinite(policy.grad).all())

    # With k=n, both policies select the sole possible set with probability one.
    torch.testing.assert_close(
        exact_set_kl(policy.detach(), reference, 4),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_exact_set_kl_matches_bruteforce_distribution_kl():
    policy = torch.tensor([0.4, -0.2, 0.9], dtype=torch.float64)
    reference = torch.tensor([-0.1, 0.5, 0.2], dtype=torch.float64)
    subsets = list(itertools.combinations(range(3), 2))
    policy_lp = torch.stack([set_logprob(policy, subset) for subset in subsets])
    reference_lp = torch.stack([set_logprob(reference, subset) for subset in subsets])
    expected = (policy_lp.exp() * (policy_lp - reference_lp)).sum()
    torch.testing.assert_close(exact_set_kl(policy, reference, 2), expected)


def test_exact_set_entropy_matches_bruteforce_and_normalizes_uniform_policy():
    scores = torch.tensor([0.4, -0.2, 0.9], dtype=torch.float64)
    subsets = list(itertools.combinations(range(3), 2))
    log_probs = torch.stack([set_logprob(scores, subset) for subset in subsets])
    expected = -(log_probs.exp() * log_probs).sum()
    torch.testing.assert_close(exact_set_entropy(scores, 2), expected)

    uniform = torch.zeros(4, dtype=torch.float64)
    torch.testing.assert_close(
        exact_set_entropy(uniform, 2, normalize=True),
        torch.tensor(1.0, dtype=torch.float64),
    )


def test_grpo_clipped_loss_has_expected_unclipped_gradient_and_detaches_advantage():
    current = torch.tensor([-0.7, -0.7], requires_grad=True)
    old = current.detach().clone()
    advantages = torch.tensor([1.0, -1.0], requires_grad=True)
    loss = grpo_clipped_loss(current, old, advantages)
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([-0.5, 0.5]))
    assert advantages.grad is None


def test_grpo_clipping_stops_updates_past_the_bad_side_of_clip_range():
    current = torch.tensor([math.log(2.0), math.log(0.5)], requires_grad=True)
    old = torch.zeros(2)
    advantages = torch.tensor([1.0, -1.0])
    loss = grpo_clipped_loss(current, old, advantages, clip_epsilon=0.2, reduction="sum")
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros(2))


def test_grpo_adds_kl_outside_reward_advantage_and_kl_supplies_gradient():
    current = torch.zeros(4, requires_grad=True)
    old = torch.zeros(4)
    flat_advantages = torch.zeros(4)
    policy_logits = torch.tensor([1.0, -1.0], requires_grad=True)
    reference_logits = torch.tensor([0.0, 0.0])
    kl = categorical_kl(policy_logits, reference_logits)
    loss = grpo_clipped_loss(
        current, old, flat_advantages, kl=kl, beta=0.3, reduction="mean"
    )
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros(4))
    assert bool(torch.isfinite(policy_logits.grad).all())
    assert float(policy_logits.grad.abs().sum()) > 0
    with pytest.raises(ValueError):
        grpo_clipped_loss(current.detach(), old, flat_advantages, beta=0.1)


def test_set_policy_gradient_smoke_test_is_finite_and_favors_rewarded_set():
    scores = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    selected = [0, 1]
    current_logprob = set_logprob(scores, selected).reshape(1)
    old_logprob = current_logprob.detach().clone()
    loss = grpo_clipped_loss(current_logprob, old_logprob, torch.ones(1))
    loss.backward()
    assert bool(torch.isfinite(scores.grad).all())
    # Gradient descent increases the total score of selected candidates relative
    # to unselected candidates.
    assert float(scores.grad[:2].mean()) < float(scores.grad[2:].mean())
