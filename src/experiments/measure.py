"""
measure.py — compute two encoding-time salience signals on an OPEN model, per
item, for every episode in the battery:

  W (workspace salience)  : latent signal. Read the residual stream at the END of
     reading the context (the workspace state after encoding) and take the peak
     logit-lens probability of the item concept across layers. High => the concept
     is held in the workspace after reading, even if never said.

  V (verbal-reflection salience) : the fair black-box baseline. Ask the SAME model
     "is <concept> important to remember for future questions?" and read
     P(yes)/(P(yes)+P(no)) from its output distribution. This is what the model
     EXPLICITLY reports as important.

Neither signal sees the probe question (encoding-time prediction). Ground-truth
utility (load_bearing vs not) comes from the battery labels.

W is read on the RAW context (no chat template) so it is comparable between base
and instruct checkpoints. V is only meaningful on instruct models.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jlens import WorkspaceLens


YES_VARIANTS = ("yes", " yes", "Yes", " Yes", "YES")
NO_VARIANTS = ("no", " no", "No", " No", "NO")


def file_sha256(path):
    """Return a streaming SHA-256 digest for an immutable artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def concept_token_id(lens, concept):
    """First token id of the concept; prefer the leading-space (mid-text) variant."""
    for variant in (" " + concept, concept):
        ids = lens.tok.encode(variant, add_special_tokens=False)
        if ids:
            return ids[0]
    return None


def concept_token_ids(lens, concept):
    """Candidate first-token ids for a concept: leading-space and CAPITALIZED
    variants. Batteries generate lowercase concepts, but proper nouns ('italy',
    'madrid') live in the vocab as ' Italy'/' Madrid'; the lowercase encoding
    starts with a junk fragment (' it', ' mad') that the workspace never ranks
    highly. Scoring max over the variants reads the token the model actually
    uses. Applied uniformly to every item, so AUC comparisons stay fair."""
    cands = []
    for variant in (" " + concept, " " + concept.capitalize(),
                    concept, concept.capitalize()):
        ids = lens.tok.encode(variant, add_special_tokens=False)
        if ids:
            cands.append(ids[0])
    return list(dict.fromkeys(cands))


@torch.no_grad()
def workspace_salience(lens, context, concept_ids, end_only=False):
    """
    Return {concept_id: (W_end, W_max, W_rr)} in one forward pass.
      W_end : peak softmax prob of the concept token at the END position, over layers
      W_max : peak softmax prob over ALL positions & layers (noisier; surface confound)
      W_rr  : peak RECIPROCAL RANK (1/rank) of the concept token at the END position,
              over layers. Scale-free, so it is not swamped by the 150k-vocab softmax
              the way raw probability is -> a far more discriminative workspace signal.
    """
    hs, ids = lens.hidden_states(context)
    last = len(ids) - 1
    layers = range(1, lens.n_layers + 1)
    end_probs = {cid: 0.0 for cid in concept_ids}
    max_probs = {cid: 0.0 for cid in concept_ids}
    rr = {cid: 0.0 for cid in concept_ids}
    for L in layers:
        logits_end = lens.logitlens(hs[L, last]).float()
        p_end = F.softmax(logits_end, dim=-1)
        for cid in concept_ids:
            end_probs[cid] = max(end_probs[cid], p_end[cid].item())
            rank = int((logits_end > logits_end[cid]).sum().item()) + 1
            rr[cid] = max(rr[cid], 1.0 / rank)
        if not end_only:
            logits_all = lens.logitlens(hs[L])        # [seq, vocab]
            p_all = F.softmax(logits_all.float(), dim=-1).max(dim=0).values
            for cid in concept_ids:
                max_probs[cid] = max(max_probs[cid], p_all[cid].item())
    return end_probs, max_probs, rr


def yes_no_ids(lens):
    yes, no = [], []
    for v in YES_VARIANTS:
        e = lens.tok.encode(v, add_special_tokens=False)
        if e:
            yes.append(e[0])
    for v in NO_VARIANTS:
        e = lens.tok.encode(v, add_special_tokens=False)
        if e:
            no.append(e[0])
    return sorted(set(yes)), sorted(set(no))


def _yes_vs_no(logits, yes_ids, no_ids):
    """P(yes) normalised against P(no), from logits, with no epsilon.

    The previous form was ``py / (py + pn + 1e-9)`` on full-vocabulary softmax
    probabilities.  For this probe the yes/no mass is often 1e-13 or smaller --
    the model spends its probability elsewhere -- so the guard epsilon dominated
    the denominator and the function silently returned ``py * 1e9``: a monotone
    function of the ABSOLUTE yes probability, not the yes-versus-no ratio it
    documented.  Working in log space removes the epsilon and is exact at any
    scale of the underlying mass.
    """
    yes = torch.logsumexp(logits[yes_ids], dim=0)
    no = torch.logsumexp(logits[no_ids], dim=0)
    return float(torch.sigmoid(yes - no))


@torch.no_grad()
def verbal_salience(lens, context, concept, yes_ids, no_ids):
    """P(yes)/(P(yes)+P(no)) to 'is <concept> important to remember?'"""
    q = (f"{context}\n\nBased only on the passage above, is the concept "
         f"\"{concept}\" one of the most important things to remember in order to "
         f"answer possible future questions? Answer with a single word: yes or no.")
    msgs = [{"role": "user", "content": q}]
    prompt = lens.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = lens.tok(prompt, return_tensors="pt").to(lens.device)
    logits = lens.model(**enc).logits[0, -1].float()
    return _yes_vs_no(logits, yes_ids, no_ids)


@torch.no_grad()
def verbal_salience_raw(lens, context, concept, yes_ids, no_ids):
    """Chat-template-free variant of V, so BASE checkpoints can be probed too.
    This is what tests (rather than assumes) that post-training installs the
    verbal-report layer: if base models show the same V(mis)calibration, the
    'installed by post-training' story is wrong."""
    q = (f"{context}\n\nQuestion: Based only on the passage above, is the concept "
         f"\"{concept}\" one of the most important things to remember in order to "
         f"answer possible future questions? Answer yes or no.\nAnswer:")
    enc = lens.tok(q, return_tensors="pt").to(lens.device)
    logits = lens.model(**enc).logits[0, -1].float()
    return _yes_vs_no(logits, yes_ids, no_ids)


def has_chat_template(lens):
    return getattr(lens.tok, "chat_template", None) is not None


def ensure_output_paths_absent(paths):
    """Reject the whole raw/metadata output set before loading a model."""
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing output: {rendered}")


def write_output_pair_exclusive(outputs):
    """Create a complete artifact pair, cleaning up only files created here.

    All potentially fallible metadata construction happens before this helper.
    Exclusive creation still protects against a writer racing the initial
    preflight check.  If either write fails, a newly created partial pair is
    removed so the immutable run can be retried.
    """
    paths = [path for path, _payload in outputs]
    ensure_output_paths_absent(paths)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, payload in outputs:
            with path.open("x", encoding="utf-8") as handle:
                created.append(path)
                handle.write(payload)
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def _token_record(tokenizer, token_id):
    try:
        vocab_token = tokenizer.convert_ids_to_tokens(token_id)
    except (AttributeError, KeyError, TypeError, ValueError):
        vocab_token = None
    try:
        decoded = tokenizer.decode([token_id])
    except (AttributeError, KeyError, TypeError, ValueError):
        decoded = None
    return {"id": int(token_id), "vocab_token": vocab_token, "decoded": decoded}


def _variant_records(tokenizer, variants):
    records = []
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        records.append(
            {
                "text": variant,
                "encoded_ids": [int(token_id) for token_id in encoded],
                "scored_first_token_id": int(encoded[0]) if encoded else None,
            }
        )
    return records


def yes_no_token_metadata(lens, yes_ids, no_ids):
    """Describe the exact next-token sets used to compute verbal salience."""
    return {
        "scoring": "sum next-token probability over unique first-token ids",
        "yes": {
            "ids": [int(token_id) for token_id in yes_ids],
            "tokens": [_token_record(lens.tok, token_id) for token_id in yes_ids],
            "variants": _variant_records(lens.tok, YES_VARIANTS),
        },
        "no": {
            "ids": [int(token_id) for token_id in no_ids],
            "tokens": [_token_record(lens.tok, token_id) for token_id in no_ids],
            "variants": _variant_records(lens.tok, NO_VARIANTS),
        },
    }


def chat_template_metadata(tokenizer):
    """Return a hash and enough provenance to identify the rendered template."""
    template = getattr(tokenizer, "chat_template", None)
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    if template is None:
        template_bytes = None
    elif isinstance(template, str):
        template_bytes = template.encode("utf-8")
    else:
        template_bytes = json.dumps(
            template,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    return {
        "present": template is not None,
        "source": "tokenizer.chat_template" if template is not None else None,
        "sha256": (
            hashlib.sha256(template_bytes).hexdigest()
            if template_bytes is not None
            else None
        ),
        "bytes": len(template_bytes) if template_bytes is not None else 0,
        "template_type": type(template).__name__ if template is not None else None,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "declared_in_tokenizer_init_kwargs": "chat_template" in init_kwargs,
    }


def _package_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata(lens, requested_device):
    """Capture software and accelerator details without invoking subprocesses."""
    cuda_available = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpus = []
    for index in range(gpu_count):
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        gpus.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    driver_path = Path("/proc/driver/nvidia/version")
    nvidia_driver = None
    if driver_path.is_file():
        try:
            nvidia_driver = driver_path.read_text(encoding="utf-8").strip()
        except OSError:
            nvidia_driver = None
    cudnn_version = None
    if getattr(torch.backends, "cudnn", None) is not None:
        cudnn_version = torch.backends.cudnn.version()
    return {
        "device_requested": requested_device,
        "device_resolved": str(lens.device),
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
            "accelerate": _package_version("accelerate"),
            "peft": _package_version("peft"),
            "cuda_runtime": torch.version.cuda,
            "cudnn": cudnn_version,
            "nvidia_driver": nvidia_driver,
        },
        "gpu": {
            "available": cuda_available,
            "count": gpu_count,
            "devices": gpus,
        },
    }


def revision_metadata(lens, args):
    model_resolved = getattr(lens, "model_revision_resolved", None)
    tokenizer_resolved = getattr(lens, "tokenizer_revision_resolved", None)
    tokenizer_effective = getattr(
        lens,
        "tokenizer_revision_effective",
        args.tokenizer_revision or args.model_revision,
    )
    return {
        "model": {
            "identifier": args.model,
            "revision_requested": args.model_revision,
            "revision_resolved": model_resolved,
        },
        "tokenizer": {
            "identifier": getattr(lens.tok, "name_or_path", args.model),
            "revision_requested": args.tokenizer_revision,
            "revision_effective": tokenizer_effective,
            "revision_resolved": tokenizer_resolved,
        },
    }


def count_metadata(full_battery, effective_battery, rows):
    full_candidates = sum(len(episode.get("items", [])) for episode in full_battery)
    effective_items = [
        item for episode in effective_battery for item in episode.get("items", [])
    ]
    input_labels = Counter(str(item.get("label")) for item in effective_items)
    output_labels = Counter(str(row.get("label")) for row in rows)
    return {
        "episodes_in_battery": len(full_battery),
        "episodes_evaluated": len(effective_battery),
        "episodes_with_output": len({row["episode"] for row in rows}),
        "candidates_in_battery": full_candidates,
        "candidates_evaluated": len(effective_items),
        "candidate_rows_written": len(rows),
        "candidates_skipped_no_token": len(effective_items) - len(rows),
        "input_labels": dict(sorted(input_labels.items())),
        "output_labels": dict(sorted(output_labels.items())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument(
        "--model-revision",
        default=None,
        help="optional immutable Hugging Face model revision (prefer a commit SHA)",
    )
    ap.add_argument(
        "--tokenizer-revision",
        default=None,
        help=(
            "optional immutable tokenizer revision; defaults to --model-revision "
            "when that pin is supplied"
        ),
    )
    ap.add_argument("--adapter", default=None,
                    help="optional LoRA adapter; keeps the base checkpoint explicit")
    ap.add_argument("--battery", default="data/benchmarks/battery.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-verbal", action="store_true", help="skip chat-template V (e.g. base models)")
    ap.add_argument("--no-verbal-raw", action="store_true", help="skip template-free V_raw")
    ap.add_argument("--device", default=None, help="cuda / cpu / auto (multi-GPU sharding)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                    help="use bfloat16 for 7B+ so it fits in unified memory")
    ap.add_argument("--limit-episodes", type=int, default=0,
                    help="0 = full battery; positive values are for smoke tests")
    ap.add_argument("--end-only", action="store_true",
                    help="skip W_max over all positions; W_end/W_rr are unchanged")
    args = ap.parse_args()

    if args.adapter and not args.out:
        ap.error("--adapter requires an explicit --out to protect base result files")
    if args.limit_episodes < 0:
        ap.error("--limit-episodes must be non-negative")

    out_path = Path(
        args.out or f"data/results/results_{args.model.split('/')[-1]}.json"
    )
    metadata_path = Path(f"{out_path}.metadata")
    try:
        ensure_output_paths_absent([out_path, metadata_path])
    except FileExistsError as exc:
        ap.error(str(exc))

    battery_path = Path(args.battery)
    battery_file_sha256 = file_sha256(battery_path)
    with battery_path.open(encoding="utf-8") as f:
        full_battery = json.load(f)
    battery = full_battery
    if args.limit_episodes:
        battery = battery[:args.limit_episodes]
    lens = WorkspaceLens(
        args.model,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        adapter_path=args.adapter,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
    )
    do_verbal = (not args.no_verbal) and has_chat_template(lens)
    do_verbal_raw = not args.no_verbal_raw
    yes_ids, no_ids = yes_no_ids(lens)
    print(f"model={args.model} adapter={args.adapter} layers={lens.n_layers} verbal={do_verbal} "
          f"verbal_raw={do_verbal_raw} device={lens.device} episodes={len(battery)}")

    rows = []
    for ei, ep in enumerate(battery):
        ctx = ep["context"]
        items = ep["items"]
        cids = {}
        for it in items:
            cand = concept_token_ids(lens, it["concept"])
            if cand:
                cids[it["concept"]] = cand
        all_ids = sorted({i for v in cids.values() for i in v})
        end_probs, max_probs, rr = workspace_salience(
            lens, ctx, all_ids, end_only=args.end_only
        )
        for candidate_index, it in enumerate(items):
            c = it["concept"]
            if c not in cids:
                continue
            cand = cids[c]
            row = {"episode": ei, "candidate_index": candidate_index,
                   "concept": c, "label": it["label"],
                   "W_end": max(end_probs[i] for i in cand),
                   "W_max": max(max_probs[i] for i in cand),
                   "W_rr": max(rr[i] for i in cand)}
            if do_verbal:
                row["V"] = verbal_salience(lens, ctx, c, yes_ids, no_ids)
            if do_verbal_raw:
                row["V_raw"] = verbal_salience_raw(lens, ctx, c, yes_ids, no_ids)
            rows.append(row)
        if (ei + 1) % 5 == 0:
            print(f"  {ei+1}/{len(battery)} episodes")

    raw_payload = json.dumps(rows, indent=2) + "\n"
    raw_output_sha256 = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
    revisions = revision_metadata(lens, args)
    chat_template = chat_template_metadata(lens.tok)
    counts = count_metadata(full_battery, battery, rows)
    candidate_order = [
        [episode_index, candidate_index, item.get("concept"), item.get("label")]
        for episode_index, episode in enumerate(battery)
        for candidate_index, item in enumerate(episode.get("items", []))
    ]
    metadata = {
        "schema_version": "workspace_measurement_metadata.v3",
        "verbal_score_definition": (
            "sigmoid(logsumexp(logits[yes_ids]) - logsumexp(logits[no_ids])); "
            "exact yes-vs-no ratio, no guard epsilon. v2 artifacts used "
            "py/(py+pn+1e-9) on full-vocabulary probabilities, which the guard "
            "epsilon dominated whenever py+pn << 1e-9, and are NOT comparable."
        ),
        "model": args.model,
        "adapter": args.adapter,
        "model_revision": revisions["model"]["revision_resolved"]
        or revisions["model"]["revision_requested"],
        "model_revision_requested": revisions["model"]["revision_requested"],
        "model_revision_resolved": revisions["model"]["revision_resolved"],
        "tokenizer_revision": revisions["tokenizer"]["revision_resolved"]
        or revisions["tokenizer"]["revision_effective"],
        "tokenizer_revision_requested": revisions["tokenizer"]["revision_requested"],
        "tokenizer_revision_effective": revisions["tokenizer"]["revision_effective"],
        "tokenizer_revision_resolved": revisions["tokenizer"]["revision_resolved"],
        "revisions": revisions,
        "battery": str(battery_path.resolve()),
        "output": str(out_path.resolve()),
        "dtype": args.dtype,
        "device": str(lens.device),
        "rows": len(rows),
        "episodes": len(battery),
        "candidates": counts["candidates_evaluated"],
        "counts": counts,
        "end_only": bool(args.end_only),
        "limit_episodes": args.limit_episodes,
        "verbal_enabled": do_verbal,
        "verbal_raw_enabled": do_verbal_raw,
        "policy_input_includes_probe": False,
        "workspace_readout": {
            "position": "final context token",
            "layer_aggregation": "maximum over transformer layers 1..n",
            "candidate_token_variants": [
                "space+lowercase",
                "space+capitalized",
                "lowercase",
                "capitalized",
            ],
            "final_norm": type(lens.final_norm).__name__,
            "unembedding": type(lens.unembed).__name__,
            "n_layers": int(lens.n_layers),
        },
        "yes_no_token_sets": yes_no_token_metadata(lens, yes_ids, no_ids),
        "yes_token_ids": [int(token_id) for token_id in yes_ids],
        "no_token_ids": [int(token_id) for token_id in no_ids],
        "chat_template": chat_template,
        "chat_template_sha256": chat_template["sha256"],
        "runtime": runtime_metadata(lens, args.device),
        "hashes": {
            "battery_file_sha256": battery_file_sha256,
            "effective_battery_canonical_sha256": canonical_json_sha256(battery),
            "candidate_order_sha256": canonical_json_sha256(candidate_order),
            "raw_output_sha256": raw_output_sha256,
            "measure_source_sha256": file_sha256(Path(__file__).resolve()),
            "workspace_lens_source_sha256": file_sha256(
                Path(__file__).resolve().parents[1] / "jlens.py"
            ),
        },
    }
    # Deliberately avoid a trailing .json so existing result-globbing scripts do
    # not mistake provenance metadata for per-item measurement rows.
    metadata_payload = json.dumps(metadata, indent=2) + "\n"
    write_output_pair_exclusive(
        [(out_path, raw_payload), (metadata_path, metadata_payload)]
    )
    print(f"saved {len(rows)} item rows -> {out_path}")


if __name__ == "__main__":
    main()
