#!/usr/bin/env bash
# Run exactly one memory-RL condition family at a time, serially.
#
# Usage:
#   bash scripts/run_memory_rl_mvp.sh dry-run MODE
#   bash scripts/run_memory_rl_mvp.sh smoke  MODE
#   bash scripts/run_memory_rl_mvp.sh formal MODE
#
# MODE: sft-w | rl-w | rl-qa | rl-hybrid
#
# Comma-separated SEEDS, BETAS, and BUDGETS are expanded serially. This script
# never launches the next scientific stage; a new explicit invocation is
# required after the corresponding manual gate has been reviewed.

set -Eeuo pipefail

usage() {
    sed -n '2,14p' "${BASH_SOURCE[0]}" >&2
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

PROFILE=$1
STAGE=$2

case "$PROFILE" in
    dry-run|smoke|formal) ;;
    *)
        echo "error: PROFILE must be dry-run, smoke, or formal" >&2
        usage
        exit 2
        ;;
esac

case "$STAGE" in
    sft-w|rl-w|rl-qa|rl-hybrid) ;;
    *)
        echo "error: MODE must be sft-w, rl-w, rl-qa, or rl-hybrid" >&2
        usage
        exit 2
        ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
OUT_ROOT=${OUT_ROOT:-data/results/memory_rl_runs}
SEEDS_RAW=${SEEDS:-0}
SPLIT_SEED=${SPLIT_SEED:-0}
BETAS_RAW=${BETAS:-0.03}
BUDGETS_RAW=${BUDGETS:-2}
LAMBDA_QA=${LAMBDA_QA:-1.0}
LAMBDA_W=${LAMBDA_W:-0.5}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
LORA_RANK=${LORA_RANK:-32}
TEMPERATURE=${TEMPERATURE:-0.7}
WORKSPACE_SET_REWARD=${WORKSPACE_SET_REWARD:-mean}
WORKSPACE_OBJECTIVE=${WORKSPACE_OBJECTIVE:-rank-continuous}
WORKSPACE_TOP_K=${WORKSPACE_TOP_K:-2}
ANSWER_TOKENS=${ANSWER_TOKENS:-64}
DIAGNOSTIC_EVERY=${DIAGNOSTIC_EVERY:-25}
REPORTER_CORRELATION_BOOTSTRAP_SAMPLES=${REPORTER_CORRELATION_BOOTSTRAP_SAMPLES:-4000}
REPORTER_CORRELATION_BOOTSTRAP_SEED=${REPORTER_CORRELATION_BOOTSTRAP_SEED:-0}
DTYPE=${DTYPE:-bfloat16}
DEVICE=${DEVICE:-cuda}

case "$PROFILE" in
    dry-run)
        MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
        TEACHER_TAG=${TEACHER_TAG:-7B-Instruct}
        MAX_STEPS=${MAX_STEPS:-1}
        GROUP_SIZE=${GROUP_SIZE:-4}
        EVAL_EVERY=${EVAL_EVERY:-1}
        SAVE_EVERY=${SAVE_EVERY:-0}
        LIMIT_TRAIN_EPISODES=${LIMIT_TRAIN_EPISODES:-0}
        LIMIT_VALIDATION_EPISODES=${LIMIT_VALIDATION_EPISODES:-0}
        VAL_EVAL_EPISODES=${VAL_EVAL_EPISODES:-0}
        MAX_LENGTH=${MAX_LENGTH:-512}
        ALLOW_TEACHER_MISMATCH=0
        ;;
    smoke)
        MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
        # The repository has no 3B W_ref files. Smoke reuses the immutable 7B
        # rows only to test wiring and records this explicit mismatch override.
        TEACHER_TAG=${TEACHER_TAG:-7B-Instruct}
        MAX_STEPS=${MAX_STEPS:-10}
        GROUP_SIZE=${GROUP_SIZE:-4}
        EVAL_EVERY=${EVAL_EVERY:-5}
        SAVE_EVERY=${SAVE_EVERY:-0}
        LIMIT_TRAIN_EPISODES=${LIMIT_TRAIN_EPISODES:-8}
        LIMIT_VALIDATION_EPISODES=${LIMIT_VALIDATION_EPISODES:-4}
        VAL_EVAL_EPISODES=${VAL_EVAL_EPISODES:-4}
        MAX_LENGTH=${MAX_LENGTH:-512}
        ALLOW_TEACHER_MISMATCH=1
        ;;
    formal)
        MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
        TEACHER_TAG=${TEACHER_TAG:-7B-Instruct}
        MAX_STEPS=${MAX_STEPS:-300}
        GROUP_SIZE=${GROUP_SIZE:-8}
        EVAL_EVERY=${EVAL_EVERY:-100}
        SAVE_EVERY=${SAVE_EVERY:-300}
        LIMIT_TRAIN_EPISODES=${LIMIT_TRAIN_EPISODES:-0}
        LIMIT_VALIDATION_EPISODES=${LIMIT_VALIDATION_EPISODES:-0}
        VAL_EVAL_EPISODES=${VAL_EVAL_EPISODES:-0}
        MAX_LENGTH=${MAX_LENGTH:-2048}
        ALLOW_TEACHER_MISMATCH=0
        ;;
esac

if [[ "$WORKSPACE_SET_REWARD" != "mean" && "$WORKSPACE_SET_REWARD" != "contrastive" ]]; then
    echo "error: WORKSPACE_SET_REWARD must be mean or contrastive" >&2
    exit 2
fi
if [[ "$WORKSPACE_OBJECTIVE" != "rank-continuous" && "$WORKSPACE_OBJECTIVE" != "top-k" ]]; then
    echo "error: WORKSPACE_OBJECTIVE must be rank-continuous or top-k" >&2
    exit 2
fi

SEEDS_NORMALIZED=${SEEDS_RAW//,/ }
BETAS_NORMALIZED=${BETAS_RAW//,/ }
BUDGETS_NORMALIZED=${BUDGETS_RAW//,/ }
read -r -a SEED_VALUES <<< "$SEEDS_NORMALIZED"
read -r -a BETA_VALUES <<< "$BETAS_NORMALIZED"
read -r -a BUDGET_VALUES <<< "$BUDGETS_NORMALIZED"

if [[ ${#SEED_VALUES[@]} -eq 0 || ${#BETA_VALUES[@]} -eq 0 || ${#BUDGET_VALUES[@]} -eq 0 ]]; then
    echo "error: SEEDS, BETAS, and BUDGETS must be non-empty" >&2
    exit 2
fi

for seed in "${SEED_VALUES[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "error: invalid seed: $seed" >&2
        exit 2
    fi
done
if [[ ! "$SPLIT_SEED" =~ ^[0-9]+$ ]]; then
    echo "error: invalid split seed: $SPLIT_SEED" >&2
    exit 2
fi
for beta in "${BETA_VALUES[@]}"; do
    if [[ ! "$beta" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
        echo "error: invalid beta: $beta" >&2
        exit 2
    fi
done
for budget in "${BUDGET_VALUES[@]}"; do
    if [[ ! "$budget" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: invalid budget: $budget" >&2
        exit 2
    fi
done

if [[ "$PROFILE" != "dry-run" ]]; then
    "$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [name for name in ("torch", "transformers", "peft", "numpy", "sklearn")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing required Python packages: " + ", ".join(missing))
PY
fi

mkdir -p "$OUT_ROOT"

MODEL_SLUG=${MODEL##*/}
MODEL_SLUG=${MODEL_SLUG//[^A-Za-z0-9_.-]/_}
LQ_SLUG=${LAMBDA_QA//./p}
LW_SLUG=${LAMBDA_W//./p}
WO_SLUG=${WORKSPACE_OBJECTIVE//-/_}

echo "profile=$PROFILE mode=$STAGE model=$MODEL teacher=$TEACHER_TAG"
echo "seeds=${SEED_VALUES[*]} split_seed=$SPLIT_SEED betas=${BETA_VALUES[*]} budgets=${BUDGET_VALUES[*]}"
echo "This invocation runs only '$STAGE'; it will not cross a scientific gate."

for seed in "${SEED_VALUES[@]}"; do
    for beta in "${BETA_VALUES[@]}"; do
        for budget in "${BUDGET_VALUES[@]}"; do
            BETA_SLUG=${beta//./p}
            RUN_NAME="${PROFILE}_${STAGE}_${MODEL_SLUG}_${WO_SLUG}_split${SPLIT_SEED}_s${seed}_beta${BETA_SLUG}_k${budget}_lq${LQ_SLUG}_lw${LW_SLUG}"
            RUN_DIR="$OUT_ROOT/$RUN_NAME"
            LOG_PATH="$OUT_ROOT/$RUN_NAME.log"

            if [[ -e "$RUN_DIR" || -e "$LOG_PATH" ]]; then
                echo "error: refusing to overwrite existing run artifact: $RUN_NAME" >&2
                exit 1
            fi

            COMMAND=(
                "$PYTHON_BIN" src/experiments/train_memory_rl.py
                --mode "$STAGE"
                --model "$MODEL"
                --teacher-tag "$TEACHER_TAG"
                --workspace-objective "$WORKSPACE_OBJECTIVE"
                --workspace-top-k "$WORKSPACE_TOP_K"
                --out-dir "$RUN_DIR"
                --seed "$seed"
                --split-seed "$SPLIT_SEED"
                --budget "$budget"
                --max-steps "$MAX_STEPS"
                --group-size "$GROUP_SIZE"
                --temperature "$TEMPERATURE"
                --learning-rate "$LEARNING_RATE"
                --beta "$beta"
                --lambda-qa "$LAMBDA_QA"
                --lambda-w "$LAMBDA_W"
                --workspace-set-reward "$WORKSPACE_SET_REWARD"
                --lora-rank "$LORA_RANK"
                --dtype "$DTYPE"
                --device "$DEVICE"
                --answer-tokens "$ANSWER_TOKENS"
                --diagnostic-every "$DIAGNOSTIC_EVERY"
                --reporter-correlation-bootstrap-samples "$REPORTER_CORRELATION_BOOTSTRAP_SAMPLES"
                --reporter-correlation-bootstrap-seed "$REPORTER_CORRELATION_BOOTSTRAP_SEED"
                --max-length "$MAX_LENGTH"
                --eval-every "$EVAL_EVERY"
                --save-every "$SAVE_EVERY"
                --limit-train-episodes "$LIMIT_TRAIN_EPISODES"
                --limit-validation-episodes "$LIMIT_VALIDATION_EPISODES"
                --val-eval-episodes "$VAL_EVAL_EPISODES"
            )

            if [[ -n "${WORKSPACE_TEACHER_MODEL:-}" ]]; then
                COMMAND+=(--workspace-teacher-model "$WORKSPACE_TEACHER_MODEL")
            fi
            if [[ "$ALLOW_TEACHER_MISMATCH" == 1 ]]; then
                COMMAND+=(--allow-teacher-mismatch)
            fi

            if [[ "$PROFILE" == "dry-run" ]]; then
                COMMAND+=(--dry-run)
            fi

            echo
            echo "Starting serial run: $RUN_NAME"
            printf '  %q' "${COMMAND[@]}"
            printf '\n'
            "${COMMAND[@]}" 2>&1 | tee "$LOG_PATH"
            echo "Completed: $RUN_NAME"
        done
    done
done

echo
if [[ "$PROFILE" == "dry-run" ]]; then
    echo "STOP: dry-run complete; inspect run_config.json and split_manifest.json."
else
    echo "STOP: inspect summary.json, metrics.jsonl, and rollouts.jsonl."
fi
echo "No later stage has been launched. Start it only after the manual gate passes."
