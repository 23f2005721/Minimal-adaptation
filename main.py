from fastapi import FastAPI
from fastapi.responses import JSONResponse

import math
import re


app = FastAPI()

SAFE_INT_MAX = 9007199254740991

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

INTERVENTION_ORDER = [
    "prompt_only",
    "retrieval",
    "lora",
    "qlora",
]

VALID_ROLES = {
    "system",
    "user",
    "assistant",
}


def utf8_key(value):
    return value.encode("utf-8")


def unique_sorted(codes):
    return sorted(
        set(codes),
        key=utf8_key
    )


def is_safe_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def is_positive_safe_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= SAFE_INT_MAX
    )


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def nonnegative_finite(value):
    return finite(value) and float(value) >= 0


def valid_digest40(value):
    return (
        isinstance(value, str)
        and HEX40.fullmatch(value) is not None
    )


def valid_digest64(value):
    return (
        isinstance(value, str)
        and HEX64.fullmatch(value) is not None
    )


# ============================================================
# CHOOSE
# ============================================================

def choose(payload):

    required = {
        "operation",
        "policy",
        "candidates",
    }

    if not required.issubset(payload.keys()):
        return None, 400

    if payload.get("operation") != "choose":
        return None, 400

    policy = payload.get("policy")
    candidates = payload.get("candidates")

    if not isinstance(policy, dict):
        return None, 400

    if not isinstance(candidates, list):
        return None, 400

    policy_required = {
        "minQuality",
        "freshnessRequired",
        "maxLatencyMs",
        "maxMemoryMb",
        "maxLabeledExamples",
        "maxTotalCost",
        "horizonRequests",
    }

    if not policy_required.issubset(policy.keys()):
        return None, 400

    # --------------------------------------------------------
    # Validate policy
    # --------------------------------------------------------

    if not finite(policy["minQuality"]):
        return None, 400

    if not 0 <= float(policy["minQuality"]) <= 1:
        return None, 400

    if not isinstance(
        policy["freshnessRequired"],
        bool
    ):
        return None, 400

    if not nonnegative_finite(
        policy["maxLatencyMs"]
    ):
        return None, 400

    if not nonnegative_finite(
        policy["maxMemoryMb"]
    ):
        return None, 400

    if not is_safe_int(
        policy["maxLabeledExamples"]
    ):
        return None, 400

    if not nonnegative_finite(
        policy["maxTotalCost"]
    ):
        return None, 400

    if not is_safe_int(
        policy["horizonRequests"]
    ):
        return None, 400

    # Exactly one candidate per intervention.
    if len(candidates) != 4:
        result = {
            "selected": None,
            "eligible": [],
            "totalCosts": {
                name: None
                for name in INTERVENTION_ORDER
            },
            "reasonCodes": {
                name: ["INVALID_INPUT"]
                for name in INTERVENTION_ORDER
            },
        }
        return result, 200

    by_name = {}

    # --------------------------------------------------------
    # Validate candidate list
    # --------------------------------------------------------

    for candidate in candidates:

        if not isinstance(candidate, dict):
            result = {
                "selected": None,
                "eligible": [],
                "totalCosts": {
                    name: None
                    for name in INTERVENTION_ORDER
                },
                "reasonCodes": {
                    name: ["INVALID_INPUT"]
                    for name in INTERVENTION_ORDER
                },
            }
            return result, 200

        required_candidate = {
            "name",
            "available",
            "quality",
            "freshness",
            "latencyMs",
            "memoryMb",
            "labeledExamples",
            "oneTimeCost",
            "recurringCost",
        }

        if not required_candidate.issubset(
            candidate.keys()
        ):
            result = {
                "selected": None,
                "eligible": [],
                "totalCosts": {
                    name: None
                    for name in INTERVENTION_ORDER
                },
                "reasonCodes": {
                    name: ["INVALID_INPUT"]
                    for name in INTERVENTION_ORDER
                },
            }
            return result, 200

        name = candidate["name"]

        if (
            not isinstance(name, str)
            or name not in INTERVENTION_ORDER
            or name in by_name
        ):
            result = {
                "selected": None,
                "eligible": [],
                "totalCosts": {
                    name: None
                    for name in INTERVENTION_ORDER
                },
                "reasonCodes": {
                    name: ["INVALID_INPUT"]
                    for name in INTERVENTION_ORDER
                },
            }
            return result, 200

        if not isinstance(
            candidate["available"],
            bool
        ):
            result = {
                "selected": None,
                "eligible": [],
                "totalCosts": {
                    name: None
                    for name in INTERVENTION_ORDER
                },
                "reasonCodes": {
                    name: ["INVALID_INPUT"]
                    for name in INTERVENTION_ORDER
                },
            }
            return result, 200

        if not finite(candidate["quality"]):
            result = {
                "selected": None,
                "eligible": [],
                "totalCosts": {
                    name: None
                    for name in INTERVENTION_ORDER
                },
                "reasonCodes": {
                    name: ["INVALID_INPUT"]
                    for name in INTERVENTION_ORDER
                },
            }
            return result, 200

        if not 0 <= float(candidate["quality"]) <= 1:
            result = {
                "selected": None,
                "eligible": [],
                "totalCosts": {
                    name: None
                    for name in INTERVENTION_ORDER
                },
                "reasonCodes": {
                    name: ["INVALID_INPUT"]
                    for name in INTERVENTION_ORDER
                },
            }
            return result, 200

        if not isinstance(
            candidate["freshness"],
            bool
        ):
            result = {
                "selected": None,
                "eligible": [],
                "totalCosts": {
                    name: None
                    for name in INTERVENTION_ORDER
                },
                "reasonCodes": {
                    name: ["INVALID_INPUT"]
                    for name in INTERVENTION_ORDER
                },
            }
            return result, 200

        if not nonnegative_finite(
            candidate["latencyMs"]
        ):
            result = {
                "selected": None,
                "eligible": [],
                "totalCosts": {
                    name: None
                    for name in INTERVENTION_ORDER
                },
                "reasonCodes": {
                    name: ["INVALID_INPUT"]
                    for name in INTERVENTION_ORDER
                },
            }
            return result, 200

        if not nonnegative_finite(
            candidate["memoryMb"]
        ):
            result = {
                "selected": None,
                "eligible": [],
                "totalCosts": {
                    name: None
                    for name in INTERVENTION_ORDER
                },
                "reasonCodes": {
                    name: ["INVALID_INPUT"]
                    for name in INTERVENTION_ORDER
                },
            }
            return result, 200

        if not is_safe_int(
            candidate["labeledExamples"]
        ):
            result = {
                "selected": None,
                "eligible": [],
                "totalCosts": {
                    name: None
                    for name in INTERVENTION_ORDER
                },
                "reasonCodes": {
                    name: ["INVALID_INPUT"]
                    for name in INTERVENTION_ORDER
                },
            }
            return result, 200

        if not nonnegative_finite(
            candidate["oneTimeCost"]
        ):
            result = {
                "selected": None,
                "eligible": [],
                "totalCosts": {
                    name: None
                    for name in INTERVENTION_ORDER
                },
                "reasonCodes": {
                    name: ["INVALID_INPUT"]
                    for name in INTERVENTION_ORDER
                },
            }
            return result, 200

        if not nonnegative_finite(
            candidate["recurringCost"]
        ):
            result = {
                "selected": None,
                "eligible": [],
                "totalCosts": {
                    name: None
                    for name in INTERVENTION_ORDER
                },
                "reasonCodes": {
                    name: ["INVALID_INPUT"]
                    for name in INTERVENTION_ORDER
                },
            }
            return result, 200

        by_name[name] = candidate

    # --------------------------------------------------------
    # Make sure exactly one of EACH four interventions exists.
    # --------------------------------------------------------

    if set(by_name.keys()) != set(
        INTERVENTION_ORDER
    ):
        result = {
            "selected": None,
            "eligible": [],
            "totalCosts": {
                name: None
                for name in INTERVENTION_ORDER
            },
            "reasonCodes": {
                name: ["INVALID_INPUT"]
                for name in INTERVENTION_ORDER
            },
        }

        return result, 200

    total_costs = {}
    reason_codes = {}
    eligible = []

    # --------------------------------------------------------
    # Evaluate in published priority order.
    # --------------------------------------------------------

    for name in INTERVENTION_ORDER:

        candidate = by_name[name]

        codes = []

        total_cost = round(
            float(candidate["oneTimeCost"])
            + (
                int(policy["horizonRequests"])
                * float(candidate["recurringCost"])
            ),
            12
        )

        total_costs[name] = total_cost

        # Availability
        if not candidate["available"]:
            codes.append("UNAVAILABLE")

        # Quality
        if float(candidate["quality"]) < float(
            policy["minQuality"]
        ):
            codes.append("QUALITY_FLOOR")

        # Freshness
        if (
            policy["freshnessRequired"]
            and candidate["freshness"] is not True
        ):
            codes.append("FRESHNESS_REQUIRED")

        # Latency
        if float(candidate["latencyMs"]) > float(
            policy["maxLatencyMs"]
        ):
            codes.append("LATENCY_LIMIT")

        # Memory
        if float(candidate["memoryMb"]) > float(
            policy["maxMemoryMb"]
        ):
            codes.append("MEMORY_LIMIT")

        # Labeled data
        if (
            candidate["labeledExamples"]
            > policy["maxLabeledExamples"]
        ):
            codes.append("DATA_LIMIT")

        # Total cost
        if total_cost > float(
            policy["maxTotalCost"]
        ):
            codes.append("COST_LIMIT")

        codes = unique_sorted(codes)

        reason_codes[name] = codes

        if len(codes) == 0:
            eligible.append(name)

    selected = (
        eligible[0]
        if eligible
        else None
    )

    return {
        "selected": selected,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reason_codes,
    }, 200


# ============================================================
# REPAIR
# ============================================================

def repair(payload):

    required = {
        "operation",
        "tokens",
        "templateApplications",
        "parameters",
        "allowedTargets",
        "inferenceMode",
        "trainRowIds",
        "evalRowIds",
        "dropoutActiveDuringEval",
        "artifactFiles",
        "baseRevision",
        "datasetDigest",
        "codeDigest",
        "configDigest",
        "expectedDigests",
        "microBatch",
        "gradientAccumulation",
        "replicas",
        "expectedEffectiveBatch",
        "checkpoint",
        "uninterruptedWeights",
        "resumedWeights",
        "resumeTolerance",
    }

    if not required.issubset(payload.keys()):
        return None, 400

    if payload.get("operation") != "repair":
        return None, 400

    reason_codes = set()

    # ========================================================
    # TOKENS
    # ========================================================

    tokens = payload["tokens"]

    labels = []

    tokens_valid = (
        isinstance(tokens, list)
        and len(tokens) > 0
    )

    if tokens_valid:

        for token in tokens:

            if not isinstance(token, dict):
                reason_codes.add(
                    "INVALID_TOKEN"
                )
                tokens_valid = False
                break

            token_required = {
                "id",
                "role",
                "padding",
                "text",
            }

            if not token_required.issubset(
                token.keys()
            ):
                reason_codes.add(
                    "INVALID_TOKEN"
                )
                tokens_valid = False
                break

            if not is_safe_int(
                token["id"]
            ):
                reason_codes.add(
                    "INVALID_TOKEN"
                )
                tokens_valid = False
                break

            if token["role"] not in VALID_ROLES:
                reason_codes.add(
                    "INVALID_TOKEN"
                )
                tokens_valid = False
                break

            if not isinstance(
                token["padding"],
                bool
            ):
                reason_codes.add(
                    "INVALID_TOKEN"
                )
                tokens_valid = False
                break

            if not isinstance(
                token["text"],
                str
            ):
                reason_codes.add(
                    "INVALID_TOKEN"
                )
                tokens_valid = False
                break

    if tokens_valid:

        for token in tokens:

            if (
                token["role"] == "assistant"
                and token["padding"] is False
            ):
                labels.append(
                    token["id"]
                )
            else:
                labels.append(-100)

    else:
        labels = [-100] * (
            len(tokens)
            if isinstance(tokens, list)
            else 0
        )

    # ========================================================
    # CHAT TEMPLATE
    # ========================================================

    template_applications = (
        payload["templateApplications"]
    )

    template_pass = (
        template_applications == 1
    )

    if not template_pass:
        reason_codes.add(
            "CHAT_TEMPLATE_COUNT"
        )

    # ========================================================
    # PARAMETERS / PEFT
    # ========================================================

    parameters = payload["parameters"]
    allowed_targets = payload["allowedTargets"]

    peft_valid = True
    trainable_names = []
    trainable_count = 0

    if not isinstance(
        parameters,
        list
    ):
        peft_valid = False

    if not isinstance(
        allowed_targets,
        list
    ):
        peft_valid = False

    if (
        not isinstance(
            allowed_targets,
            list
        )
        or len(allowed_targets) == 0
    ):
        peft_valid = False
    else:
        if any(
            not isinstance(x, str)
            or x == ""
            for x in allowed_targets
        ):
            peft_valid = False

        if len(set(allowed_targets)) != len(
            allowed_targets
        ):
            peft_valid = False

    parameter_names = set()
    has_lora = False

    if isinstance(parameters, list):

        for parameter in parameters:

            if not isinstance(
                parameter,
                dict
            ):
                peft_valid = False
                continue

            required_parameter = {
                "name",
                "target",
                "numel",
            }

            if not required_parameter.issubset(
                parameter.keys()
            ):
                peft_valid = False
                continue

            name = parameter["name"]
            target = parameter["target"]
            numel = parameter["numel"]

            if not isinstance(name, str):
                peft_valid = False
                continue

            if name in parameter_names:
                peft_valid = False

            parameter_names.add(name)

            if not isinstance(
                target,
                str
            ):
                peft_valid = False
                continue

            if not is_positive_safe_int(
                numel
            ):
                peft_valid = False
                continue

            is_lora_parameter = (
                target in allowed_targets
                and (
                    name.endswith(
                        ".lora_A.weight"
                    )
                    or name.endswith(
                        ".lora_B.weight"
                    )
                )
            )

            if is_lora_parameter:
                has_lora = True
                trainable_names.append(name)

                # Avoid overflow.
                if (
                    trainable_count
                    > SAFE_INT_MAX - numel
                ):
                    peft_valid = False
                else:
                    trainable_count += numel

    if not has_lora:
        peft_valid = False

    if not peft_valid:
        reason_codes.add(
            "INVALID_PARAMETER"
        )

    trainable_names.sort(
        key=utf8_key
    )

    # ========================================================
    # INFERENCE MODE
    # ========================================================

    inference_mode = payload["inferenceMode"]

    if inference_mode is not False:
        reason_codes.add(
            "INFERENCE_MODE"
        )

    # ========================================================
    # FULL MODEL ARTIFACT / ADAPTER FILES
    # ========================================================

    artifact_files = payload["artifactFiles"]

    adapter_files = []

    artifact_pass = True

    if not isinstance(
        artifact_files,
        list
    ):
        artifact_pass = False
    else:
        expected_files = [
            "adapter_config.json",
            "adapter_model.safetensors",
        ]

        if len(artifact_files) != 2:
            artifact_pass = False

        if sorted(
            artifact_files,
            key=utf8_key
        ) != sorted(
            expected_files,
            key=utf8_key
        ):
            artifact_pass = False

        if len(set(artifact_files)) != 2:
            artifact_pass = False

        if artifact_pass:
            adapter_files = sorted(
                artifact_files,
                key=utf8_key
            )

    if not artifact_pass:
        reason_codes.add(
            "ADAPTER_FILE_SET"
        )

    # ========================================================
    # OPTIONAL FULL MODEL DETECTION
    # ========================================================

    # A full model artifact is not allowed.
    # The expected files must be exactly the adapter pair.

    if isinstance(
        artifact_files,
        list
    ):
        full_model_names = {
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.safetensors",
            "model.bin",
        }

        if any(
            isinstance(f, str)
            and f in full_model_names
            for f in artifact_files
        ):
            reason_codes.add(
                "FULL_MODEL_ARTIFACT"
            )

    # ========================================================
    # TRAIN / EVAL DATA ISOLATION
    # ========================================================

    train_ids = payload["trainRowIds"]
    eval_ids = payload["evalRowIds"]

    eval_isolated = True

    if not isinstance(
        train_ids,
        list
    ) or not isinstance(
        eval_ids,
        list
    ):
        eval_isolated = False

    else:

        if len(train_ids) == 0:
            eval_isolated = False

        if len(eval_ids) == 0:
            eval_isolated = False

        for value in train_ids + eval_ids:
            if (
                not isinstance(value, str)
                or value == ""
            ):
                eval_isolated = False

        if len(set(train_ids)) != len(
            train_ids
        ):
            eval_isolated = False

        if len(set(eval_ids)) != len(
            eval_ids
        ):
            eval_isolated = False

        if set(train_ids) & set(eval_ids):
            eval_isolated = False

    if not eval_isolated:
        reason_codes.add(
            "EVAL_LEAKAGE"
        )

    # ========================================================
    # DROPOUT
    # ========================================================

    dropout_active = payload[
        "dropoutActiveDuringEval"
    ]

    eval_deterministic = (
        isinstance(dropout_active, bool)
        and dropout_active is False
    )

    if not eval_deterministic:
        reason_codes.add(
            "EVAL_DROPOUT_ACTIVE"
        )

    # ========================================================
    # BASE + LINEAGE
    # ========================================================

    base_revision = payload[
        "baseRevision"
    ]

    dataset_digest = payload[
        "datasetDigest"
    ]

    code_digest = payload[
        "codeDigest"
    ]

    config_digest = payload[
        "configDigest"
    ]

    expected_digests = payload[
        "expectedDigests"
    ]

    lineage_pass = True

    if not valid_digest40(
        base_revision
    ):
        lineage_pass = False
        reason_codes.add(
            "MUTABLE_BASE_REVISION"
        )

    if not valid_digest64(
        dataset_digest
    ):
        lineage_pass = False
        reason_codes.add(
            "LINEAGE_MISMATCH"
        )

    if not valid_digest64(
        code_digest
    ):
        lineage_pass = False
        reason_codes.add(
            "LINEAGE_MISMATCH"
        )

    if not valid_digest64(
        config_digest
    ):
        lineage_pass = False
        reason_codes.add(
            "LINEAGE_MISMATCH"
        )

    if not isinstance(
        expected_digests,
        dict
    ):
        lineage_pass = False
        reason_codes.add(
            "LINEAGE_MISMATCH"
        )
    else:

        expected_dataset = expected_digests.get(
            "datasetDigest"
        )
        expected_code = expected_digests.get(
            "codeDigest"
        )
        expected_config = expected_digests.get(
            "configDigest"
        )
        expected_base = expected_digests.get(
            "baseRevision"
        )

        if (
            expected_dataset != dataset_digest
            or expected_code != code_digest
            or expected_config != config_digest
            or expected_base != base_revision
        ):
            lineage_pass = False
            reason_codes.add(
                "LINEAGE_MISMATCH"
            )

    # ========================================================
    # BATCH
    # ========================================================

    micro_batch = payload[
        "microBatch"
    ]

    gradient_accumulation = payload[
        "gradientAccumulation"
    ]

    replicas = payload[
        "replicas"
    ]

    expected_effective_batch = payload[
        "expectedEffectiveBatch"
    ]

    batch_pass = (
        is_positive_safe_int(micro_batch)
        and is_positive_safe_int(
            gradient_accumulation
        )
        and is_positive_safe_int(
            replicas
        )
        and is_positive_safe_int(
            expected_effective_batch
    )

    if batch_pass:

        computed_batch = (
            micro_batch
            * gradient_accumulation
            * replicas
        )

        if (
            computed_batch
            != expected_effective_batch
        ):
            batch_pass = False

    if not batch_pass:
        reason_codes.add(
            "EFFECTIVE_BATCH_MISMATCH"
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint = payload[
        "checkpoint"
    ]

    checkpoint_complete = (
        isinstance(
            checkpoint,
            dict
        )
        and {
            "model",
            "optimizer",
            "scheduler",
            "step",
            "rng",
            "dataPosition",
        }.issubset(
            checkpoint.keys()
        )
    )

    if not checkpoint_complete:
        reason_codes.add(
            "INCOMPLETE_CHECKPOINT"
        )

    # ========================================================
    # RESUME
    # ========================================================

    uninterrupted = payload[
        "uninterruptedWeights"
    ]

    resumed = payload[
        "resumedWeights"
    ]

    tolerance = payload[
        "resumeTolerance"
    ]

    resume_pass = True

    if not isinstance(
        uninterrupted,
        list
    ) or not isinstance(
        resumed,
        list
    ):
        resume_pass = False

    else:

        if len(uninterrupted) == 0:
            resume_pass = False

        if len(resumed) == 0:
            resume_pass = False

        if len(uninterrupted) != len(
            resumed
        ):
            resume_pass = False

        if any(
            not finite(x)
            for x in uninterrupted
        ):
            resume_pass = False

        if any(
            not finite(x)
            for x in resumed
        ):
            resume_pass = False

    if not nonnegative_finite(
        tolerance
    ):
        resume_pass = False

    else:
        if (
            isinstance(uninterrupted, list)
            and isinstance(resumed, list)
            and len(uninterrupted)
            == len(resumed)
        ):

            for a, b in zip(
                uninterrupted,
                resumed
            ):

                if (
                    not finite(a)
                    or not finite(b)
                    or abs(float(a) - float(b))
                    > float(tolerance)
                ):
                    resume_pass = False
                    break

    if not resume_pass:
        reason_codes.add(
            "RESUME_DIVERGENCE"
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_names,
        "trainableCount": trainable_count,
        "peftConfigPass": peft_valid,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": eval_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": unique_sorted(
            reason_codes
        ),
    }, 200


# ============================================================
# ROUTE
# ============================================================

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/adapt")
async def adapt(payload: dict):

    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    operation = payload.get("operation")

    if operation == "choose":
        result, status = choose(payload)

    elif operation == "repair":
        result, status = repair(payload)

    else:
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    if result is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    return JSONResponse(
        status_code=status,
        content=result
    )
