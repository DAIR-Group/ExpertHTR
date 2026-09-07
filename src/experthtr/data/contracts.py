"""data.contracts phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~experthtr.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from experthtr.runtime.context import RuntimeState, export_runtime_locals


def validate_data_contract(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # --- Canonical seven-source dataset / source-agnostic MoE validation ---------
    # EPARCHOS is filtered from every split. The routed-expert count is independent
    # of metadata groups because router CE is disabled.
    if CFG.get("prompt_text"):
        raise ValueError("page_multitask v4 requires record-specific prompts; leave HWVLM_PROMPT_TEXT empty.")

    EXPECTED_SOURCES = {
        "BRESSAY", "Bentham", "HWDB2.0", "IAM",
        "ICFHR2016", "Images_Courriers", "ScribbleLens",
    }
    if set(CFG.get("excluded_sources", [])) != {"Eparchos"}:
        raise ValueError(
            "This final MoE protocol must exclude exactly EPARCHOS. "
            f"Got: {CFG['excluded_sources']}"
        )
    CFG["expected_sources"] = sorted(EXPECTED_SOURCES)
    CFG["expected_split_records"] = {"train": 20388, "valid": 962, "test": 1327}
    CFG["routing_grouping"] = ROUTING_GROUPING
    CFG["routing_group_description"] = str(ROUTING_GROUP_SPEC["description"])
    CFG["language_groups"] = list(LANGUAGE_GROUPS)
    CFG["moe_mlp_language_experts"] = int(
        os.getenv("HWVLM_MOE_LANGUAGE_EXPERTS", str(ROUTING_EXPERTS[ROUTING_GROUPING]))
    )
    if CFG["moe_mlp_router_ce_coef"] != 0.0:
        raise ValueError("Router CE must remain disabled for this final MoE protocol.")
    if CFG["moe_mlp_language_experts"] != 4:
        raise ValueError(
            f"This final MoE protocol requires four routed experts, got "
            f"{CFG['moe_mlp_language_experts']}"
        )
    print("Routing mode:", ROUTING_GROUPING, "| routed experts:", CFG["moe_mlp_language_experts"])
    print("Router CE:", CFG["moe_mlp_router_ce_coef"], "| shared expert is configured per ablation arm.")

    return export_runtime_locals(state, locals())
