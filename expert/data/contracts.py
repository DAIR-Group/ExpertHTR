"""data.contracts phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~expert.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from expert.runtime.context import RuntimeState, export_runtime_locals


def validate_data_contract(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # --- Dataset scope and routing metadata -------------------------------------
    EXPECTED_SOURCES = {
        "BRESSAY", "Bentham", "HWDB2.0", "IAM",
        "ICFHR2016", "Images_Courriers", "ScribbleLens",
    }
    CFG["expected_sources"] = sorted(EXPECTED_SOURCES)
    CFG["expected_split_records"] = {"train": 20388, "valid": 962, "test": 1327}
    CFG["routing_grouping"] = ROUTING_GROUPING
    CFG["routing_group_description"] = str(ROUTING_GROUP_SPEC["description"])
    CFG["language_groups"] = list(LANGUAGE_GROUPS)
    CFG["moe_mlp_language_experts"] = int(
        os.getenv("HWVLM_MOE_LANGUAGE_EXPERTS", str(ROUTING_EXPERTS[ROUTING_GROUPING]))
    )
    print("Routing mode:", ROUTING_GROUPING, "| routed experts:", CFG["moe_mlp_language_experts"])
    print("Router CE:", CFG["moe_mlp_router_ce_coef"], "| shared experts:", CFG["moe_mlp_shared_experts"])

    return export_runtime_locals(state, locals())
