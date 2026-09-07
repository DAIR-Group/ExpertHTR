"""Command-aware output protection for separate CLI invocations."""

from pathlib import Path

from .context import RuntimeState


def prepare_output_directory(state: RuntimeState) -> Path:
    state.require("CFG")
    config = state["CFG"]
    output_dir = Path(config["output_dir"])
    if (
        state.command == "train"
        and output_dir.exists()
        and any(output_dir.iterdir())
        and not config["dense_init_allow_existing_output"]
    ):
        raise FileExistsError(
            f"Refusing to mix training outputs in a nonempty directory: {output_dir}. "
            "Choose a new run tag or output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
