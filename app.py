"""Launch the CloudInsight Streamlit workspace."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

APP_ENTRYPOINT = Path(__file__).resolve().parent / "app" / "streamlit_app.py"


def _launched_by_streamlit() -> bool:
    """Return True when this file is running inside a Streamlit script context."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False

    return get_script_run_ctx() is not None


def _load_streamlit_main():
    """Load the Streamlit app entrypoint without relying on package imports."""
    spec = spec_from_file_location("cloudinsight_streamlit_app", APP_ENTRYPOINT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Streamlit app from {APP_ENTRYPOINT}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


if __name__ == "__main__":
    if _launched_by_streamlit():
        _load_streamlit_main()()
    else:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(APP_ENTRYPOINT)],
            check=True,
        )
