from pathlib import Path
import importlib
import logging
import os
import warnings
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
PARAMS_PATH = ROOT_DIR / "config" / "params.env"

_LOADED = False


def _resolve_warning_class(path: str) -> type[Warning] | None:
    if not path:
        return None
    try:
        module_name, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        warning_class = getattr(module, class_name)
    except Exception:
        return None
    if isinstance(warning_class, type) and issubclass(warning_class, Warning):
        return warning_class
    return None


def _configure_warnings_from_env() -> None:
    warnings_spec = os.getenv("PYTHONWARNINGS")
    if not warnings_spec:
        return
    for entry in warnings_spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        parts += [""] * (5 - len(parts))
        action, message, category_str, module, lineno = parts[:5]
        category = _resolve_warning_class(category_str) or Warning
        lineno_value = int(lineno) if lineno.isdigit() else 0
        warnings.filterwarnings(
            action or "default",
            message=message or "",
            category=category,
            module=module or "",
            lineno=lineno_value,
        )


def _configure_default_logger_levels() -> None:
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("faiss").setLevel(logging.ERROR)
    logging.getLogger("faiss.loader").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)


def load_environment() -> None:
    """Load secrets from .env first, then tunable params from config/params.env."""
    global _LOADED
    if _LOADED:
        return

    load_dotenv(dotenv_path=ENV_PATH)
    load_dotenv(dotenv_path=PARAMS_PATH)
    _configure_warnings_from_env()
    _configure_default_logger_levels()
    _LOADED = True
