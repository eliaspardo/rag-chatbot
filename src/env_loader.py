from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
PARAMS_PATH = ROOT_DIR / "config" / "params.env"

_LOADED = False


def load_environment() -> None:
    """Load secrets from .env first, then tunable params from config/params.env."""
    global _LOADED
    if _LOADED:
        return

    load_dotenv(dotenv_path=ENV_PATH)
    load_dotenv(dotenv_path=PARAMS_PATH)
    _LOADED = True
