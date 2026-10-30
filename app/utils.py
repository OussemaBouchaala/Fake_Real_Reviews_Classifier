import json
from pathlib import Path
from typing import Any


def load_config() -> dict[str, Any]:
    """Load configuration from config.json file."""
    config_path = Path(__file__).parent.parent / "config.json"
    #print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)
