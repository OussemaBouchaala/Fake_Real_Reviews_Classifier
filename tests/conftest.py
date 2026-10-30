import sys
from pathlib import Path

# The app modules import each other as top-level modules (`from utils import ...`), like the notebook does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
