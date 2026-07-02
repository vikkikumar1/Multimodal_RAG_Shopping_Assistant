import sys
from pathlib import Path

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add the project root to Python path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))
