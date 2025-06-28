# Add the project root to Python path
import sys
import os
from pathlib import Path

# Get the project root (master_thesis directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "code" / "python"))
sys.path.insert(0, str(project_root / "code" / "python" / "scripts"))
