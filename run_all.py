"""
Root execution entry point for the NASA C-MAPSS Research Project.
Allows anyone to run the full pipeline directly after cloning the repo.
"""
import sys
from pathlib import Path

# Add project/src to sys.path
project_src = Path(__file__).resolve().parent / "project" / "src"
sys.path.insert(0, str(project_src))

from run_all import main

if __name__ == "__main__":
    sys.exit(main())
