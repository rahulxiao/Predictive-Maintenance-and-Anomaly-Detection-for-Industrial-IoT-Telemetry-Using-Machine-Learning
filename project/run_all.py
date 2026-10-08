"""
Root convenience entry point for the NASA C-MAPSS Research Project.
Simply forwards execution to src/run_all.py.
"""
import sys
from pathlib import Path

# Add src to sys.path and run main()
src_dir = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_dir))

from run_all import main

if __name__ == "__main__":
    sys.exit(main())
