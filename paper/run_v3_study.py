"""Persistent serial study driver; failed stages remain inspectable for diagnosis."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
for script in ('run_v3_matrix.py','run_v3_evaluation_matrix.py'):
    print('Stage:',script,flush=True)
    subprocess.run([sys.executable,'-u','-X','utf8',str(ROOT/'paper'/script)],cwd=ROOT,check=True)
