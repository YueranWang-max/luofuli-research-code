"""Evaluate every frozen training endpoint and both bases; no best-seed selection."""
import json
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
from run_v3_matrix import sha
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v3'

def main():
    train=json.loads((OUT/'training_manifest.json').read_text(encoding='utf-8'))
    assert (OUT/'training_completed.json').exists(),'Full training matrix is required'
    assert json.loads((OUT/'training_completed.json').read_text())['completed_runs']==len(train['jobs'])
    models=list(dict.fromkeys(j['model'] for j in train['jobs']))
    jobs=[dict(model=m,key=m+'_base',adapter=None) for m in models]
    for j in train['jobs']:
        key=f"{j['model']}_{j['arm']}_s{j['seed']}"
        jobs.append(dict(model=j['model'],key=key,adapter=str(OUT/'runs'/key)))
    env=os.environ.copy();env['PYTHONUTF8']='1';env['TEMP']=env['TMP']=str(OUT/'tmp')
    logs=OUT/'eval_logs';logs.mkdir(exist_ok=True)
    for i,j in enumerate(jobs):
        assert sys.version==train['python_version'],'Python runtime drift'
        assert all(importlib.metadata.version(name)==version for name,version in train['runtime_versions'].items()),'Runtime dependency drift'
        assert all(sha(ROOT/p)==h for p,h in train['hashes'].items()),'Frozen source/data drift before evaluation'
        assert all(sha(ROOT/p)==h for p,h in train['model_hashes'].items() if j['model'] in p),'Base model drift'
        dest=OUT/'evaluations'/j['key']
        cmd=[sys.executable,'-u','-X','utf8',str(ROOT/'paper/evaluate_v3_batched.py'),
            '--model',j['model'],'--output',str(dest),'--prompt-styles','compact']
        if j['adapter']:cmd+=['--adapter',j['adapter']]
        print(f"[eval {i+1}/{len(jobs)}] {j['key']}",flush=True)
        with (logs/(j['key']+'.log')).open('a',encoding='utf-8') as f:
            # Evaluator validates its frozen manifest even when all rows already exist.
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    (OUT/'evaluation_completed.json').write_text(json.dumps(dict(completed_evaluations=len(jobs))))

if __name__=='__main__':main()
