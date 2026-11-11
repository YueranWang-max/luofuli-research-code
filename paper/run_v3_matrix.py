"""Fixed serial matrix; halt on technical failure, never on measured accuracy."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v3'
ARMS=['binary','grs_only','centered_budget']
SEEDS=(42,123,2026)
MODELS=['Qwen2.5-1.5B-Instruct','Qwen3-4B-Instruct']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    # Both engineering runs must have succeeded before the official manifest exists.
    assert not (ROOT/'artifacts/.gpu_lock').exists(),'GPU engineering/training job is still active'
    for prefix in ('qwen15','qwen4'):
        done=list((OUT/'smoke').glob(prefix+'_attempt*/completed.json'))
        assert done and any(json.loads(p.read_text())['steps']==2 for p in done),prefix+' smoke incomplete'
    eval_smoke=OUT/'engineering/eval_batch8_qwen4'
    assert json.loads((eval_smoke/'completed.json').read_text())['n']==16,'Long evaluation smoke incomplete'
    assert not (eval_smoke/'prefix_failure.json').exists(),'Evaluation prefixes failed'
    rng=random.Random(20261017)
    jobs=[]
    for model in MODELS:
        for seed in SEEDS:
            arms=ARMS.copy();rng.shuffle(arms)
            jobs.extend(dict(model=model,seed=seed,arm=arm) for arm in arms)
    inputs=['src/train/run_grpo_v3.py','src/rewards/v3.py','src/environments/math_env.py','paper/run_v3_matrix.py','paper/CONFIRMATORY_PROTOCOL_V3.md',
        'paper/STATISTICAL_ANALYSIS_V3.md','paper/stats_v3.py','paper/evaluate_v3_batched.py',
        'paper/analyze_v3_results.py','paper/analyze_v3_secondary.py','paper/render_v3_figures.R','paper/FIGURE_CAPTIONS.md',
        'src/evaluation/answer_contracts.py','paper/run_v3_evaluation_matrix.py','paper/run_v3_study.py','paper/audit_v3_groups.py',
        'paper/build_v3_data.py','paper/audit_near_duplicates_v3.py','paper/SIGN_MARGIN_NOTE.md','paper/TECHNICAL_AMENDMENT_R1.md',
        'paper/TECHNICAL_AMENDMENT_R2.md','paper/TECHNICAL_AMENDMENT_R3.md',
        'artifacts/v3/data/train.jsonl','artifacts/v3/data/manifest.json',
        'artifacts/v3/data/analysis_subsets.json','artifacts/v3/data/near_duplicate_screen.json',
        'artifacts/v3/data/confirm_math.jsonl','artifacts/v3/data/confirm_gsm.jsonl']
    model_inputs=[p for m in MODELS for p in (ROOT/'models'/m).iterdir() if p.suffix in ('.json','.safetensors','.txt','.model','.jinja')]
    frozen=dict(jobs=jobs,runtime_versions={name:importlib.metadata.version(name)
        for name in ('torch','transformers','trl','peft','accelerate','datasets','numpy')},
        python_version=sys.version,hashes={p:sha(ROOT/p) for p in inputs},
        model_hashes={str(p.relative_to(ROOT)):sha(p) for p in model_inputs})
    manifest=OUT/'training_manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==frozen,'Frozen inputs changed'
    else:
        manifest.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
        archive=OUT/'frozen_sources';archive.mkdir()
        for p in inputs:
            target=archive/p;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/p).read_bytes())
    env=os.environ.copy();env['PYTHONUTF8']='1'
    tmp=OUT/'tmp';tmp.mkdir(exist_ok=True)
    env['TEMP']=env['TMP']=str(tmp)
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    for i,j in enumerate(jobs):
        assert all(importlib.metadata.version(name)==version for name,version in frozen['runtime_versions'].items()),'Runtime dependency drift'
        assert all(sha(ROOT/p)==h for p,h in frozen['hashes'].items()),'Source/data drift'
        assert all(sha(ROOT/p)==h for p,h in frozen['model_hashes'].items() if j['model'] in p),'Base model drift'
        key=f"{j['model']}_{j['arm']}_s{j['seed']}"
        dest=OUT/'runs'/key
        if (dest/'completed.json').exists():
            assert json.loads((dest/'completed.json').read_text())['steps']==100
            continue
        resume_plan=dest/'resume_plan.json'
        if dest.exists():
            assert resume_plan.exists(),f'Incomplete attempt requires diagnosis: {dest}'
            resume_info=json.loads(resume_plan.read_text())
            checkpoint=dest/resume_info['checkpoint']
            assert checkpoint.is_dir() and resume_info['resume_global_step']==50
        else:
            checkpoint=None
        cmd=[sys.executable,'-u','-X','utf8',str(ROOT/'src/train/run_grpo_v3.py'),
            '--model',j['model'],'--arm',j['arm'],'--seed',str(j['seed']),'--output',str(dest)]
        if checkpoint:cmd.extend(['--resume-from',str(checkpoint)])
        print(f'[{i+1}/{len(jobs)}] {key}',flush=True)
        with (logs/(key+'.log')).open('a' if checkpoint else 'x',encoding='utf-8') as f:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
        if result.returncode:
            if dest.exists():
                (dest/'process_failure.json').write_text(json.dumps(dict(exit_code=result.returncode,
                    observed_unix=time.time(),command=cmd),indent=2),encoding='utf-8')
            raise subprocess.CalledProcessError(result.returncode,cmd)
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/'paper/audit_v3_groups.py'),str(dest)],
            cwd=ROOT,env=env,stdout=subprocess.DEVNULL,check=True)
    (OUT/'training_completed.json').write_text(json.dumps(dict(completed_runs=len(jobs))))

if __name__=='__main__':main()
