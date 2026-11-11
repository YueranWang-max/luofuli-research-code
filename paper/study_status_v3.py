"""Compact health/status report without selecting runs by measured performance."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v3'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    state=dict(unix=time.time(),training_complete=0,evaluations_complete=0,
               incomplete_runs=[],failures=[],engineering_completed=[])
    lock=ROOT/'artifacts/.gpu_lock'
    state['gpu_owner']=lock.read_text(encoding='utf-8') if lock.exists() else None
    manifest=OUT/'training_manifest.json'
    state['formal_manifest_exists']=manifest.exists()
    if manifest.exists():
        for job in read(manifest)['jobs']:
            key=f"{job['model']}_{job['arm']}_s{job['seed']}"
            path=OUT/'runs'/key
            if (path/'completed.json').exists():
                state['training_complete']+=1
            elif path.exists():
                groups=path/'groups.jsonl'
                last=None;count=0
                if groups.exists():
                    for line in groups.read_text(encoding='utf-8').split('\n'):
                        if not line.strip():continue
                        try:last=json.loads(line);count+=1
                        except json.JSONDecodeError:break  # live final write may be partial
                state['incomplete_runs'].append(dict(key=key,logged_groups=count,
                    last_generation_step=last['global_step'] if last else None,
                    last_group_mtime=groups.stat().st_mtime if groups.exists() else None))
    if manifest.exists():
        frozen=read(manifest);models=list(dict.fromkeys(j['model'] for j in frozen['jobs']))
        evaluation_keys={m+'_base' for m in models}|{
            f"{j['model']}_{j['arm']}_s{j['seed']}" for j in frozen['jobs']}
        state['evaluations_complete']=sum((OUT/'evaluations'/key/'completed.json').exists()
                                          for key in evaluation_keys)
    state['failures']=[str(p.relative_to(ROOT)) for location in ('runs','evaluations','smoke','engineering','failed_attempts')
                       for p in (OUT/location).glob('*/*failure*') if p.name!='gpu_lock_at_failure.txt']
    for location in ('smoke','engineering'):
        state['engineering_completed'] += [str(p.parent.relative_to(ROOT)) for p in (OUT/location).glob('*/completed.json')]
    print(json.dumps(state,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
