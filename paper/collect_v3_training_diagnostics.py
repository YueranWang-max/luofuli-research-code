"""Collect complete endpoints and keep offline reward counterfactuals explicitly labelled."""
import csv
import hashlib
import json
from pathlib import Path
import math
from audit_v3_groups import audit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v3'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_csv(path,records):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def main():
    import torch
    from safetensors.torch import load_file
    frozen=read(OUT/'training_manifest.json')
    assert len(frozen['jobs'])==70 and read(OUT/'training_completed.json')['completed_runs']==70
    assert all(sha(ROOT/path)==value for path,value in frozen['hashes'].items()),'Frozen inputs changed'
    summaries=[];composition=[];counterfactuals=[];inputs={}
    for job in frozen['jobs']:
        key=f"{job['model']}_{job['arm']}_s{job['seed']}";run=OUT/'runs'/key
        done=read(run/'completed.json');request=read(run/'request.json')
        assert done['steps']==request['args']['steps']==100
        assert all(request['args'][field]==job[field] for field in ('model','arm','seed'))
        for name,value in done['endpoint_hashes'].items():assert sha(run/name)==value,'Endpoint drift'
        tensors=load_file(str(run/'adapter_model.safetensors'),device='cpu')
        assert all(torch.isfinite(value).all().item() for value in tensors.values()),f'Nonfinite endpoint: {key}'
        del tensors
        report=audit(run);assert report['complete']
        assert report['groups']==200 and report['rollouts']==800,'Unequal fixed rollout budget'
        groups=[json.loads(line) for line in (run/'groups.jsonl').read_text(encoding='utf-8').split('\n') if line.strip()]
        history=read(run/'trainer_state.json')['log_history']
        grad=[float(row['grad_norm']) for row in history if 'grad_norm' in row]
        assert len(grad)==100 and all(math.isfinite(value) for value in grad),f'Invalid gradient history: {key}'
        runtimes=[row['train_runtime'] for row in history if 'train_runtime' in row]
        assert len(runtimes)==1 and runtimes[0]>0
        metrics=report['counterfactual_metrics'][job['arm']]
        successful_tokens=sum(len(ids) for g in groups for ids,passed in zip(g['completion_ids'],g['passed']) if passed)
        correct=metrics['correct_rollouts'];total=report['total_generated_tokens'];n=report['rollouts']
        mixed_correct=sum(g['successes'] for g in groups if 0<g['successes']<4)
        summaries.append(dict(run=key,**job,steps=100,groups=report['groups'],rollouts=n,
            generated_tokens=total,correct_rollouts=correct,truncated_rollouts=report['truncated_rollouts'],
            mean_all_tokens=total/n,mean_success_tokens=successful_tokens/correct if correct else None,
            mean_failure_tokens=(total-successful_tokens)/(n-correct) if n>correct else None,
            tokens_per_success=total/correct if correct else None,
            nonzero_reward_sd_groups=metrics['nonzero_sd_groups'],
            unmasked_nonzero_advantage_groups=metrics['unmasked_nonzero_advantage_groups'],
            correct_negative_advantages=metrics['correct_negative_advantages'],
            mixed_correct_rollouts=mixed_correct,
            mixed_correct_negative_advantages=metrics['mixed_correct_negative_advantages'],
            logged_nonzero_preclip_gradient_steps=sum(value>0 for value in grad),
            trainer_seconds=runtimes[0],script_elapsed_seconds=done['elapsed_seconds'],
            peak_allocated_bytes=done['peak_allocated_bytes'],
            cost_scope='Observed local runtime; excludes parent hashing, analysis and human work; not an isolated hardware benchmark'))
        for k,count in report['success_composition'].items():
            composition.append(dict(run=key,**job,successful_responses_in_group=k,groups=count))
        for evaluated_arm,values in report['counterfactual_metrics'].items():
            counterfactuals.append(dict(origin_run=key,origin_model=job['model'],origin_arm=job['arm'],
                origin_seed=job['seed'],offline_reward_arm=evaluated_arm,**values,
                interpretation='Same observed rollouts rescored offline; not outcomes of training the counterfactual policy'))
        for name in ('groups.jsonl','trainer_state.json','completed.json','request.json'):
            inputs[str(run/name)]=sha(run/name)
    destination=OUT/'training_analysis';destination.mkdir(exist_ok=False)
    write_csv(destination/'run_summary.csv',summaries)
    write_csv(destination/'group_composition.csv',composition)
    write_csv(destination/'offline_reward_counterfactuals.csv',counterfactuals)
    failures=[]
    for location in ('failed_attempts','smoke','engineering'):
        for path in (OUT/location).glob('*/*failure*'):
            if path.name=='gpu_lock_at_failure.txt':continue
            failures.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),
                cost_status='Retained attempt; cost may be unknown and must not be silently set to zero'))
    (destination/'manifest.json').write_text(json.dumps(dict(input_hashes=inputs,failures=failures,
        script_sha256=sha(Path(__file__)),completed_runs=70,all_endpoint_tensors_finite=True,
        scope='Training diagnostics only; no confirmation accuracy claim. Training uses equal steps/rollout counts, not equal realized tokens or FLOPs.'),indent=2),encoding='utf-8')
    print('Complete training diagnostics exported; failed attempts remain separately identified.')


if __name__=='__main__':main()
