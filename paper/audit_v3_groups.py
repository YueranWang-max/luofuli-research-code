"""Validate raw rollout groups and compute descriptive, mask-aware diagnostics."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rewards.v3 import ARMS,reward_group

def audit(run):
    request=json.loads((run/'request.json').read_text(encoding='utf-8'))
    arm=request['args']['arm']
    rows=[json.loads(l) for l in (run/'groups.jsonl').read_text(encoding='utf-8').split('\n') if l.strip()]
    assert [r['group'] for r in rows]==list(range(len(rows)))
    counts={k:0 for k in range(5)}
    metrics={a:dict(groups=0,nonzero_sd_groups=0,unmasked_nonzero_advantage_groups=0,
        correct_negative_advantages=0,correct_rollouts=0,negative_correct_unmasked=0,
        mixed_correct_negative_advantages=0,nonzero_delta_sum_groups=0) for a in ARMS}
    csr_eligible=csr_satisfied=0
    tokens=truncated=0
    for row in rows:
        lengths=[len(x) for x in row['completion_ids']]
        assert len(lengths)==4 and all(0<t<=768 for t in lengths)
        assert len(row['texts'])==len(row['passed'])==len(row['truncated'])==4
        k=sum(row['passed']);assert k==row['successes'];counts[k]+=1
        actual=reward_group(row['passed'],lengths,arm)
        for field in ('rewards','grs','delta','normalized_advantage'):
            assert all(math.isclose(a,b,abs_tol=1e-10) for a,b in zip(actual[field],row[field])),field
        tokens+=sum(lengths);truncated+=sum(row['truncated'])
        if 0<k<4:
            csr_eligible+=1
            mean_length=sum(lengths)/4
            raw=[1/(1+.33*t/mean_length) if p else 0 for p,t in zip(row['passed'],lengths)]
            csr_satisfied+=1/1.33>=sum(raw)/4
        for a in ARMS:
            counter=reward_group(row['passed'],lengths,a)
            m=metrics[a];adv=counter['normalized_advantage'];m['groups']+=1
            assert max(abs(x) for x in adv)<=1.5+1e-10,'Sample-SD normalized bound failed'
            m['nonzero_sd_groups']+=counter['reward_sd']>1e-12
            m['unmasked_nonzero_advantage_groups']+=any(abs(v)>1e-12 and not z for v,z in zip(adv,row['truncated']))
            m['correct_rollouts']+=k
            m['correct_negative_advantages']+=sum(p and v<0 for p,v in zip(row['passed'],adv))
            if 0<k<4:
                m['mixed_correct_negative_advantages']+=sum(p and v<0 for p,v in zip(row['passed'],adv))
                if a=='centered_budget':assert all(v>0 for p,v in zip(row['passed'],adv) if p),'Mixed-group sign theorem violated'
            m['negative_correct_unmasked']+=sum(p and v<0 and not z for p,v,z in zip(row['passed'],adv,row['truncated']))
            m['nonzero_delta_sum_groups']+=abs(counter['delta_sum'])>1e-10
    report=dict(run=str(run),arm=arm,groups=len(rows),success_composition=counts,total_generated_tokens=tokens,
        truncated_rollouts=truncated,rollouts=len(rows)*4,complete=(run/'completed.json').exists(),
        counterfactual_metrics=metrics,
        gr3_fixed_csr=dict(eligible_mixed_groups=csr_eligible,satisfied=csr_satisfied,
            rate=csr_satisfied/csr_eligible if csr_eligible else None),
        interpretation='Same observed rollouts, offline reward counterfactuals; not separate trained policies. Unmasked nonzero advantage is not proof of useful or nonzero parameter gradients.')
    if report['complete']:
        done=json.loads((run/'completed.json').read_text())
        assert done['groups']==len(rows)
        assert done['steps']==request['args']['steps']
    (run/'group_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run');a=ap.parse_args()
    print(json.dumps(audit(Path(a.run)),indent=2))
