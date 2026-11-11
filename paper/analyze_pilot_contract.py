"""Paired exploratory protocol contrasts, including exact prefix verification."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rewards.v3 import verify_final
OUT=ROOT/'artifacts/protocol_pilot_20261017'
rows=[json.loads(l) for l in (OUT/'rows.jsonl').read_text(encoding='utf-8').split('\n') if l.strip()]
assert len(rows)==128 and len({r['key'] for r in rows})==128
index={(r['model'],r['task_id'],r['prompt_style'],r['max_new_tokens']):r for r in rows}
models=sorted({r['model'] for r in rows});tasks=sorted({r['task_id'] for r in rows})
rng=np.random.default_rng(20261017)
boot=rng.integers(0,16,size=(20000,16))
cells=[];contrasts=[];disagreements=[];prefix=[]
for r in rows:
    r['strict']=verify_final(r['completion'],r['reference_answer'])[0]
    if r['strict']!=r['passed_legacy_verifier']:disagreements.append(r)
for m in models:
    for style in ('compact','unconstrained'):
        for cap in (256,512):
            rr=[index[m,t,style,cap] for t in tasks]
            cells.append(dict(model=m,prompt=style,cap=cap,n=16,
                strict_correct=sum(r['strict'] for r in rr),legacy_correct=sum(r['passed_legacy_verifier'] for r in rr),
                truncated=sum(r['stop_reason']=='length' for r in rr)))
        for t in tasks:
            a=index[m,t,style,256]['generated_ids'];b=index[m,t,style,512]['generated_ids']
            prefix.append(dict(model=m,task=t,prompt=style,match=a==b[:len(a)]))
    comparisons=[('budget_compact',('compact',512),('compact',256)),
        ('budget_unconstrained',('unconstrained',512),('unconstrained',256)),
        ('prompt_256',('compact',256),('unconstrained',256)),
        ('prompt_512',('compact',512),('unconstrained',512))]
    for label,a,b in comparisons:
        for score in ('strict','passed_legacy_verifier'):
            d=np.array([int(index[m,t,*a][score])-int(index[m,t,*b][score]) for t in tasks])
            ci=np.quantile(d[boot].mean(axis=1),[.025,.975])
            contrasts.append(dict(model=m,contrast=label,score=score,difference=float(d.mean()),
                paired_item_percentile95=ci.tolist(),gains=int(sum(d==1)),losses=int(sum(d==-1))))
report=dict(scope='Exploratory n=16 reused development questions; intervals are unadjusted and not confirmatory.',
    cells=cells,contrasts=contrasts,prefix_checks=prefix,all_prefix_match=all(r['match'] for r in prefix),
    legacy_strict_disagreements=len(disagreements))
(OUT/'contract_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(OUT/'contract_disagreements.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in disagreements),encoding='utf-8')
print(json.dumps(dict(cells=cells,disagreements=len(disagreements),prefix_match=report['all_prefix_match']),indent=2))
