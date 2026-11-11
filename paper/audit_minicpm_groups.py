import json,math,pathlib,sys,hashlib
from collections import Counter
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rewards.v3 import reward_group,verify_final
R=ROOT/'artifacts/minicpm5_external'; M=json.loads((R/'manifest.json').read_text(encoding='utf8')); train={x['task_id']:x for x in (json.loads(l) for l in (ROOT/'artifacts/v3/data/train.jsonl').read_text(encoding='utf8').splitlines() if l)}
def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf8').splitlines() if l]
reports=[]
for job in M['jobs']:
 p=ROOT/job['output'];allrows=rows(p/'groups.jsonl');archived=0
 if p.name=='centered_budget_s42':
  old=rows(R/'interrupted_attempts/centered_budget_s42_20261028_1415/groups.jsonl'); assert allrows[:len(old)]==old and len(old)==190; selected=[x for x in old if x['global_step']<50]+allrows[len(old):];archived=len(old)-100
 else:selected=allrows
 counts=Counter(x['global_step'] for x in selected);assert len(selected)==200 and counts==Counter({i:2 for i in range(100)})
 truncated=nonzero=0
 for x in selected:
  lengths=[len(t) for t in x['completion_ids']];assert len(lengths)==4 and all(0<n<=768 for n in lengths); assert x['passed']==[verify_final(t,train[x['task_id']]['reference_answer'])[0] for t in x['texts']]; q=reward_group(x['passed'],lengths,job['arm'])
  for f in ('rewards','grs','delta','normalized_advantage'):assert all(math.isclose(a,b,abs_tol=1e-10) for a,b in zip(x[f],q[f]))
  truncated+=sum(x['truncated']);nonzero+=q['reward_sd']>1e-12
 reports.append(dict(run=p.name,selected_groups=len(selected),superseded_groups=archived,nonzero_reward_sd_groups=nonzero,truncated_rollouts=truncated,total_rollouts=800,groups_sha256=hashlib.sha256((p/'groups.jsonl').read_bytes()).hexdigest()))
dest=R/'analysis'/'training_diagnostics.json';assert not dest.exists();dest.write_text(json.dumps(reports,indent=2),encoding='utf8');print('Audited',len(reports),'runs and 1200 selected groups')
