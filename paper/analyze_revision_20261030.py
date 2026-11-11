import json,csv,collections,pathlib,statistics,hashlib
root=pathlib.Path.cwd(); out={}; seeds=(42,123,2026)
for ds in ('math','gsm'):
 out[ds]={}
 for arm in ('binary','centered_budget'):
  v=[]
  for seed in seeds:
   p=root/f'artifacts/minicpm5_external/evaluations/{arm}_s{seed}/rows.jsonl'; rows=[json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]; rs=[next(z for z in x['scores'] if z['cap']==1024) for x in rows if x['key'].startswith(ds+'/')]; assert len(rs)==300
   v.append({'seed':seed,'strict':sum(s['strict_passed'] for s in rs),'tolerant':sum(s['formatted_final_passed'] for s in rs),'strict_fail_tolerant_pass':sum(not s['strict_passed'] and s['formatted_final_passed'] for s in rs),'strict_pass_tolerant_fail':sum(s['strict_passed'] and not s['formatted_final_passed'] for s in rs),'truncated':sum(s['stop_reason']=='length' for s in rs)})
  out[ds][arm]=v
out['qwen']=[r for r in csv.DictReader((root/'artifacts/v3/analysis/all_seed_cells.csv').open(encoding='utf8')) if r['cap']=='1024' and r['prompt_style']=='compact' and r['metric'] in ('strict_passed','formatted_final_passed') and r['arm'] in ('binary','grs_only','centered_budget')]
out['groups']=[]
for folder in [root/'artifacts/v3/runs',root/'artifacts/minicpm5_external/runs']:
 for p in sorted(folder.glob('*/groups.jsonl')):
  if not any((p.parent.name.startswith(a+'_s') or '_'+a+'_s' in p.parent.name) for a in ('binary','grs_only','centered_budget')):continue
  rows=[json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
  if folder.name=='runs' and 'minicpm5_external' in str(folder) and p.parent.name=='centered_budget_s42':
   old=[json.loads(x) for x in (root/'artifacts/minicpm5_external/interrupted_attempts/centered_budget_s42_20261028_1415/groups.jsonl').read_text(encoding='utf8').splitlines() if x.strip()]; assert rows[:len(old)]==old;rows=[x for x in old if x['global_step']<50]+rows[len(old):]
  counts=collections.Counter(x['global_step'] for x in rows)
  if len(rows)!=200 or counts!=collections.Counter({i:2 for i in range(100)}):
   out['groups'].append({'run':str(p.parent.relative_to(root)),'excluded_ambiguous':True,'records':len(rows)});continue
  dist=collections.Counter(x['successes'] for x in rows); k3=[x for x in rows if x['successes']==3]; retained=sum(any(abs(a)>1e-12 and not t for a,t in zip(x['normalized_advantage'],x['truncated'])) for x in k3)
  out['groups'].append({'run':str(p.parent.relative_to(root)),'n':len(rows),'k':{str(k):dist[k] for k in range(5)},'k3_any_unmasked_nonzero_advantage':retained,'note':'nonzero advantage is not a causal estimate of centering difference','sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
p=root/'paper/revision_evidence_20261030.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in out.items() if k in ('math','gsm')},indent=2));print('group records',len(out['groups']),'ambiguous',sum(x.get('excluded_ambiguous',False) for x in out['groups']))

