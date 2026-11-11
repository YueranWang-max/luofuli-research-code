import hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'paper')]
from stats_v3 import paired_effect,holm
from rewards.v3 import verify_final
from evaluation.answer_contracts import verify_formatted_final
from environments.math_env import MathVerifier
from tokenizers import Tokenizer
R=ROOT/'artifacts/minicpm5_external';M=json.loads((R/'manifest.json').read_text(encoding='utf8'));T=Tokenizer.from_file(str(ROOT/'models/MiniCPM5-2B/tokenizer.json'));V=MathVerifier()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
data={d:lines(ROOT/f'artifacts/v3/data/confirm_{d}.jsonl') for d in ('math','gsm')}; refs={x['task_id']:x['reference_answer'] for v in data.values() for x in v}; expected={f"{d}/{x['task_id']}/compact" for d,v in data.items() for x in v}; assert len(expected)==600
out={};inputs={}
for name in M['evaluation_endpoints']:
 p=R/'evaluations'/name; records=lines(p/'rows.jsonl'); conf=json.loads((p/'completed.json').read_text()); manifest=json.loads((p/'manifest.json').read_text(encoding='utf8')); assert conf['n']==len(records)==600 and set(x['key'] for x in records)==expected and len({x['key'] for x in records})==600 and {x['key'] for x in manifest['jobs']}==expected
 for row in records:
  assert row['reference_answer']==refs[row['task_id']]; ids=row['generated_ids']; assert 0<len(ids)<=1024 and not any(x in (1,130073) for x in ids[:-1]); assert {s['cap'] for s in row['scores']}=={256,512,1024}
  for score in row['scores']:
   part=ids[:score['cap']];text=T.decode(part,skip_special_tokens=True);assert text==score['text'] and score['strict_passed']==verify_final(text,row['reference_answer'])[0] and score['formatted_final_passed']==verify_formatted_final(text,row['reference_answer'])[0] and score['legacy_passed']==V.verify(text,row['reference_answer']).passed and score['stop_reason']==('eos' if part[-1] in (1,130073) else 'length')
 out[name]={x['key']:x for x in records}; inputs[name]={f:sha(p/f) for f in ('rows.jsonl','manifest.json','completed.json')}
for job in M['jobs']:
 p=ROOT/job['output'];c=json.loads((p/'completed.json').read_text());assert c['steps']==100 and all(sha(p/n)==h for n,h in c['endpoint_hashes'].items())
results=[];cells=[]
for ds in ('math','gsm'):
 keys=[f"{ds}/{x['task_id']}/compact" for x in data[ds]]
 for name in M['evaluation_endpoints']:
  scores=[next(s for s in out[name][k]['scores'] if s['cap']==1024) for k in keys]; good=[s['strict_passed'] for s in scores];tokens=[s['generated_tokens'] for s in scores];cells.append(dict(dataset=ds,endpoint=name,n=300,correct=sum(good),accuracy=sum(good)/300,total_tokens=sum(tokens),tokens_per_correct=sum(tokens)/sum(good) if sum(good) else None,truncated=sum(s['stop_reason']=='length' for s in scores)))
 diff=np.array([[int(next(s for s in out[f'centered_budget_s{seed}'][k]['scores'] if s['cap']==1024)['strict_passed'])-int(next(s for s in out[f'binary_s{seed}'][k]['scores'] if s['cap']==1024)['strict_passed']) for k in keys] for seed in (42,123,2026)]);results.append(dict(dataset=ds,**paired_effect(diff,[x.get('difficulty','gsm') for x in data[ds]])))
for x,p in zip(results,holm([x['conditional_item_permutation_p'] for x in results])):x['holm_p_two_datasets']=p
dest=R/'analysis';dest.mkdir(exist_ok=False);(dest/'results.json').write_text(json.dumps(dict(results=results,cells=cells,inputs=inputs,analysis_sha256=sha(pathlib.Path(__file__)),stats_sha256=sha(ROOT/'paper/stats_v3.py')),indent=2),encoding='utf8');print('verified 4200 records; wrote',dest/'results.json')
