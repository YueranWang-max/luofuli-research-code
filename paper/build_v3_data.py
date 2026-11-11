"""Freeze unused confirmation problems before any v3 inference; preserve source hashes."""
import hashlib
import json
import random
import re
from pathlib import Path
import pandas as pd
from datasets import Dataset

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v3/data'
SUFFIX="\n\nSolve this problem with at most three short steps (under 120 words total), then end your reply with exactly 'Answer: <final number>'. The final answer must be a plain number (no units, no commas)."


def norm(q):
    q=re.split(r'\n\n(?:Solve this problem|Reason briefly)',q)[0]
    return re.sub(r'\s+',' ',q).strip()


def load(p):
    return [json.loads(l) for l in p.read_text(encoding='utf-8').split('\n') if l.strip()]


def main():
    sources={}
    def register(p):
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    excluded=set()
    for bench in ('mathhard','mathtrain','math500','gsm8k'):
        for p in (ROOT/'artifacts/benchmarks'/bench).glob('*.jsonl'):
            register(p)
            excluded.update(norm(r['prompt']) for r in load(p))
    p=ROOT/'artifacts/distill/sft_data.jsonl'
    register(p); excluded.update(norm(r['prompt']) for r in load(p))
    pool={}
    for p in sorted((ROOT/'artifacts/benchmarks/raw/math_train').glob('*.parquet')):
        register(p)
        for index,r in enumerate(pd.read_parquet(p).to_dict('records')):
            q=norm(r['problem'])
            a=re.findall(r'\\boxed\{([^{}]+)\}',str(r['solution']))
            if q in excluded or not a or not re.fullmatch(r'-?\d+(?:\.\d+)?',a[-1].strip()): continue
            if r['level'] not in ('Level 4','Level 5'):continue
            if q in pool: assert pool[q]['reference_answer']==a[-1].strip()
            pool[q]=dict(task_id='confirm_math_'+hashlib.sha256(q.encode()).hexdigest()[:16],
                question=r['problem'].strip(),prompt=r['problem'].strip()+SUFFIX,
                reference_answer=a[-1].strip(),difficulty=r['level'].replace('Level ','L'),domain='math',
                source=p.name,source_row=index)
    rng=random.Random(20261017)
    chosen=[]
    for level in ('L4','L5'):
        eligible=sorted((r for r in pool.values() if r['difficulty']==level),key=lambda x:norm(x['question']))
        assert len(eligible)>=150,(level,len(eligible))
        chosen+=rng.sample(eligible,150)
    gsm=Path('C:/Users/dragoncat/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow')
    register(gsm)
    gsm_pool={}
    for i,r in enumerate(Dataset.from_file(str(gsm))):
        q=norm(r['question'])
        if q in excluded or q in {norm(x['question']) for x in chosen}:continue
        a=r['answer'].split('####')[-1].strip().replace(',','')
        assert re.fullmatch(r'-?\d+(?:\.\d+)?',a),a
        gsm_pool[q]=dict(task_id='confirm_gsm_'+hashlib.sha256(q.encode()).hexdigest()[:16],
            question=r['question'].strip(),prompt=r['question'].strip()+SUFFIX,reference_answer=a,
            difficulty='gsm8k',domain='math',source='openai/gsm8k main official test',source_row=i)
    assert len(gsm_pool)>=300
    chosen_gsm=rng.sample(sorted(gsm_pool.values(),key=lambda x:norm(x['question'])),300)
    train=load(ROOT/'artifacts/benchmarks/mathhard/train.jsonl')
    confirm=chosen+chosen_gsm
    assert len({norm(x['prompt']) for x in confirm})==600
    assert not {norm(x['prompt']) for x in confirm}&excluded
    OUT.mkdir(parents=True,exist_ok=True)
    files={'train':train,'confirm_math':chosen,'confirm_gsm':chosen_gsm}
    hashes={}
    for name, rr in files.items():
        content=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rr)
        target=OUT/(name+'.jsonl')
        if target.exists():assert target.read_text(encoding='utf-8')==content,'Frozen data differ'
        else:target.write_text(content,encoding='utf-8')
        hashes[name]=hashlib.sha256(content.encode()).hexdigest()
    manifest=dict(seed=20261017,source_sha256=sources,output_sha256=hashes,
        excluded_unique_questions=len(excluded),unused_math_candidates=len(pool),unused_gsm_candidates=len(gsm_pool),
        n_train=len(train),n_confirm_math=len(chosen),n_confirm_gsm=len(chosen_gsm),
        limitation='Project-unseen exact-normalized questions; pretraining exposure and near duplicates not excluded.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('source_sha256','output_sha256')},indent=2))


if __name__=='__main__':main()
