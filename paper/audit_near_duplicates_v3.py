"""Pre-outcome screening: textual similarity candidates, not semantic adjudication."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT=Path(__file__).resolve().parents[1]

def norm(q):
    q=re.split(r'\n\n(?:Solve this problem|Reason briefly)',q)[0]
    return re.sub(r'\s+',' ',q).strip()

def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').split('\n') if l.strip()]

def main():
    files=[p for b in ('mathhard','mathtrain','math500','gsm8k') for p in (ROOT/'artifacts/benchmarks'/b).glob('*.jsonl')]
    files.append(ROOT/'artifacts/distill/sft_data.jsonl')
    historic={}
    for p in files:
        for r in load(p):
            q=norm(r['prompt']);historic.setdefault(q,[]).append(str(p.relative_to(ROOT)))
    old=sorted(historic)
    confirm=[]
    for dataset in ('math','gsm'):
        p=ROOT/f'artifacts/v3/data/confirm_{dataset}.jsonl';files.append(p)
        confirm.extend(dict(dataset=dataset,**r) for r in load(p))
    vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),lowercase=True,min_df=1,dtype=np.float32)
    matrix=vectorizer.fit_transform(old+[norm(r['question']) for r in confirm])
    similarities=(matrix[len(old):]@matrix[:len(old)].T).toarray()
    nearest=[]
    for i,r in enumerate(confirm):
        j=int(np.argmax(similarities[i]));score=float(similarities[i,j])
        nearest.append(dict(task_id=r['task_id'],dataset=r['dataset'],similarity=score,
            question=r['question'],historical_question=old[j],historical_sources=historic[old[j]]))
    nearest.sort(key=lambda r:(-r['similarity'],r['task_id']))
    out=ROOT/'artifacts/v3/data/near_duplicate_screen.json'
    out.write_text(json.dumps(dict(method='TF-IDF char_wb 3-5 cosine; maximum among 4890 historical normalized questions',
        threshold=.9,n_historical=len(old),n_confirmation=len(confirm),
        n_at_least_threshold=sum(r['similarity']>=.9 for r in nearest),
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        candidates=nearest,limitation='Low textual similarity cannot rule out semantic near-duplicates or pretraining exposure.'),indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(n_historical=len(old),flagged=sum(r['similarity']>=.9 for r in nearest),top=nearest[:12]),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
