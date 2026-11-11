"""Unadjusted descriptive robustness intervals; the four primary tests stay separate."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v3'
SEEDS=(42,123,2026)


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_only(differences,strata):
    d=np.asarray(differences,dtype=float);s=np.asarray(strata)
    assert d.ndim==2 and d.shape[0]>=2 and d.shape[1]==len(s) and np.isfinite(d).all()
    n_seeds=d.shape[0]
    rng=np.random.default_rng(20261017);item=d.mean(axis=0)
    groups=[np.flatnonzero(s==value) for value in np.unique(s)]
    conditional=[];crossed=[]
    for start in range(0,50000,256):
        n=min(256,50000-start)
        columns=np.concatenate([rng.choice(g,size=(n,len(g)),replace=True) for g in groups],axis=1)
        seeds=rng.integers(0,n_seeds,size=(n,n_seeds))
        conditional.extend(item[columns].mean(axis=1).tolist())
        crossed.extend(d[seeds[:,:,None],columns[:,None,:]].mean(axis=(1,2)).tolist())
    return dict(difference=float(d.mean()),seed_differences=d.mean(axis=1).tolist(),
        seed_sd=float(d.mean(axis=1).std(ddof=1)),n_items=d.shape[1],n_seeds=n_seeds,
        stratum_sizes={str(value):int(sum(s==value)) for value in np.unique(s)},
        conditional_item_ci95=np.quantile(conditional,[.025,.975]).tolist(),
        crossed_seed_item_ci95=np.quantile(crossed,[.025,.975]).tolist(),bootstraps=50000)


def main():
    analysis=OUT/'analysis';manifest=read(analysis/'analysis_manifest.json')
    assert manifest['n_runs']==20
    assert all(sha(analysis/name)==value for name,value in manifest['output_hashes'].items()),'Primary analysis output changed'
    assert all(sha(Path(name))==value for name,value in manifest['input_hashes'].items()),'Evaluation records changed'
    keys=read(analysis/'item_outcome_keys.json')
    with np.load(analysis/'item_outcomes.npz') as archive:
        cells={tuple(key):archive[f'cell_{i}'] for i,key in enumerate(keys)}
    data={name:[json.loads(line) for line in (OUT/f'data/confirm_{name}.jsonl').read_text(encoding='utf-8').split('\n') if line.strip()]
          for name in ('math','gsm')}
    subsets=read(OUT/'data/analysis_subsets.json')['datasets']
    models=sorted({key[0] for key in cells})
    arms=sorted({key[1] for key in cells}-{'base','binary'})
    primary={(x['model'],x['comparator']):x for x in read(analysis/'primary.json')}
    results=[];cache={}

    def calculate(model,treatment,comparator,dataset,style,cap,metric,subset):
        allowed=set(subsets[dataset][subset]);indices=[i for i,row in enumerate(data[dataset]) if row['task_id'] in allowed]
        assert len(indices)==len(allowed)
        cache_key=(model,treatment,comparator,dataset,style,cap,metric,tuple(indices))
        if cache_key not in cache:
            d=np.stack([cells[model,treatment,seed,dataset,style,cap,metric].astype(float)
                        -cells[model,comparator,seed,dataset,style,cap,metric].astype(float) for seed in SEEDS])[:,indices]
            strata=[data[dataset][i].get('difficulty','gsm') for i in indices]
            if (treatment=='centered_budget' and comparator in ('binary','grs_only') and dataset=='math'
                and style=='compact' and cap==1024 and metric=='strict_passed' and len(indices)==300):
                p=primary[model,comparator]
                cache[cache_key]={k:p[k] for k in ('difference','seed_differences','seed_sd','n_items','n_seeds',
                    'conditional_item_ci95','crossed_seed_item_ci95','bootstraps','stratum_sizes')}
            else:cache[cache_key]=bootstrap_only(d,strata)
        results.append(dict(model=model,treatment=treatment,comparator=comparator,dataset=dataset,
            prompt_style=style,cap=cap,metric=metric,subset=subset,**cache[cache_key]))

    for model in models:
        for comparator in ('binary','grs_only'):
            for dataset in ('math','gsm'):
                for style in ('compact',):
                    for cap in (256,512,1024):
                        for metric in ('strict_passed','formatted_final_passed','legacy_passed'):
                            for subset in subsets[dataset]:
                                calculate(model,'centered_budget',comparator,dataset,style,cap,metric,subset)
        for treatment in arms:
            if treatment!='centered_budget':
                calculate(model,treatment,'binary','math','compact',1024,'strict_passed','all_ids')
    destination=analysis/'secondary.json'
    assert not destination.exists(),'Archive prior analysis before rerunning'
    destination.write_text(json.dumps(dict(results=results,script_sha256=sha(Path(__file__)),
        primary_manifest_sha256=sha(analysis/'analysis_manifest.json'),
        interpretation='Descriptive unadjusted intervals, not a family of confirmatory discoveries. Four primary tests remain in primary.json; small-seed bootstrap limitations apply.'),indent=2),encoding='utf-8')
    print(f'Wrote {len(results)} prespecified descriptive robustness cells.')


if __name__=='__main__':main()
