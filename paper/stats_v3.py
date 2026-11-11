"""Predeclared paired inference; rows are seeds and columns are shared items."""
import numpy as np

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if p.ndim!=1 or not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('Invalid p-values')
    order=np.argsort(p,kind='stable');adjusted=np.empty_like(p);previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(p)-rank)*p[index]))
        adjusted[index]=previous
    return adjusted.tolist()

def paired_effect(differences,strata,seed=20261017,permutations=200000,bootstraps=50000):
    d=np.asarray(differences,dtype=float)
    strata=np.asarray(strata)
    if d.ndim!=2 or min(d.shape)<2 or len(strata)!=d.shape[1] or not np.all(np.isfinite(d)):
        raise ValueError('Expected finite seed x shared-item paired differences')
    if permutations<1 or bootstraps<1:raise ValueError('Positive repetition count required')
    rng=np.random.default_rng(seed)
    n_seeds,n_items=d.shape
    item_difference=d.mean(axis=0);estimate=float(item_difference.mean())
    threshold=abs(estimate)-1e-15;extreme=0
    for start in range(0,permutations,512):
        n=min(512,permutations-start)
        signs=rng.integers(0,2,size=(n,n_items),dtype=np.int8)*2-1
        statistics=np.abs((signs*item_difference).mean(axis=1))
        extreme+=int(np.sum(statistics>=threshold))
    p=(extreme+1)/(permutations+1)
    groups=[np.flatnonzero(strata==s) for s in np.unique(strata)]
    conditional=[];crossed=[]
    for start in range(0,bootstraps,256):
        n=min(256,bootstraps-start)
        items=np.concatenate([rng.choice(group,size=(n,len(group)),replace=True) for group in groups],axis=1)
        seeds=rng.integers(0,n_seeds,size=(n,n_seeds))
        conditional.extend(item_difference[items].mean(axis=1).tolist())
        crossed.extend(d[seeds[:,:,None],items[:,None,:]].mean(axis=(1,2)).tolist())
    seed_effects=d.mean(axis=1)
    return dict(difference=estimate,seed_differences=seed_effects.tolist(),seed_sd=float(seed_effects.std(ddof=1)),
        conditional_item_ci95=np.quantile(conditional,[.025,.975]).tolist(),
        crossed_seed_item_ci95=np.quantile(crossed,[.025,.975]).tolist(),
        conditional_item_permutation_p=p,permutations=permutations,bootstraps=bootstraps,
        n_seeds=n_seeds,n_items=n_items,stratum_sizes={str(s):int(sum(strata==s)) for s in np.unique(strata)},
        scope='P-value conditional on trained seed pairs and item-level label exchangeability. Crossed bootstrap is approximate with few seeds; not an exact training-population test.')
