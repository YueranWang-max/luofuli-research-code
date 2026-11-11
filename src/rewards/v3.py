"""Prospective v3 ablations. Historical reward code is intentionally untouched."""
from __future__ import annotations
from decimal import Decimal, InvalidOperation
import math
import re
import statistics

ARMS=('binary','success_length','grs_only','legacy_budget','centered_budget','centered_fixed','gr3_fixed')


def verify_final(text: str, reference: str) -> tuple[bool,str|None]:
    """Numeric terminal Answer contract, no accidental intermediate-number credit."""
    m=re.search(r'(?<!\w)Answer:\s*(-?\d+(?:\.\d+)?)\s*$',text,re.I)
    if not m:return False,None
    try:
        a,b=Decimal(m[1]),Decimal(reference.strip())
        if not(a.is_finite() and b.is_finite()):return False,m[1]
        # References are finite plain decimals; require exact numeric equality.
        return a==b,m[1]
    except InvalidOperation:return False,m[1]


def reward_group(passed: list[bool], tokens: list[int], arm: str, cap: int=768) -> dict:
    if arm not in ARMS:raise ValueError(arm)
    if len(passed)!=len(tokens) or not tokens:raise ValueError('invalid group')
    if any(t<1 for t in tokens):raise ValueError('invalid token count')
    n=len(tokens); good=[i for i,p in enumerate(passed) if p]; k=len(good)
    grs=[(.859+.135*24/(24+t)) if p else 0. for p,t in zip(passed,tokens)]
    delta=[0.]*n
    if arm=='binary':r=[float(p) for p in passed]
    elif arm=='success_length':r=[(1-.1*min(t/cap,1)) if p else 0. for p,t in zip(passed,tokens)]
    elif arm=='gr3_fixed':
        # GR^3 Eq.15, alpha .33 from its mathematical-reasoning recipe.
        # All-correct groups contribute zero advantage (Appendix C.2).
        # Controlled fixed-coefficient adaptation: no claim of full calibrated reproduction.
        mean_length=statistics.fmean(tokens)
        r=[(1/(1+.33*t/mean_length)) if p else 0. for p,t in zip(passed,tokens)] if k<n else [1.]*n
    else:
        r=grs.copy()
        if arm!='grs_only' and k>=2:
            costs=[tokens[i] for i in good]; lo,hi=min(costs),max(costs)
            if hi>lo:
                eff=[1/(.5*(t-lo)/(hi-lo)+1e-6) for t in costs]
                z=[(x-min(eff))/(max(eff)-min(eff)) for x in eff]
                center=.5 if arm=='legacy_budget' else statistics.fmean(z)
                lam=.3 if arm=='centered_fixed' else .3*k/n
                for i,v in zip(good,z):delta[i]=lam*(v-center);r[i]+=delta[i]
    sd=statistics.stdev(r) if n>1 else 0.
    avg=statistics.fmean(r)
    advantage=[(v-avg)/(sd+1e-4) for v in r]
    return dict(rewards=r,grs=grs,delta=delta,successes=k,reward_sd=sd,
        normalized_advantage=advantage,delta_sum=sum(delta))
