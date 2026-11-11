"""Constructed mechanism diagnostics; no claim of training performance."""
from __future__ import annotations
import itertools
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rewards.groupwise import _centered_unit


def centered_success_scores(values):
    """Candidate v3: zero mean and max absolute magnitude <= 0.5.

    Not a replacement for historical training. Mean-center before scaling;
    constant scores receive zero. This changes the old method's semantics.
    """
    x=np.asarray(values,dtype=float)
    if not len(x):
        return x
    if not np.isfinite(x).all():
        raise ValueError('non-finite score')
    z=x-x.mean()
    scale=np.max(np.abs(z))
    return np.zeros_like(z) if scale<1e-12 else z/(2*scale)


def normalize(rewards,eps=1e-4):
    x=np.asarray(rewards,dtype=float)
    return (x-x.mean())/(x.std(ddof=1)+eps)


def main():
    cases=[]
    for k in (2,3,4):
        for scores in itertools.product((0.,.1,.5,1.),repeat=k):
            old=np.array(_centered_unit(list(scores)))
            new=centered_success_scores(scores)
            assert abs(new.sum())<1e-10
            assert np.max(np.abs(new))<=.5+1e-10
            assert all(new[i]<=new[j]+1e-10 for i in range(k) for j in range(k) if scores[i]<=scores[j])
            cases.append(dict(k=k,scores=scores,legacy_sum=float(old.sum()),corrected_sum=float(new.sum())))
    # All-success pure binary+shaping: amplitude cancels with zero epsilon.
    z=centered_success_scores([0.,.1,.5,1.])
    scales=(.01,.05,.1,.3)
    curves=[]
    for scale in scales:
        a=normalize(1+scale*z)
        exact=normalize(1+scale*z,eps=0)
        assert np.allclose(exact,normalize(z,eps=0))
        curves.append(dict(scale=scale,advantages=a.tolist(),norm=float(np.linalg.norm(a))))
    mixed=[]
    for scale in scales:
        reward=np.concatenate([1+scale*centered_success_scores([0.,.1,1.]),[0.]])
        mixed.append(dict(scale=scale,reward=reward.tolist(),advantages=normalize(reward).tolist()))
    out=ROOT/'artifacts/paper_audit_20261017/reward_mechanism_probe.json'
    out.write_text(json.dumps(dict(scope='Constructed cases, not rollouts or empirical improvement',
        tested_cases=len(cases),zero_sum_assertions_passed=True,all_success_scaling=curves,
        mixed_success_scaling=mixed,cases=cases),indent=2),encoding='utf-8')
    print(json.dumps(dict(tested_cases=len(cases),all_success_scaling=curves),indent=2))


if __name__=='__main__':
    main()
