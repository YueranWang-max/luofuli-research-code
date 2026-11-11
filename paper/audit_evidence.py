"""Versioned post-hoc audit. Never writes legacy inputs or re-labels them in place."""
from __future__ import annotations
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from environments.math_env import MathVerifier
from rewards.groupwise import _centered_unit

OUT = ROOT / 'artifacts/paper_audit_20261017'
SEEDS = (42, 123, 2026)
ARMS = ('e0', 'e2', 'e3', 'e5', 'e6')
MANIFEST = {}


def read(path):
    path = ROOT / path
    raw = path.read_bytes()
    MANIFEST[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8-sig')


def rows(path):
    return [json.loads(x) for x in read(path).split('\n') if x.strip()]


def question(prompt):
    return re.sub(r'\s+', ' ', re.split(r'\n\n(?:Solve this problem|Reason briefly)', prompt)[0]).strip()


def strict_answer(text, reference):
    # Diagnostic of the specified output contract, not a semantic gold standard.
    match = re.search(r"(?<!\w)Answer:\s*(-?\d+(?:\.\d+)?)\s*$", text, re.I)
    return bool(match and MathVerifier().verify('Answer: ' + match[1], reference).passed)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in [Path(__file__), ROOT/'src/rewards/groupwise.py', ROOT/'src/train/run_grpo.py',
              ROOT/'src/environments/math_env.py', ROOT/'paper/EXPERIMENT_PROTOCOL_20261017.md']:
        read(p)
    ds = rows('artifacts/benchmarks/math500/test.jsonl')
    ids = sorted(r['task_id'] for r in ds)
    refs = {r['task_id']: r for r in ds}
    tok = AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3-4B-Instruct'), local_files_only=True)
    for p in (ROOT/'models/Qwen3-4B-Instruct').glob('*token*.json'):
        read(p)
    arrays, summary, disagreements = {}, {}, []
    names = ['e0_math500_256', 'e0_math500', 'e0_distilled']
    names += [f'mathhard_{a}_s{s}' for a in ARMS[1:] for s in SEEDS]
    verifier = MathVerifier()
    derived = []
    for name in names:
        rr = rows(f'artifacts/bench_rows/{name}.jsonl')
        by_id = {r['task_id']: r for r in rr}
        assert len(rr) == len(by_id) == len(ids) and set(by_id) == set(ids), name
        rr = [by_id[i] for i in ids]
        lens = [len(x) for x in tok([r['completion'] for r in rr], add_special_tokens=False)['input_ids']]
        y = np.array([int(r['passed']) for r in rr])
        arrays[name] = y
        strict = []
        for r, length in zip(rr, lens):
            ref = refs[r['task_id']]['reference_answer']
            rescored = verifier.verify(r['completion'], ref)
            strict_pass = strict_answer(r['completion'], ref)
            strict.append(strict_pass)
            if rescored.passed != r['passed'] or strict_pass != r['passed']:
                disagreements.append(dict(run=name, task_id=r['task_id'], reference=ref,
                    legacy_pass=r['passed'], recheck_pass=rescored.passed, extracted=rescored.extracted,
                    strict_format_pass=strict_pass, completion=r['completion']))
            derived.append(dict(run=name, task_id=r['task_id'], passed=r['passed'],
                whitespace_units=r['tokens'], reencoded_text_tokens=length, strict_pass=strict_pass))
        lens = np.array(lens)
        summary[name] = dict(n=len(rr), correct=int(y.sum()), accuracy=float(y.mean()),
            strict_format_correct=sum(strict),
            legacy_recheck_mismatches=sum(verifier.verify(r['completion'], refs[r['task_id']]['reference_answer']).passed != r['passed'] for r in rr),
            whitespace_matches=sum(r['tokens'] == max(1, len(r['completion'].split())) for r in rr),
            mean_whitespace=float(np.mean([r['tokens'] for r in rr])),
            mean_reencoded_tokens=float(lens.mean()),
            mean_reencoded_success=float(lens[y == 1].mean()) if y.sum() else None,
            total_reencoded_per_success=float(lens.sum()/y.sum()) if y.sum() else None,
            levels={level: float(np.mean([r['passed'] for r in rr if refs[r['task_id']]['difficulty'] == level])) for level in ('L4','L5')})
    def name(arm, seed):
        return 'e0_math500_256' if arm == 'e0' else f'mathhard_{arm}_s{seed}'
    tests, effects = [], []
    rng = np.random.default_rng(20261017)
    draws = rng.integers(0, len(ids), size=(20000, len(ids)))
    for b, a in itertools.combinations(ARMS, 2):
        diffs = []
        for s in SEEDS:
            va, vb = arrays[name(a,s)], arrays[name(b,s)]
            wins, losses = int(((va==1)&(vb==0)).sum()), int(((va==0)&(vb==1)).sum())
            n = wins + losses
            tests.append(dict(a=a, b=b, seed=s, wins=wins, losses=losses,
                delta=float(np.mean(va-vb)), p=float(binomtest(wins,n).pvalue) if n else 1.0))
            diffs.append(va-vb)
        d = np.mean(diffs, axis=0)
        ci = np.quantile(d[draws].mean(axis=1), [.025,.975])
        effects.append(dict(a=a,b=b,delta=float(d.mean()), ci95=ci.tolist(),
            interpretation='paired item bootstrap of mean over the three fixed seeds; excludes seed-population uncertainty'))
    ordered = sorted(range(len(tests)), key=lambda i: tests[i]['p'])
    prev = 0.0
    for rank, i in enumerate(ordered):
        prev = max(prev, min(1., tests[i]['p']*(len(tests)-rank)))
        tests[i]['p_holm_30'] = prev
    sets = {}
    for bench in ('mathhard','mathtrain','math500','gsm8k'):
        for split in ('train','val','test'):
            p = f'artifacts/benchmarks/{bench}/{split}.jsonl'
            if (ROOT/p).exists():
                rr = rows(p)
                sets[f'{bench}/{split}'] = [question(r['prompt']) for r in rr]
    overlap = []
    for (a, aa),(b,bb) in itertools.combinations(sets.items(),2):
        common = sorted(set(aa)&set(bb))
        if common:
            overlap.append(dict(a=a,b=b,n=len(common),example=common[:2]))
    training = {}
    for arm in ARMS[1:]:
        for s in SEEDS:
            run = name(arm,s)
            state_path = f'artifacts/bench_ckpt/{run}/checkpoint-100/trainer_state.json'
            st = json.loads(read(state_path))
            logs = [r for r in st['log_history'] if 'frac_reward_zero_std' in r]
            stats = rows(f'artifacts/bench_reward_stats/{run}.jsonl')
            args_path = ROOT/f'artifacts/bench_ckpt/{run}/checkpoint-100/training_args.bin'
            MANIFEST[str(args_path.relative_to(ROOT))] = hashlib.sha256(args_path.read_bytes()).hexdigest()
            saved_args = torch.load(args_path,weights_only=False,map_location='cpu')
            keys = ('frac_reward_zero_std','completions/clipped_ratio','completions/mean_length','grad_norm')
            training[run] = dict(global_step=st['global_step'],logged_steps=len(logs),
                saved_config={k:getattr(saved_args,k,None) for k in ('loss_type','scale_rewards','max_steps','max_completion_length','num_generations','seed','beta')},
                reward_callback_records=len(stats), logged_groups=sum(r['n_groups'] for r in stats),
                metric_means={k:float(np.mean([r[k] for r in logs if k in r])) for k in keys},
                note='trainer-state means are descriptive; callback file may contain earlier attempts')
    constructed = []
    for n in (2,3,4):
        sums = [sum(_centered_unit(list(x))) for x in itertools.product((0.,.1,.5,1.),repeat=n)]
        constructed.append(dict(n_success=n,n_cases=len(sums),nonzero_sum=sum(abs(v)>1e-10 for v in sums),max_abs_sum=max(abs(v) for v in sums)))
    report = dict(summary=summary,pairwise=tests,conditional_effects=effects,overlap=overlap,
        duplicate_question_counts={k:len(v)-len(set(v)) for k,v in sets.items()},training=training,
        constructed_zero_sum_check=constructed,
        scope='Post-hoc audit; reencoded tokens omit generation-only IDs; strict scoring is a format sensitivity diagnostic.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    for filename, data in [('disagreements.jsonl',disagreements),('derived_rows.jsonl',derived)]:
        (OUT/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
    (OUT/'input_manifest.json').write_text(json.dumps(MANIFEST,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'overlap':overlap,'zero_sum':constructed,'effects':effects},indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
