"""Summarize only complete factorial cells; never fill missing outputs with zero."""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/protocol_pilot_20261017'


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    expected={j['key']:j for j in manifest['jobs']}
    path=OUT/'rows.jsonl'
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').split('\n') if x.strip()] if path.exists() else []
    by={r['key']:r for r in rows}
    assert len(by)==len(rows), 'Duplicate request keys'
    assert set(by)<=set(expected), 'Unexpected request keys'
    grouped=defaultdict(list)
    for r in rows:
        e=expected[r['key']]
        assert all(r[k]==e[k] for k in e), 'Manifest mismatch'
        assert len(r['generated_ids'])==r['generated_tokens']
        grouped[(r['model'],r['prompt_style'],r['max_new_tokens'])].append(r)
    cells=[]
    for model in dict.fromkeys(j['model'] for j in manifest['jobs']):
        for prompt in ('compact','unconstrained'):
            for cap in (256,512):
                rr=grouped[(model,prompt,cap)]
                complete=len(rr)==16
                cells.append(dict(model=model,prompt=prompt,cap=cap,n=len(rr),expected_n=16,complete=complete,
                    correct=sum(r['passed_legacy_verifier'] for r in rr) if complete else None,
                    accuracy=float(np.mean([r['passed_legacy_verifier'] for r in rr])) if complete else None,
                    truncated=sum(r['stop_reason']=='length' for r in rr) if complete else None,
                    mean_generated_tokens=float(np.mean([r['generated_tokens'] for r in rr])) if complete else None,
                    mean_generation_seconds=float(np.mean([r['generation_seconds'] for r in rr])) if complete else None))
    report=dict(scope='Exploratory engineering pilot, not confirmatory or independent cross-family evidence',
        completed=len(rows),expected=len(expected),all_complete=len(rows)==len(expected),cells=cells,
        unknown_keys=sorted(set(expected)-set(by)))
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
