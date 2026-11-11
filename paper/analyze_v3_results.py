"""Analyze the complete frozen study only; never silently drop failed/missing runs."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from stats_v3 import holm, paired_effect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rewards.v3 import verify_final
from evaluation.answer_contracts import verify_formatted_final
from environments.math_env import MathVerifier
OUT = ROOT / 'artifacts/v3'
SEEDS = (42, 123, 2026)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, records):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    frozen = read_json(OUT / 'training_manifest.json')
    assert len(frozen['jobs']) == 18
    assert read_json(OUT / 'training_completed.json')['completed_runs'] == 18
    assert read_json(OUT / 'evaluation_completed.json')['completed_evaluations'] == 20
    for relative, expected in frozen['hashes'].items():
        assert digest(ROOT / relative) == expected, f'Frozen source/data changed: {relative}'
    data = {name: rows(OUT / f'data/confirm_{name}.jsonl') for name in ('math', 'gsm')}
    ids = {name: [row['task_id'] for row in values] for name, values in data.items()}
    assert all(len(value) == len(set(value)) == 300 for value in ids.values())
    models = list(dict.fromkeys(job['model'] for job in frozen['jobs']))
    tokenizers={model:Tokenizer.from_file(str(ROOT/'models'/model/'tokenizer.json')) for model in models}
    eos_tokens={model:read_json(ROOT/'models'/model/'generation_config.json')['eos_token_id'] for model in models}
    eos_tokens={model:set(value if isinstance(value,list) else [value]) for model,value in eos_tokens.items()}
    references={row['task_id']:row['reference_answer'] for dataset in data.values() for row in dataset}
    legacy_verifier=MathVerifier()
    jobs = [dict(model=m, arm='base', seed=None) for m in models] + frozen['jobs']
    expected_keys = {f'{dataset}/{item}/{style}' for dataset in ids for item in ids[dataset]
                     for style in ('compact',)}
    outcomes, summaries, disagreement_rows, inputs = {}, [], [], {}
    for job in jobs:
        model, arm, seed = job['model'], job['arm'], job['seed']
        key = model + '_base' if arm == 'base' else f'{model}_{arm}_s{seed}'
        directory = OUT / 'evaluations' / key
        assert read_json(directory / 'completed.json')['n'] == 600, key
        records = rows(directory / 'rows.jsonl')
        by_key = {row['key']: row for row in records}
        assert len(records) == len(by_key) == 600 and set(by_key) == expected_keys, key
        manifest = read_json(directory / 'manifest.json')
        assert manifest['model'] == model and not manifest['development_smoke']
        assert manifest['adapter']==(None if arm=='base' else str((OUT/'runs'/key).resolve()))
        assert {row['key'] for row in manifest['jobs']} == expected_keys
        job_metadata={row['key']:row for row in manifest['jobs']}
        for row in records:
            meta=job_metadata[row['key']]
            assert row['reference_answer']==references[row['task_id']]==meta['reference_answer']
            assert row['prompt']==meta['prompt'] and row['prompt_style']==meta['prompt_style']
            assert {s['cap'] for s in row['scores']}=={256,512,1024} and len(row['scores'])==3
            generated=row['generated_ids']
            assert 0<len(generated)<=1024 and not any(t in eos_tokens[model] for t in generated[:-1])
            for score in row['scores']:
                part=generated[:score['cap']]
                decoded=tokenizers[model].decode(part,skip_special_tokens=True)
                assert decoded==score['text'],'Saved text differs from raw token IDs'
                assert score['stop_reason']==('eos' if part[-1] in eos_tokens[model] else 'length')
                expected=dict(strict_passed=verify_final(decoded,row['reference_answer'])[0],
                    formatted_final_passed=verify_formatted_final(decoded,row['reference_answer'])[0],
                    legacy_passed=legacy_verifier.verify(decoded,row['reference_answer']).passed)
                assert all(isinstance(score[name],bool) and score[name]==value for name,value in expected.items()),'Scoring mismatch'
        inputs[str(directory / 'rows.jsonl')] = digest(directory / 'rows.jsonl')
        inputs[str(directory / 'manifest.json')] = digest(directory / 'manifest.json')
        for dataset in ids:
            for style in ('compact',):
                selected = [by_key[f'{dataset}/{item}/{style}'] for item in ids[dataset]]
                for cap in (256, 512, 1024):
                    scores = [next(s for s in row['scores'] if s['cap'] == cap) for row in selected]
                    tokens = np.array([score['generated_tokens'] for score in scores])
                    assert np.all((tokens > 0) & (tokens <= cap))
                    assert all(s['generated_tokens'] == min(cap, len(row['generated_ids']))
                               for row, s in zip(selected, scores))
                    for metric in ('strict_passed', 'formatted_final_passed', 'legacy_passed'):
                        passed = np.array([score[metric] for score in scores], dtype=bool)
                        outcomes[model, arm, seed, dataset, style, cap, metric] = passed
                        correct = int(passed.sum())
                        summaries.append(dict(model=model, arm=arm, seed=seed, dataset=dataset,
                            prompt_style=style, cap=cap, metric=metric, n=300, correct=correct,
                            accuracy=float(passed.mean()), mean_tokens=float(tokens.mean()),
                            mean_success_tokens=float(tokens[passed].mean()) if correct else None,
                            mean_failure_tokens=float(tokens[~passed].mean()) if correct < 300 else None,
                            total_tokens_per_success=float(tokens.sum()/correct) if correct else None,
                            truncated=sum(s['stop_reason'] == 'length' for s in scores),
                            cost_scope='completion IDs including EOS; prefixes have no independent wall time'))
                    for row, score in zip(selected, scores):
                        if len({score[m] for m in ('strict_passed', 'formatted_final_passed', 'legacy_passed')}) > 1:
                            disagreement_rows.append(dict(run=key, key=row['key'], cap=cap,
                                reference_answer=row['reference_answer'], **{k:v for k,v in score.items() if k!='cap'}))
    # The only confirmatory family: two contrasts per model, fixed endpoint.
    primary = []
    for model in models:
        for comparator in ('binary', 'grs_only'):
            differences = np.stack([
                outcomes[model, 'centered_budget', seed, 'math', 'compact', 1024, 'strict_passed'].astype(float)
                - outcomes[model, comparator, seed, 'math', 'compact', 1024, 'strict_passed'].astype(float)
                for seed in SEEDS])
            result = paired_effect(differences, [row['difficulty'] for row in data['math']])
            primary.append(dict(model=model, treatment='centered_budget', comparator=comparator,
                                endpoint='math/compact/1024/strict_passed', **result))
    for result, adjusted in zip(primary, holm([r['conditional_item_permutation_p'] for r in primary])):
        result['holm_p_four_primary_comparisons'] = adjusted
    destination = OUT / 'analysis'
    destination.mkdir(exist_ok=False)
    write_csv(destination / 'all_seed_cells.csv', summaries)
    (destination / 'primary.json').write_text(json.dumps(primary, indent=2), encoding='utf-8')
    outcome_keys=list(outcomes)
    np.savez_compressed(destination / 'item_outcomes.npz',
                        **{f'cell_{i}':outcomes[key] for i,key in enumerate(outcome_keys)})
    (destination / 'item_outcome_keys.json').write_text(json.dumps(outcome_keys),encoding='utf-8')
    with (destination / 'scoring_disagreements.jsonl').open('w', encoding='utf-8') as handle:
        for row in disagreement_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    (destination / 'analysis_manifest.json').write_text(json.dumps(dict(
        input_hashes=inputs, script_sha256=digest(Path(__file__)),
        stats_sha256=digest(ROOT / 'paper/stats_v3.py'), n_runs=len(jobs),
        output_hashes={p.name:digest(p) for p in destination.iterdir() if p.is_file()},
        secondary_inference_status='Separate prespecified subset/contrast analysis still required'), indent=2), encoding='utf-8')
    print('Complete raw matrix and four primary contrasts exported; secondary analyses remain.')


if __name__ == '__main__':
    main()
