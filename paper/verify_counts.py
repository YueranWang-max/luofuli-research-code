"""Read-only label aggregation audit; does not rerun inference or validate grading."""
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
master = json.loads((root / 'artifacts/master_table_v2.json').read_text(encoding='utf-8'))
reference_ids = None
for arm, spec in master['arms'].items():
    scores = []
    for seed in spec['per_seed']:
        path = root / 'artifacts/bench_rows' / (seed['name'] + '.jsonl')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        ids = [row['task_id'] for row in rows]
        assert len(rows) == len(set(ids)) == 158, path
        if reference_ids is None:
            reference_ids = set(ids)
        assert set(ids) == reference_ids, path
        assert all(type(row['passed']) is bool for row in rows), path
        correct = sum(row['passed'] for row in rows)
        score = correct / len(rows)
        assert abs(score - seed['pass_at_1']) < 1e-12, path
        scores.append(score)
        print(arm, seed['name'], correct, '/', len(rows))
    assert abs(statistics.mean(scores) - spec['pass_mean']) < 1e-12
    if len(scores) > 1:
        assert abs(statistics.stdev(scores) - spec['pass_std']) < 1e-12
print('PASS: counts, unique/shared IDs, means and sample SDs match the master table.')
