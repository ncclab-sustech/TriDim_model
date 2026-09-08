"""Recompute the archived Full table; no training or EEG data required."""
import json,statistics,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
protocol=json.loads((root/'reported/protocol.json').read_text())
records=json.loads((root/'reported/full_24_runs.json').read_text())
expected={(d,s) for d in protocol['configs'] for s in [5,42,43]}
assert len(records)==24 and {(r['dataset'],r['seed']) for r in records}==expected
for r in records:
    assert r['model']==protocol['model'] and 0<=r['test_accuracy']<=1
means=[]
for ds in protocol['configs']:
    values=[r['test_accuracy']*100 for r in records if r['dataset']==ds]
    mean=statistics.mean(values);means.append(mean)
    print(f'{ds}: {mean:.2f} +/- {statistics.stdev(values):.2f}%')
macro=statistics.mean(means)
assert abs(macro-protocol['macro_accuracy_percent'])<1e-10
assert round(macro,2)==57.70
print(f'Full, eight-dataset macro average: {macro:.10f}% -> {macro:.2f}%')
hashes=root/'release_sha256.json'
if hashes.exists():
    for rel,digest in json.loads(hashes.read_text()).items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
    print('Release file hashes verified.')
