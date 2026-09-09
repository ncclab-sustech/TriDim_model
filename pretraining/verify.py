"""Verify published source hashes, presets and archived convergence coverage."""
import ast,csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    hashes=json.loads((ROOT/'sha256.json').read_text())
    for rel,digest in hashes.items():
        path=ROOT/rel
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,rel
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'),filename=rel)
    for rel,spec in json.loads((ROOT/'SOURCE_MANIFEST.json').read_text()).items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==spec['sha256'],rel
    from run import build_job
    for family,world,batch in [('cbramod',1,128),('csbrain',4,64),('reve',4,16)]:
        j=build_job(family,Path('/dry-data'),ROOT/'upstream',Path('/dry-output'))
        assert j['world_size']==world and j['per_gpu_batch_size']==batch
        assert Path(j['command'][6]).suffix=='.py'
    with (ROOT/'evidence/pretrain_convergence_data.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    keys={(r['host'],r['variant'],int(r['epoch'])) for r in rows}
    assert len(keys)==len(rows)==236
    for host in ['reve','cbramod','csbrain']:
        for variant in ['official','tridim']:
            epochs={e for h,v,e in keys if (h,v)==(host,variant)}
            expected=set(range(1,41)) if host=='reve' else (set(range(1,41))-{39} if host=='cbramod' else set(range(2,41)))
            assert epochs==expected,(host,variant)
    print(f'PASS {len(hashes)} file hashes, three launch presets, and 236 archived curve points (documented omissions retained).')
if __name__=='__main__':main()
