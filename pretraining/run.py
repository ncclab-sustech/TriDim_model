"""Portable launcher for the three retained TriDim pretraining integrations."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FAMILIES=['cbramod','csbrain','reve']
def build_job(family,data_root,upstream_root,output,gpus=None):
    preset=ROOT/'configs'/f'{family}_tridim_tuh2k.json'
    cfg=json.loads(preset.read_text());params=dict(cfg['params'])
    devices=gpus or [str(i) for i in range(cfg['historical_world_size'])]
    if not devices or len(set(devices))!=len(devices):raise ValueError('GPU identifiers must be nonempty and unique')
    project=upstream_root/cfg['project'];entry=project/cfg['entry']
    params['dataset_dir']=str(data_root.resolve())
    params['output_dir' if family=='reve' else 'model_dir']=str(output.resolve())
    if family=='reve':
        params['coord_csv']=str(ROOT/'overlays/cbramod/standard_1020_uppercase_ch_pos.csv')
        params['triaxis_module_dir']=str(upstream_root/'CBraMod')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc-per-node={len(devices)}',str(entry)]
    for key,value in params.items():command.extend(['--'+key,str(value)])
    command.extend('--'+flag for flag in cfg['flags'])
    pythonpath=[str(project/'src'),str(project)] if family=='reve' else [str(project)]
    return dict(family=family,command=command,cwd=str(project),gpus=devices,pythonpath=pythonpath,world_size=len(devices),historical_world_size=cfg['historical_world_size'],per_gpu_batch_size=params['batch_size'],global_batch_size=params['batch_size']*len(devices),preset_sha256=hashlib.sha256(preset.read_bytes()).hexdigest())
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--family',choices=FAMILIES,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--upstream-root',type=Path,default=ROOT/'upstream')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--gpus',nargs='+',help='Override historical GPU count; changes global batch size')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();upstream=a.upstream_root.resolve()
    output=(a.output_dir or ROOT.parent/'runs/pretraining'/a.family/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    job=build_job(a.family,a.data_root,upstream,output,a.gpus)
    print(json.dumps(job,indent=2),flush=True)
    if a.dry_run:return 0
    if not a.data_root.is_dir():p.error('H5 data root does not exist')
    if not Path(job['command'][6]).is_file():p.error('Prepared entry is missing; run pretraining/prepare_upstreams.py first')
    output.mkdir(parents=True,exist_ok=False)
    (output/'launch.json').write_text(json.dumps(job,indent=2)+'\n')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(job['gpus']);env['PYTHONUNBUFFERED']='1'
    env['PYTHONPATH']=os.pathsep.join(job['pythonpath']+[env.get('PYTHONPATH','')])
    with (output/'train.log').open('w') as log:
        result=subprocess.run(job['command'],cwd=job['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
    (output/'exit_code.txt').write_text(str(result.returncode)+'\n')
    print(f"Finished with exit {result.returncode}; logs: {output/'train.log'}")
    return result.returncode
if __name__=='__main__':raise SystemExit(main())
