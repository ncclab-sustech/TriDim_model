"""Original Full protocol, isolated working directory for each dataset/seed."""
import argparse,concurrent.futures,datetime,json,os,queue,subprocess,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=json.loads((ROOT/'reported/protocol.json').read_text())

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',choices=list(PROTOCOL['configs']),default=list(PROTOCOL['configs']))
    p.add_argument('--gpus',nargs='+',default=['0'])
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if len(a.gpus)!=len(set(a.gpus)):p.error('GPU identifiers must be unique')
    if len(a.datasets)!=len(set(a.datasets)):p.error('Datasets must be unique')
    data=os.environ.get('TRIDIM_DATA_ROOT')
    if not data and not a.dry_run:p.error('Set TRIDIM_DATA_ROOT to the prepared dataset parent directory')
    output=(a.output_dir or ROOT/'runs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    if not a.dry_run:output.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for ds in a.datasets:
        original=ROOT/PROTOCOL['configs'][ds]
        for seed in PROTOCOL['seeds']:
            cfg=yaml.safe_load(original.read_text());params=cfg['params']
            assert params['model']==PROTOCOL['model'] and params['seeds']==[5,42,43]
            assert params['train_ratio']==.8 and params['val_ratio']==.1 and params['select_metric']=='Accuracy'
            target=output/f'{ds}_seed{seed}'
            cfg['root_path']=str(Path(cfg['root_path'].replace('${TRIDIM_DATA_ROOT}',data or '/path/to/preprocessed/eeg')).resolve())
            params.update(seeds=[seed],seed=seed,seed_start=seed,itr=1,gpu=0,gpu_idx=[0],use_multi_gpu=False)
            for mapping in [cfg,params]:
                for key in ['electrode_csv','canonical_channel_coord_path','input_channel_coord_path','external_split_manifest']:
                    if isinstance(mapping.get(key),str):
                        mapping[key]=str((ROOT/mapping[key].format(seed=seed)).resolve())
                        assert Path(mapping[key]).is_file(),mapping[key]
            command=[sys.executable,str(ROOT/'run.py'),'--model',PROTOCOL['model'],'--data',ds,'--dataset_paths_yaml',str(target/'config.yaml'),'--root_path',cfg['root_path'],'--seeds',str(seed),'--seed_start',str(seed),'--gpu','0','--gpu_idx','0']
            if a.dry_run:
                print(json.dumps({'dataset':ds,'seed':seed,'cwd':str(target),'config':str(original),'command':command}));continue
            assert Path(cfg['root_path']).is_dir(),cfg['root_path']
            target.mkdir();(target/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
            jobs.append((target,command))
    if a.dry_run:return 0
    pending=queue.Queue()
    for j in jobs:pending.put(j)
    def worker(gpu):
        failures=0
        while True:
            try:target,command=pending.get_nowait()
            except queue.Empty:return failures
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT)+os.pathsep+env.get('PYTHONPATH',''))
            (target/'command.json').write_text(json.dumps({'command':command,'cwd':str(target),'gpu':gpu},indent=2))
            with (target/'train.log').open('x') as log:
                result=subprocess.run(command,cwd=target,env=env,stdout=log,stderr=subprocess.STDOUT)
            (target/'exit_code.txt').write_text(str(result.returncode));failures+=int(result.returncode!=0)
            print(target.name,'exit',result.returncode,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        failed=sum(pool.map(worker,a.gpus))
    print(f'Finished {len(jobs)} runs; {failed} failed. Outputs: {output}')
    return int(failed>0)

if __name__=='__main__':raise SystemExit(main())
