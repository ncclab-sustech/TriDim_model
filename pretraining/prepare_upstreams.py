"""Obtain exact upstream revisions and apply only the pretraining overlays."""
import argparse,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAMES={'cbramod':'CBraMod','csbrain':'CSBrain','reve':'reve_eeg'}
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--destination',type=Path,default=ROOT/'upstream')
    p.add_argument('--families',nargs='+',choices=NAMES,default=list(NAMES))
    a=p.parse_args();manifest=json.loads((ROOT/'upstream.json').read_text())
    a.destination.mkdir(parents=True,exist_ok=True)
    for family in a.families:
        spec=manifest[family];target=a.destination/NAMES[family]
        if target.exists():
            raise FileExistsError(f'{target} exists; use a new destination to preserve any local changes')
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',spec['url'],str(target)],check=True)
        subprocess.run(['git','checkout','--detach',spec['commit']],cwd=target,check=True)
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()
        assert actual==spec['commit']
        shutil.copytree(ROOT/spec['overlay'],target,dirs_exist_ok=True)
        print(f'Prepared {family}: {actual}')
if __name__=='__main__':main()
