"""Publish only validated pretraining additions, preserving every prior file."""
import ast,hashlib,json,os,subprocess,sys
from pathlib import Path,PurePosixPath
ROOT=Path(os.environ['GITHUB_WORKSPACE']).resolve()
TEMP=Path(os.environ['RUNNER_TEMP']).resolve()
TARGET=TEMP/'tridim-pretraining-check'
STATE=TEMP/'tridim-pretraining-state.json'
WF='.github/workflows/add-pretraining.yml'
HELPER='.github/workflows/add_pretraining.py'
PAYLOAD='.github/workflows/pretraining_payload.json'
def git(*args,data=None,env=None):return subprocess.check_output(['git',*args],cwd=ROOT,input=data,env=env)
def prepare():
    payload=json.loads((ROOT/PAYLOAD).read_text())
    head=git('rev-parse','HEAD').decode().strip();base=git('rev-parse','HEAD^').decode().strip()
    assert base.startswith(payload['expected_base_prefix']),('Unexpected base',base)
    assert set(git('diff','--name-only',base,head).decode().splitlines())=={WF,HELPER,PAYLOAD}
    before=set(git('ls-tree','-r','--name-only',base).decode().splitlines())
    assert not any(p.startswith('pretraining/') for p in before)
    TARGET.mkdir(exist_ok=False)
    for rel,content in payload['files'].items():
        p=PurePosixPath(rel)
        assert rel.startswith('pretraining/') and not p.is_absolute() and '..' not in p.parts and '.git' not in p.parts and '\\' not in rel
        target=TARGET/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content.encode('utf-8'))
        if target.suffix=='.py':ast.parse(content,filename=rel)
    (TARGET/'requirements-runtime.txt').write_bytes((ROOT/'requirements-runtime.txt').read_bytes())
    subprocess.run([sys.executable,str(TARGET/'pretraining/verify.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_reported_results.py')],check=True)
    STATE.write_text(json.dumps({'head':head,'base':base,'before':sorted(before),'added':sorted(payload['files'])}))
    print('Prepared additive pretraining tree; all existing Full files are retained.')
def commit():
    s=json.loads(STATE.read_text());assert git('ls-remote','origin','refs/heads/main').decode().split()[0]==s['head'],'main changed; refusing to overwrite'
    subprocess.run([sys.executable,str(TARGET/'pretraining/verify.py')],check=True)
    env=os.environ.copy();env['GIT_INDEX_FILE']=str(TEMP/'tridim-pretraining-index')
    git('read-tree',s['head'],env=env)
    git('update-index','--force-remove',HELPER,PAYLOAD,env=env)
    for rel in s['added']:
        blob=git('hash-object','-w','--stdin',data=(TARGET/rel).read_bytes()).decode().strip()
        git('update-index','--add','--cacheinfo','100644,'+blob+','+rel,env=env)
    tree=git('write-tree',env=env).decode().strip()
    expected=set(s['before'])|set(s['added'])|{WF}
    assert set(git('ls-tree','-r','--name-only',tree).decode().splitlines())==expected
    changed=set(git('diff-tree','--no-commit-id','--name-only','-r',s['base'],tree).decode().splitlines())
    assert changed==set(s['added'])|{WF},changed
    assert git('show',tree+':README.md')==git('show',s['base']+':README.md')
    env.update(GIT_AUTHOR_NAME='github-actions[bot]',GIT_AUTHOR_EMAIL='41898282+github-actions[bot]@users.noreply.github.com',GIT_COMMITTER_NAME='github-actions[bot]',GIT_COMMITTER_EMAIL='41898282+github-actions[bot]@users.noreply.github.com')
    msg=b'Add audited TriDim pretraining integrations\n\nAdd CBraMod, CSBrain and REVE TriDim pretraining source overlays, explicit presets, launch/verification tools and archived convergence evidence. Preserve all root Full code, configs, logs and README. No weight binaries or EEG datasets. CSBrain reconstruction and historical provenance limits are documented.\n'
    new=git('commit-tree',tree,'-p',s['head'],data=msg,env=env).decode().strip()
    git('push','origin',new+':refs/heads/main')
    print('Published pretraining:',new,'; preserved',len(s['before']),'existing files; added',len(s['added']))
if __name__=='__main__':{'prepare':prepare,'commit':commit}[sys.argv[1]]()
