"""Validate and atomically publish the scoped TriDim identifier rename."""
import ast,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path,PurePosixPath
ROOT=Path(os.environ['GITHUB_WORKSPACE']).resolve()
TEMP=Path(os.environ['RUNNER_TEMP']).resolve()
TARGET=TEMP/'tridim-rename-check'
STATE=TEMP/'tridim-rename-state.json'
WF='.github/workflows/rename-tridim.yml'
HELPER='.github/workflows/rename_model.py'
PAYLOAD='.github/workflows/rename_payload.json'
def git(*args,data=None,env=None):return subprocess.check_output(['git',*args],cwd=ROOT,input=data,env=env)
def prepare():
    payload=json.loads((ROOT/PAYLOAD).read_text(encoding='utf-8'))
    head=git('rev-parse','HEAD').decode().strip();base=payload['expected_base']
    assert git('merge-base',base,head).decode().strip()==base
    assert set(git('diff','--name-only',base,head).decode().splitlines())=={WF,HELPER,PAYLOAD}
    before=set(git('ls-tree','-r','--name-only',base).decode().splitlines())
    TARGET.mkdir(exist_ok=False)
    for rel in before:
        p=TARGET/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p)
    for rel,digest in payload['before_hashes'].items():
        p=PurePosixPath(rel)
        assert not p.is_absolute() and '..' not in p.parts and '.git' not in p.parts and '\\' not in rel
        assert rel!='README.md' and not rel.startswith('logs/') and rel!='reported/full_24_runs.json'
        if digest is None:assert rel not in before
        else:assert hashlib.sha256((TARGET/rel).read_bytes()).hexdigest()==digest,rel
    assert set(payload['delete'])==set(payload['renames'])
    for old,new in payload['renames'].items():assert (TARGET/old).read_bytes()==payload['files'][new].encode('utf-8')
    for rel in payload['delete']:
        p=(TARGET/rel).resolve();assert p.is_relative_to(TARGET)
        p.unlink()
    for rel,content in payload['files'].items():
        assert rel in payload['before_hashes']
        p=TARGET/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode('utf-8'))
        if p.suffix=='.py':ast.parse(content,filename=rel)
    for script in ['scripts/verify_reported_results.py','pretraining/verify.py']:
        subprocess.run([sys.executable,str(TARGET/script)],cwd=TARGET,check=True)
    s={'head':head,'base':base,'before':sorted(before),'changed':sorted(payload['files']),'deleted':payload['delete']}
    STATE.write_text(json.dumps(s),encoding='utf-8')
    print('Prepared validated naming changes; model implementation, historical logs and metrics unchanged.')
def commit():
    s=json.loads(STATE.read_text(encoding='utf-8'))
    assert git('ls-remote','origin','refs/heads/main').decode().split()[0]==s['head'],'main changed'
    for script in ['scripts/verify_reported_results.py','pretraining/verify.py']:
        subprocess.run([sys.executable,str(TARGET/script)],cwd=TARGET,check=True)
    env=os.environ.copy();env['GIT_INDEX_FILE']=str(TEMP/'tridim-rename-index')
    git('read-tree',s['head'],env=env)
    git('update-index','--force-remove',HELPER,PAYLOAD,*s['deleted'],env=env)
    for rel in s['changed']:
        blob=git('hash-object','-w','--stdin',data=(TARGET/rel).read_bytes()).decode().strip()
        git('update-index','--add','--cacheinfo','100644,'+blob+','+rel,env=env)
    tree=git('write-tree',env=env).decode().strip()
    expected=(set(s['before'])-set(s['deleted']))|set(s['changed'])|{WF}
    assert set(git('ls-tree','-r','--name-only',tree).decode().splitlines())==expected
    changed=set(git('diff-tree','--no-commit-id','--name-only','-r',s['base'],tree).decode().splitlines())
    assert changed==set(s['changed'])|set(s['deleted'])|{WF},changed
    assert git('show',tree+':README.md')==git('show',s['base']+':README.md')
    env.update(GIT_AUTHOR_NAME='github-actions[bot]',GIT_AUTHOR_EMAIL='41898282+github-actions[bot]@users.noreply.github.com',GIT_COMMITTER_NAME='github-actions[bot]',GIT_COMMITTER_EMAIL='41898282+github-actions[bot]@users.noreply.github.com')
    msg=b'Rename Full model module and identifier to tridim\n\nUpdate Full configs, imports, defaults, pretraining integrations and source checksums. Preserve model implementation bytes, training settings, archived metrics, logs and root README. Map historical result identifiers explicitly.\n'
    new=git('commit-tree',tree,'-p',s['head'],data=msg,env=env).decode().strip()
    git('push','origin',new+':refs/heads/main')
    print('Published TriDim rename:',new)
if __name__=='__main__':{'prepare':prepare,'commit':commit}[sys.argv[1]]()
