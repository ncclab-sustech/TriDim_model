"""One-time, non-force migration to the reviewed paper Full tree."""
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(os.environ['GITHUB_WORKSPACE']).resolve()
TEMP = Path(os.environ['RUNNER_TEMP']).resolve()
TARGET = TEMP / 'tridim-flat-full-validated'
STATE = TEMP / 'tridim-flat-full-state.json'
WORKFLOW = '.github/workflows/flatten-paper-full.yml'
PAYLOAD = ROOT / '.github/workflows/paper_full_migration.json'

def git(*args, data=None, env=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, input=data, env=env)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def safe_path(rel):
    p = PurePosixPath(rel)
    assert not p.is_absolute() and '..' not in p.parts and '.git' not in p.parts
    assert '\\' not in rel and '\n' not in rel and '\0' not in rel
    return p

def prepare():
    payload = json.loads(PAYLOAD.read_text())
    head = git('rev-parse', 'HEAD').decode().strip()
    base = git('rev-parse', 'HEAD^').decode().strip()
    assert base.startswith(payload['expected_base_prefix']), ('Unexpected base', base)
    added = set(git('diff', '--name-only', base, head).decode().splitlines())
    assert added == {WORKFLOW, '.github/workflows/apply_paper_full.py', '.github/workflows/paper_full_migration.json'}, added
    source = ROOT / 'TriDim_paper_811_full_5770'
    for rel, digest in payload['source_hashes'].items():
        assert sha((source / safe_path(rel)).read_bytes()) == digest, rel
    source_files = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}
    assert source_files == set(payload['source_hashes']) | {'release_sha256.json'}
    readme = git('show', head + ':README.md')
    TARGET.mkdir(exist_ok=False)
    for rel, content in payload['files'].items():
        path = TARGET / safe_path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode('utf-8'))
        if path.suffix == '.py':
            ast.parse(content, filename=rel)
    (TARGET / 'README.md').write_bytes(readme)
    hashes = json.loads((TARGET / 'release_sha256.json').read_text())
    hashes['README.md'] = sha(readme)
    (TARGET / 'release_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    protected = ['models/eeg_mixer_v11_1_spatial_multilevel.py'] + [str(p.relative_to(source)) for p in (source / 'configs').rglob('*') if p.is_file()]
    for rel in protected:
        assert (TARGET / rel).read_bytes() == (source / rel).read_bytes(), ('Changed scientific source', rel)
    desired = {p.relative_to(TARGET).as_posix() for p in TARGET.rglob('*') if p.is_file()}
    assert len(desired) == 103
    assert len(list((TARGET / 'logs/full').glob('*.log'))) == 24
    assert len(list((TARGET / 'models').glob('*.py'))) == 2
    assert not any(p.startswith(('TriDim_paper_', 'checkpoints/', 'configs.bak/', 'results/', 'CSBrain_pretrain/', 'cbramod_pretrain/')) for p in desired)
    STATE.write_text(json.dumps({'head': head, 'readme_sha256': sha(readme), 'desired': sorted(desired)}))
    print('Prepared 103 final files; 24 Full logs; root README unchanged; original Full model/config/splits unchanged.')

def commit():
    state = json.loads(STATE.read_text())
    assert git('rev-parse', 'HEAD').decode().strip() == state['head']
    remote = git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0]
    assert remote == state['head'], 'main changed during validation; refusing to overwrite'
    assert sha((TARGET / 'README.md').read_bytes()) == state['readme_sha256']
    subprocess.run([sys.executable, str(TARGET / 'scripts/verify_reported_results.py')], cwd=TARGET, check=True)
    env = os.environ.copy()
    env['GIT_INDEX_FILE'] = str(TEMP / 'tridim-flat-full-index')
    git('read-tree', '--empty', env=env)
    paths = state['desired'] + [WORKFLOW]
    for rel in paths:
        content = (ROOT / rel).read_bytes() if rel == WORKFLOW else (TARGET / rel).read_bytes()
        blob = git('hash-object', '-w', '--stdin', data=content).decode().strip()
        git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + rel, env=env)
    tree = git('write-tree', env=env).decode().strip()
    assert set(git('ls-tree', '-r', '--name-only', tree).decode().splitlines()) == set(paths)
    assert git('show', tree + ':README.md') == git('show', state['head'] + ':README.md')
    env.update(GIT_AUTHOR_NAME='github-actions[bot]', GIT_AUTHOR_EMAIL='41898282+github-actions[bot]@users.noreply.github.com', GIT_COMMITTER_NAME='github-actions[bot]', GIT_COMMITTER_EMAIL='41898282+github-actions[bot]@users.noreply.github.com')
    message = b'Keep only paper 8:1:1 Full code and logs at repository root\n\nPreserve root README. Remove legacy 4:3:3 code, backups, pretraining placeholders, checkpoints and ablations. Retain eight original Full configs, six exact split manifests and 24 hash-verified archived logs. No new full training.\n'
    commit_id = git('commit-tree', tree, '-p', state['head'], data=message, env=env).decode().strip()
    git('push', 'origin', commit_id + ':refs/heads/main')
    print('Pushed verified root tree:', commit_id)
    print('The one-time workflow is retained unchanged for permission compatibility; remove it via the GitHub UI next.')

if __name__ == '__main__':
    {'prepare': prepare, 'commit': commit}[sys.argv[1]]()
