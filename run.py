"""Verify the release, reconstruct large records, and run offline reproduction."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def verify_manifest():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        path=(ROOT/record['path']).resolve()
        assert path.is_relative_to(ROOT.resolve()),'Manifest path escapes release'
        assert path.is_file(),f'Missing release file: {record["path"]}'
        assert path.stat().st_size==record['bytes'],f'File size changed: {record["path"]}'
        assert digest(path)==record['sha256'],f'File hash changed: {record["path"]}'
    return manifest


def materialize(destination,manifest):
    destination.mkdir(parents=True,exist_ok=True)
    base=destination.resolve()
    for record in manifest['files']:
        relative=record['path']
        if not relative.startswith('task-compiler/'):continue
        target=(destination/relative).resolve()
        assert target.is_relative_to(base),'Workspace path escapes destination'
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    parts=json.loads((ROOT/'DATA_PARTS.json').read_text(encoding='utf-8'))
    for record in parts['files']:
        target=(destination/record['path']).resolve()
        assert target.is_relative_to(base),'Chunk destination escapes workspace'
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('wb') as output:
            for part in record['parts']:
                path=ROOT/part['path']
                assert digest(path)==part['sha256']
                with path.open('rb') as source:shutil.copyfileobj(source,output)
        assert target.stat().st_size==record['bytes'] and digest(target)==record['sha256']
    external=ROOT/'external_data'
    if external.exists():
        pins=json.loads((ROOT/'external/superni.json').read_text(encoding='utf-8'))
        for record in pins['tasks']:
            source=external/record['path']
            if not source.is_file():continue
            assert digest(source)==record['recorded_input_sha256'],'Downloaded upstream data changed'
            target=destination/'task-compiler/data/superni'/record['path']
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    return destination/'task-compiler'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--materialize-only',action='store_true',help='Prepare the complete working tree without running calculations.')
    args=parser.parse_args()
    output=args.output.resolve()
    if output==ROOT or output==ROOT/'task-compiler':raise ValueError('Use a separate output directory')
    manifest=verify_manifest()
    print(f'Checked {len(manifest["files"])} release files.',flush=True)
    working=materialize(output/'workspace',manifest)
    print('Reconstructed working tree:',working,flush=True)
    if args.materialize_only:return
    results=output/'results';results.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-B','-S',str(ROOT/'check_core.py'),'--root',str(working),'--output',str(results/'core_checks.json')],check=True)
    subprocess.run([sys.executable,'-B','-S',str(ROOT/'reproduce.py'),'--root',str(working),'--output',str(results)],check=True)
    print('Offline reproduction completed. Results:',results,flush=True)


if __name__=='__main__':main()
