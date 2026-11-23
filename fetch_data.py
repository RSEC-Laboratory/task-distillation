"""Fetch pinned SuperNI source files for optional upstream revalidation."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT=Path(__file__).resolve().parent


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'external_data')
    args=parser.parse_args()
    pins=json.loads((ROOT/'external/superni.json').read_text(encoding='utf-8'))
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    for record in pins['tasks']:
        target=(output/record['path']).resolve()
        assert target.is_relative_to(output),'External data path escapes destination'
        if target.exists() and sha(target.read_bytes())==record['recorded_input_sha256']:
            print('Present:',record['path']);continue
        request=urllib.request.Request(record['fetch_url'],headers={'User-Agent':'research-artifact-reproduction'})
        with urllib.request.urlopen(request,timeout=120) as response:raw=response.read()
        assert sha(raw)==record['download_sha256'],'Pinned upstream download differs: '+record['path']
        recorded=raw
        if sha(recorded)!=record['recorded_input_sha256']:
            recorded=raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
        assert sha(recorded)==record['recorded_input_sha256'],'Recorded input normalization differs'
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(recorded)
        print('Fetched and verified:',record['path'],flush=True)
    print('Upstream data ready. Rerun python run.py to include optional source checks.')


if __name__=='__main__':main()
