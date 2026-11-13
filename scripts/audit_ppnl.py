"""Audit seven local PPNL files; save metadata/issues and at most 8 converted samples."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.load_ppnl import FILES,REQUIRED,load_ppnl

def audit(source,output,samples):
    stats=[]; examples=[]; issues=[]
    for name in FILES:
        p=source/name
        if not p.exists():
            stats.append({'file':name,'exists':False});continue
        try:
            raw=json.loads(p.read_text()); loaded=load_ppnl(p)
        except (ValueError,TypeError) as error:
            stats.append({'file':name,'exists':True,'parse_success':False,'error':str(error)});continue
        missing=sum(not isinstance(r,dict) or any(k not in r for k in REQUIRED) for r in raw)
        stats.append({'file':name,'exists':True,'parse_success':True,'records':loaded['total'],
                      'missing_required_fields':missing,'conversion_issues':len(loaded['issues']),
                      'converted':len(loaded['records']),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        issues.extend(loaded['issues'])
        examples.extend(loaded['records'][:1])
    output.write_text(json.dumps({'files':stats,'issues':issues},indent=2)+'\n')
    samples.write_text(''.join(json.dumps(r)+'\n' for r in examples[:8]))
    print(json.dumps(stats,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=Path,required=True)
    args=p.parse_args();audit(args.source,args.output,args.samples)
