"""Stable environment identity and deterministic environment-level splitting."""
from __future__ import annotations
import hashlib
import json
import random
from typing import Iterable

def environment_key(rows: int, cols: int, obstacles: Iterable[Iterable[int]]) -> str:
    payload={"rows":int(rows),"cols":int(cols),"obstacles":sorted([list(map(int,p)) for p in obstacles])}
    return json.dumps(payload,sort_keys=True,separators=(",",":"))

def environment_id(rows: int, cols: int, obstacles: Iterable[Iterable[int]]) -> str:
    return "ppnl_"+hashlib.sha256(environment_key(rows,cols,obstacles).encode()).hexdigest()

def base_sample_id(commit: str, source_file: str, source_index: int) -> str:
    raw=json.dumps([commit,source_file,int(source_index)],separators=(",",":"))
    return "base_"+hashlib.sha256(raw.encode()).hexdigest()

def split_environments(ids: Iterable[str], seed: int=42) -> dict[str,list[str]]:
    unique=sorted(set(ids)); random.Random(seed).shuffle(unique); n=len(unique)
    raw=[n*.7,n*.1,n*.2]; counts=[int(x) for x in raw]
    for idx in sorted(range(3),key=lambda i:(raw[i]-counts[i],-i),reverse=True)[:n-sum(counts)]: counts[idx]+=1
    a,b,c=counts
    return {"train":unique[:a],"dev":unique[a:a+b],"heldout_test":unique[a+b:a+b+c]}
