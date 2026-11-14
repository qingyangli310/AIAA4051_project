from src.data.environment import split_environments

def test_seed_42_maximum_remainder_split_and_no_leakage():
    ids=[f'e{i}' for i in range(101)]
    a=split_environments(ids,42);b=split_environments(ids,42)
    assert a==b
    assert {k:len(v) for k,v in a.items()}=={'train':71,'dev':10,'heldout_test':20}
    assert not(set(a['train'])&set(a['dev']) or set(a['train'])&set(a['heldout_test']) or set(a['dev'])&set(a['heldout_test']))
    assert set.union(*(set(v) for v in a.values()))==set(ids)
