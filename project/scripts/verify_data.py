import pandas as pd
from pathlib import Path

COLS = ['unit','cycle','op1','op2','op3'] + [f's{i}' for i in range(1,22)]
EXPECTED = {
    1: dict(train_rows=20631, train_units=100, test_rows=13096, test_units=100, rul_lines=100),
    2: dict(train_rows=53759, train_units=260, test_rows=33991, test_units=259, rul_lines=259),
    3: dict(train_rows=24720, train_units=100, test_rows=16596, test_units=100, rul_lines=100),
    4: dict(train_rows=61249, train_units=249, test_rows=41214, test_units=248, rul_lines=248),
}
raw = Path(r'h:\clint-paper\paper_work\CMAPSSData')
all_ok = True
for s in [1,2,3,4]:
    tag = f'FD00{s}'
    tr  = pd.read_csv(raw / f'train_{tag}.txt', sep=r'\s+', header=None, names=COLS)
    te  = pd.read_csv(raw / f'test_{tag}.txt',  sep=r'\s+', header=None, names=COLS)
    rul = pd.read_csv(raw / f'RUL_{tag}.txt',   sep=r'\s+', header=None, names=['RUL'])
    exp = EXPECTED[s]
    ok_tr  = (len(tr) == exp['train_rows'])   and (tr['unit'].nunique() == exp['train_units'])
    ok_te  = (len(te) == exp['test_rows'])    and (te['unit'].nunique() == exp['test_units'])
    ok_rul = (len(rul) == exp['rul_lines'])
    status_tr  = "OK"      if ok_tr  else f"MISMATCH(got {len(tr)}/{tr['unit'].nunique()})"
    status_te  = "OK"      if ok_te  else f"MISMATCH(got {len(te)}/{te['unit'].nunique()})"
    status_rul = "OK"      if ok_rul else f"MISMATCH(got {len(rul)})"
    if not (ok_tr and ok_te and ok_rul):
        all_ok = False
    print(
        f"{tag}: "
        f"train {len(tr):6d} rows/{tr['unit'].nunique():3d} units [{status_tr}]  "
        f"test {len(te):6d} rows/{te['unit'].nunique():3d} units [{status_te}]  "
        f"RUL {len(rul):3d} lines [{status_rul}]"
    )

print()
if all_ok:
    print("ALL REAL FILES VERIFIED OK - ready to use CMAPSSData/")
else:
    print("PROBLEMS FOUND - see above")
