"""Differential check against supplied author notebooks and accounting identities."""
import contextlib, io, json, os
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'文献和数据/replication_package_MS-FTBDA-20-01585'
OUT=ROOT/'t1/results'
os.chdir(DATA)
def original(name):
    path=DATA/'.ipynb_checkpoints'/name
    notebook=json.loads(path.read_text())
    source='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    # Compatibility only: remove unused plotting/progress imports and LaTeX export.
    source='\n'.join(line for line in source.splitlines() if not any(x in line for x in ['import seaborn','from tqdm','import matplotlib','to_latex']))
    source=source.replace("df_lr.drop('y',1)","df_lr.drop(columns='y')")
    env={}
    with contextlib.redirect_stdout(io.StringIO()):exec(compile(source,str(path),'exec'),env)
    return env
s=original('E2E_Section4-checkpoint.ipynb')
actual=pd.read_csv(OUT/'offline_summary.csv');actual=actual[actual.scope=='author'].set_index('method')
rename={'OPT':'OPT','E2E_RNN':'E2E_RNN','Bm1':'BM1','Bm2':'BM2','Normal':'Normal','Gamma':'Gamma','gbm':'E2E_GBM'}
for old,new in rename.items():
    for author,metric in [('Total cost','total'),('Holding cost','holding'),('Stockout cost','stockout')]:
        np.testing.assert_allclose(actual.loc[new,metric],s['df_aggcom'].loc[author,old],rtol=1e-12)
f=original('Field_experiments_summary 2-checkpoint.ipynb')
for group,arrays in [('T_post',f['post_exp_e2e']),('C_post',f['post_exp_jd'])]:
    v=pd.read_csv(OUT/f'field_{group}.csv')
    for i,m in enumerate(['holding','stockout','total','turnover','stockout_ratio']):
        target=v.turnover_raw if group=='C_post' and m=='turnover' else v[m]
        np.testing.assert_allclose(target,arrays[i],rtol=1e-12,equal_nan=True)
for p in OUT.glob('field_*.csv'):
    if not any(p.name.startswith('field_'+g) for g in ['T_pre','T_post','C_pre','C_post']):continue
    v=pd.read_csv(p)
    np.testing.assert_allclose(v.total,v.holding+v.stockout)
    assert v.stockout_ratio.between(0,1).all()
did=pd.read_csv(OUT/'did.csv');np.testing.assert_allclose(did.did,did.did_cluster,atol=1e-9)
print('PASS: seven offline methods match author notebooks; post-period field metrics match; cost identities, proportions and DiD regression coefficients verified.')
