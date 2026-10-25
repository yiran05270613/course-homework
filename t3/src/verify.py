"""Verify the accelerated backend against an independent NumPy implementation."""
from pathlib import Path
import json, subprocess, tempfile
import numpy as np
import pandas as pd
from replicate import HERE, OUT, market, cycle, rollout
pn,pm,prices,payoff,pi_n,pi_m=market()
np.testing.assert_allclose([pn,pm],[1.4729266600306221,1.9249809190177618],rtol=1e-10)
np.testing.assert_allclose(payoff[0],payoff[1].T,atol=1e-15)
assert pi_m>pi_n>0 and np.all(np.diff(prices)>0)
# A deterministic sequence exercises the actual backend update loop; seed=0 is test-only.
with tempfile.TemporaryDirectory(prefix='t3-check-') as folder:
    prefix=Path(folder)/'check'
    subprocess.run([str(HERE/'src/qlearn'),'1','.95','.15','0.000004','300','100000','0',repr(pn),repr(pm),str(prefix)],check=True)
    q=np.empty((2,225,15))
    for i in range(2):
        initial=(payoff[0].mean(axis=1) if i==0 else payoff[1].mean(axis=0))/.05
        q[i]=initial
    state=7
    for t in range(300):
        actions=[t%15,(t*7+3)%15];nxt=15*actions[0]+actions[1]
        for i in range(2):
            old=q[i,state,actions[i]]
            q[i,state,actions[i]]=.85*old+.15*(payoff[i,actions[0],actions[1]]+.95*q[i,nxt].max())
        state=nxt
    actual=np.fromfile(str(prefix)+'_q_0.bin',dtype=np.float64).reshape(2,225,15)
    np.testing.assert_allclose(actual,q,rtol=1e-12,atol=1e-12)
    # Same seed and binary must reproduce exactly, not merely similar means.
    args=[str(HERE/'src/qlearn'),'2','.95','.15','0.000004','5000','100000','9876',repr(pn),repr(pm)]
    subprocess.run(args+[str(Path(folder)/'a')],check=True)
    subprocess.run(args+[str(Path(folder)/'b')],check=True)
    for suffix in ['_sessions.csv','_learning.csv','_q_0.bin','_q_1.bin']:
        assert (Path(folder)/('a'+suffix)).read_bytes()==(Path(folder)/('b'+suffix)).read_bytes()
# Independently reconstruct reported long-run outcomes and the forced best response.
main=pd.read_csv(OUT/'baseline_sessions.csv')
for row in main.itertuples():
    q=np.fromfile(OUT/'raw'/f'baseline_q_{row.session}.bin',dtype=np.float64).reshape(2,225,15)
    policy=q.argmax(axis=2)
    raw=pd.read_csv(OUT/'raw/baseline_sessions.csv').iloc[row.session]
    states=cycle(policy,int(raw.last_state));a,b=policy[:,states]
    profit=payoff[:,a,b].mean()
    np.testing.assert_allclose(row.profit_gain,(profit-pi_n)/(pi_m-pi_n),atol=1e-12)
    assert row.cycle_length==len(states)
    for state in states:
        p,r=rollout(policy,state,prices,payoff,force=True)
        opponent=policy[1,state];best=payoff[0,:,opponent].max()
        np.testing.assert_allclose(r[0,0],best,atol=1e-14)
        assert r[0,0]>=payoff[0,policy[0,state],opponent]-1e-14
meta=json.loads((OUT/'audit.json').read_text())
assert len(main)==meta['sessions']
assert main.steps.le(meta['max_steps']).all()
assert main.loc[main.converged,'steps'].ge(meta['stable_steps']).all()
assert np.isfinite(main.select_dtypes(include='number')).all().all()
print('PASS: 300-step C++/NumPy Q-table equality; fixed-seed reproducibility; market benchmarks; all baseline cycle outcomes; unilateral static best responses; run limits.')
