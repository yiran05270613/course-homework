"""Accounting, simulator, policy-structure and independent small-DP checks."""
from pathlib import Path
import itertools, json
import numpy as np
import pandas as pd
import torch
from model import Config,Policy,patterns,sample,simulate,simulate_numpy
from dp import solve,DEMAND
OUT=Path(__file__).resolve().parents[1]/'results'
torch.set_num_threads(1)
# Independent hand calculation: first two periods cannot receive any order.
c=Config('hand',L=2,T=5,discount=1.)
class Constant(torch.nn.Module):
    def __init__(self,c):super().__init__();self.c=c
    def forward(self,i,p):return torch.full_like(i,3.)
p=Constant(c);d=torch.ones((1,5));y=torch.tensor([[1.,1.,1.,0.,1.]])
total,parts,tr=simulate(p,d,y,True)
np.testing.assert_allclose(tr[0,:,2],[0,0,3,0,3])
np.testing.assert_allclose(tr[0,:,3],[-1,-2,0,-1,1])
np.testing.assert_allclose(tr[0,:,1],[3,3,3,0,0])
np.testing.assert_allclose(parts.numpy(),[[1.,4*c.b,6*c.co]],rtol=1e-6)
# WNH enumerates all Bernoulli paths; initial DRL exactly matches its teacher.
for L in [2,3,5]:
    c=Config('structure',L=L,p=.8,T=12)
    bits,prob=patterns(c);np.testing.assert_allclose(prob.sum(),1.,atol=1e-6)
    assert len(bits)==2**(L-1) and len(np.unique(bits.numpy(),axis=0))==len(bits)
    inv=torch.tensor([-20.,0.,30.]);pipe=torch.arange(3*(L-1),dtype=torch.float32).reshape(3,L-1)*4
    wnh=Policy(c,'WNH');network=Policy(c,'DRL');static=Policy(c,'GDSH')
    expected=[]
    for i in range(3):
        value=0.
        for b in itertools.product([0,1],repeat=L-1):
            weight=np.prod([c.p if x else 1-c.p for x in b]);value+=weight*max(c.target-float(inv[i])-sum(float(pipe[i,j])*b[j] for j in range(L-1)),0)
        expected.append(value)
    np.testing.assert_allclose(wnh(inv,pipe).detach(),expected,rtol=2e-6)
    torch.testing.assert_close(network(inv,pipe),wnh(inv,pipe));torch.testing.assert_close(static(inv,pipe),wnh(inv,pipe))
    weights=network.weight_values(inv,pipe).detach();assert (weights>=prob-c.delta-1e-6).all() and (weights<=prob+c.delta+1e-6).all()
    d,y=sample(c,24,np.random.default_rng(123))
    for k in ['LIR','WNH','GDSH','DRL']:
        policy=Policy(c,k)
        with torch.no_grad():t,parts,tr=simulate(policy,d,y,True)
        nt,np_,ntr=simulate_numpy(policy,d.numpy(),y.numpy())
        np.testing.assert_allclose(t,nt,rtol=3e-6,atol=.01);np.testing.assert_allclose(tr,ntr,rtol=3e-6,atol=.001)
# Check that pathwise gradients include effects propagated through future inventory.
c=Config('gradient',L=2,T=9,p=.8)
m=Policy(c,'GDSH').double()
d,y=sample(c,20,np.random.default_rng(47));d=d.double();y=y.double()
objective=simulate(m,d,y)[0].mean();objective.backward();analytic=m.weights.grad[0].item()
base=m.weights.detach().clone();eps=1e-5
with torch.no_grad():
    m.weights.copy_(base);m.weights[0]+=eps;plus=simulate(m,d,y)[0].mean().item()
    m.weights.copy_(base);m.weights[0]-=eps;minus=simulate(m,d,y)[0].mean().item()
np.testing.assert_allclose(analytic,(plus-minus)/(2*eps),rtol=1e-4,atol=1e-4)
# DP under T=3, L=2 has only one effective ordering decision. Enumerate independently.
c=Config('bruteforce',L=2,T=3,p=.8,distribution='discrete',mu=2,sigma=np.sqrt(.5),discount=.95,action_cap=3)
values=[]
for q in range(4):
    expected=0.
    for choices in itertools.product(DEMAND,repeat=3):
        ds=[x[0] for x in choices];pd_=np.prod([x[1] for x in choices])
        for y,py in [(0,1-c.p),(1,c.p)]:
            end0=-ds[0];end1=end0-ds[1];end2=end1+y*q-ds[2]
            h=lambda v:c.h*max(v,0)+c.b*max(-v,0)
            expected+=pd_*py*(h(end0)+c.discount*h(end1)+c.discount**2*(h(end2)+c.co*y*q))
    values.append(expected)
optimal,_,_=solve(c);np.testing.assert_allclose(optimal,min(values),atol=1e-10)
print('PASS: manual delivery timing, successful-arrival payment, terminal order cutoff; exhaustive WNH; teacher initialization; independent NumPy queue; DP vs full scenario enumeration.',flush=True)
if (OUT/'audit.json').exists():
    result=pd.read_csv(OUT/'case_results.csv')
    np.testing.assert_allclose(result.mean_cost,result.holding+result.backlog+result.purchase,rtol=2e-6)
    assert np.isfinite(result.select_dtypes(include='number')).all().all()
    small=pd.read_csv(OUT/'dp_results.csv');assert small.gap_pct.ge(-1e-7).all()
    meta=json.loads((OUT/'audit.json').read_text())
    for path in (OUT/'models').glob('*.pt'):torch.load(path,weights_only=True)
    core=result[(result['T']==50)&np.isclose(result.critical_ratio,.9)]
    if len(meta['configs'])==18:assert len(core)==48
    for conf in meta['configs']:
        c=Config(**conf);d,y=sample(c,meta['test_paths'],np.random.default_rng(c.seed+3000))
        with torch.no_grad():
            for k in ['GDSH','DRL']:
                checkpoint=torch.load(OUT/'models'/f'{c.name}_{k}.pt',weights_only=True)
                m=Policy(c,k);m.load_state_dict(checkpoint['state']);cost,_,_=simulate(m,d,y)
                saved=pd.read_csv(OUT/'raw'/f'{c.name}_test_costs.csv')[k]
                np.testing.assert_allclose(cost,saved,rtol=2e-6,atol=.01)
    print('PASS: saved model reload reproduces all test costs; cost accounting; nonnegative discrete DP gaps.',flush=True)
