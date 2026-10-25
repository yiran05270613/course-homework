"""Differentiable all-or-nothing inventory simulator and paper-structured policies."""
from dataclasses import dataclass, asdict
import itertools
import numpy as np
from scipy.stats import norm
import torch
from torch import nn

@dataclass
class Config:
    name: str
    L: int = 2
    T: int = 50
    p: float = .8
    distribution: str = 'normal'
    mu: float = 10.
    sigma: float = 2.
    gamma_scale: float = 1.6
    critical_ratio: float = .9
    co: float = 2.
    h: float = 1.
    discount: float = .99
    seed: int = 20261024
    action_cap: int = 0
    @property
    def b(self):return self.h*self.critical_ratio/(1-self.critical_ratio)
    @property
    def std(self):return np.sqrt(self.mu*self.gamma_scale) if self.distribution=='gamma' else self.sigma
    @property
    def target(self):
        rho2=(1-self.p)/self.p
        return (self.L+1)*self.mu+norm.ppf(self.critical_ratio)*np.sqrt((self.L+1)*self.std**2+max(self.L,1)*rho2/(1-rho2)*(self.mu**2+self.std**2))
    @property
    def delta(self):return .0884 if self.L==5 else .25


def patterns(c):
    # Bit i corresponds to q_(t-i-1), newest order first.
    bits=np.array([[(j>>i)&1 for i in range(c.L-1)] for j in range(2**(c.L-1))],dtype=np.float32)
    weights=np.prod(np.where(bits==1,c.p,1-c.p),axis=1)
    return torch.tensor(bits),torch.tensor(weights,dtype=torch.float32)

class Policy(nn.Module):
    def __init__(self,c,kind):
        super().__init__();self.c=c;self.kind=kind
        bits,prob=patterns(c);self.register_buffer('bits',bits);self.register_buffer('prob',prob)
        if kind=='GDSH':self.weights=nn.Parameter(prob.clone())
        if kind=='DRL':
            # Include S_L as the paper's augmented state; it is fixed within a case.
            self.network=nn.Sequential(nn.Linear(c.L+1,8),nn.ReLU(),nn.Linear(8,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU(),nn.Linear(16,2*len(prob)))
            nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
    def weight_values(self,inv,pipe):
        if self.kind=='GDSH':return self.weights.expand(len(inv),-1)
        if self.kind=='DRL':
            state=torch.cat([torch.full_like(inv[:,None],self.c.target),inv[:,None],pipe],dim=1)/self.c.target
            eta=self.network(state).reshape(-1,len(self.prob),2).softmax(dim=-1)
            return eta[:,:,0]*(self.prob-self.c.delta)+eta[:,:,1]*(self.prob+self.c.delta)
        return self.prob.expand(len(inv),-1)
    def forward(self,inv,pipe):
        if self.kind=='LIR':q=torch.relu(self.c.target-inv-self.c.p*pipe.sum(dim=1))
        else:
            gaps=torch.relu(self.c.target-inv[:,None]-pipe@self.bits.T)
            q=torch.relu((gaps*self.weight_values(inv,pipe)).sum(dim=1))
        if self.c.action_cap:
            q=q.clamp(max=self.c.action_cap)
            # Straight-through rounding for training; exact integer actions forward.
            q=q+(q.round()-q).detach()
        return q


def sample(c,n,rng):
    if c.distribution=='normal':d=np.maximum(0,rng.normal(c.mu,c.sigma,(n,c.T)))
    elif c.distribution=='gamma':d=rng.gamma(c.mu/c.gamma_scale,c.gamma_scale,(n,c.T))
    elif c.distribution=='discrete':d=rng.choice([1.,2.,3.],size=(n,c.T),p=[.25,.5,.25])
    else:raise ValueError(c.distribution)
    y=(rng.random((n,c.T))<c.p).astype(np.float32)
    return torch.tensor(d,dtype=torch.float32),torch.tensor(y)


def simulate(policy,demand,yields,trace=False):
    c=policy.c;n=len(demand);zero=torch.zeros(n,dtype=demand.dtype)
    end=zero;orders=[];holding=zero;backlog=zero;purchase=zero;rows=[]
    for t in range(c.T):
        arrived=orders[t-c.L]*yields[:,t] if t>=c.L else zero
        inv=end+arrived
        pipe=torch.stack([orders[t-i] if t-i>=0 else zero for i in range(1,c.L)],dim=1)
        q=policy(inv,pipe) if t<c.T-c.L else zero
        orders.append(q);end=inv-demand[:,t]
        factor=c.discount**t
        holding=holding+factor*c.h*torch.relu(end)
        backlog=backlog+factor*c.b*torch.relu(-end)
        purchase=purchase+factor*c.co*arrived
        if trace:rows.append(torch.stack([inv,q,arrived,end,demand[:,t],yields[:,t]],dim=1))
    parts=torch.stack([holding,backlog,purchase],dim=1)
    return parts.sum(dim=1),parts,torch.stack(rows,dim=1) if trace else None


def simulate_numpy(policy,demand,yields):
    """Independent queue-based evaluator for accounting verification."""
    c=policy.c;n=len(demand);end=np.zeros(n);queue=np.zeros((n,c.L));total=np.zeros(n);parts=np.zeros((n,3));traces=[]
    for t in range(c.T):
        arrived=queue[:,0]*yields[:,t];inv=end+arrived
        pending=queue[:,1:][:,::-1].copy()
        with torch.no_grad():q=policy(torch.tensor(inv,dtype=torch.float32),torch.tensor(pending,dtype=torch.float32)).numpy() if t<c.T-c.L else np.zeros(n)
        end=inv-demand[:,t];stage=np.stack([c.h*np.maximum(end,0),c.b*np.maximum(-end,0),c.co*arrived],axis=1)
        parts+=c.discount**t*stage
        queue=np.c_[queue[:,1:],q]
        traces.append(np.stack([inv,q,arrived,end,demand[:,t],yields[:,t]],axis=1))
    return parts.sum(axis=1),parts,np.stack(traces,axis=1)


def train(c,epochs=300,batch=500,validation=256,drl_aggregation="trajectory"):
    torch.manual_seed(c.seed)
    models={k:Policy(c,k) for k in ['LIR','WNH','GDSH','DRL']}
    opts={k:torch.optim.Adam(models[k].parameters(),lr=.003) for k in ['GDSH','DRL']}
    rng=np.random.default_rng(c.seed+1000)
    vd,vy=sample(c,validation,np.random.default_rng(c.seed+2000))
    with torch.no_grad():initial=float(simulate(models['WNH'],vd,vy)[0].mean())
    best={k:initial for k in opts};states={k:{n:v.detach().clone() for n,v in models[k].state_dict().items()} for k in opts};selected={k:0 for k in opts}
    history=[dict(epoch=0,method=k,validation_cost=initial,train_cost=np.nan,loss=np.nan,lr=.003) for k in opts]
    for epoch in range(1,epochs+1):
        d,y=sample(c,batch,rng)
        with torch.no_grad():teacher=simulate(models['WNH'],d,y)[0]
        for k,opt in opts.items():
            lr=max(.0001,.003*.98**(epoch-1))
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad();cost=simulate(models[k],d,y)[0]
            if k=='DRL' and drl_aggregation=='batch':
                loss=100*(cost.mean()-teacher.mean())/teacher.mean()
            elif k=='DRL':
                ratio=100*(cost-teacher)/teacher.clamp_min(1e-6)
                # Same minimizer as mean(exp(ratio)); log prevents numeric overflow.
                loss=torch.logsumexp(ratio,dim=0)-np.log(batch)
            else:loss=cost.mean()/teacher.mean()
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training loss {c.name} {k} {epoch}')
            loss.backward();torch.nn.utils.clip_grad_norm_(models[k].parameters(),10.);opt.step()
            if epoch%10==0 or epoch==epochs:
                with torch.no_grad():val=float(simulate(models[k],vd,vy)[0].mean())
                if val<best[k]:
                    best[k]=val;states[k]={n:v.detach().clone() for n,v in models[k].state_dict().items()};selected[k]=epoch
                history.append(dict(epoch=epoch,method=k,validation_cost=val,train_cost=float(cost.detach().mean()),loss=float(loss.detach()),lr=lr))
    for k in opts:models[k].load_state_dict(states[k]);models[k].eval()
    return models,history,selected
