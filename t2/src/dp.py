"""Exact finite-horizon recursion for an explicitly discretized, bounded action model."""
from functools import lru_cache
import numpy as np
import torch

DEMAND=[(1,.25),(2,.5),(3,.25)]

def solve(c,policy=None):
    decisions={}
    @lru_cache(None)
    def value(t,inv,pipe):
        if t==c.T:return 0.
        if t>=c.T-c.L:actions=[0]
        elif policy is None:actions=range(c.action_cap+1)
        else:
            with torch.no_grad():q=policy(torch.tensor([inv],dtype=torch.float32),torch.tensor([pipe],dtype=torch.float32))
            actions=[int(q.item())]
        best=float('inf');bestq=0
        for q in actions:
            total=0.
            for d,pd in DEMAND:
                end=inv-d;current=c.h*max(end,0)+c.b*max(-end,0)
                continuation=0.
                if t<c.T-1:
                    oldest=pipe[-1];newpipe=(q,)+pipe[:-1]
                    for y,py in [(0,1-c.p),(1,c.p)]:
                        arrived=y*oldest
                        continuation+=py*(c.co*arrived+value(t+1,end+arrived,newpipe))
                total+=pd*(current+c.discount*continuation)
            if total<best:best=total;bestq=q
        if policy is None:decisions[(t,inv,pipe)]=bestq
        return best
    result=value(0,0,tuple([0]*(c.L-1)))
    return result,decisions,value.cache_info().currsize
