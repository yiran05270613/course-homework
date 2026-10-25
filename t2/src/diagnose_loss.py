"""Explicitly labelled aggregation sensitivity; never replaces the primary DRL result."""
import argparse, json, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
from model import Config,sample,simulate,train
OUT=Path(__file__).resolve().parents[1]/'results'
def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=300);p.add_argument('--batch',type=int,default=500);p.add_argument('--validation',type=int,default=256);p.add_argument('--test-paths',type=int,default=1000);p.add_argument('--seed',type=int,default=20261024);a=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter();rows=[]
    for L,offset in [(2,10000),(5,40000)]:
        c=Config(f'normal_L{L}_p0.8',L=L,p=.8,seed=a.seed+offset)
        models,history,selected=train(c,a.epochs,a.batch,a.validation,drl_aggregation='batch')
        d,y=sample(c,a.test_paths,np.random.default_rng(c.seed+3000))
        with torch.no_grad():cost=simulate(models['DRL'],d,y)[0].numpy();base=simulate(models['LIR'],d,y)[0].numpy()
        torch.save(dict(config=asdict(c),kind='DRL',aggregation='batch',state=models['DRL'].state_dict()),OUT/'models'/f'{c.name}_DRL_batchmean.pt')
        pd.DataFrame(history).to_csv(OUT/'raw'/f'{c.name}_batchmean_training.csv',index=False)
        pd.DataFrame(dict(DRL_batchmean=cost,LIR=base)).to_csv(OUT/'raw'/f'{c.name}_batchmean_test.csv',index=False)
        rows.append(dict(case=c.name,aggregation='batch_mean_before_exp',mean_cost=cost.mean(),change_pct=100*(cost.mean()/base.mean()-1),selected_epoch=selected['DRL']))
        print(rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'loss_aggregation_sensitivity.csv',index=False)
    (OUT/'loss_sensitivity_audit.json').write_text(json.dumps(dict(**vars(a),elapsed_seconds=time.perf_counter()-start),indent=2))
if __name__=='__main__':main()
