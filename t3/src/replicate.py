"""Calvano et al. (2020) learning-scale replication with exact tabular Q-learning."""
from pathlib import Path
import argparse, base64, hashlib, json, platform, shutil, subprocess, time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'t3';OUT=HERE/'results'

def market():
    # Continuous symmetric one-shot Bertrand and joint-monopoly first-order conditions.
    def share(p):return np.exp((2-p)/.25)/(1+2*np.exp((2-p)/.25))
    pn=brentq(lambda p:1-(p-1)*(1-share(p))/.25,1,3)
    pm=brentq(lambda p:1-(p-1)*(1-2*share(p))/.25,1,3)
    prices=np.linspace(pn-.1*(pm-pn),pm+.1*(pm-pn),15)
    u=np.exp((2-prices)/.25);d=1+u[:,None]+u[None,:]
    payoff=np.stack([(prices[:,None]-1)*u[:,None]/d,(prices[None,:]-1)*u[None,:]/d])
    return pn,pm,prices,payoff,(pn-1)*share(pn),(pm-1)*share(pm)

def cycle(policy,start):
    seen={};states=[];s=int(start)
    while s not in seen:
        seen[s]=len(states);states.append(s);a,b=policy[:,s];s=int(15*a+b)
    return states[seen[s]:]

def rollout(policy,start,prices,payoff,horizon=400,force=False):
    s=int(start);p=[];r=[]
    for t in range(horizon):
        a,b=map(int,policy[:,s])
        if force and t==0:a=int(np.argmax(payoff[0,:,b]))
        p.append([prices[a],prices[b]]);r.append(payoff[:,a,b]);s=15*a+b
    return np.array(p),np.array(r)

def evaluate(prefix,delta,prices,payoff,pn_profit,pm_profit):
    sessions=pd.read_csv(str(prefix)+'_sessions.csv');records=[];impulses=[]
    for r in sessions.itertuples():
        q=np.fromfile(str(prefix)+f'_q_{r.session}.bin',dtype=np.float64).reshape(2,225,15)
        policy=q.argmax(axis=2);states=cycle(policy,r.last_state)
        actions=policy[:,states];profits=np.array([payoff[:,a,b] for a,b in actions.T]);ps=prices[actions]
        gains=[];impulse=[]
        for s in states:
            baseline_p,baseline_r=rollout(policy,s,prices,payoff)
            dev_p,dev_r=rollout(policy,s,prices,payoff,force=True)
            # t=0 is pre-deviation; t=1 is forced best-response deviation.
            impulse.append(np.c_[np.vstack([prices[[s//15,s%15]],dev_p[:40]]),np.vstack([prices[[s//15,s%15]],baseline_p[:40]])])
            gains.append(float(np.dot(delta**np.arange(len(dev_r)),dev_r[:,0]-baseline_r[:,0])))
        impulses.append(np.mean(impulse,axis=0))
        records.append(dict(session=r.session,seed=r.seed,steps=r.steps,converged=bool(r.converged),cycle_length=len(states),symmetric_constant=bool(len(states)==1 and actions[0,0]==actions[1,0]),price=float(ps.mean()),profit=float(profits.mean()),profit_gain=float((profits.mean()-pn_profit)/(pm_profit-pn_profit)),deviation_npv_gain=np.mean(gains),unprofitable_deviation_share=float(np.mean(np.array(gains)<=1e-10))))
    result=pd.DataFrame(records);result.to_csv(str(prefix)+'_evaluated.csv',index=False)
    impulses=np.array(impulses)
    return result,impulses

def simulate(exe,folder,name,n,delta,alpha,beta,cap,seed,pn,pm):
    prefix=folder/name
    subprocess.run([str(exe),str(n),str(delta),str(alpha),str(beta),str(cap),'100000',str(seed),repr(pn),repr(pm),str(prefix)],check=True)
    return prefix

def figure(name):
    plt.tight_layout();plt.savefig(OUT/f'{name}.png',dpi=160);plt.savefig(OUT/f'{name}.svg');plt.close()

def ci(values):
    values=np.asarray(values,float)
    return float(1.96*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.

def plots(main,impulse,learning,discount,grid,pn,pm,pi_n,pi_m):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.figsize':(9,4.5),'axes.titleweight':'bold'})
    # No post-stop imputation: each point uses sessions with observations in that bin.
    learning['bin']=np.ceil(learning.step/5000).astype(int)
    points=learning.groupby('bin').agg(step=('step','mean'),profit=('profit','mean'),n=('session','count'))
    gain=(points.profit-pi_n)/(pi_m-pi_n)
    fig,(ax,counts)=plt.subplots(2,1,sharex=True,figsize=(9,5.8),gridspec_kw={'height_ratios':[3,1]});counts.step(points.step/1e6,points.n,where='mid',color='#777');counts.set(xlabel='Learning periods (million)',ylabel='Sessions');ax.plot(points.step/1e6,gain,color='#247c91',lw=2);ax.axhline(0,color='#999',ls='--',label='One-shot Nash');ax.axhline(1,color='#d38a4a',ls='--',label='Joint monopoly');ax.set(ylabel='Normalized profit gain',title='Learning dynamics: 5,000-period window averages');ax.legend();figure('learning')
    pd.DataFrame(dict(step=points.step,gain=gain,contributing_sessions=points.n)).to_csv(OUT/'learning_summary.csv',index=False)
    mean=impulse.mean(axis=0);err=1.96*impulse.std(axis=0,ddof=1)/np.sqrt(len(impulse));t=np.arange(len(mean))
    fig,ax=plt.subplots()
    for i,label in enumerate(['Deviating firm','Other firm']):
        ax.plot(t,mean[:,i],label=label,color=['#247c91','#d38a4a'][i]);ax.fill_between(t,mean[:,i]-err[:,i],mean[:,i]+err[:,i],alpha=.13,color=['#247c91','#d38a4a'][i])
    ax.plot(t,mean[:,2:4].mean(axis=1),color='#777',ls=':',label='No-deviation counterfactual');ax.axhline(pn,color='#888',ls='--',label='Nash price');ax.axvline(1,color='#aaa',lw=.8);ax.set(xlabel='Periods (forced deviation at 1)',ylabel='Price',title='Frozen-policy response to a one-period deviation');ax.legend(ncol=2);figure('deviation')
    pd.DataFrame(dict(period=t,deviator=mean[:,0],other=mean[:,1],counterfactual_deviator=mean[:,2],counterfactual_other=mean[:,3],ci_deviator=err[:,0],ci_other=err[:,1])).to_csv(OUT/'impulse_response.csv',index=False)
    fig,ax=plt.subplots();ax.errorbar(discount.delta,discount.profit_gain,yerr=discount.ci95,fmt='o-',capsize=4,color='#247c91');ax.axhline(0,color='#aaa',ls='--');ax.set(xlabel='Discount factor',ylabel='Normalized profit gain',title='Patience and learned pricing (approximate 95% intervals)');figure('discount')
    pivot=grid.pivot(index='alpha',columns='beta',values='profit_gain');fig,ax=plt.subplots();im=ax.imshow(pivot.to_numpy(),origin='lower',cmap='YlGnBu',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(len(pivot.columns)),[f'{x:.0e}' for x in pivot.columns]);ax.set_yticks(range(len(pivot.index)),[f'{x:g}' for x in pivot.index]);ax.set(xlabel='Exploration decay beta',ylabel='Learning rate alpha',title='Reduced parameter grid: mean normalized profit gain')
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):ax.text(j,i,f'{pivot.iloc[i,j]:.2f}',ha='center',va='center',color='white' if pivot.iloc[i,j]>.55 else '#18343e')
    fig.colorbar(im,ax=ax);figure('parameter_grid')
    fig,ax=plt.subplots();ax.hist(main.price,bins=np.linspace(pn-.1*(pm-pn),pm+.1*(pm-pn),16),color='#247c91',alpha=.8);ax.axvline(pn,ls='--',color='#777',label=f'Nash {pn:.3f}');ax.axvline(pm,ls='--',color='#d38a4a',label=f'Monopoly {pm:.3f}');ax.set(xlabel='Mean on-cycle price per session',ylabel='Sessions',title='Learned outcomes across independent seeds');ax.legend();figure('price_distribution')

def report(main,discount,grid,meta):
    def img(name):return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((OUT/f'{name}.png').read_bytes()).decode()+'">'
    def table(df):return df.to_html(index=False,border=0,float_format=lambda x:f'{x:,.4g}')
    converged=main[main.converged]
    descriptive=pd.DataFrame(dict(statistic=['Mean profit gain','Std. dev. profit gain','Constant symmetric frequency','Constant asymmetric frequency','All constant frequency','Cycle length 2 frequency','Cycle length >=3 frequency'],replication=[main.profit_gain.mean(),main.profit_gain.std(ddof=1),main.symmetric_constant.mean(),((main.cycle_length==1)&~main.symmetric_constant).mean(),(main.cycle_length==1).mean(),(main.cycle_length==2).mean(),(main.cycle_length>=3).mean()],paper=[.849,.112,.277,.366,.643,.238,.119]))
    descriptive.to_csv(OUT/'paper_table1_comparison.csv',index=False)
    body=f'''<header><small>AER 2020 · T3 · 可运行学习版复现</small><h1>算法如何学会定价？</h1><p>独立 Q-learning · 超竞争价格 · 偏离后的反应</p><p>基于论文公式自行实现，所有结果均来自本次模拟。未使用论文图表数据作为复现结果。</p></header>
    <section><h2>01 / 实验设定与结果概览</h2><p>两家企业，Logit 需求，成本 c=1、产品质量 a=2、外部选项 a₀=0、μ=0.25。15 个离散价格，记忆上一期双方价格：225 个状态，每家 Q 表 225×15。折现因子 δ=0.95，学习率 α=0.15，探索率 εₜ=exp(−4×10⁻⁶t)。采用原文 Q 初始化及全状态策略稳定 100,000 期的停止条件。</p><div class="cards"><article><b>{main.price.mean():.3f}</b><span>平均周期价格</span></article><article><b>{main.profit_gain.mean():.1%}</b><span>标准化利润增益 Δ</span></article><article><b>{main.converged.sum()}/{len(main)}</b><span>达到原文稳定判据</span></article><article><b>{meta['elapsed_seconds']:.1f} 秒</b><span>编译、模拟、分析耗时</span></article></div><p>连续动作竞争均衡价格 pᴺ={meta['pn']:.6f}，联合垄断价格 pᴹ={meta['pm']:.6f}。Δ=(平均单企业利润−竞争利润)/(联合垄断时单企业利润−竞争利润)，不是价格涨幅。100% 表示联合垄断利润，0% 表示连续价格竞争利润。</p><p>基准 {len(main)} 次独立实验；每次最多 {meta['max_steps']:,} 期。平均运行 {main.steps.mean():,.0f} 期。已收敛样本平均 Δ={converged.profit_gain.mean():.1%}。所有图默认保留全部样本；未收敛者的结果仅指训练停止时的贪心策略。</p></section>
    <section><h2>02 / 学习过程 · 对应图 10 的思路</h2>{img('learning')}<p>每条实验按 5,000 期窗口汇总后跨实验平均，不是论文原图的 100 期移动平均。提前停止后不补造训练数据，下方同步显示参与样本数：后段均值会受样本退出影响，不能把尾部下降直接解释为同一组算法降低价格，详见 learning_summary.csv。</p></section>
    <section><h2>03 / 偏离、惩罚与恢复 · 对应图 4 的思路</h2>{img('deviation')}<p>训练停止后冻结 Q 表并关闭探索，先找到贪心策略的最终周期。对每个周期相位，在第 1 期强制企业 1 采用针对对手原定价格的网格内静态最佳反应，之后双方恢复所学策略。先对周期相位平均，再对独立实验平均。虚线反事实控制周期本身引起的价格变化；阴影为跨实验均值的近似 95% 区间。</p><p>按各周期相位计算，偏离折现净收益不为正的平均比例为 {main.unprofitable_deviation_share.mean():.1%}；平均偏离净收益为 {main.deviation_npv_gain.mean():.6f}。折现比较累计 400 期，δ=0.95 时尾项很小。它只检验一种单期偏离，不等同于对所有状态和所有策略证明纳什均衡；也没有预设必须出现惩罚。</p></section>
    <section><h2>04 / 折现因子 · 对应图 3 的思路</h2>{img('discount')}{table(discount)}<p>每组重新初始化并训练，而不是修改已有策略的利润计算。误差线为 1.96×样本标准误；独立实验次数较少，属于探索性比较，不要求每个相邻点单调。</p></section>
    <section><h2>05 / 学习参数网格 · 对应图 1 的缩减实验</h2>{img('parameter_grid')}{table(grid)}<p>3×3 网格，各点独立运行 {meta['grid_sessions']} 次；原论文为 100×100 网格、每点 1,000 次，因此这里只复现分析方法与代表性结果，不替代全网格结论。</p></section>
    <section><h2>06 / 独立实验的异质性 · 对应表 1 部分统计</h2>{img('price_distribution')}{table(descriptive)}<p>paper 列为原论文表 1 的人工录入参考值（1,000 次实验），replication 为本次模拟；原值未参与学习、调参或结果计算。</p>{table(main[['session','steps','converged','cycle_length','price','profit_gain','deviation_npv_gain']])}<p>最终策略可能形成常数价格，也可能形成周期。这里按最终周期平均评价，不把某一时点的高价格作为长期结果。价格超过竞争水平本身不充分证明合谋，需结合偏离反应与激励检查理解。</p></section>
    <section><h2>07 / 复现边界与运行</h2><p>保留核心经济环境、Q 更新、最低价格打破平局、原文初始化、探索递减和稳定停止条件。缩减独立重复次数与参数网格，并将原文十亿期上限降为 {meta['max_steps']:,} 期。当前未复现均衡最优响应完整检验、多人企业、非对称环境、需求不确定性、重新配对及全部稳健性实验。</p><pre>python -m pip install -r t3/requirements.txt\npython t3/src/replicate.py\npython t3/src/verify.py</pre><p>需要 Python 3.13 及 C++17 编译器。Python 负责市场求解、实验组织、统计与图表；C++ 负责大量逐期 Q 更新以缩短计算时间。所有代码和原始 Q 表保存在项目内，运行不需要联网。详见 README.md。</p><p>参考文献：Calvano, Calzolari, Denicolò &amp; Pastorello (2020), Artificial Intelligence, Algorithmic Pricing, and Collusion, AER 110(10):3267–3297. DOI: 10.1257/aer.20190623。</p></section>'''
    css='body{margin:0;background:#f2f5f5;color:#203b45;font:16px/1.8 system-ui,-apple-system,"PingFang SC",sans-serif}header,section{max-width:1050px;margin:26px auto;padding:32px 40px;background:white;border-radius:16px}header{background:#173e49;color:white}h1{font-size:44px;margin:14px 0}h2{font-size:23px;color:#246c7e}small{letter-spacing:2px}img{width:100%;height:auto}.cards{display:flex;gap:12px;flex-wrap:wrap}.cards article{flex:1;min-width:160px;background:#edf5f5;padding:18px;border-radius:10px}.cards b{display:block;font-size:28px;color:#216a7c}.cards span{font-size:13px}table{display:block;overflow:auto;border-collapse:collapse;font-size:13px}th,td{padding:9px 12px;border-bottom:1px solid #dce6e9;text-align:right;white-space:nowrap}th{background:#edf5f5}pre{background:#edf5f5;padding:20px;overflow:auto}@media(max-width:700px){header,section{margin:12px;padding:20px}h1{font-size:30px}}'
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>T3 算法定价复现</title><style>'+css+'</style>'+body+'</html>',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sessions',type=int,default=40);parser.add_argument('--sensitivity-sessions',type=int,default=16);parser.add_argument('--grid-sessions',type=int,default=8);parser.add_argument('--max-steps',type=int,default=5000000);parser.add_argument('--seed',type=int,default=20261024);args=parser.parse_args()
    if min(args.sessions,args.sensitivity_sessions,args.grid_sessions)<2:parser.error('each experiment requires at least two sessions')
    if args.seed<=0:parser.error('seed must be positive; zero is reserved for deterministic verification')
    if args.max_steps<100000:parser.error('max steps must be at least 100000')
    start=time.perf_counter();OUT.mkdir(exist_ok=True,parents=True);raw=OUT/'raw';raw.mkdir(exist_ok=True)
    compiler=shutil.which('clang++') or shutil.which('g++')
    if not compiler:raise RuntimeError('C++ compiler not found; install Apple command line tools or g++.')
    exe=HERE/'src/qlearn';subprocess.run([compiler,'-O3','-std=c++17',str(HERE/'src/qlearn.cpp'),'-o',str(exe)],check=True)
    pn,pm,prices,payoff,pi_n,pi_m=market()
    baseline=simulate(exe,raw,'baseline',args.sessions,.95,.15,4e-6,args.max_steps,args.seed,pn,pm)
    main_results,impulse=evaluate(baseline,.95,prices,payoff,pi_n,pi_m);main_results.to_csv(OUT/'baseline_sessions.csv',index=False);print('Baseline:',main_results[['steps','converged','price','profit_gain']].mean().to_dict(),flush=True)
    discount=[]
    for j,delta in enumerate([0,.35,.65,.85,.95]):
        if delta==.95:res=main_results
        else:
            prefix=simulate(exe,raw,f'delta_{delta:g}',args.sensitivity_sessions,delta,.15,4e-6,args.max_steps,args.seed+10000*(j+1),pn,pm)
            res,_=evaluate(prefix,delta,prices,payoff,pi_n,pi_m)
        discount.append(dict(delta=delta,sessions=len(res),profit_gain=res.profit_gain.mean(),ci95=ci(res.profit_gain),price=res.price.mean(),convergence_rate=res.converged.mean()))
        print('Discount',delta,discount[-1],flush=True)
    grid=[]
    for i,alpha in enumerate([.05,.15,.25]):
        for j,beta in enumerate([2e-6,4e-6,1e-5]):
            prefix=simulate(exe,raw,f'grid_{i}_{j}',args.grid_sessions,.95,alpha,beta,args.max_steps,args.seed+100000+i*10000+j*1000,pn,pm)
            res,_=evaluate(prefix,.95,prices,payoff,pi_n,pi_m)
            grid.append(dict(alpha=alpha,beta=beta,sessions=len(res),profit_gain=res.profit_gain.mean(),ci95=ci(res.profit_gain),convergence_rate=res.converged.mean()))
            print('Grid',alpha,beta,grid[-1]['profit_gain'],flush=True)
    discount=pd.DataFrame(discount);grid=pd.DataFrame(grid);discount.to_csv(OUT/'discount_summary.csv',index=False);grid.to_csv(OUT/'grid_summary.csv',index=False)
    meta=dict(**vars(args),pn=pn,pm=pm,profit_nash=pi_n,profit_monopoly=pi_m,prices=prices.tolist(),elapsed_seconds=time.perf_counter()-start,python=platform.python_version(),compiler=compiler,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'src/qlearn.cpp',Path(__file__)]},total_sessions=args.sessions+4*args.sensitivity_sessions+9*args.grid_sessions,stable_steps=100000)
    plots(main_results,impulse,pd.read_csv(str(baseline)+'_learning.csv'),discount,grid,pn,pm,pi_n,pi_m);report(main_results,discount,grid,meta)
    (OUT/'audit.json').write_text(json.dumps(meta,indent=2));print('Completed:',OUT/'index.html','seconds',meta['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
