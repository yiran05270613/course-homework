"""Runnable learning-scale replication of Yan et al. (2026)."""
from pathlib import Path
from dataclasses import asdict
import argparse, base64, hashlib, json, os, platform, time
os.environ.setdefault('MPLCONFIGDIR','/tmp/t2-matplotlib')
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Config,Policy,sample,simulate,train
from dp import solve

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'t2';OUT=HERE/'results'
METHODS=['LIR','WNH','GDSH','DRL'];COLORS=['#87969a','#d49a5c','#7b71a3','#247d91']

def cases(seed):
    result=[]
    for distribution in ['normal','gamma']:
        for L in [2,5]:
            for p in [.6,.8,.95]:result.append(Config(f'{distribution}_L{L}_p{p:g}',L=L,p=p,distribution=distribution))
    for L in [2,5]:
        for ratio in [.85,.95]:result.append(Config(f'normal_L{L}_r{ratio:g}',L=L,critical_ratio=ratio))
        result.append(Config(f'normal_L{L}_T100',L=L,T=100))
    for j,c in enumerate(result):c.seed=seed+j*10000
    return result

def evaluate_case(c,models,test,selected):
    d,y=sample(c,test,np.random.default_rng(c.seed+3000));allcost={};components={};traces={}
    with torch.no_grad():
        for k,m in models.items():
            cost,parts,tr=simulate(m,d,y,trace=True);allcost[k]=cost.numpy();components[k]=parts.numpy();traces[k]=tr[0].numpy()
    rows=[]
    for k in METHODS:
        cost=allcost[k];difference=cost-allcost['LIR']
        rows.append(dict(case=c.name,method=k,L=c.L,T=c.T,p=c.p,distribution=c.distribution,critical_ratio=c.critical_ratio,target=c.target,mean_cost=cost.mean(),se_cost=cost.std(ddof=1)/np.sqrt(test),holding=components[k][:,0].mean(),backlog=components[k][:,1].mean(),purchase=components[k][:,2].mean(),change_pct=100*(cost.mean()/allcost['LIR'].mean()-1),paired_difference=difference.mean(),paired_ci95=1.96*difference.std(ddof=1)/np.sqrt(test),selected_epoch=selected.get(k,0)))
    pd.DataFrame(allcost).to_csv(OUT/'raw'/f'{c.name}_test_costs.csv',index=False)
    for k,tr in traces.items():pd.DataFrame(tr,columns=['inventory_before_demand','order','arrival','ending_inventory','demand','yield']).to_csv(OUT/'raw'/f'{c.name}_{k}_trajectory.csv',index=False)
    return rows

def save_models(c,models):
    config={k:(v.item() if isinstance(v,np.generic) else v) for k,v in asdict(c).items()}
    for k in ['GDSH','DRL']:torch.save(dict(config=config,kind=k,state=models[k].state_dict()),OUT/'models'/f'{c.name}_{k}.pt')

def savefig(name):
    plt.tight_layout();plt.savefig(OUT/f'{name}.png',dpi=160);plt.savefig(OUT/f'{name}.svg');plt.close()

def figures(result,history,small):
    plt.rcParams.update({'figure.figsize':(9,4.8),'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    means=result.groupby('method').mean_cost.mean().reindex(METHODS);fig,ax=plt.subplots();ax.bar(METHODS,means,color=COLORS);ax.set(ylabel='Mean discounted cost',title=f'Four policies: equal-weight average over {result.case.nunique()} representative cases');savefig('cost_comparison')
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,L in zip(axs,[2,5]):
        sel=result[(result.L==L)&(result['T']==50)&(result.critical_ratio==.9)]
        for k,color in zip(METHODS,COLORS):
            v=sel[sel.method==k].groupby('p').change_pct.mean();ax.plot(v.index,v.values,'o-',label=k,color=color)
        ax.axhline(0,ls=':',color='#777');ax.set(xlabel='Supply success probability p',title=f'Leadtime L={L}',ylabel='Cost change vs LIR (%)')
    axs[1].legend();savefig('supply_sensitivity')
    ref='normal_L2_p0.8';hist=history[history.case==ref];fig,ax=plt.subplots()
    for k,color in [('GDSH',COLORS[2]),('DRL',COLORS[3])]:
        h=hist[hist.method==k];ax.plot(h.epoch,h.validation_cost,label=k,color=color)
        r=result[(result.case==ref)&(result.method==k)].iloc[0];best=h[h.epoch==r.selected_epoch].iloc[0];ax.scatter(best.epoch,best.validation_cost,color=color,s=40)
    ax.set(xlabel='Training epoch',ylabel='Validation discounted cost',title='Training and checkpoint selection: normal, L=2, p=0.8');ax.legend();savefig('training')
    fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for k,col in zip(METHODS,COLORS):
        tr=pd.read_csv(OUT/'raw'/f'{ref}_{k}_trajectory.csv');t=np.arange(1,len(tr)+1);axs[0].plot(t,tr.ending_inventory,label=k,color=col);axs[1].plot(t,tr.order,label=k,color=col)
    axs[0].axhline(0,color='#aaa',ls=':');axs[0].set(ylabel='End-of-period net inventory',title='One shared test path (illustration, not an average)');axs[0].legend(ncol=4);axs[1].set(xlabel='Period',ylabel='Order quantity');savefig('inventory_path')
    fig,ax=plt.subplots();x=np.arange(len(small.p.unique()));width=.18
    for j,(k,col) in enumerate(zip(METHODS,COLORS)):
        s=small[small.method==k].sort_values('p');ax.bar(x+(j-1.5)*width,s.gap_pct,width,label=k,color=col)
    ax.set_xticks(x,[f'p={p:g}' for p in sorted(small.p.unique())]);ax.set(ylabel='Cost gap to discrete DP (%)',title='Exact DP within the stated discrete, bounded-action model');ax.legend();savefig('dp_comparison')
    # A state slice is illustrative; it is not the full state space.
    c=Config(ref,L=2,p=.8);checkpoint=torch.load(OUT/'models'/f'{ref}_DRL.pt',weights_only=True);c=Config(**checkpoint['config']);m=Policy(c,'DRL');m.load_state_dict(checkpoint['state'])
    inv=np.linspace(-80,100,90);pipe=np.linspace(0,140,90);ii,qq=np.meshgrid(inv,pipe)
    with torch.no_grad():z=m(torch.tensor(ii.ravel(),dtype=torch.float32),torch.tensor(qq.ravel()[:,None],dtype=torch.float32)).numpy().reshape(ii.shape)
    fig,ax=plt.subplots();im=ax.imshow(z,origin='lower',extent=[inv.min(),inv.max(),pipe.min(),pipe.max()],aspect='auto',cmap='YlGnBu');ax.set(xlabel='Inventory after arrival',ylabel='Outstanding order',title='Learned DRL ordering policy: L=2, p=0.8');fig.colorbar(im,ax=ax,label='Order quantity');savefig('policy_surface')

def report(result,history,small,meta):
    means=result.groupby('method').mean_cost.mean().reindex(METHODS);reduction=100*(1-means['DRL']/means['LIR']);g_reduction=100*(1-means['GDSH']/means['LIR'])
    summary=pd.DataFrame(dict(method=METHODS,mean_cost=means.values,change_pct=100*(means.values/means['LIR']-1)));summary.to_csv(OUT/'overall_summary.csv',index=False)
    def img(name):return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((OUT/f'{name}.png').read_bytes()).decode()+'">'
    def table(df):return df.to_html(index=False,border=0,float_format=lambda x:f'{x:,.4g}')
    selected=result[result.method.isin(['DRL','GDSH'])]
    diagnostic_html=''
    if (OUT/'loss_aggregation_sensitivity.csv').exists() and (OUT/'loss_sensitivity_audit.json').exists() and all(json.loads((OUT/'loss_sensitivity_audit.json').read_text()).get(key)==meta.get(key) for key in ['epochs','batch','validation','test_paths','seed']):
        diagnostic=pd.read_csv(OUT/'loss_aggregation_sensitivity.csv')
        reference=result[(result.method=='DRL')&result.case.isin(diagnostic.case)][['case','mean_cost','change_pct']].rename(columns={'mean_cost':'primary_trajectory_loss_cost','change_pct':'primary_change_pct'})
        diagnostic=diagnostic.merge(reference,on='case')
        diagnostic_html='<section><h2>07 / 损失聚合方式的独立诊断</h2>'+table(diagnostic)+'<p>主结果按每条轨迹先计算指数损失，再对轨迹平均。论文没有提供代码说明批量聚合的全部细节，因此额外对两个预先指定案例测试“先对成本求均值、再取相对指数”的版本；该版本的最优解等价于最小化平均成本。它另行训练、仍由验证集选检查点，不替换上方主 DRL 结果。不能据此断言作者原始实现采用哪种方式。可运行 diagnose_loss.py 单独重算。</p></section>'
    ranking_note='本次主 DRL 没有优于 GDSH，因此未完整复现论文中 DRL 排名最优的结论。' if means['DRL']>means['GDSH'] else '本次平均成本排序中 DRL 优于 GDSH。'
    body=f'''<header><small>T2 · COMPUTERS &amp; OPERATIONS RESEARCH · 2026</small><h1>供货全成或全败时，如何补货？</h1><p>基于论文公式的独立实现 · 四种策略 · 实际神经网络训练</p><p>公开文献没有随项目提供 T2 源代码。本页数字由本次模拟与训练生成，没有用论文数字替换实验输出。</p></header>
    <section><h2>01 / 实验范围与结果</h2><div class="cards"><article><b>{len(result.case.unique())}</b><span>代表性连续需求案例</span></article><article><b>{reduction:.2f}%</b><span>DRL 相对 LIR 平均成本降低</span></article><article><b>{g_reduction:.2f}%</b><span>GDSH 相对 LIR 平均成本降低</span></article><article><b>{meta['elapsed_seconds']/60:.1f} 分钟</b><span>训练、评估及 DP 分析耗时</span></article></div><p>{ranking_note} 正数降低代表成本更低，负数代表更高。基准设定：μ=10，单位采购成本 2、持有成本 1；缺货成本由临界比例 b/(b+h) 决定。覆盖 L=2/5、p=0.6/0.8/0.95、正态/伽马需求，另补充临界比例和 T=100 对照。每个实例独立训练，未使用一个模型通吃所有参数。</p><p>每个实例、每种可训练方法使用 {meta['epochs']} 个训练 epoch，每次生成 {meta['batch']} 条新轨迹；固定独立验证集 {meta['validation']} 条，独立测试集 {meta['test_paths']} 条，四种方法共享同一测试需求和供货成功样本。仅一个训练随机种子/实例，测试路径区间不涵盖训练种子不确定性。</p></section>
    <section><h2>02 / 四种方法比较 · 表 2 / 表 3 的缩减复现</h2>{img('cost_comparison')}{table(summary)}<p>对本次各实例的平均成本等权平均，再计算相对 LIR 的变化；这不是逐实例百分比的平均。论文为 432 个大规模案例，报告 DRL 平均降低约 3.6%、GDSH 约 3.44%；实验网格与随机数不同，当前数字不能作为原表的精确复刻。该论文数字仅用于解释，没有参与优化。</p>{table(result[['case','method','mean_cost','change_pct','paired_difference','paired_ci95','selected_epoch']])}<p>paired_difference 为同一测试路径上“该方法−LIR”的均值；paired_ci95 是近似 95% 半宽，使用独立测试路径的标准误。</p></section>
    <section><h2>03 / 供货成功率与提前期 · 图 5 / 表 3 的分析思路</h2>{img('supply_sensitivity')}<p>图中仅比较 T=50、临界比例 0.9 的 12 个核心案例；正态与伽马案例按相同供货概率平均。更长提前期与更低供货概率增加风险，改善幅度由实际训练结果决定，没有预设方法排名。</p></section>
    <section><h2>04 / 真实训练与检查点</h2>{img('training')}<p>网络隐藏层为 [8,16,16]，ReLU；每条供货组合输出两个 logits，经行 Softmax 映射到 WNH 权重 ±δ。δ 在 L=2 时为 0.25，L=5 时为 0.0884。权重不是在全部组合上归一化的概率。网络末层零初始化，使初始策略严格等于 WNH；输入除以 S_L，属于本实现的数值稳定处理。</p><p>DRL 使用论文的 exp(100×(C_DRL−C_WNH)/C_WNH) 轨迹相对损失，计算时以 log-mean-exp 等价单调变换避免溢出；梯度经完整库存路径反向传播。GDSH 直接优化样本平均折现成本，权重不随状态变化。Adam 初始学习率 0.003，每 epoch 乘 0.98、下限 0.0001；梯度范数上限 10 为本实现稳定措施。</p><p>每 10 epoch 在独立验证集选择平均成本最低的检查点，包含初始 WNH 候选。共 {len(selected)} 个连续需求模型，其中 {int((selected.selected_epoch==0).sum())} 个保留初始检查点；这是公开的验证选择，不是依据测试结果替换策略。完整训练历史和模型权重已保存。</p></section>
    <section><h2>05 / 库存、订货与策略</h2>{img('inventory_path')}{img('policy_surface')}<p>轨迹为预先固定测试集的第一条，没有挑选最好看的案例。策略图仅是 L=2 的状态切片；不是论文原图像素重绘。期初先接收到期合格订单，再决策，再发生需求，期末计持有/积压成本；采购费只在整批合格到货时收取。最后 L 期不再下单。</p></section>
    <section><h2>06 / 小规模动态规划 · 图 4 的近最优性检验思路</h2>{img('dp_comparison')}{table(small)}<p>这里明确改变为离散小模型：T=8、L=2，D∈{{1,2,3}}、概率 (0.25,0.5,0.25)，整数行动 q∈{{0,…,12}}，p=0.7/0.8/0.9，折现因子 0.95。DP 使用精确概率递推，无库存边界截断；四种策略采用同样的整数取整与上限，在该模型下也以精确期望递推评估，消除蒙特卡洛误差。</p><p>小模型的网络独立重训，训练采用直通估计器处理取整；DP 是这个受限离散问题的最优解，不能称为原论文连续正态需求问题的精确最优解。论文原文的 972 个小案例未全量复现。</p></section>
    {diagnostic_html}<section><h2>08 / 运行与复现边界</h2><pre>python -m pip install -r t2/requirements.txt\npython t2/src/replicate.py\npython t2/src/verify.py</pre><p>Python + PyTorch 在 CPU 上训练，无需外部数据、API 或 GPU。HTML 内嵌图片，可离线查看；各图另存 PNG/SVG，模型在 models/，逐路径成本、训练历史、配置与审计数据均保存。</p><p>主要实现约定：对权重组合得到的订货量施加非负约束，GDSH 权重自由优化但行动仍非负；权重区间本身可含负数，未擅自将其裁剪成概率。S_L 使用已知模拟分布的参数；正态需求在 0 截断但计算 S_L 使用截断前参数。不同于原文最后 epoch 的做法，本版使用验证集检查点；这些改动及有限实例与单种子限制详见 README。</p><p>文献：Yan, Chen, Fu &amp; Bi, Heuristics and deep reinforcement learning for the inventory problem with an all-or-nothing yield pattern and non-zero leadtimes, Computers &amp; Operations Research 186 (2026), 107305. DOI: 10.1016/j.cor.2025.107305。</p></section>'''
    css='body{margin:0;background:#f3f5f4;color:#203841;font:16px/1.8 system-ui,-apple-system,"PingFang SC",sans-serif}header,section{max-width:1080px;margin:26px auto;padding:32px 40px;background:white;border-radius:16px}header{background:#173d48;color:white}h1{font-size:42px;margin:14px 0}h2{font-size:23px;color:#246c7c}small{letter-spacing:2px}img{width:100%;height:auto}.cards{display:flex;gap:12px;flex-wrap:wrap}.cards article{flex:1;min-width:160px;background:#edf4f4;padding:18px;border-radius:10px}.cards b{display:block;font-size:28px;color:#246c7c}.cards span{font-size:13px}table{display:block;overflow:auto;border-collapse:collapse;font-size:13px}th,td{padding:9px 12px;border-bottom:1px solid #dce5e7;white-space:nowrap;text-align:right}th{background:#edf4f4}pre{padding:20px;background:#edf4f4;overflow:auto}@media(max-width:700px){header,section{margin:12px;padding:20px}h1{font-size:28px}}'
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>T2 随机供货库存复现</title><style>'+css+'</style>'+body+'</html>',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report-only',action='store_true');parser.add_argument('--epochs',type=int,default=300);parser.add_argument('--batch',type=int,default=500);parser.add_argument('--validation',type=int,default=256);parser.add_argument('--test-paths',type=int,default=1000);parser.add_argument('--seed',type=int,default=20261024);parser.add_argument('--case-limit',type=int,default=0);args=parser.parse_args()
    if min(args.epochs,args.batch,args.validation,args.test_paths)<2:parser.error('counts must be at least 2')
    if args.case_limit and args.case_limit<3:parser.error('case limit must be 0 or at least 3')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if args.report_only:
        result=pd.read_csv(OUT/'case_results.csv');history=pd.read_csv(OUT/'training_history.csv');small=pd.read_csv(OUT/'dp_results.csv');meta=json.loads((OUT/'audit.json').read_text());figures(result,history,small);report(result,history,small,meta);return
    start=time.perf_counter()
    for name in ['raw','models']: (OUT/name).mkdir(parents=True,exist_ok=True)
    configs=cases(args.seed)
    if args.case_limit:configs=configs[:args.case_limit]
    allrows=[];histories=[];small=[];dp_audit=[]
    for j,c in enumerate(configs):
        tick=time.perf_counter();models,hist,selected=train(c,args.epochs,args.batch,args.validation);rows=evaluate_case(c,models,args.test_paths,selected);save_models(c,models)
        allrows.extend(rows);histories.extend(dict(case=c.name,**r) for r in hist)
        pd.DataFrame(allrows).to_csv(OUT/'case_results.csv',index=False);pd.DataFrame(histories).to_csv(OUT/'training_history.csv',index=False)
        description=', '.join('{}={:.2f}'.format(r['method'],r['mean_cost']) for r in rows)
        print(f'[{j+1}/{len(configs)}] {c.name}: {description} ({time.perf_counter()-tick:.1f}s)',flush=True)
    for j,p in enumerate([.7,.8,.9]):
        c=Config(f'discrete_p{p:g}',L=2,T=8,p=p,distribution='discrete',mu=2,sigma=np.sqrt(.5),discount=.95,action_cap=12,seed=args.seed+500000+j*10000)
        models,hist,selected=train(c,args.epochs,args.batch,args.validation);save_models(c,models);histories.extend(dict(case=c.name,**r) for r in hist)
        optimal,decisions,nstates=solve(c)
        for k,m in models.items():
            expected,_,visited=solve(c,m);gap=100*(expected/optimal-1)
            if gap < -1e-7:raise AssertionError('Policy is below exact DP')
            small.append(dict(case=c.name,p=p,method=k,expected_cost=expected,dp_cost=optimal,gap_pct=gap,selected_epoch=selected.get(k,0)))
        dp_audit.append(dict(config=asdict(c),optimal=optimal,states=nstates))
        print(f'Discrete DP p={p}: optimal={optimal:.6f}, states={nstates}',flush=True)
    result=pd.DataFrame(allrows);history=pd.DataFrame(histories);small=pd.DataFrame(small)
    result.to_csv(OUT/'case_results.csv',index=False);history.to_csv(OUT/'training_history.csv',index=False);small.to_csv(OUT/'dp_results.csv',index=False)
    meta=dict(**vars(args),configs=[asdict(c) for c in configs],dp=dp_audit,elapsed_seconds=time.perf_counter()-start,torch=torch.__version__,python=platform.python_version(),threads=torch.get_num_threads(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'src').glob('*.py')})
    (OUT/'audit.json').write_text(json.dumps(meta,indent=2));figures(result,history,small);report(result,history,small,meta);print('Completed',OUT/'index.html','elapsed seconds',meta['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
