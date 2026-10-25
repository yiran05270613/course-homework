"""T1 public-sample replication. No training or network access is required."""
from pathlib import Path
import argparse, ast, hashlib, json, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
METRICS = ['holding', 'stockout', 'total', 'turnover', 'stockout_ratio']
LABELS = ['持有成本', '缺货成本', '总成本', '库存周转天数', '缺货比例']

def costs(values, h, b):
    a = np.asarray(values, dtype=float)
    return h * np.maximum(a, 0).sum(), b * np.maximum(-a, 0).sum()

def inventory(row, name):
    # Preserve the author's notebook, including rounding and negative replenishment.
    full, evaluated = [row['initial_stock'][0]], []
    fallback = 0
    for r, demand in enumerate(row['demand_hist']):
        length = len(demand) - (r < len(row['demand_hist']) - 1)
        lead = int(round(row['vlt'][r]))
        for t in range(length):
            if t == 0:
                if r == 0:
                    ref = full[0]
                else:
                    try: ref = full[-lead-1]
                    except IndexError:
                        ref = full[1]; fallback += 1
                replen = int(round(row[name][r] - ref))
            if t < lead:
                if r == 0: full.append(full[-1] - demand[t])
            elif t == lead:
                v = max(full[-1], 0) + replen - demand[t]
                full.append(v); evaluated.append(v)
            else:
                v = full[-1] - demand[t]
                full.append(v); evaluated.append(v)
    return full[1:], evaluated, fallback

def offline(data, out, h, b):
    df = pd.read_csv(data/'df_test.csv')
    df['sf_rnn'] = df.sf_rnn.map(ast.literal_eval)
    df['demand_hist'] = df.demand_hist.map(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    forecast = df.sf_rnn.map(np.mean)
    rl = df.review_period + df.vlt
    df['BM1'] = forecast * rl.astype(int)
    df['BM2'] = forecast * np.where(rl <= b, rl.astype(int), rl.astype(int)-rl//(b+h))
    df['Normal'] = (df.demand_mean*(df.review_period+df.vendor_vlt_mean) + 1.64*np.sqrt((df.review_period+df.vendor_vlt_mean)*df.demand_std**2+df.demand_std**2*df.vendor_vlt_std)).astype(int)
    theta = df.demand_std**2/(df.demand_mean+1e-4)
    shape = (df.review_period+df.vendor_vlt_mean).astype(int)*df.demand_mean/(theta+1e-4)
    df['Gamma'] = np.nan_to_num(stats.gamma.ppf(.9, a=shape, scale=theta), nan=0).astype(int)
    methods = {'OPT':'OPT_pred', 'E2E_RNN':'E2E_RNN_pred', 'E2E_GBM':'gbm_pred','BM1':'BM1','BM2':'BM2','Normal':'Normal','Gamma':'Gamma'}
    grouped = df.groupby('SKU', sort=True).agg(list)
    records=[]; trace=None
    for sku, row in grouped.iterrows():
        for method, col in methods.items():
            full, selected, fallback = inventory(row, col)
            for scope, inv in [('author',selected),('full_generated',full)]:
                hold, shortage = costs(inv,h,b)
                records.append(dict(SKU=sku,method=method,scope=scope,holding=hold,stockout=shortage,total=hold+shortage,days=len(inv),fallback=fallback))
            if trace is None and method=='E2E_RNN': trace=dict(SKU=sku,inventory=full)
    raw=pd.DataFrame(records);raw.to_csv(out/'offline_per_sku.csv',index=False)
    summary=raw.groupby(['scope','method'])[['holding','stockout','total','days']].mean().reset_index()
    summary.to_csv(out/'offline_summary.csv',index=False)
    return summary, dict(rows=len(df),unique_skus=len(grouped),zero_evaluated_skus=int(((raw.scope=='author')&(raw.method=='OPT')&(raw.days==0)).sum()),fallbacks=int(raw[raw.scope=='author'].fallback.sum()),zero_evaluated=int(((raw.scope=='author')&(raw.days==0)).sum())), trace

def field_metrics(df, h, b, include_last=False):
    rows=[]
    for _, r in df.iterrows():
        inv=np.asarray(ast.literal_eval(r.test_inv),float)
        used=inv if include_last else inv[:-1]
        hold, shortage=costs(used,h,b)
        ratio=float(r.ave_inv/r.ave_demand) if r.ave_demand>0 else np.nan
        rows.append(dict(pair=r.sku_dc_pair_index,holding=hold,stockout=shortage,total=hold+shortage,turnover=max(0,ratio) if np.isfinite(ratio) else np.nan,turnover_raw=ratio,stockout_ratio=float((used<0).sum()/len(inv)),demand=r.ave_demand,vlt=float(str(r.vlt).split()[0]),days=len(inv)))
    return pd.DataFrame(rows)

def field(data,out,h,b):
    groups={}
    for group,file in [('T_pre','order_e2e_pre_500.csv'),('T_post','order_e2e_post_500.csv'),('C_pre','order_jd_pre_500.csv'),('C_post','order_jd_post_500.csv')]:
        source=pd.read_csv(data/file)
        groups[group]=field_metrics(source,h,b)
        groups[group].to_csv(out/f'field_{group}.csv',index=False)
        field_metrics(source,h,b,True).to_csv(out/f'field_{group}_include_last.csv',index=False)
    
    daily=[]
    for g,v in groups.items():
        exposure=v.days-1
        for m in ['holding','stockout','total']:
            daily.append(dict(group=g,metric=m,mean_per_evaluated_day=(v[m]/exposure.replace(0,np.nan)).mean(),excluded_zero_days=int((exposure==0).sum())))
    pd.DataFrame(daily).to_csv(out/'field_daily_sensitivity.csv',index=False)
    rows=[]; regressions=[]; did=[]
    for metric in METRICS:
        t,c=groups['T_post'][metric],groups['C_post'][metric]
        rows.append(dict(metric=metric,treatment=t.mean(),control=c.mean(),difference=t.mean()-c.mean(),change_pct=100*(t.mean()/c.mean()-1),p_equal_var=stats.ttest_ind(t,c,nan_policy='omit').pvalue,p_welch=stats.ttest_ind(t,c,equal_var=False,nan_policy='omit').pvalue))
        combined=pd.concat([groups['T_post'].assign(treat=1),groups['C_post'].assign(treat=0)],ignore_index=True)
        for intercept in [False,True]:
            x=combined[['treat','demand','vlt']]
            if intercept:x=sm.add_constant(x)
            model=sm.OLS(combined[metric],x,missing='drop').fit()
            regressions.append(dict(metric=metric,intercept=intercept,coefficient=model.params['treat'],p=model.pvalues['treat'],n=int(model.nobs)))
        means=[groups[g][metric].mean() for g in ['T_pre','T_post','C_pre','C_post']]
        delta=means[1]-means[0]-means[3]+means[2]
        # The author pairs rows by position for its t-test, without joining identifiers.
        post=groups['T_post'][metric].to_numpy()-groups['C_post'][metric].to_numpy()
        pre=groups['T_pre'][metric].to_numpy()-groups['C_pre'][metric].to_numpy()
        if metric=='turnover':post=groups['T_post'][metric].to_numpy()-groups['C_post'].turnover_raw.to_numpy()
        panel=pd.concat([v.assign(treat=int(g[0]=='T'),post=int(g.endswith('post')),unit=g[0]+'_'+v.pair.astype(str)) for g,v in groups.items()],ignore_index=True)
        panel['interaction']=panel.treat*panel.post
        valid=panel[[metric,'treat','post','interaction','unit']].dropna()
        model=sm.OLS(valid[metric],sm.add_constant(valid[['treat','post','interaction']])).fit(cov_type='cluster',cov_kwds={'groups':valid.unit})
        did.append(dict(metric=metric,T_pre=means[0],T_post=means[1],C_pre=means[2],C_post=means[3],did=delta,p_author=stats.ttest_ind(post,pre,nan_policy='omit').pvalue,did_cluster=model.params['interaction'],p_cluster=model.pvalues['interaction']))
    tables=[pd.DataFrame(x) for x in [rows,regressions,did]]
    for name,table in zip(['field_comparison','regression','did'],tables):table.to_csv(out/f'{name}.csv',index=False)
    return groups,tables

def matching(data,out):
    treat=pd.read_csv(data/'order_e2e_post_500.csv')[['ave_demand','vlt_num']]
    control=pd.read_csv(data/'control_demand_vlt.csv')[['ave_demand','vlt_num']]
    x=pd.concat([treat,control],ignore_index=True);y=np.r_[np.ones(len(treat)),np.zeros(len(control))]
    model=LogisticRegression(max_iter=1000).fit(x,y)
    # Author actually matches raw covariates, not the propensity score.
    distance,idx=NearestNeighbors(n_neighbors=1).fit(control).kneighbors(treat)
    selected=control.iloc[idx[:,0]]
    scores=[model.predict_proba(v)[:,1] for v in [treat,control,selected]]
    pd.DataFrame(dict(treatment_row=np.arange(len(treat)),control_row=idx[:,0],distance=distance[:,0])).to_csv(out/'matching_pairs.csv',index=False)
    balance=[]
    for col in treat:
        for stage,c in [('before',control),('after',selected)]:
            den=np.sqrt((treat[col].var()+c[col].var())/2)
            balance.append(dict(feature=col,stage=stage,treatment_mean=treat[col].mean(),control_mean=c[col].mean(),smd=(treat[col].mean()-c[col].mean())/den))
    pd.DataFrame(balance).to_csv(out/'matching_balance.csv',index=False)
    return scores,dict(candidates=len(control),matched=len(selected),unique_matches=len(np.unique(idx)))

def table_html(df):
    return df.to_html(index=False,border=0,float_format=lambda x:f'{x:,.4g}',classes='data')

def bars(df, columns, labels, title):
    colors=['#317d91','#dd8c50','#7b75b7']; width=940;height=340;left=85;bottom=270;top=40
    values=df[columns].to_numpy(float);maximum=max(values.max()*1.15,1);n=len(df);step=810/max(n,1);bw=min(26,step/(len(columns)+1))
    s=[f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}"><rect width="100%" height="100%" fill="white"/>']
    for j in range(5):
        value=maximum*j/4;yy=bottom-(bottom-top)*j/4
        s.append(f'<path d="M{left},{yy} H910" stroke="#e2e8eb"/><text x="75" y="{yy+4}" text-anchor="end">{value:,.0f}</text>')
    for i,(_,r) in enumerate(df.iterrows()):
        for j,col in enumerate(columns):
            v=float(r[col]);hh=max(v,0)/maximum*(bottom-top);xx=left+i*step+12+j*bw
            s.append(f'<rect x="{xx}" y="{bottom-hh}" width="{bw-2}" height="{hh}" fill="{colors[j]}"><title>{r.iloc[0]} {labels[j]}: {v:.3f}</title></rect>')
        s.append(f'<text x="{left+i*step+step/2}" y="292" text-anchor="middle">{r.iloc[0]}</text>')
    for j,l in enumerate(labels):s.append(f'<rect x="{110+j*240}" y="315" width="13" height="13" fill="{colors[j]}"/><text x="{130+j*240}" y="327">{l}</text>')
    return ''.join(s)+'</svg>'

def hist_svg(scores):
    edges=np.linspace(min(map(np.min,scores)),max(map(np.max,scores)),31)
    parts=['<svg viewBox="0 0 940 330" role="img" aria-label="匹配前后倾向得分分布">']
    for panel,other in enumerate([scores[1],scores[2]]):
        xx=panel*470
        maxfreq=max(np.histogram(v,edges)[0].max()/len(v) for v in [scores[0],other])*1.1
        scale=220/max(maxfreq,1e-9)
        parts.append(f'<text x="{xx+80}" y="25">匹配{["前","后"][panel]}（各组频率）</text>')
        for vals,color in [(scores[0],'#317d91'),(other,'#dd8c50')]:
            counts=np.histogram(vals,edges)[0]/len(vals)
            for i,v in enumerate(counts):parts.append(f'<rect x="{xx+45+i*13}" y="{270-v*scale}" width="12" height="{v*scale}" fill="{color}" opacity=".5"/>')
        for k in [0,15,30]:parts.append(f'<text x="{xx+45+k*13}" y="295" text-anchor="middle">{edges[k]:.3f}</text>')
    return ''.join(parts)+'</svg><p>蓝色：处理组；橙色：控制组。匹配有放回，使用原始协变量的欧氏距离。</p>'

def report(out,summary,groups,tables,scores,meta):
    comp,reg,did=tables
    offline_table=summary[summary.scope=='author'].drop(columns='scope').sort_values('total')
    paper=pd.DataFrame(dict(metric=METRICS,paper_treatment=[598.20,496.34,1094.54,18.59,.17],paper_control=[809.54,1027.93,1837.47,20.39,.26],paper_did=[-112.79,-524.45,-637.24,-1.81,-.11]))
    comparison=comp.merge(paper,on='metric');comparison.to_csv(out/'paper_comparison.csv',index=False)
    labelmap=dict(zip(METRICS,LABELS))
    def display(df):return table_html(df.replace({'metric':labelmap}))
    body=f'''<header><small>MANAGEMENT SCIENCE · T1 · 公开样本复现</small><h1>从预测到库存决策</h1><p>端到端库存管理 · 可重复计算的学习报告</p><p>本页所有“复现”数字来自本地 CSV 的实际计算；E2E 和 GBM 使用作者提供的预测，未重新训练神经网络。</p></header>
    <section><h2>01 / 数据与复现范围</h2><p>离线数据 {meta['offline']['rows']:,} 条记录，实际 {meta['offline']['unique_skus']:,} 个唯一 SKU 标识。现场实验四组各 500 条记录。控制候选 {meta['matching']['candidates']:,} 条。计算耗时 {meta['elapsed_seconds']:.2f} 秒（不含安装依赖与开发）。</p><p>固定 h={meta['h']:g}，b={meta['b']:g}。公开材料是抽样数据，不能期望与原论文全样本数值完全一致。原论文图 1 是网络结构示意图，本项目不将结构重绘冒充模型训练复现。</p></section>
    <section><h2>02 / 离线模型比较 · 对应图 2(b)</h2>{bars(offline_table[['method','holding','stockout','total']],['holding','stockout','total'],['持有成本','缺货成本','总成本'],'离线模型成本')}{display(offline_table)}<p>审计发现 {meta['offline']['zero_evaluated_skus']} 个 SKU 没有到货后可评估点，按作者代码其成本计为零并进入均值。该现象以及 77 次备用库存引用（跨七种方法计数）保留在主复现，不能视作完善的生产级模拟。主结果严格沿用作者 Notebook：仅统计每段预测到货时点及之后的库存。完整生成序列成本另见 offline_summary.csv 的 full_generated 行；这仍是作者模拟器的口径对照，不是另行实现的完整库存模型。</p><p>BM1/BM2 使用公开 sf_rnn 的均值构造；原论文还涉及多分位数预测选择，当前数据无法独立恢复全部候选模型。Normal 保留作者代码的 1.64 系数与方差公式，Gamma 使用 0.9 分位数。OPT 为公开文件中的离线最优预测参考，不代表我们重新求解并验证了全局最优。</p></section>
    <section><h2>03 / 现场实验 · 对应表 2</h2>{bars(pd.DataFrame([['E2E',*comp.treatment[:3]],['JD',*comp.control[:3]]],columns=['group','holding','stockout','total']),['holding','stockout','total'],['持有成本','缺货成本','总成本'],'现场成本')}{display(comp)}<p>p_equal_var 为作者使用的双侧等方差独立样本 t 检验，p_welch 为不等方差对照。成本按库存序列去掉最后一点计算；缺货比例分母保留原序列长度。库存周转指标 max(0, ave_inv / ave_demand) 的单位接近天数，不是年度周转次数。各组库存记录长度不一致（前期处理组均为 30 点，后期处理组为 2–32 点，后期控制组为 1–32 点）；总成本对比受到暴露时长影响，不能直接解释为相同时长下的因果节约。field_daily_sensitivity.csv 提供每个计费点的平均成本对照，排除零计费点记录。含最后一点的敏感性结果见 field_*_include_last.csv；无效周转值排除并记录在审计文件。</p></section>
    <section><h2>04 / 控制协变量的回归 · 对应表 3</h2>{display(reg)}<p>解释变量：E2E 指示变量、平均需求、提前期。intercept=False 对应作者代码；True 对应论文公式中含截距的规格。此处报告常规 OLS 标准误。</p></section>
    <section><h2>05 / 双重差分 · 对应表 4</h2>{display(did)}<p>DiD = (处理组后−处理组前) − (控制组后−控制组前)。p_author 保留作者按行位置相减后进行独立 t 检验的算法，其中后期控制组周转量未截零；这不能视为验证了匹配关系的面板检验。did_cluster 与 p_cluster 使用两期回归并按“组别+公开 pair 标识”聚类；标识跨期稳定性依赖数据说明，不能由文件独立确认。仅有一个前期汇总，无法检验平行趋势。</p></section>
    <section><h2>06 / 匹配分布 · 对应图 3</h2>{hist_svg(scores)}<p>作者 Notebook 标题为倾向得分匹配，但实际选邻居依据是平均需求与提前期的原始数值，倾向得分用于画图。这里忠实重现这一行为。500 次匹配涉及 {meta['matching']['unique_matches']} 个不同候选行。重算匹配只用于本节；现场统计继续使用作者提供的控制组文件，未擅自将候选行号与匿名配对标识关联。</p></section>
    <section><h2>07 / 与论文报告值对照</h2>{display(comparison)}<p>paper_* 为论文表 2、表 4 人工录入的参考数值，其他列为本次实际计算。未将原值写入复现计算，也未为逼近论文数值调整数据。</p></section>
    <section><h2>08 / 如何重新运行</h2><pre>python -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r t1/requirements.txt\npython t1/src/replicate.py</pre><p>输出：t1/results/index.html、逐样本计算、汇总 CSV、匹配索引与审计 JSON。可以离线打开本页；无 CDN、在线字体或远程脚本依赖。</p><p>方法与限制详见 README.md。源文献 DOI：10.1287/mnsc.2022.4564。</p></section>'''
    import re
    for name,svg in zip(['offline_costs','field_costs','matching_scores'],re.findall(r'<svg.*?</svg>',body,flags=re.S)):
        (out/f'{name}.svg').write_text(svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ',1),encoding='utf-8')
    style='body{margin:0;background:#f3f5f4;color:#20343c;font:16px/1.8 system-ui,-apple-system,"PingFang SC",sans-serif}header,section{max-width:1080px;margin:28px auto;padding:32px 42px;background:white;border-radius:16px}header{background:#173d48;color:white;padding-top:48px}h1{font-size:44px;letter-spacing:2px;margin:12px 0}h2{font-size:23px;color:#226778}small{letter-spacing:2px}p{color:inherit}svg{width:100%;height:auto}svg text{font:13px system-ui;fill:#334b54}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto;width:100%}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #dde5e8;white-space:nowrap}th{background:#eaf2f3}pre{background:#edf3f4;padding:20px;overflow:auto}@media(max-width:700px){header,section{margin:12px;padding:20px}h1{font-size:30px}}'
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>T1 库存管理复现</title><style>'+style+'</style><main>'+body+'</main></html>',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=ROOT/'文献和数据/replication_package_MS-FTBDA-20-01585');parser.add_argument('--output-dir',type=Path,default=ROOT/'t1/results');parser.add_argument('--holding',type=float,default=1);parser.add_argument('--stockout',type=float,default=9);args=parser.parse_args()
    if args.holding<=0 or args.stockout<=0:parser.error('costs must be positive')
    start=time.perf_counter();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    summary,audit,trace=offline(args.data_dir,out,args.holding,args.stockout)
    groups,tables=field(args.data_dir,out,args.holding,args.stockout)
    scores,match=matching(args.data_dir,out)
    meta=dict(h=args.holding,b=args.stockout,offline=audit,matching=match,field={g:dict(n=len(v),invalid_turnover=int(v.turnover.isna().sum()),unique_ids=int(v.pair.nunique()),days_min=int(v.days.min()),days_max=int(v.days.max())) for g,v in groups.items()},inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data_dir.glob('*.csv')},elapsed_seconds=time.perf_counter()-start)
    (out/'audit.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));(out/'inventory_example.json').write_text(json.dumps(trace))
    report(out,summary,groups,tables,scores,meta)
    print(summary.to_string(index=False));print(tables[0].to_string(index=False));print(json.dumps(meta,ensure_ascii=False,indent=2));print('Report:',out/'index.html')
if __name__=='__main__':main()
