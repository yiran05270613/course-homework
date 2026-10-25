# T1：端到端库存管理公开样本复现

论文：Qi et al., *A Practical End-to-End Inventory Management Model with Deep Learning*, Management Science 69(2), 759–773. DOI: 10.1287/mnsc.2022.4564。

这是学习用的结果评估复现：使用作者公开预测和样本，重算主要离线比较、现场实验指标、回归、双重差分与匹配分布。不包含从零训练 E2E 网络、重新求解最优标签、多分位预测重训以及附录训练敏感性实验。

## 运行

从仓库根目录执行，推荐 Python 3.13（本次验证版本 3.13）：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r t1/requirements.txt
python t1/src/replicate.py
python t1/src/verify.py
```

直接打开 `results/index.html` 即可查看。HTML 内嵌 SVG 图，无外部网络依赖。依赖安装需要网络，实验运行不需要网络。目录默认以程序文件位置解析，从其他工作目录运行也可。

自定义输入与输出：

```sh
python t1/src/replicate.py --data-dir /absolute/path/to/data --output-dir /absolute/path/to/output
```

`--holding` / `--stockout` 可更改 h、b，默认 1、9；这仅改变评估成本及 BM2 公式，**不会重训作者预测，也不会重算 OPT 标签**，因此改变参数不能称为完整的成本比敏感性实验。HTML 的 paper_* 永远是论文默认参数参考值。

## 文件

- `src/replicate.py`：输入解析、库存模拟、指标、统计推断、匹配和报告生成。
- `src/verify.py`：直接读取并运行原作者 Notebook（只修复旧版 pandas 调用及移除不用的绘图库导入），与当前结果逐项核对。
- `results/index.html`：自包含中文结果页。
- `results/offline_summary.csv`：七种方法的作者计费时段与完整生成序列对照。
- `results/offline_per_sku.csv`：逐 SKU 库存成本与有效时段数。
- `results/field_comparison.csv`：现场实验五项指标、等方差 t 检验与 Welch 对照。
- `results/regression.csv`：作者无截距规格、论文公式含截距规格。
- `results/did.csv`：均值双重差分、作者 t 检验、按组别与单位标识聚类的两期回归。
- `results/field_daily_sensitivity.csv`：按实际计费点数归一化的成本对照；零计费点记录排除。
- `results/field_*_include_last.csv`：纳入库存最后一点的敏感性对照，其他定义不变。
- `results/matching_pairs.csv` / `matching_balance.csv`：重算匹配行号、距离与标准化均值差。
- `results/paper_comparison.csv`：公开样本计算值与论文表 2、表 4 参考值。
- `results/audit.json`：样本数量、记录长度、输入 SHA-256 与计算耗时。

## 忠实复现与限制

1. README 称离线样本为 3,000 个 SKU，但实际 CSV 是 3,000 行、2,607 个唯一 SKU 标识；程序按原作者方式按 SKU 聚合，不擅自补全缺失订单。
2. 离线预测列在 README 中称补货量，Notebook 实际按目标库存扣减参考库存计算补货；这里忠实使用 Notebook 的行为。
3. 模拟保留作者的到货前负库存处理、整数舍入、允许负补货、历史库存索引不足时的备用引用。主成本只统计到货后点；完整生成序列对照仍沿用同一模拟器，不能视为经过修正的完整库存模型。
4. 部分 SKU 没有到货后计费点，作者均值仍将其零成本纳入。审计文件明确记录；OPT 是既有标签，不能据此宣称重新证明最优性。
5. Normal 方法保留作者 Z90=1.64（并非标准正态 90% 分位数）及原方差表达式；Gamma 使用 0.9 分位。BM1/BM2 基于公开 sf_rnn 均值，无法恢复原文全套分位选择。
6. 现场实验成本与缺货天数使用 `test_inv[:-1]`，缺货比例分母使用原序列长度。周转指标使用 `max(0, ave_inv/ave_demand)`。它更接近周转天数，不是年度周转次数。
7. **各组库存序列长度不一致，原始累计成本比较受到观测时长影响。** 每日成本敏感性只是一项对照，不自动解决选择偏差或因果识别问题。
8. 论文回归公式含截距，作者 Notebook 漏掉截距；两种结果均报告。作者 DiD t 检验使用行位置相减，没有核对单位关系，后期控制组周转量还未截零。这里将该检验单独标为作者算法，并提供统一截零指标的回归结果。
9. 聚类 DiD 使用公开配对 ID 的跨期稳定性假设；数据说明未提供原始身份映射，无法独立确认。仅有一个前期汇总，无法检查平行趋势。
10. 作者所谓倾向得分匹配实际按需求与提前期原始尺度做最近邻，允许重复控制行；逻辑回归倾向得分只用于绘图。重算匹配不能与另一个文件的匿名单位 ID 自动关联；现场统计始终使用提供的控制组。
11. 论文全样本与当前公开子样本不同，未要求或调参使结果接近原表。本次得到的现场成本改善比例不能代替论文总体结论。

## GitHub 保存

保留 `t1/` 及原数据包的相对目录即可重新运行。HTML 可单独上传并展示；默认生成到 `t1/results/index.html`。原文 PDF 无需随代码上传。原始数据与作者 Notebook 的再分发遵循其原有授权，本项目不另行宣称拥有它们的许可。
