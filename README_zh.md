# 自然观看眼动交叉零模型再分析

本目录保存论文 **“Crossed randomization distinguishes global ocular-score differences from event-boundary-specific effects in naturalistic eye tracking”** 的本地投稿稿和可复现分析包。稿件按 *Behavior Research Methods* 的方法论文要求准备，目前没有投稿、发布预印本或上传公共仓库。

## 一句话结果

随机短视频组在全片尺度上出现了明显的队列相对眼动评分差异，但事件边界对比在移动边界位置后并不异常，因此公开数据不支持事件边界特异的注意或事件分割机制。

## 结果及解释边界

BBBD 基准分析包含 52 名参与者、251 个完整的参与者—视频配对和 502 条记录。模型区分了公开数据中的首次专注观看与分心重看标签，AUC 为 .876653，95% CI [.817378, .932143]。由于观看条件与顺序、重复观看完全混杂，本文把它称为“分心场次基准”，不称为注意生物标志物。

固定模型随后应用于 104 份 Sherlock 公开眼动记录。Short-R 组的全片对比为 -.753856，95% CI [-.773701, -.728420]，p = .000100。18 个标注边界上的对比为 -.042841；参与者标签检验得到 p = .001300，边界位置检验和交并检验均为 p = .522548。也就是说，数据支持全片尺度的队列相对评分分离，却没有显示该差异集中在标注的事件边界。

在作者指定的随机种子下，BBBD 眼动—EEG 条件调整关联为正，但 D10 种子纠正审计显示，其常用显著性阈值状态会随持出折分配改变。这项结果只能作为复用同一队列的次要、折分敏感关联，不能视为独立神经验证。记忆关联没有明确证据。单一商业影片、Sherlock 原始质量控制资料缺失、样本复用、跨数据集迁移以及窗口和通道敏感性，都限制了结论的推广范围。

## 目录说明

- `manuscript/`：匿名正文、标题页模板、补充材料、表格、图注、APA 参考文献、RIS、投稿信和投稿清单。
- `results/`：结果表、统计审计、作图源数据，以及 SVG、PDF 和 600 dpi PNG 三种格式的四张主图。
- `src/boundaryisc/`：Python 包和命令行程序。
- `tests/`：同步性、交叉验证、边界效应、缺失数据、CorrCA、模拟和固定随机种子的单元测试。
- `config/`、`analysis_plan.md`：冻结的分析定义和已披露的 D10 种子修正。
- `provenance/`：输入散列、数据归档审计、D01—D11 偏离记录、文献核验和两次复现散列。
- `evidence/`：文献注册表、新颖性审计和论断—证据矩阵。
- `literature/`：通过合法开放渠道取得的出版社 PDF、PMC 全文 XML 和下载清单。
- `data/data_dictionary.md`：原始、派生和报告字段的定义。
- `de_ai_audit.md`：中英文文字终审和不变量复核记录。

## 环境和运行

需要 Python 3.10 或更高版本。本次完整运行使用 macOS arm64 和 Python 3.13.14；精确版本见 `requirements-lock.txt`。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e .
PYTHONPATH=src .venv/bin/python -m boundaryisc.cli --config config/analysis.json validate-config
PYTHONPATH=src .venv/bin/python -m boundaryisc.cli --config config/analysis.json reproduce
PYTHONPATH=src .venv/bin/python -m boundaryisc.cli --config config/analysis.json analyze
PYTHONPATH=src .venv/bin/python -m boundaryisc.cli --config config/analysis.json simulate
PYTHONPATH=src .venv/bin/python -m boundaryisc.cli --config config/analysis.json report
.venv/bin/python -m pytest -q
python3 scripts/build_markdown_manuscript.py
python3 scripts/audit_manuscript.py
```

冻结配置记录的是本机绝对路径。其他使用者应保留 `config/analysis.json` 作为来源记录，另复制一份配置并填写自己合法取得的数据路径。BBBD 实验 2 和实验 3 可从原 Zenodo 记录获取，Sherlock 处理数据来自原 OSF 项目；本包不重新分发商业影片和第三方原始数据。种子 20260816 的两次完整运行生成了相同的数值表和作图源数据聚合散列，预期值见 `provenance/reproducibility_check.md`。

## 投稿前仍需处理

不要把 `data/raw/`、商业影片、原始 MAT 文件或 BBBD ZIP 文件上传到新仓库。正文中的 `AUTHOR_INPUT_NEEDED` 需要由作者填写，包括作者姓名、单位、通讯信息、ORCID、伦理认定、同意说明、资助、利益冲突、CRediT 分工、公共仓库地址、软件许可证和归档 DOI。本文没有公开预注册；分析计划在推断性再分析前完成本地冻结，之后的偏离均已记录。
