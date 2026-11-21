# 第9课作业提交说明

**宋昶辉｜20251030008｜金融学＋计算机科学与技术**

本项目使用课程提供的 `mall_customers_sim.csv` 与 `mall_customers_yuan.csv` 完成客户分群实验。主要成果包括KMeans聚类、肘部法、客户画像、特征对比、稳定性检验、二维散点图、量纲实验及三项独立研究。

核心聚类结果：300位客户分为4簇，人数分别为50、54、102、94。三组随机种子比较的ARI均为1.0000。

详细结果见 `generated/mall_report.md`、`generated/feature_comparison.md`、`generated/stability_analysis.md` 和 `generated/research_*.md`。

注意：数据为教学模拟数据；当前数据字段与实践指南示例存在差异，`account_age_months` 扩展项尚未按原字段完成，提交前需确认教师口径。
