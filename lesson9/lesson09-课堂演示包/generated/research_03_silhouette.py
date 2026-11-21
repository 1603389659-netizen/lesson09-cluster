# -*- coding: utf-8 -*-
"""
独立研究三：轮廓系数（silhouette_score）

基于 data/mall_customers_sim.csv 的四个数值特征（age、income、balance、product_count），
StandardScaler 标准化后，对 K=2..9 分别做 KMeans 并计算轮廓系数，
打印真实数值、保存 CSV，并生成分析报告。

产出：
    generated/silhouette_results.csv
    generated/research_03_silhouette.md
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

CSV_PATH = "data/mall_customers_sim.csv"
OUT_CSV = "generated/silhouette_results.csv"
OUT_MD = "generated/research_03_silhouette.md"

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
FEATURES = ["age", "income", "balance", "product_count"]
X = StandardScaler().fit_transform(df[FEATURES])
print("=" * 68)
print(f"数据：{CSV_PATH}   样本数：{len(df)}   特征：{FEATURES}")
print("=" * 68)

rows = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    s = silhouette_score(X, labels)
    rows.append({"K": k, "轮廓系数": round(float(s), 6), "inertia": round(float(km.inertia_), 4)})
    print(f"K={k}：轮廓系数 = {s:.6f}   inertia_ = {km.inertia_:.4f}")

res = pd.DataFrame(rows)
res.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n已保存：{OUT_CSV}")

best = res.loc[res["轮廓系数"].idxmax()]
k4 = res.loc[res["K"] == 4].iloc[0]
k3 = res.loc[res["K"] == 3].iloc[0]
print(f"最高轮廓系数：K={int(best['K'])}，{best['轮廓系数']:.6f}")
print(f"K=3：{k3['轮廓系数']:.6f}；K=4：{k4['轮廓系数']:.6f}")

md = f"""# 独立研究三：轮廓系数（Silhouette Score）

> 数据：课程提供的**模拟客户数据** `mall_customers_sim.csv`（300 位客户），非真实银行业务数据。
> 特征：`age`、`income`、`balance`、`product_count`，StandardScaler 标准化。
> 聚类：`KMeans(n_clusters=K, random_state=42, n_init=10)`，K = 2..9。

## 一、轮廓系数是什么

对每个样本 i：

- 记 a(i) 为它到**同簇其他样本**的平均距离（凝聚度，越小越好）；
- 记 b(i) 为它到**最近的其他簇**所有样本的平均距离（分离度，越大越好）；
- 单个样本的轮廓系数：s(i) = (b(i) − a(i)) / max(a(i), b(i))。

整个数据集的 `silhouette_score` 就是所有样本 s(i) 的平均值。
它同时衡量"簇内够不够紧"和"簇间够不够开"，不需要真实标签。

## 二、取值范围与含义

- 取值范围：**−1 ≤ s ≤ 1**。
- **接近 1**：样本离本簇很近、离其他簇很远，聚类结构清晰。
- **接近 0**：样本处在两簇边界上，归属模糊。
- **小于 0**：样本可能被分错簇（离别的簇比本簇更近）。

因此，同一份数据下**分数越大通常意味着簇划分越合理**，但要注意：
分数高不等于业务上有意义（例如把数据切成很多很小的簇，分数可能虚高）。

## 三、本次数据的真实计算结果

| K | 轮廓系数 | inertia_ |
|---|---|---|
"""
for _, r in res.iterrows():
    md += f"| {int(r['K'])} | {r['轮廓系数']:.6f} | {r['inertia']:.4f} |\n"

md += f"""
- 轮廓系数最高：**K = {int(best['K'])}（{best['轮廓系数']:.6f}）**
- K=3：{k3['轮廓系数']:.6f}；K=4：{k4['轮廓系数']:.6f}
- 变化并非单调：K=2 最高（{res.iloc[0]['轮廓系数']:.4f}），K=3 回落到 {k3['轮廓系数']:.4f}，
  K=4、K=5 小幅回升（{k4['轮廓系数']:.4f}、{res.loc[res['K'] == 5].iloc[0]['轮廓系数']:.4f}），
  K=6 之后稳定下降到 0.36~0.37 区间。
  说明在这套四特征数据上，K=2 的二分最"干净"，但二分会把不同客群硬合并；
  这与"inertia 随 K 单调下降"并不矛盾——inertia 只看簇内，轮廓系数还看簇间。

## 四、轮廓系数与肘部法的区别

| 对比项 | 肘部法 | 轮廓系数 |
|---|---|---|
| 依据 | 簇内平方和 SSE（inertia）随 K 的下降曲线 | 每个样本的簇内紧密度与簇间分离度 |
| 是否看簇间关系 | 否，只看簇内误差 | 是，同时看 a(i) 与 b(i) |
| 输出 | 一条 SSE-K 曲线，靠肉眼找拐点 | 一个可比较的数值（−1~1），越大越好 |
| 主观性 | 拐点位置常不唯一（本次 K=3~4 均可解释） | 数值可直接排序，但也要结合业务判断 |
| 计算成本 | 低 | 较高（需样本两两距离） |

**结论**：肘部法回答"再加一个簇还值不值"，轮廓系数回答"当前划分够不够干净"，
两者应结合使用；本次数据上两者给出的信号并不完全一致，需要结合业务可解释性定 K。

## 五、参考资料

1. Rousseeuw, P. J. (1987). *Silhouettes: A graphical aid to the interpretation and validation
   of cluster analysis.* Journal of Computational and Applied Mathematics, 20, 53–65.
2. scikit-learn 官方文档：silhouette_score —
   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
3. scikit-learn 用户指南：聚类性能评估 —
   https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
"""
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)
print(f"已保存：{OUT_MD}")
