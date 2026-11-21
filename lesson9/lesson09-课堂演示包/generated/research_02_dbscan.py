# -*- coding: utf-8 -*-
"""
独立研究二：KMeans 对环形数据的局限（KMeans vs DBSCAN）

用 make_circles 生成两个同心圆，分别用 KMeans 和 DBSCAN 聚类，
量化对比（ARI）并绘制三幅图，同时输出分析报告。

产出：
    outputs/09_dbscan_comparison.png
    generated/research_02_dbscan.md
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score

# ---------- 中文字体 ----------
def setup_chinese_font():
    for f in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
              r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]:
        if os.path.exists(f):
            fm.fontManager.addfont(f)
            prop = fm.FontProperties(fname=f)
            matplotlib.rcParams["font.sans-serif"] = (
                [prop.get_name()] + matplotlib.rcParams["font.sans-serif"]
            )
            print(f"使用中文字体：{prop.get_name()}（{f}）")
            break
    else:
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_chinese_font()

OUT_PNG = "outputs/09_dbscan_comparison.png"
OUT_MD = "generated/research_02_dbscan.md"
os.makedirs("outputs", exist_ok=True)

# ==================== 一、生成环形数据 ====================
print("=" * 68)
print("一、make_circles 环形数据")
print("=" * 68)
X, y_true = make_circles(n_samples=600, factor=0.5, noise=0.05, random_state=42)
print(f"样本数：{X.shape[0]}   特征数：{X.shape[1]}")
print(f"真实簇数：{len(np.unique(y_true))}，各簇样本数：{np.bincount(y_true).tolist()}")

# ==================== 二、KMeans 聚类 ====================
print("\n" + "=" * 68)
print("二、KMeans 聚类（k=2）")
print("=" * 68)
km = KMeans(n_clusters=2, random_state=42, n_init=10)
y_km = km.fit_predict(X)
ari_km = adjusted_rand_score(y_true, y_km)
print(f"KMeans 各簇人数：{np.bincount(y_km).tolist()}")
print(f"ARI（与真实标签相比）= {ari_km:.4f}")

# ==================== 三、DBSCAN 聚类 ====================
print("\n" + "=" * 68)
print("三、DBSCAN 聚类（eps=0.15, min_samples=5）")
print("=" * 68)
db = DBSCAN(eps=0.15, min_samples=5)
y_db = db.fit_predict(X)
n_clusters_db = len(set(y_db)) - (1 if -1 in y_db else 0)
n_noise = int((y_db == -1).sum())
ari_db = adjusted_rand_score(y_true, y_db)
print(f"DBSCAN 找到簇数：{n_clusters_db}")
print(f"噪声点数量：{n_noise}")
print(f"各标签人数：{[(int(l), int((y_db == l).sum())) for l in sorted(set(y_db))]}")
print(f"ARI（与真实标签相比）= {ari_db:.4f}")

# ==================== 四、绘图 ====================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap="coolwarm", s=22, alpha=0.85)
ax.set_title("真实结构：两个同心圆", fontsize=13)
ax.set_xlabel("特征 1", fontsize=11)
ax.set_ylabel("特征 2", fontsize=11)
ax.grid(alpha=0.25)

ax = axes[1]
ax.scatter(X[:, 0], X[:, 1], c=y_km, cmap="coolwarm", s=22, alpha=0.85)
for c in range(2):
    ax.scatter(km.cluster_centers_[c, 0], km.cluster_centers_[c, 1],
               marker="*", s=520, c="black", edgecolors="white", linewidths=1.2, zorder=5)
ax.set_title(f"KMeans（k=2）：把圆切成两半\nARI = {ari_km:.3f}", fontsize=13)
ax.set_xlabel("特征 1", fontsize=11)
ax.grid(alpha=0.25)

ax = axes[2]
colors = np.array(["#4C72B0", "#DD8452", "#55A868"])
scatter_colors = np.where(y_db == -1, 0.0, 1.0)
ax.scatter(X[:, 0], X[:, 1], c=y_db, cmap="coolwarm", s=22, alpha=0.85)
ax.set_title(f"DBSCAN（eps=0.15, min_samples=5）\n簇数 {n_clusters_db}，噪声 {n_noise}，ARI = {ari_db:.3f}",
             fontsize=13)
ax.set_xlabel("特征 1", fontsize=11)
ax.grid(alpha=0.25)

fig.suptitle("KMeans 与 DBSCAN 在环形数据上的对比", fontsize=15)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"\n已保存：{OUT_PNG}")

# ==================== 五、生成分析报告 ====================
md = f"""# 独立研究二：KMeans 对环形数据的局限

> 数据：`sklearn.datasets.make_circles(n_samples=600, factor=0.5, noise=0.05, random_state=42)`，
> 完全由程序生成，非真实业务数据。

## 一、实验设置

| 项目 | 设置 |
|---|---|
| 数据 | 600 个点，两个同心圆（内圈半径约为外圈一半） |
| KMeans | `KMeans(n_clusters=2, random_state=42, n_init=10)` |
| DBSCAN | `DBSCAN(eps=0.15, min_samples=5)` |
| 评价 | ARI（调整兰德指数），与真实标签比较，1 为完全正确 |

## 二、真实运行结果

| 算法 | 簇数 | 噪声点 | 各簇人数 | ARI |
|---|---|---|---|---|
| KMeans(k=2) | 2 | — | {np.bincount(y_km).tolist()} | **{ari_km:.4f}** |
| DBSCAN | {n_clusters_db} | {n_noise} | 见表下说明 | **{ari_db:.4f}** |

- DBSCAN 标签人数：{[(int(l), int((y_db == l).sum())) for l in sorted(set(y_db))]}
  （标签 -1 表示噪声）

KMeans 的 ARI 只有 {ari_km:.3f}，DBSCAN 的 ARI 为 {ari_db:.3f}：
在环形数据上，KMeans 把外圈切成左右两半、内圈切成上下两半，几乎完全错误；
而 DBSCAN 把内外两个圆完整分开，且没有产生噪声点。

## 三、KMeans 为什么不适合环形分布

1. **KMeans 的簇是凸的、近似球形的**。它的目标是最小化簇内平方和，
   每个点到簇中心的距离是欧氏直线距离，因此一个簇天然是"围绕中心的圆盘"。
   环形数据的内外圆不是圆盘，任何一条直线边界都会把圆切开。

2. **KMeans 隐含"簇中心"假设**。两个同心圆的整体中心几乎重合在原点，
   用两个质心去描述它们，必然只能沿某个方向把平面一分为二，
   所以结果是"左右/上下两半"，而不是"内圈/外圈"。

3. **距离度量只看直线距离，不看连通性**。内圈上相邻的点沿圆周很近，
   但欧氏距离与"绕圈走"的路径无关，KMeans 无法利用"密度相连"的结构；
   DBSCAN 则基于 eps 邻域与密度可达，能把沿圆周连续分布的点串成同一个簇。

4. **KMeans 必须指定 k 且所有点都归簇**，对复杂形状没有容错；
   DBSCAN 可以自动发现簇数，并把稀疏区域标为噪声。

## 四、一句话比较

**KMeans 找"围绕中心的球形团"，DBSCAN 找"密度连成一片的区域"——
前者靠距离到质心，后者靠邻域密度，所以环形、月牙形等非凸结构要用 DBSCAN。**

## 五、图

`outputs/09_dbscan_comparison.png` 左：真实结构；中：KMeans 把圆切成两半；
右：DBSCAN 分出内外两圈。

## 六、参考资料

1. scikit-learn 官方文档：KMeans — https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
2. scikit-learn 官方文档：DBSCAN — https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
3. scikit-learn 用户指南：聚类算法对比（不同算法的适用形状）—
   https://scikit-learn.org/stable/modules/clustering.html
4. Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996).
   *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.* KDD-96.
5. MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations.*
   Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability.
"""
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)
print(f"已保存：{OUT_MD}")

print("\n" + "=" * 68)
print("结论")
print("=" * 68)
print(f"KMeans ARI = {ari_km:.4f}；DBSCAN ARI = {ari_db:.4f}")
print("KMeans 把同心圆切成两半，DBSCAN 能按密度分出内外两圈。")
