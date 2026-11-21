# -*- coding: utf-8 -*-
"""
第 2 步：K-Means 客户聚类（k=4）

使用 data/mall_customers_sim.csv 中四个数值特征：
    age、income、balance、product_count

约定：
- customer_id 仅作客户编号，绝不进入模型；
- 先 StandardScaler 标准化，再做 KMeans(n_clusters=4, random_state=42, n_init=10)；
- 簇编号写回数据框；
- 不修改原始 CSV（结果另存为 generated/ 下的新文件）。

产出：
    generated/clustered_customers.csv   聚类结果（含 cluster 列）
    generated/cluster_profile.csv       四簇画像（人数、占比、特征均值）
    outputs/02_clustered_scatter.png    按簇着色的二维散点图（含簇中心）
"""
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------- 中文字体设置（优先 Windows 系统字体） ----------
def setup_chinese_font():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑 Bold
        r"C:\Windows\Fonts\simhei.ttf",   # 黑体
        r"C:\Windows\Fonts\simsun.ttc",   # 宋体
    ]
    for f in candidates:
        if os.path.exists(f):
            fm.fontManager.addfont(f)
            prop = fm.FontProperties(fname=f)
            matplotlib.rcParams["font.sans-serif"] = (
                [prop.get_name()] + matplotlib.rcParams["font.sans-serif"]
            )
            print(f"使用中文字体：{prop.get_name()}（{f}）")
            break
    else:
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "SimSun",
            "Songti SC", "STHeiti", "PingFang SC", "Arial Unicode MS",
        ]
        print("未找到 Windows 中文字体文件，使用字体名候选列表")
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_chinese_font()

# ---------- 路径 ----------
CSV_PATH = "data/mall_customers_sim.csv"
OUT_SCATTER = "outputs/02_clustered_scatter.png"
OUT_CLUSTERED = "generated/clustered_customers.csv"
OUT_PROFILE = "generated/cluster_profile.csv"
os.makedirs("outputs", exist_ok=True)

# ==================== 一、读取数据 & 选定特征 ====================
print("=" * 68)
print("一、读取数据与特征选择")
print("=" * 68)
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print(f"数据文件：{CSV_PATH}")
print(f"行数：{df.shape[0]}   列数：{df.shape[1]}")

FEATURES = ["age", "income", "balance", "product_count"]
# customer_id 只作编号，绝不进入模型
X_raw = df[FEATURES].copy()
print(f"建模特征（4 个）：{FEATURES}")
print("customer_id 已排除，不参与标准化与聚类。")
print(f"特征矩阵形状：{X_raw.shape}")

# ==================== 二、标准化 ====================
print("\n" + "=" * 68)
print("二、StandardScaler 标准化")
print("=" * 68)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

scaled_check = pd.DataFrame(X, columns=FEATURES)
print("标准化后各列均值 / 标准差（应约等于 0 / 1）：")
print(scaled_check.agg(["mean", "std"]).round(4).to_string())

# ==================== 三、KMeans 聚类 ====================
print("\n" + "=" * 68)
print("三、KMeans 聚类")
print("=" * 68)
km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X)
df["cluster"] = labels
print("模型参数：KMeans(n_clusters=4, random_state=42, n_init=10)")
print(f"迭代次数（n_iter_）：{km.n_iter_}")
print(f"簇内平方和（inertia_）：{km.inertia_:.4f}")

# ==================== 四、各簇人数与占比 ====================
print("\n" + "=" * 68)
print("四、各簇人数与占比")
print("=" * 68)
counts = df["cluster"].value_counts().sort_index()
total = len(df)
for c, n in counts.items():
    print(f"  簇 {c}：{n:>3d} 人，占比 {n / total * 100:5.2f}%")
print(f"\n四簇人数之和：{int(counts.sum())}  （原始行数：{total}）")
print(f"校验结果：{'通过 ✅' if int(counts.sum()) == 300 else '不通过 ❌'}")

# ==================== 五、各簇特征均值 ====================
print("\n" + "=" * 68)
print("五、各簇特征均值")
print("=" * 68)
profile = df.groupby("cluster")[FEATURES].mean().round(3)
profile.insert(0, "人数", counts)
profile.insert(1, "占比(%)", (counts / total * 100).round(2))
print(profile.to_string())

# 簇中心（标准化空间）逆变换回原始量纲，便于解释
centers_raw = pd.DataFrame(
    scaler.inverse_transform(km.cluster_centers_), columns=FEATURES
).round(3)
print("\n簇中心（逆标准化回原始量纲）：")
print(centers_raw.to_string())

# ==================== 六、按簇着色散点图（收入-余额平面） ====================
print("\n" + "=" * 68)
print("六、生成按簇着色的散点图")
print("=" * 68)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
fig, ax = plt.subplots(figsize=(10, 7))
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    ax.scatter(
        sub["income"], sub["balance"],
        s=42, alpha=0.7, color=colors[c % len(colors)],
        edgecolors="white", linewidths=0.5,
        label=f"簇 {c}（{len(sub)} 人）",
    )
# 簇中心（用原始量纲的 income / balance 画星形）
ax.scatter(
    centers_raw["income"], centers_raw["balance"],
    marker="*", s=650, c="red", edgecolors="black", linewidths=1.2,
    zorder=5, label="簇中心",
)
ax.set_title("K-Means 聚类结果（k=4，收入-余额平面，四特征标准化后聚类）", fontsize=14)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=12)
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_SCATTER, dpi=150)
plt.close()
print(f"已保存：{OUT_SCATTER}")

# ==================== 七、保存结果 CSV ====================
df.to_csv(OUT_CLUSTERED, index=False, encoding="utf-8-sig")
print(f"已保存聚类结果：{OUT_CLUSTERED}（{df.shape[0]} 行 × {df.shape[1]} 列）")

profile_out = profile.copy()
profile_out.index.name = "cluster"
profile_out.to_csv(OUT_PROFILE, encoding="utf-8-sig")
print(f"已保存簇画像：{OUT_PROFILE}（{profile_out.shape[0]} 行 × {profile_out.shape[1]} 列）")

print("\n完成：k=4 聚类结果已写出，原始 CSV 未被修改。")
