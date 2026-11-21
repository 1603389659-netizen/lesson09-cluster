# -*- coding: utf-8 -*-
"""
第 8 步：量纲陷阱实验（不标准化 vs 标准化）

数据：
    data/mall_customers_sim.csv  income 单位=万元，balance 单位=万元
    data/mall_customers_yuan.csv income 单位=万元，balance 单位=元

核心演示：同一批客户，仅因 balance 的单位从万元变元（数值放大 1 万倍），
不标准化时平衡余额会在欧氏距离中压倒收入，聚类结果随之偏移；
StandardScaler 把两个特征拉到同一量纲，问题消失。

产出：
    generated/scale_comparison.csv    两种情况下的簇人数与画像
    outputs/08_scale_comparison.png   原始坐标 vs 标准化坐标 + 距离贡献对比
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

SIM = "data/mall_customers_sim.csv"
YUAN = "data/mall_customers_yuan.csv"
OUT_CSV = "generated/scale_comparison.csv"
OUT_PNG = "outputs/08_scale_comparison.png"
os.makedirs("outputs", exist_ok=True)

sim = pd.read_csv(SIM, encoding="utf-8-sig")
yua = pd.read_csv(YUAN, encoding="utf-8-sig")

# ==================== 一、两份数据是否对应同一批客户 ====================
print("=" * 72)
print("一、两份数据对应关系检查")
print("=" * 72)
print(f"sim  行数={len(sim)}   列={list(sim.columns)}")
print(f"yuan 行数={len(yua)}   列={list(yua.columns)}")

same_ids = set(sim["customer_id"]) == set(yua["customer_id"])
same_order = sim["customer_id"].equals(yua["customer_id"])
print(f"customer_id 集合相同：{same_ids}；顺序完全相同：{same_order}")

# 按 customer_id 对齐后逐列比较
m = sim.merge(yua, on="customer_id", suffixes=("_sim", "_yuan"))
checks = {
    "age 完全相同": (m["age_sim"] == m["age_yuan"]).all(),
    "income 完全相同": np.allclose(m["income_sim"], m["income_yuan"]),
    "product_count 完全相同": (m["product_count_sim"] == m["product_count_yuan"]).all(),
    "gender 完全相同": (m["gender_sim"] == m["gender_yuan"]).all(),
    "city 完全相同": (m["city_sim"] == m["city_yuan"]).all(),
    "balance_yuan = balance_sim × 10000": np.allclose(
        m["balance_yuan"], m["balance_sim"] * 10000, rtol=1e-9),
}
for k, v in checks.items():
    print(f"  {k}：{v}")
print(f"结论：两份文件为同一批 {len(m)} 位客户的两种金额单位版本。"
      if all(checks.values()) else "结论：两份文件存在不一致，见上表。")

# ==================== 二、单位与数值范围 ====================
print("\n" + "=" * 72)
print("二、income / balance 的单位与数值范围")
print("=" * 72)
print(f"{'文件':<10}{'字段':<10}{'均值':>14}{'最小':>14}{'最大':>14}")
for label, d in [("sim", sim), ("yuan", yua)]:
    for col in ["income", "balance"]:
        print(f"{label:<10}{col:<10}{d[col].mean():>14.2f}"
              f"{d[col].min():>14.2f}{d[col].max():>14.2f}")
print("\nsim ：income 与 balance 均为「万元」；")
print("yuan：income 为「万元」，balance 为「元」（数值放大 10000 倍，量纲不一致）。")
print(f"balance 均值放大倍数：{yua['balance'].mean() / sim['balance'].mean():.1f}")

# ==================== 三、不标准化聚类（原始量纲） ====================
print("\n" + "=" * 72)
print("三、不标准化：直接用 yuan 版原始数值做 KMeans")
print("=" * 72)
X_raw = yua[["income", "balance"]].to_numpy()
km_raw = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_raw = km_raw.fit_predict(X_raw)
print(f"inertia_ = {km_raw.inertia_:.4e}（单位为 元²，与标准化后不可直接比较）")

# 各特征对欧氏平方距离的平均贡献（mean((xi-xj)^2) = 2*var）
def dist_share(Xmat, names):
    var = Xmat.var(axis=0)
    share = var / var.sum()
    return pd.Series(share * 100, index=names)

raw_share = dist_share(X_raw, ["income（万元）", "balance（元）"])
print("\n原始量纲下，各特征对平方欧氏距离的贡献占比：")
for k, v in raw_share.items():
    print(f"  {k}：{v:6.2f}%")

# ==================== 四、标准化后聚类 ====================
print("\n" + "=" * 72)
print("四、标准化后：StandardScaler + KMeans（同一份 yuan 数据）")
print("=" * 72)
scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)
km_std = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_std = km_std.fit_predict(X_std)
print(f"标准化后各列均值≈{X_std.mean(axis=0).round(6)}，标准差≈{X_std.std(axis=0, ddof=1).round(4)}")
print(f"inertia_ = {km_std.inertia_:.4f}（无量纲）")

std_share = dist_share(X_std, ["income", "balance"])
print("\n标准化后，各特征对平方欧氏距离的贡献占比：")
for k, v in std_share.items():
    print(f"  {k}：{v:6.2f}%")

# ==================== 五、结果比较 ====================
print("\n" + "=" * 72)
print("五、标准化前后聚类结果比较")
print("=" * 72)
ari = adjusted_rand_score(labels_raw, labels_std)
print(f"两种划分的 ARI = {ari:.4f}")
raw_sizes = pd.Series(labels_raw).value_counts().sort_index()
std_sizes = pd.Series(labels_std).value_counts().sort_index()
print(f"原始量纲簇人数：{raw_sizes.tolist()}")
print(f"标准化后簇人数：{std_sizes.tolist()}")

ct = pd.crosstab(pd.Series(labels_raw, name="原始量纲簇"),
                 pd.Series(labels_std, name="标准化簇"))
print("\n交叉表（行=原始量纲簇，列=标准化簇）：")
print(ct.to_string())

# 画像（余额统一换算成万元展示，便于阅读）
yua["cluster_raw"] = labels_raw
yua["cluster_std"] = labels_std
yua["balance_wan"] = yua["balance"] / 10000

rows = []
for scen, col in [("原始量纲（不标准化）", "cluster_raw"), ("标准化后", "cluster_std")]:
    g = yua.groupby(col)
    prof = pd.DataFrame({
        "人数": g.size(),
        "占比(%)": (g.size() / len(yua) * 100).round(2),
        "平均年龄(岁)": g["age"].mean().round(2),
        "平均收入(万元)": g["income"].mean().round(2),
        "平均余额(万元)": g["balance_wan"].mean().round(2),
        "平均持有产品数(个)": g["product_count"].mean().round(2),
    })
    print(f"\n--- {scen} ---")
    print(prof.to_string())
    for c in prof.index:
        r = prof.loc[c]
        rows.append({
            "场景": scen, "簇": int(c), "人数": int(r["人数"]), "占比(%)": r["占比(%)"],
            "平均年龄(岁)": r["平均年龄(岁)"], "平均收入(万元)": r["平均收入(万元)"],
            "平均余额(万元)": r["平均余额(万元)"],
            "平均持有产品数(个)": r["平均持有产品数(个)"],
        })

pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n已保存：{OUT_CSV}")

# ==================== 六、对比图 ====================
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
fig, axes = plt.subplots(1, 3, figsize=(19, 6))

# (1) 原始量纲：横轴 income（万元），纵轴 balance（元）
ax = axes[0]
for c in sorted(set(labels_raw)):
    sub = yua[labels_raw == c]
    ax.scatter(sub["income"], sub["balance"], s=34, alpha=0.7,
               color=colors[c % 4], edgecolors="white", linewidths=0.4,
               label=f"簇 {c}（{len(sub)} 人）")
    ax.scatter(sub["income"].mean(), sub["balance"].mean(), marker="*", s=480,
               color=colors[c % 4], edgecolors="black", linewidths=1.0, zorder=5)
ax.set_title("原始量纲（不标准化）：纵轴余额单位为元", fontsize=12)
ax.set_xlabel("年收入 income（万元）", fontsize=11)
ax.set_ylabel("账户平均余额 balance（元）", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3)
ax.text(0.98, 0.02, f"距离贡献：收入 {raw_share.iloc[0]:.1f}%\n余额 {raw_share.iloc[1]:.1f}%",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round", fc="#FFF3CD", ec="#E0B400"))

# (2) 标准化后：坐标仍用万元展示，颜色按标准化聚类
ax = axes[1]
for c in sorted(set(labels_std)):
    sub = yua[labels_std == c]
    ax.scatter(sub["income"], sub["balance_wan"], s=34, alpha=0.7,
               color=colors[c % 4], edgecolors="white", linewidths=0.4,
               label=f"簇 {c}（{len(sub)} 人）")
    ax.scatter(sub["income"].mean(), sub["balance_wan"].mean(), marker="*", s=480,
               color=colors[c % 4], edgecolors="black", linewidths=1.0, zorder=5)
ax.set_title("标准化后：两特征等权（坐标按万元展示）", fontsize=12)
ax.set_xlabel("年收入 income（万元）", fontsize=11)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3)
ax.text(0.98, 0.02, f"距离贡献：收入 {std_share.iloc[0]:.1f}%\n余额 {std_share.iloc[1]:.1f}%",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round", fc="#D4EDDA", ec="#28A745"))

# (3) 距离贡献占比对比
ax = axes[2]
xpos = np.arange(2)
ax.bar(xpos - 0.19, raw_share.values, width=0.36, color="#C44E52", alpha=0.85,
       label="原始量纲（不标准化）")
ax.bar(xpos + 0.19, std_share.values, width=0.36, color="#4C72B0", alpha=0.85,
       label="标准化后")
for i, (a, b) in enumerate(zip(raw_share.values, std_share.values)):
    ax.text(xpos[i] - 0.19, a + 1, f"{a:.1f}%", ha="center", fontsize=10)
    ax.text(xpos[i] + 0.19, b + 1, f"{b:.1f}%", ha="center", fontsize=10)
ax.set_xticks(xpos)
ax.set_xticklabels(["income", "balance"], fontsize=11)
ax.set_ylabel("对平方欧氏距离的贡献占比（%）", fontsize=11)
ax.set_ylim(0, 105)
ax.set_title("量纲决定谁在距离里说话", fontsize=12)
ax.legend(fontsize=9.5)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("量纲陷阱：标准化不是修改客户，而是改变距离计算的度量衡", fontsize=15)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"已保存：{OUT_PNG}")

# ==================== 七、结论 ====================
print("\n" + "=" * 72)
print("六、结论")
print("=" * 72)
print(f"1. 两份文件对应同一批 {len(m)} 位客户，只有 balance 单位不同（万元 vs 元）。")
print(f"2. 原始量纲下，balance（元）贡献 {raw_share.iloc[1]:.2f}% 的距离，"
      f"income 仅 {raw_share.iloc[0]:.2f}%，聚类几乎只听余额的。")
print(f"3. 标准化后，两特征贡献 {std_share.iloc[0]:.2f}% / {std_share.iloc[1]:.2f}%，等权参与距离。")
print(f"4. 两种划分 ARI = {ari:.4f}，簇人数由 {raw_sizes.tolist()} 变为 {std_sizes.tolist()}。")
print("5. 标准化不改变任何客户的收入或余额，只改变计算距离时的度量衡。")
