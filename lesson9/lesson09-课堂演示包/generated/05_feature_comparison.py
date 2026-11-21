# -*- coding: utf-8 -*-
"""
第 5 步（自主实践一）：更换特征集，比较特征选择对客户分群的影响

实验一：特征 = income、balance
实验二：特征 = income、balance、age

两组均：StandardScaler 标准化 + KMeans(n_clusters=4, random_state=42, n_init=10)

产出：
    generated/feature_comparison.csv  两组各簇人数与画像对比
    generated/feature_comparison.md   对比分析报告
    outputs/05_feature_comparison.png 两组聚类散点图 + 规模对比
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

# ---------- 中文字体设置（优先 Windows 系统字体） ----------
def setup_chinese_font():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
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
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_chinese_font()

CSV_PATH = "data/mall_customers_sim.csv"
OUT_CSV = "generated/feature_comparison.csv"
OUT_MD = "generated/feature_comparison.md"
OUT_PNG = "outputs/05_feature_comparison.png"
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
N = len(df)
print("=" * 72)
print("数据：%s   行数：%d" % (CSV_PATH, N))
print("=" * 72)

# ==================== 两组实验 ====================
EXPERIMENTS = [
    ("实验一", ["income", "balance"]),
    ("实验二", ["income", "balance", "age"]),
]
PROFILE_COLS = ["age", "income", "balance", "product_count"]

results = {}
for exp_name, feats in EXPERIMENTS:
    print("\n" + "=" * 72)
    print(f"{exp_name}：特征 = {feats}")
    print("=" * 72)

    X = StandardScaler().fit_transform(df[feats])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    # 簇编号按簇中心第一特征（income）升序重排，便于两组对照阅读
    order = np.argsort(km.cluster_centers_[:, feats.index("income")])
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])

    tmp = df.copy()
    tmp["cluster"] = labels

    sizes = tmp.groupby("cluster").size()
    prof = tmp.groupby("cluster")[PROFILE_COLS].mean().round(2)
    prof.insert(0, "人数", sizes)
    prof.insert(1, "占比(%)", (sizes / N * 100).round(2))

    print(f"inertia_：{km.inertia_:.4f}")
    print("\n各簇人数：")
    for c in sizes.index:
        print(f"  簇 {c}：{sizes[c]:>3d} 人（{sizes[c] / N * 100:5.2f}%）")
    print(f"合计：{int(sizes.sum())}")
    print("\n各簇画像：")
    print(prof.to_string())

    results[exp_name] = {"labels": labels, "profile": prof, "features": feats,
                         "inertia": km.inertia_}

# ==================== 一致性比较 ====================
print("\n" + "=" * 72)
print("两组分区一致性")
print("=" * 72)
l1 = results["实验一"]["labels"]
l2 = results["实验二"]["labels"]
ari = adjusted_rand_score(l1, l2)
ct = pd.crosstab(pd.Series(l1, name="实验一簇"),
                 pd.Series(l2, name="实验二簇"))
print(f"调整兰德指数 ARI = {ari:.4f}（1 表示完全一致，0 表示与随机相当）")
print("\n交叉表（行=实验一分簇，列=实验二分簇）：")
print(ct.to_string())

# 每个实验一的簇在实验二中被打散到什么程度
print("\n实验一的各簇在实验二中的去向（行内占比 %）：")
print((ct.div(ct.sum(axis=1), axis=0) * 100).round(1).to_string())

# ==================== 保存 CSV ====================
rows = []
for exp_name, feats in EXPERIMENTS:
    prof = results[exp_name]["profile"]
    for c in prof.index:
        rows.append({
            "实验": exp_name,
            "特征集": "+".join(feats),
            "簇": c,
            "人数": int(prof.loc[c, "人数"]),
            "占比(%)": prof.loc[c, "占比(%)"],
            "平均年龄(岁)": prof.loc[c, "age"],
            "平均收入(万元)": prof.loc[c, "income"],
            "平均余额(万元)": prof.loc[c, "balance"],
            "平均持有产品数(个)": prof.loc[c, "product_count"],
        })
comp = pd.DataFrame(rows)
comp.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n已保存：{OUT_CSV}（{len(comp)} 行）")

# ==================== 绘图 ====================
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))

for ax, (exp_name, feats) in zip(axes[:2], EXPERIMENTS):
    labels = results[exp_name]["labels"]
    for c in range(4):
        sub = df[labels == c]
        ax.scatter(sub["income"], sub["balance"], s=38, alpha=0.7,
                   color=colors[c], edgecolors="white", linewidths=0.5,
                   label=f"簇 {c}（{len(sub)} 人）")
        ax.scatter(sub["income"].mean(), sub["balance"].mean(),
                   marker="*", s=520, color=colors[c],
                   edgecolors="black", linewidths=1.0, zorder=5)
    ax.set_title(f"{exp_name}：{' + '.join(feats)}", fontsize=13)
    ax.set_xlabel("年收入 income（万元）", fontsize=11)
    ax.set_ylabel("账户平均余额 balance（万元）", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

# 第三幅：两组各簇规模对比（按人数降序）
ax = axes[2]
for i, (exp_name, feats) in enumerate(EXPERIMENTS):
    s = sorted(results[exp_name]["profile"]["人数"].tolist(), reverse=True)
    ax.bar(np.arange(4) + (i - 0.5) * 0.38, s, width=0.36,
           label=exp_name, color=["#4C72B0", "#DD8452"][i], alpha=0.85)
    for x, v in zip(np.arange(4) + (i - 0.5) * 0.38, s):
        ax.text(x, v + 1, str(v), ha="center", fontsize=9)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(["第1大簇", "第2大簇", "第3大簇", "第4大簇"])
ax.set_ylabel("人数", fontsize=11)
ax.set_title("两组各簇规模对比（按人数降序）", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("特征集更换对 K-Means 客户分群的影响（k=4，两组均标准化）", fontsize=15)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"已保存：{OUT_PNG}")

# ==================== 生成报告 ====================
p1 = results["实验一"]["profile"]
p2 = results["实验二"]["profile"]

lines = [
    "# 自主实践一：更换特征集对客户分群的影响", "",
    "> 数据：课程提供的**模拟客户数据** `mall_customers_sim.csv`（300 位客户），"
    "非真实银行业务数据。原始数据未被修改。", "",
    "## 一、实验设置", "",
    "| 实验 | 特征集 | 预处理 | 聚类参数 | inertia_ |",
    "|---|---|---|---|---|",
]
for exp_name, feats in EXPERIMENTS:
    lines.append(
        f"| {exp_name} | {' + '.join(feats)} | StandardScaler | "
        f"KMeans(n_clusters=4, random_state=42, n_init=10) | "
        f"{results[exp_name]['inertia']:.2f} |"
    )

lines += ["", "## 二、实验一：income + balance", "",
          "| 簇 | 人数 | 占比 | 平均年龄 | 平均收入 | 平均余额 | 平均产品数 |",
          "|---|---|---|---|---|---|---|"]
for c in p1.index:
    r = p1.loc[c]
    lines.append(f"| {c} | {int(r['人数'])} | {r['占比(%)']:.2f}% | {r['age']:.2f} | "
                 f"{r['income']:.2f} | {r['balance']:.2f} | {r['product_count']:.2f} |")

lines += ["", "## 三、实验二：income + balance + age", "",
          "| 簇 | 人数 | 占比 | 平均年龄 | 平均收入 | 平均余额 | 平均产品数 |",
          "|---|---|---|---|---|---|---|"]
for c in p2.index:
    r = p2.loc[c]
    lines.append(f"| {c} | {int(r['人数'])} | {r['占比(%)']:.2f}% | {r['age']:.2f} | "
                 f"{r['income']:.2f} | {r['balance']:.2f} | {r['product_count']:.2f} |")

lines += ["", "## 四、两组分区一致性", "",
          f"- 调整兰德指数 **ARI = {ari:.4f}**（同簇编号已按收入中心排序对齐）。",
          "- 交叉表（行=实验一，列=实验二；数值为人数）：", "",
          "| 实验一 \\ 实验二 | " + " | ".join(str(c) for c in ct.columns) + " |",
          "|" + "---|" * (len(ct.columns) + 1)]
for idx, row in ct.iterrows():
    lines.append(f"| 簇 {idx} | " + " | ".join(str(v) for v in row.values) + " |")

# 规模变化文字
s1 = sorted(p1["人数"].tolist(), reverse=True)
s2 = sorted(p2["人数"].tolist(), reverse=True)

# 根据交叉表自动找出"被拆分"与"被合并"的簇
share = ct.div(ct.sum(axis=1), axis=0)          # 实验一每簇的去向占比
split_desc = []
for idx, row in share.iterrows():
    main_col = row.idxmax()
    main_share = row.max()
    if main_share < 0.9:
        parts = [f"{int(ct.loc[idx, c])} 人（实验二簇 {c}）"
                 for c in ct.columns if ct.loc[idx, c] / ct.loc[idx].sum() > 0.1]
        split_desc.append(f"  - 实验一簇 {idx}（{int(ct.loc[idx].sum())} 人）被拆分到："
                          + "、".join(parts) + f"，最大一块仅占 {main_share * 100:.1f}%")

merge_desc = []
for c in ct.columns:
    src = [i for i in ct.index if ct.loc[i, c] / ct[c].sum() > 0.1]
    if len(src) > 1:
        merge_desc.append(
            f"  - 实验二簇 {c}（{int(ct[c].sum())} 人）由实验一的簇 "
            + "、".join(str(i) for i in src) + " 合并而来")

lines += ["", "## 五、加入 age 后分群的变化", "",
          f"- 簇规模分布：实验一（降序）{s1} → 实验二（降序）{s2}。",
          f"- 两组划分的一致程度 ARI = {ari:.4f}；编号已对齐，但分区并非简单平移。",
          "- 具体变动（按交叉表自动识别）："]
lines += split_desc + merge_desc
if not split_desc and not merge_desc:
    lines.append("  - 交叉表中每个簇均能一一对应，未被拆分或合并。")

# 用实际数据描述年龄分层程度
age_range1 = p1["age"].max() - p1["age"].min()
age_range2 = p2["age"].max() - p2["age"].min()
if abs(age_range2 - age_range1) < 1:
    age_note = (f"各簇平均年龄极差几乎不变（实验一 {age_range1:.2f} 岁 → "
                f"实验二 {age_range2:.2f} 岁），说明年龄主要改变的是簇的切分位置，"
                "而不是把年龄差异整体拉大。")
else:
    age_note = (f"各簇平均年龄极差由 {age_range1:.2f} 岁变为 {age_range2:.2f} 岁，"
                + ("簇间年龄差异被拉大。" if age_range2 > age_range1 else "簇间年龄差异收窄。"))
lines += [
    f"- {age_note}",
    "- 实验一的簇本质上只按「收入-余额」二维位置切分，年龄只作为事后描述；"
    "实验二让 age 直接参与距离计算，标准化后年龄与收入、余额等权，"
    "因此同一个资产层内部会再按年龄段分开。",
    "",
    "## 六、特征选择对业务含义的影响", "",
    "- **实验一（income + balance）**：客群定义是纯资产维度——「高净值 / 中等 / 低净值」。"
    "解读直接，适合做资产分层、产品配置与额度策略；但同簇内可能混杂不同人生阶段的客户，"
    "营销话术难以统一。",
    "- **实验二（income + balance + age）**：客群同时带上生命周期标签——"
    "「年轻高潜」「中年稳健」「年长高净值」等。适合做生命周期营销（首卡、家庭配置、养老规划），"
    "但簇边界不再严格等价于资产等级，不能直接用于资产分层考核。",
    "- 结论：特征选择决定了「业务问题的定义」。要回答资产分层问题，用实验一；"
    "要回答客户生命周期经营问题，用实验二；两者不可互相替代。",
    "",
    "> 说明：本次两组实验均为 k=4、两/三特征，与课程实践指南的两特征 k=3 口径不同，"
    "人数与簇结构不可直接与指南结果对比。",
]

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"已保存：{OUT_MD}")

# ==================== 终端总结 ====================
print("\n" + "=" * 72)
print("结论摘要")
print("=" * 72)
print(f"实验一（income+balance）四簇人数：{p1['人数'].tolist()}")
print(f"实验二（income+balance+age）四簇人数：{p2['人数'].tolist()}")
print(f"两组分区 ARI = {ari:.4f}")
print(f"实验一各簇平均年龄：{p1['age'].tolist()}")
print(f"实验二各簇平均年龄：{p2['age'].tolist()}")
