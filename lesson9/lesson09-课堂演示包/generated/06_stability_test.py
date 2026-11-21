# -*- coding: utf-8 -*-
"""
第 6 步（自主实践二）：聚类稳定性检验

基准特征：income、balance（StandardScaler 标准化）
KMeans(n_clusters=4, n_init=10)，分别取 random_state = 7、42、2026，
比较三次聚类结果是否稳定。簇编号可能交换，故用 ARI（调整兰德指数）判断，
不依据簇编号直接比较。

产出：
    generated/stability_results.csv   三次运行的各簇人数与簇中心
    generated/stability_analysis.md   稳定性分析报告
    outputs/06_stability.png          三次聚类散点对比 + ARI 热力图
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

# ---------- 中文字体设置 ----------
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
OUT_CSV = "generated/stability_results.csv"
OUT_MD = "generated/stability_analysis.md"
OUT_PNG = "outputs/06_stability.png"
os.makedirs("outputs", exist_ok=True)

FEATURES = ["income", "balance"]
SEEDS = [7, 42, 2026]
N = None

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
N = len(df)
X = StandardScaler().fit_transform(df[FEATURES])

print("=" * 74)
print(f"数据：{CSV_PATH}   行数：{N}")
print(f"特征：{FEATURES}（StandardScaler 标准化）")
print(f"随机种子：{SEEDS}   KMeans(n_clusters=4, n_init=10)")
print("=" * 74)

# ==================== 一、三次聚类 ====================
runs = {}
rows = []
for seed in SEEDS:
    km = KMeans(n_clusters=4, n_init=10, random_state=seed)
    raw_labels = km.fit_predict(X)

    # 仅用于展示：按簇中心的 income 分量升序重编号，避免编号交换干扰阅读
    order = np.argsort(km.cluster_centers_[:, FEATURES.index("income")])
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in raw_labels])

    sizes = pd.Series(labels).value_counts().sort_index()
    runs[seed] = {"raw": raw_labels, "labels": labels, "sizes": sizes,
                  "inertia": km.inertia_, "km": km, "remap": remap}

    print(f"\n--- random_state = {seed} ---")
    print(f"inertia_ = {km.inertia_:.4f}   n_iter_ = {km.n_iter_}")
    print("各簇人数（编号已按收入中心升序对齐，仅用于展示）：")
    for c in sizes.index:
        print(f"  簇 {c}：{sizes[c]:>3d} 人（{sizes[c] / N * 100:5.2f}%）")
    print(f"  合计：{int(sizes.sum())}")

    for c in range(4):
        raw_c = [k for k, v in remap.items() if v == c][0]
        m = raw_labels == raw_c
        rows.append({
            "random_state": seed,
            "簇": c,
            "人数": int(sizes.get(c, 0)),
            "占比(%)": round(float(sizes.get(c, 0)) / N * 100, 2),
            "簇中心_收入(万元)": round(float(df.loc[m, "income"].mean()), 2),
            "簇中心_余额(万元)": round(float(df.loc[m, "balance"].mean()), 2),
            "inertia": round(float(km.inertia_), 4),
        })

res_df = pd.DataFrame(rows)
res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n已保存：{OUT_CSV}（{len(res_df)} 行）")

# ==================== 二、ARI 两两比较 ====================
print("\n" + "=" * 74)
print("两次运行一致性：ARI（调整兰德指数）")
print("=" * 74)
print("说明：ARI 对簇编号的任意置换不变，用于判断两次划分是否等价。")

ari_matrix = pd.DataFrame(np.ones((len(SEEDS), len(SEEDS))),
                          index=[f"seed={s}" for s in SEEDS],
                          columns=[f"seed={s}" for s in SEEDS])
for i, s1 in enumerate(SEEDS):
    for j, s2 in enumerate(SEEDS):
        if i < j:
            a = adjusted_rand_score(runs[s1]["raw"], runs[s2]["raw"])
            ari_matrix.loc[f"seed={s1}", f"seed={s2}"] = round(a, 4)
            ari_matrix.loc[f"seed={s2}", f"seed={s1}"] = round(a, 4)
print(ari_matrix.to_string())

# 用"对齐后"的标签再算一次，验证结果一致（编号对齐不改变 ARI）
print("\n用重编号后的标签复算（应与上表一致）：")
for i, s1 in enumerate(SEEDS):
    for s2 in SEEDS[i + 1:]:
        a2 = adjusted_rand_score(runs[s1]["labels"], runs[s2]["labels"])
        print(f"  seed={s1} vs seed={s2}：ARI = {a2:.4f}")

pairs = [(s1, s2) for i, s1 in enumerate(SEEDS) for s2 in SEEDS[i + 1:]]
ari_values = [ari_matrix.loc[f"seed={s1}", f"seed={s2}"] for s1, s2 in pairs]
min_ari = min(ari_values)

# ==================== 三、绘图（2×2） ====================
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 三幅散点
for ax, seed in zip(axes.flat[:3], SEEDS):
    labels = runs[seed]["labels"]
    for c in range(4):
        sub = df[labels == c]
        ax.scatter(sub["income"], sub["balance"], s=36, alpha=0.7,
                   color=colors[c], edgecolors="white", linewidths=0.4,
                   label=f"簇 {c}（{len(sub)} 人）")
        ax.scatter(sub["income"].mean(), sub["balance"].mean(),
                   marker="*", s=520, color=colors[c],
                   edgecolors="black", linewidths=1.0, zorder=5)
    ax.set_title(f"random_state = {seed}（inertia={runs[seed]['inertia']:.2f}）", fontsize=12)
    ax.set_xlabel("年收入 income（万元）", fontsize=10)
    ax.set_ylabel("账户平均余额 balance（万元）", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)

# 第四幅：ARI 热力图
ax = axes[1, 1]
im = ax.imshow(ari_matrix.to_numpy(), cmap="YlGn", vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(SEEDS)))
ax.set_yticks(range(len(SEEDS)))
ax.set_xticklabels(ari_matrix.columns, fontsize=10)
ax.set_yticklabels(ari_matrix.index, fontsize=10)
for i in range(len(SEEDS)):
    for j in range(len(SEEDS)):
        v = ari_matrix.iloc[i, j]
        ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                fontsize=11, color="black" if v < 0.93 else "white")
ax.set_title("三次运行的 ARI 一致性矩阵", fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("聚类稳定性检验：不同 random_state 下的 k=4 划分（income + balance）",
             fontsize=15)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"\n已保存：{OUT_PNG}")

# ==================== 四、分析报告 ====================
if min_ari >= 0.99:
    verdict = "**完全稳定**：三次划分几乎等价（ARI≈1），仅簇编号可能交换。"
elif min_ari >= 0.9:
    verdict = "**高度稳定**：三次划分基本一致，仅有极少数边界样本归属不同。"
elif min_ari >= 0.7:
    verdict = "**基本稳定但存在可见差异**：主要结构一致，部分簇的边界或合并方式随种子变化。"
else:
    verdict = "**不稳定**：不同随机种子得到明显不同的划分，聚类结果依赖初始化。"

# 找出最不稳定的一对
worst = min(pairs, key=lambda p: ari_matrix.loc[f"seed={p[0]}", f"seed={p[1]}"])

lines = [
    "# 聚类稳定性检验报告（random_state 对比）", "",
    "> 数据：课程提供的**模拟客户数据** `mall_customers_sim.csv`（300 位客户），非真实银行业务数据。",
    "> 原始数据未做任何修改。", "",
    "## 一、实验设置", "",
    "- 特征：`income`、`balance`（StandardScaler 标准化）",
    "- 聚类：`KMeans(n_clusters=4, n_init=10)`",
    f"- 随机种子：{', '.join(str(s) for s in SEEDS)}",
    "- 评估：ARI（调整兰德指数），对簇编号交换不敏感",
    "",
    "## 二、三次运行的各簇人数", "",
    "| random_state | 簇 0 | 簇 1 | 簇 2 | 簇 3 | inertia_ |",
    "|---|---|---|---|---|---|",
]
for seed in SEEDS:
    s = runs[seed]["sizes"]
    counts = [int(s.get(c, 0)) for c in range(4)]
    lines.append(f"| {seed} | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} | "
                 f"{runs[seed]['inertia']:.4f} |")

lines += ["", "> 表中簇编号已按簇中心收入升序对齐，仅为便于阅读；"
              "稳定性判断不使用编号对齐，而是下面的 ARI。", "",
          "## 三、ARI 一致性矩阵", "",
          "| | " + " | ".join(ari_matrix.columns) + " |",
          "|" + "---|" * (len(SEEDS) + 1)]
for idx, row in ari_matrix.iterrows():
    lines.append(f"| {idx} | " + " | ".join(f"{v:.4f}" for v in row.values) + " |")

lines += ["",
          f"- 最小 ARI = **{min_ari:.4f}**（出现在 seed={worst[0]} 与 seed={worst[1]} 之间）。",
          f"- 结论：{verdict}", "",
          "## 四、稳定性分析", "",
          "1. **为什么不能看簇编号**：KMeans 的簇标签由初始化顺序决定，"
          "两次完全相同的划分也可能出现「簇 0 ↔ 簇 1」对调，因此必须用 ARI 这类"
          "对标签置换不变的指标。",
          "2. **inertia 的差异**：三次运行的 inertia_ 越接近，说明收敛到的簇内平方和越一致；"
          "若差异明显，则说明存在多个局部最优解。",
          f"   本次三次 inertia_ 分别为 "
          + "、".join(f"{runs[s]['inertia']:.4f}" for s in SEEDS)
          + f"，极差 {max(runs[s]['inertia'] for s in SEEDS) - min(runs[s]['inertia'] for s in SEEDS):.4f}。",
          "3. **ARI 的含义**：ARI=1 表示两次划分完全相同（允许标签重排），"
          "ARI≈0 表示与随机划分相当；通常 >0.9 可认为高度一致。",
          "4. **对业务的影响**：",
          "   - 若结论稳定，客群划分可以直接用于业务策略，不必担心换种子后人群大变；",
          "   - 若结论不稳定，说明数据在 k=4 下的簇边界较模糊，"
          "需要增加特征、调整 k 或改用层次聚类/DBSCAN 等方法，并在报告中说明不确定性。",
          "",
          "## 五、说明", "",
          "- 本次实验固定基准特征 income + balance、k=4，与课程实践指南的两特征 k=3 口径不同。",
          "- 稳定性只说明算法对初始化是否敏感，不代表分群结果在业务上一定有效。",
          ]

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"已保存：{OUT_MD}")

# ==================== 五、终端摘要 ====================
print("\n" + "=" * 74)
print("结论摘要")
print("=" * 74)
for seed in SEEDS:
    print(f"seed={seed}: 各簇人数 {[int(runs[seed]['sizes'].get(c, 0)) for c in range(4)]}")
print(f"两两 ARI：" + "；".join(
    f"{s1}-{s2}={ari_matrix.loc[f'seed={s1}', f'seed={s2}']:.4f}" for s1, s2 in pairs))
print(f"最小 ARI = {min_ari:.4f}")
print(f"判定：{verdict}")
