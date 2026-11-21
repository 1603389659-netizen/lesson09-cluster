# -*- coding: utf-8 -*-
"""
第 7 步（自主实践三）：四簇聚类二维散点图

直接读取前序步骤真实产出的结果，不重新聚类：
    generated/clustered_customers.csv  （含真实 cluster 列，k=4）
    generated/cluster_profile.csv      （含客群名称与人数）

横轴 income、纵轴 balance，每个客户一个散点，按 cluster 着色，
图例与簇中心标签标注客群名称和人数。

产出：
    outputs/07_cluster_scatter.png
"""
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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

CLUSTERED_CSV = "generated/clustered_customers.csv"
PROFILE_CSV = "generated/cluster_profile.csv"
OUT_PNG = "outputs/07_cluster_scatter.png"
os.makedirs("outputs", exist_ok=True)

# ==================== 一、读取真实聚类结果 ====================
print("=" * 64)
print("一、读取已完成聚类结果（不重新聚类）")
print("=" * 64)
df = pd.read_csv(CLUSTERED_CSV, encoding="utf-8-sig")
profile = pd.read_csv(PROFILE_CSV, encoding="utf-8-sig").set_index("cluster")
print(f"聚类结果：{CLUSTERED_CSV}   行数：{len(df)}   列数：{df.shape[1]}")
print(f"cluster 列取值：{sorted(df['cluster'].unique().tolist())}")
print(f"各簇人数（直接统计自 cluster 列）："
      f"{df['cluster'].value_counts().sort_index().tolist()}")

# 簇名称优先取画像表，缺失则用簇编号兜底（不臆造名称）
name_map, count_map = {}, {}
for c in sorted(df["cluster"].unique()):
    cnt = int((df["cluster"] == c).sum())
    count_map[c] = cnt
    if c in profile.index and "客群名称" in profile.columns:
        name_map[c] = str(profile.loc[c, "客群名称"])
    else:
        name_map[c] = f"簇 {c}"

print("\n图例将使用的客群名称与人数：")
for c in sorted(df["cluster"].unique()):
    print(f"  簇 {c}：{name_map[c]}，{count_map[c]} 人")

# ==================== 二、绘图 ====================
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
fig, ax = plt.subplots(figsize=(11, 7.5))

for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    ax.scatter(
        sub["income"], sub["balance"],
        s=45, alpha=0.72, color=colors[c % len(colors)],
        edgecolors="white", linewidths=0.5,
        label=f"簇 {c} · {name_map[c]}（{count_map[c]} 人）",
        zorder=3,
    )

# 簇中心星形 + 文字标注（名称与人数）；按簇微调偏移，避免遮挡散点
label_offsets = {0: (14, -34), 1: (-78, 42), 2: (16, 22), 3: (10, -40)}
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    cx, cy = sub["income"].mean(), sub["balance"].mean()
    ax.scatter(cx, cy, marker="*", s=600, color=colors[c % len(colors)],
               edgecolors="black", linewidths=1.1, zorder=5)
    ax.annotate(
        f"簇 {c} {name_map[c]}\n（{count_map[c]} 人）",
        xy=(cx, cy), xytext=label_offsets.get(c, (8, 10)), textcoords="offset points",
        fontsize=10, color="#222222", zorder=6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[c % len(colors)],
                  alpha=0.85),
    )

ax.set_title("四簇客户分布：收入-余额平面（每个点为一客户）", fontsize=15)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=12)
ax.grid(alpha=0.3)
ax.legend(fontsize=9.5, loc="upper left", title="客群（簇编号 · 名称 · 人数）",
          title_fontsize=10)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()

print(f"\n已保存图片：{OUT_PNG}")
print("数据来源：generated/clustered_customers.csv 的真实 cluster 列，未重新聚类。")
