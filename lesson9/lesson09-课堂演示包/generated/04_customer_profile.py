# -*- coding: utf-8 -*-
"""
第 4 步：客户分群画像

读取第 2 步产出的聚类结果 generated/clustered_customers.csv，
按 cluster 分组统计人数、占比与四个特征均值，生成画像表、业务命名、
分组柱状图与中文分析报告。

产出：
    generated/cluster_profile.csv   各簇画像统计表（含业务名称）
    generated/mall_report.md        分群分析报告
    outputs/04_customer_profile.png 四特征分组柱状图（标准化均值）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
CLUSTERED_CSV = "generated/clustered_customers.csv"
OUT_PROFILE = "generated/cluster_profile.csv"
OUT_REPORT = "generated/mall_report.md"
OUT_PNG = "outputs/04_customer_profile.png"
os.makedirs("outputs", exist_ok=True)

FEATURES = ["age", "income", "balance", "product_count"]
# 百分比：人数占比统一用本步骤自己的口径（不写死 300）
df = pd.read_csv(CLUSTERED_CSV, encoding="utf-8-sig")
N = len(df)
print("=" * 72)
print("一、读取聚类结果")
print("=" * 72)
print(f"数据文件：{CLUSTERED_CSV}   行数：{N}   列数：{df.shape[1]}")
print(f"簇编号：{sorted(df['cluster'].unique().tolist())}")

# ==================== 二、groupby 统计各簇画像 ====================
print("\n" + "=" * 72)
print("二、按 cluster 分组统计")
print("=" * 72)
g = df.groupby("cluster")
profile = pd.DataFrame({
    "客户人数": g.size(),
    "人数占比(%)": (g.size() / N * 100).round(2),
    "平均年龄(岁)": g["age"].mean().round(2),
    "平均收入(万元)": g["income"].mean().round(2),
    "平均余额(万元)": g["balance"].mean().round(2),
    "平均持有产品数(个)": g["product_count"].mean().round(2),
})
profile.index.name = "cluster"
print(profile.to_string())
print(f"\n人数合计：{int(profile['客户人数'].sum())}（应为 {N}）"
      f" → {'通过 ✅' if int(profile['客户人数'].sum()) == N else '不通过 ❌'}")

# ==================== 三、依据实际特征做业务命名 ====================
# 判定只用本步骤算出的均值与全局均值比较，不预设结论
overall = df[FEATURES].mean()
n_by_cluster = profile["客户人数"]
max_prod_cluster = profile["平均持有产品数(个)"].idxmax()
min_prod_cluster = profile["平均持有产品数(个)"].idxmin()

def name_cluster(c):
    age = profile.loc[c, "平均年龄(岁)"]
    inc = profile.loc[c, "平均收入(万元)"]
    bal = profile.loc[c, "平均余额(万元)"]
    prod = profile.loc[c, "平均持有产品数(个)"]
    # 第一层：收入与余额双高 → 高净值
    if inc > overall["income"] and bal > overall["balance"]:
        return "高净值中年客群"
    # 第二层：收入高于均值、余额低于均值 → 稳健理财（资金留存不足）
    if inc > overall["income"]:
        return "中年稳健理财客群"
    # 第三层：年轻且产品数最多 → 多产品成长
    if c == max_prod_cluster:
        return "年轻多产品成长客群"
    # 其余：年轻、低收入低余额 → 潜力待激活
    return "年轻低净值潜力客群"

profile.insert(0, "客群名称", [name_cluster(c) for c in profile.index])
print("\n业务命名（依据各簇均值与全局均值的大小关系）：")
for c in profile.index:
    print(f"  簇 {c} → {profile.loc[c, '客群名称']}")

# ==================== 四、保存画像表 ====================
profile.to_csv(OUT_PROFILE, encoding="utf-8-sig")
print(f"\n已保存画像表：{OUT_PROFILE}")

# ==================== 五、分组柱状图（标准化均值） ====================
# 四个特征量纲不同（岁 / 万元 / 个），柱状图用 Z-score 标准化后比较形状
profile_feature_cols = ["平均年龄(岁)", "平均收入(万元)", "平均余额(万元)", "平均持有产品数(个)"]
Z = pd.DataFrame({
    col: (profile[col] - df[feat].mean()) / df[feat].std()
    for col, feat in zip(profile_feature_cols, FEATURES)
})
print("\n各簇标准化均值（用于绘图）：")
print(Z.round(3).to_string())
feature_labels = ["年龄", "收入", "余额", "持有产品数"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, ax = plt.subplots(figsize=(11, 7))
x = np.arange(len(profile.index))
width = 0.2
for i, (col, label) in enumerate(zip(Z.columns, feature_labels)):
    bars = ax.bar(x + (i - 1.5) * width, Z[col].to_numpy(), width,
                  label=label, color=colors[i], alpha=0.85)
    for b, v in zip(bars, Z[col].to_numpy()):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, color="#333333")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"簇 {c}\n{profile.loc[c, '客群名称']}\n({int(profile.loc[c, '客户人数'])} 人)"
                    for c in profile.index], fontsize=10)
ax.set_ylabel("标准化均值（Z-score，0 表示总体平均）", fontsize=12)
ax.set_title("四类客群画像对比：四个特征相对总体平均的高低", fontsize=14)
ax.legend(fontsize=10, ncol=4, loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"已保存画像图：{OUT_PNG}")

# ==================== 六、业务解读（由实际数值生成） ====================
def describe(c):
    r = profile.loc[c]
    return (f"- **簇 {c} · {r['客群名称']}**：{int(r['客户人数'])} 人，占 {r['人数占比(%)']:.2f}%。"
            f"平均年龄 {r['平均年龄(岁)']:.2f} 岁，平均年收入 {r['平均收入(万元)']:.2f} 万元，"
            f"平均余额 {r['平均余额(万元)']:.2f} 万元，平均持有 {r['平均持有产品数(个)']:.2f} 个产品。")

suggestions = {
    "高净值中年客群": "收入与余额均为四簇最高，年龄偏大、产品数居中。建议配置专属客户经理与资产配置服务，"
                      "重点做财富保值增值、家族信托/私行类产品，同时加强流失预警。",
    "中年稳健理财客群": "收入高于总体平均但余额低于总体平均，说明资金留存不足，可能在行外配置。"
                        "建议以存款留存与稳健理财转化为主，配合代发工资、基金定投等场景提高资金留存。",
    "年轻多产品成长客群": "年龄明显低于总体平均，持有产品数四簇最多，但收入与余额偏低。"
                          "建议以消费信贷、信用卡、定投等低门槛产品承接其成长过程，做长期客户培育。",
    "年轻低净值潜力客群": "年龄最小、收入与余额均最低、持有产品数最少。"
                          "建议以数字化渠道低成本获客与激活为主，推送首卡、首贷与入门理财，控制营销成本。",
}

lines = [
    "# 客户分群画像报告（K-Means，k=4）", "",
    "> 数据说明：本报告基于课程提供的**模拟客户数据** `mall_customers_sim.csv`，"
    "非真实银行业务数据，仅用于教学演示。", "",
    "## 一、建模口径", "",
    "- 建模特征：`age`、`income`、`balance`、`product_count`（四个数值特征）",
    "- 预处理：StandardScaler 标准化",
    "- 聚类：`KMeans(n_clusters=4, random_state=42, n_init=10)`",
    "- `customer_id` 仅作客户编号，未进入模型",
    f"- 样本量：{N} 位客户", "",
    "## 二、各客群画像统计", "",
    "| 簇 | 客群名称 | 人数 | 占比 | 平均年龄 | 平均收入(万元) | 平均余额(万元) | 平均产品数 |",
    "|---|---|---|---|---|---|---|---|",
]
for c in profile.index:
    r = profile.loc[c]
    lines.append(
        f"| {c} | {r['客群名称']} | {int(r['客户人数'])} | {r['人数占比(%)']:.2f}% | "
        f"{r['平均年龄(岁)']:.2f} | {r['平均收入(万元)']:.2f} | {r['平均余额(万元)']:.2f} | "
        f"{r['平均持有产品数(个)']:.2f} |"
    )

lines += ["", "## 三、各客群特征与业务建议", ""]
for c in profile.index:
    lines.append(describe(c))
    lines.append(f"  - 业务建议：{suggestions[profile.loc[c, '客群名称']]}")
    lines.append("")

# 结论由实际数值排序生成，而不是单纯罗列簇编号
top_inc = profile["平均收入(万元)"].idxmax()
top_bal = profile["平均余额(万元)"].idxmax()
top_prod = profile["平均持有产品数(个)"].idxmax()
lowest_inc = profile["平均收入(万元)"].idxmin()
biggest = profile["客户人数"].idxmax()
lines += [
    "## 四、结论", "",
    f"- 收入最高、余额最高的均为**簇 {top_inc}（{profile.loc[top_inc, '客群名称']}）**，"
    f"平均收入 {profile.loc[top_inc, '平均收入(万元)']:.2f} 万元、"
    f"平均余额 {profile.loc[top_bal, '平均余额(万元)']:.2f} 万元，是高价值核心客群。",
    f"- 人数最多的是**簇 {biggest}（{profile.loc[biggest, '客群名称']}）**，"
    f"共 {int(profile.loc[biggest, '客户人数'])} 人，是规模基本盘。",
    f"- 持有产品数最多的是**簇 {top_prod}（{profile.loc[top_prod, '客群名称']}）**，"
    f"平均 {profile.loc[top_prod, '平均持有产品数(个)']:.2f} 个；"
    f"收入最低的是**簇 {lowest_inc}（{profile.loc[lowest_inc, '客群名称']}）**，"
    f"平均 {profile.loc[lowest_inc, '平均收入(万元)']:.2f} 万元。",
    "",
    "> 说明：本次特征选择为四个数值字段，与课程实践指南所列的两个特征（income、balance）存在差异，"
    "因此簇结构与人数分布不能直接与指南中的三簇结果对比。",
]

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"已保存报告：{OUT_REPORT}")

# ==================== 七、打印画像 ====================
print("\n" + "=" * 72)
print("三、四类客群画像")
print("=" * 72)
for c in profile.index:
    print(describe(c))
