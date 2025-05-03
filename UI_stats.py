# stats.py

import os
import matplotlib.pyplot as plt

# 1. 在模块加载时，配置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']      # 指定黑体
plt.rcParams['axes.unicode_minus'] = False        # 负号正常显示

def generate_stats_charts(counts: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    labels = list(counts.keys())
    values = list(counts.values())

    # ———— 饼图 ———— #
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.8,
        wedgeprops=dict(width=0.5)
    )
    ax.axis('equal')
    # 右侧图例显示中文类别
    ax.legend(
        wedges,
        labels,
        title="类别",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        prop={'size': 10}  # 图例字体大小
    )
    plt.title("类别分布饼图")
    plt.tight_layout()
    pie_path = os.path.join(save_dir, "pie_chart.png")
    plt.savefig(pie_path)
    plt.close()

    # ———— 柱状图 ———— #
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title("类别数量柱状图")
    plt.xlabel("类别")
    plt.ylabel("数量")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    bar_path = os.path.join(save_dir, "bar_chart.png")
    plt.savefig(bar_path)
    plt.close()

    # ———— 折线图 ———— #
    plt.figure(figsize=(10, 6))
    plt.plot(labels, values, marker='o')
    plt.title("类别数量折线图")
    plt.xlabel("类别")
    plt.ylabel("数量")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    line_path = os.path.join(save_dir, "line_chart.png")
    plt.savefig(line_path)
    plt.close()

    return pie_path, bar_path, line_path
