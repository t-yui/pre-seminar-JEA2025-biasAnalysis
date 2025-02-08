#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import japanize_matplotlib
from tqdm import tqdm

np.random.seed(0)


def calcOR(X, Y):
    a = np.sum((X == 1) & (Y == 1))
    b = np.sum((X == 1) & (Y == 0))
    c = np.sum((X == 0) & (Y == 1))
    d = np.sum((X == 0) & (Y == 0))
    table = np.array([a, b, c, d])
    # カウント数がゼロのセルに0.5を足すことでゼロ除算を回避
    a, b, c, d = (x if x > 0 else 0.5 for x in (a, b, c, d))
    OR = (a * d) / (c * b)
    return OR, table


def calcB(p1, p0, RR_UY):
    B = (RR_UY * p1 + (1 - p1)) / (RR_UY * p0 + (1 - p0))
    return B


def fixedBiasAnalysisOR(calcORfunc, X, Y, p1, p0, RR_UY, N_boot=100000):
    """Bias Analysis of OR with Fixed Bias Parameter"""
    OR, _ = calcORfunc(X, Y)
    B = calcB(p1, p0, RR_UY)
    adjusted_OR = OR / B

    OR_vals = []
    indices = np.arange(len(X))
    # ブートストラップ法で信頼区間計算
    for i in range(N_boot):
        boot_idx = np.random.choice(indices, size=len(X), replace=True)
        OR_val, _ = calcORfunc(X[boot_idx], Y[boot_idx])
        OR_vals.append(OR_val / B)

    OR_lower = np.percentile(OR_vals, 2.5)
    OR_upper = np.percentile(OR_vals, 97.5)
    return adjusted_OR, OR_lower, OR_upper


def probBiasAnalysisOR(
    calcORfunc,
    X,
    Y,
    p1_unif_params=[0, 1],
    p0_unif_params=[0, 1],
    RR_unif_params=[0, 5],
    N_sim=1000,
    N_boot=1000,
):
    """Probabilistic Bias Analysis"""
    indices = np.arange(len(X))
    OR_vals = []
    for i in tqdm(range(N_sim), desc="Iteration of PBA"):
        # バイアスパラメータのサンプリング
        p1_i = np.random.uniform(p1_unif_params[0], p1_unif_params[1])
        p0_i = np.random.uniform(p0_unif_params[0], p0_unif_params[1])
        RR_ZY_i = np.random.uniform(RR_unif_params[0], RR_unif_params[1])
        B_i = (RR_ZY_i * p1_i + (1 - p1_i)) / (RR_ZY_i * p0_i + (1 - p0_i))
        # ブートストラップ再サンプリング
        for i in range(N_boot):
            boot_idx = np.random.choice(indices, size=len(X), replace=True)
            OR_val, _ = calcORfunc(X[boot_idx], Y[boot_idx])
            OR_vals.append(OR_val / B_i)
    return np.array(OR_vals)


def plot_distribution(
    x_values,
    pdf_values,
    xlabel,
    ylabel,
    filename,
    xlim,
    ylim,
    hline_end,
    save=False,
    show=True,
):
    plt.rcParams["font.size"] = 52
    plt.figure(figsize=(20, 6))
    plt.plot(x_values, pdf_values, color="skyblue", linewidth=2)
    plt.fill_between(x_values, pdf_values, alpha=0.95, color="skyblue")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.hlines(0, 0, hline_end, linewidth=8)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig(filename, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close()


def plot_histogram(
    data, median_val, lower_val, upper_val, filename, save=False, show=True
):
    plt.rcParams["font.size"] = 52
    plt.figure(figsize=(30 * 0.9, 14 * 0.9))
    plt.hist(data, bins=50, density=True, alpha=0.95, color="skyblue")
    plt.xlabel("補正後のオッズ比推定値", fontsize=80)
    plt.ylabel("頻度", fontsize=80)
    plt.axvline(
        median_val, color="red", linestyle="dashed", linewidth=15, label="中央値"
    )
    plt.axvline(
        lower_val, color="green", linestyle="dashed", linewidth=15, label="2.5%"
    )
    plt.axvline(
        upper_val, color="blue", linestyle="dashed", linewidth=15, label="97.5%"
    )
    plt.axvline(1.0, color="black", linestyle="dotted", linewidth=15, label="1.0")
    plt.legend(fontsize=80, bbox_to_anchor=(0.75, 1), loc="upper left")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if save:
        plt.savefig(filename, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":

    # データ生成
    n_total = 10000
    n_vac = 5000
    n_unvac = 5000
    X = np.concatenate((np.ones(n_vac), np.zeros(n_unvac)))
    p = np.where(X == 1, 0.08, 0.10)
    Y = np.random.binomial(1, p, n_total)

    # 粗な解析
    crude_OR, observed_table = calcOR(X, Y)
    OR_vals = []
    N_boot = 100000
    indices = np.arange(len(X))
    for i in range(N_boot):
        boot_idx = np.random.choice(indices, size=len(X), replace=True)
        OR_val, _ = calcOR(X[boot_idx], Y[boot_idx])
        OR_vals.append(OR_val)
    crude_OR_lower = np.percentile(OR_vals, 2.5)
    crude_OR_upper = np.percentile(OR_vals, 97.5)

    print("\n【テーブルデータ】")
    print(
        "\nワクチン接種群：ケース =", observed_table[0], ", 非ケース =", observed_table[1]
    )
    print("未接種群：ケース =", observed_table[2], ", 非ケース =", observed_table[3])

    print("\n【粗なオッズ比】")
    print("\nCrude OR =", crude_OR)
    print(f"95% CI = [{crude_OR_lower:.3f}, {crude_OR_upper:.3f}]")

    # 固定値によるバイアス分析
    print("\n【固定値によるバイアス分析】")

    ## setting 1
    p1 = 0.20
    p0 = 0.10
    RR_UY = 1.5
    adjusted_OR, OR_lower, OR_upper = fixedBiasAnalysisOR(calcOR, X, Y, p1, p0, RR_UY)
    print(f"\nバイアスパラメータ: p1 = {p1}, p0 = {p0}, RR_UY = {RR_UY}")
    print(f"補正後 OR = {adjusted_OR:.3f}")
    print(f"95% CI = [{OR_lower:.3f}, {OR_upper:.3f}]")

    ## setting 2
    p1 = 0.10
    p0 = 0.20
    RR_UY = 1.5
    adjusted_OR, OR_lower, OR_upper = fixedBiasAnalysisOR(calcOR, X, Y, p1, p0, RR_UY)
    print(f"\nバイアスパラメータ: p1 = {p1}, p0 = {p0}, RR_UY = {RR_UY}")
    print(f"補正後 OR = {adjusted_OR:.3f}")
    print(f"95% CI = [{OR_lower:.3f}, {OR_upper:.3f}]")

    ## setting 3
    p1 = 0.20
    p0 = 0.10
    RR_UY = 5.0
    adjusted_OR, OR_lower, OR_upper = fixedBiasAnalysisOR(calcOR, X, Y, p1, p0, RR_UY)
    print(f"\nバイアスパラメータ: p1 = {p1}, p0 = {p0}, RR_UY = {RR_UY}")
    print(f"補正後 OR = {adjusted_OR:.3f}")
    print(f"95% CI = [{OR_lower:.3f}, {OR_upper:.3f}]")

    # 確率的バイアス分析
    print("\n【確率的バイアス分析】")
    
    N_sim = 2000
    N_boot = 2000
    p1_unif_params = [0.1, 0.5]
    p0_unif_params = [0.1, 0.2]
    RR_unif_params = [1.0, 5.0]
    adjusted_OR_array = probBiasAnalysisOR(
        calcOR, X, Y, p1_unif_params, p0_unif_params, RR_unif_params, N_sim, N_boot
    )
    median_OR_sim = np.median(adjusted_OR_array)
    lower_bound_sim = np.percentile(adjusted_OR_array, 2.5)
    upper_bound_sim = np.percentile(adjusted_OR_array, 97.5)
    print(f"中央値 = {median_OR_sim:.3f}")
    print(f"95%シミュレーション区間 = [{lower_bound_sim:.3f}, {upper_bound_sim:.2f}]")

    ## バイアスパラメータの分布
    num_points = 1000
    p1_values = np.linspace(p1_unif_params[0], p1_unif_params[1], num_points)
    p0_values = np.linspace(p0_unif_params[0], p0_unif_params[1], num_points)
    RR_ZY_values = np.linspace(RR_unif_params[0], RR_unif_params[1], num_points)
    p1_pdf = uniform.pdf(p1_values, loc=0.10, scale=0.40)
    p0_pdf = uniform.pdf(p0_values, loc=0.10, scale=0.10)
    RR_ZY_pdf = uniform.pdf(RR_ZY_values, loc=1.0, scale=4.0)
    ### p1
    plot_distribution(
        p1_values,
        p1_pdf,
        "$p_1$",
        "確率密度",
        "./dist_p1.eps",
        xlim=(-0.1, 1.1),
        ylim=(-0.07, 3),
        hline_end=1,
        save=False,
        show=True,
    )
    ### p0
    plot_distribution(
        p0_values,
        p0_pdf,
        "$p_0$",
        "確率密度",
        "./dist_p0.eps",
        xlim=(-0.1, 1.1),
        ylim=(-0.3, 12),
        hline_end=1,
        save=False,
        show=True,
    )
    ### RR_ZY
    plot_distribution(
        RR_ZY_values,
        RR_ZY_pdf,
        "$\\mathrm{RR}_{ZY}$",
        "確率密度",
        "./dist_RR.eps",
        xlim=(-0.1, 6.1),
        ylim=(-0.01, 0.4),
        hline_end=6.1,
        save=False,
        show=True,
    )

    ## 補正後 OR の頻度分布
    plot_histogram(
        adjusted_OR_array,
        median_OR_sim,
        lower_bound_sim,
        upper_bound_sim,
        "./dist_OR.eps",
        save=False,
        show=True,
    )
