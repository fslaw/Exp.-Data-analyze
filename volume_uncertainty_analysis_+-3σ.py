#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
儀器體積不確定度分析腳本 (v11 - 結報濃縮版 - 內建數據 - 加入 ±3σ 線)

本腳本用於分析實驗室儀器（如滴定管、吸管）的體積測量不確定度。
它會執行單一樣本 t-檢定 (One-Sample t-Test)，並為每項儀器生成一張
重點突出的可視化圖表，包含標示 ±1σ, ±2σ, ±3σ 的垂直線。
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib.font_manager as fm
from scipy.stats import gaussian_kde # 雖然不畫 KDE，但 t-test 仍需 scipy
import sys
import os
# import csv  # 不再需要

def set_chinese_font():
    """
    自動檢測並設定可用的中文字型，以解決 Matplotlib 中文顯示問題。
    (Codespaces 強力修正版)
    """
    print("正在設定中文字型...")

    # Matplotlib 的預設 sans-serif 字體
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

    # --- 我們不再猜測名稱，而是直接檢查 Codespaces 中字型的「檔案路徑」---
    codespace_font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'

    if os.path.exists(codespace_font_path):
        print(f"成功找到字型檔案 (via path): {codespace_font_path}")
        font_prop = fm.FontProperties(fname=codespace_font_path)
        plt.rcParams['font.sans-serif'].insert(0, font_prop.get_name())
        print(f"成功設定字型: {font_prop.get_name()}")
        return True
    else:
        print(f"警告：找不到指定的字型檔案: {codespace_font_path}")
        print("圖表中的中文標籤可能顯示為方塊 (□)。")
        print("請確認您已在 Codespaces 終端機中執行過:")
        print("sudo apt-get update && sudo apt-get install -y fonts-wqy-zenhei")
        return False

def analyze_and_plot(instrument_name, target_volume, density, raw_weights):
    """

    針對單一儀器進行分析並繪圖。
    """
    print(f"\n--- 正在分析: {instrument_name} ---")

    # --- 2. 自動計算 ---
    if not isinstance(raw_weights, np.ndarray):
        raw_weights = np.array(raw_weights)

    calculated_volumes = raw_weights / density
    print(f"  {len(raw_weights)} 筆原始重量 (g): {raw_weights}")
    print(f"  換算後的 {len(raw_weights)} 筆體積 (mL): {np.round(calculated_volumes, 4)}")

    # --- 3. 執行 t-檢定 ---
    print("  --- 雙尾 t-檢定結果 ---")
    t_statistic, p_value = stats.ttest_1samp(calculated_volumes, target_volume, alternative='two-sided')
    print(f"  t 統計值 (t-statistic): {t_statistic:.4g}")
    print(f"  p 值 (p-value):         {p_value:.4g}")

    alpha = 0.05
    if p_value < alpha:
        print(f"  判讀: p < {alpha}，結果具有統計顯著性 (不準確)。")
    else:
        print(f"  判讀: p >= {alpha}，結果不具統計顯著性 (準確)。")

    # --- 4. 繪製鐘型分佈圖 ---
    data_to_plot = calculated_volumes
    n = len(data_to_plot)
    df = n - 1

    # (b) 計算統計數據
    mu = np.mean(data_to_plot)
    sigma = np.std(data_to_plot, ddof=1)
    relative_error = ((mu - target_volume) / target_volume) * 100.0

    print(f"  測量平均值 (μ): {mu:.4g} mL")
    print(f"  樣本標準偏差 (σ): {sigma:.4g} mL")
    print(f"  相對誤差 (E_rel): {relative_error:.2f} %")

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # (c) 自動化 X 軸
    x_min = mu - 4*sigma
    x_max = mu + 4*sigma
    bins_strategy = 'auto'

    # (d) 繪製直方圖 (Histogram) -
    ax.hist(data_to_plot, bins=bins_strategy, density=True, alpha=0.6,
             color='g', label="機率密度(來自測量值)", edgecolor='black', zorder=2) # zorder=2

    # (e) 繪製鐘型曲線
    x_curve = np.linspace(x_min, x_max, 1000)
    y_curve_t = stats.t.pdf(x_curve, df=df, loc=mu, scale=sigma)
    ax.plot(x_curve, y_curve_t, 'k-', linewidth=2, label=f"理論 t-分佈曲線 (df={df})", zorder=3) #

    # (f) 繪製標記線條
    # 畫在 zorder=4，確保在最上層
    ax.axvline(mu, color='r', linestyle='--', linewidth=2, label=rf'測量平均值 ($\mu$) = {mu:.4g} mL', zorder=4)  #
    ax.axvline(target_volume, color='b', linestyle=':', linewidth=2, label=f'目標體積 (理論值) = {target_volume:.2f} mL', zorder=4) #

    # !!標記 1, 2, 3 倍標準差 (使用不同透明度) !!
    # 畫在 zorder=1，線條會在柱狀圖下面
    ax.axvline(mu + 1*sigma, color='purple', linestyle=':', linewidth=1.5, alpha=1.0, label=rf'標準差 $\sigma$ 區間 (1, 2, 3)', zorder=1)
    ax.axvline(mu - 1*sigma, color='purple', linestyle=':', linewidth=1.5, alpha=1.0, zorder=1)
    ax.axvline(mu + 2*sigma, color='purple', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
    ax.axvline(mu - 2*sigma, color='purple', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
    ax.axvline(mu + 3*sigma, color='purple', linestyle=':', linewidth=1.5, alpha=0.4, zorder=1)
    ax.axvline(mu - 3*sigma, color='purple', linestyle=':', linewidth=1.5, alpha=0.4, zorder=1)

    # (g) 顯示檢定結果文字 -
    text_to_display = (
        rf"檢定方式: 單一樣本 t-檢定 (雙尾)"
        "\n"
        rf"虛無假設 ($H_0$): μ = {target_volume:.2f} mL"
        "\n"
        rf"相對誤差: {relative_error:.2f} %"
        "\n"
        rf"t-statistic: {t_statistic:.4g}"
        "\n"
        rf"p-value: {p_value:.4g}"
    )
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    ax.text(0.95, 0.95, text_to_display,
             transform=ax.transAxes,
             fontsize=12,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=bbox_props,
             zorder=10) # zorder=10 確保在最最上層

    # (h) 設定 X 軸
    ax.set_xlim(x_min, x_max)
    ax.locator_params(axis='x', nbins=10) # 讓 Matplotlib 自動找漂亮的主要刻度
    plt.xticks(rotation=30, ha='right')

    # 建立頂部的「標準差刻度」軸 -
    ax2 = ax.secondary_xaxis('top')
    sigma_ticks = [mu - 3*sigma, mu - 2*sigma, mu - 1*sigma, mu, mu + 1*sigma, mu + 2*sigma, mu + 3*sigma]
    sigma_labels = [r'$\mu-3\sigma$', r'$\mu-2\sigma$', r'$\mu-1\sigma$', r'$\mu$', r'$\mu+1\sigma$', r'$\mu+2\sigma$', r'$\mu+3\sigma$']

    ax2.set_xticks(sigma_ticks)
    ax2.set_xticklabels(sigma_labels, rotation=30, ha='left', fontsize=10)
    ax2.set_xlabel('標準差參照點', fontsize=12)


    # (i) 圖表收尾並儲存
    data_label = f"{instrument_name} 測量體積 (mL)"
    ax.set_title(f'數據分佈與標準差 ({data_label})', fontsize=16)
    ax.set_xlabel('測量體積 (mL)', fontsize=12)
    ax.set_ylabel('機率密度', fontsize=12)
    ax.legend(loc='best') # 圖例現在會更乾淨

    # !!只開啟「水平」網格線 (axis='y') !!
    # 因為垂直線已經由紫色的 sigma 線標示了
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout() # 自動調整佈局以防止標籤重疊

    # 建立一個 'output' 資料夾來存放圖片 (如果它不存在)
    output_dir = "output_condensed_with_sigma_lines" # 存到新資料夾
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"  已建立儲存資料夾: {output_dir}")
        except OSError as e:
            print(f"  錯誤：無法建立資料夾 {output_dir}: {e}")
            output_dir = "." # 如果失敗，儲存在當前目錄

    # 移除檔名中不安全的字元
    safe_filename = instrument_name.replace(' ', '_').replace('/', '_').replace('μ', 'u')
    file_name = os.path.join(output_dir, f"{safe_filename}_condensed_sigma_plot.png")

    # 儲存高解析度圖片 (300dpi)
    plt.savefig(file_name, dpi=300)
    plt.show() # 在本地執行時顯示圖表
    # plt.close() # 如果您不想看到彈出視窗，可以取消註解此行
    print(f"  繪圖完成！已儲存高解析度圖片至: {file_name}")

# --- 主程式執行 ---
def main():
    """
    主函數：定義所有實驗數據並依序執行分析。
    """
    set_chinese_font()

    # --- 1. 定義所有實驗數據 ---
    # (instrument_name, target_volume, density, weights_array)
    all_experiments = [
        (
            "1mL刻度吸管", 1.0, 0.9968,
            np.array([0.7517, 0.9122, 0.9343, 0.9174, 1.0165, 1.0071])
        ),
        (
            "10mL刻度吸管", 10.0, 0.9968,
            np.array([9.9094, 9.8970, 9.8806, 9.9107, 9.8025, 9.7799])
        ),
        (
            "1mL滴定管", 1.0, 0.9968,
            np.array([1.1935, 1.0952, 1.0515, 0.9904, 1.0004, 1.0601])
        ),
        (
            "10mL滴定管", 10.0, 0.9968,
            np.array([10.0740, 10.0338, 9.8277, 10.0358, 9.426, 10.1115])
        ),
        (
            "1000μL微量吸管", 1.0, 0.9968,
            np.array([0.9933, 0.9678, 0.9998, 0.9914, 0.9845, 0.9823])
        )
    ]

    print("--- 開始批次分析儀器不確定度 ---")

    # 循環處理所有實驗
    for experiment in all_experiments:
        try:
            name, target, density, weights = experiment
            # 呼叫核心函數
            analyze_and_plot(name, target, density, weights)
        except Exception as e:
            print(f"!!! 錯誤：處理 '{name}' 時發生未預期錯誤: {e} !!!")

    print("\n--- 所有分析已完成 ---")

if __name__ == "__main__":
    main()
