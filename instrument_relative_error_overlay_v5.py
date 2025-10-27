
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
儀器體積不確定度分析腳本 (v5 - 內建數據)

本腳本使用「相對誤差 (%)」(與目標體積的偏差) 作為 X 軸，
生成兩組疊加圖表，比較同類型儀器的相對準確度與精密度：
1. 滴定管組 (2 種)
2. 吸管組 (3 種)
每組包含 t-分佈疊圖與 KDE 疊圖，並在波峰處標記偏差數值。
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib.font_manager as fm
from scipy.stats import gaussian_kde
import os


def set_chinese_font():
    """
    自動檢測並設定可用的中文字型，以解決 Matplotlib 中文顯示問題。
    (Codespaces 強力修正版)
    """
    print("正在設定中文字型...")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

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


def generate_overlay_plots(group_name, experiments_in_group, output_dir):
    """
    為指定的一組儀器生成基於相對誤差 (%) 的 t-分佈疊圖與 KDE 疊圖，
    並在每條曲線的波峰處標記對應的偏差數值。
    """
    print(f"\n--- 開始處理分組：{group_name} ---")

    results = {}
    all_relative_errors = []

    # 計算每個儀器的相對誤差及其統計值
    for name, target, density, weights in experiments_in_group:
        print(f"  處理: {name}")
        abs_volumes = weights / density
        rel_volumes = (abs_volumes / target) * 100.0
        rel_errors = rel_volumes - 100.0  # 轉換為相對誤差 (%)
        all_relative_errors.extend(rel_errors)
        mu_err = np.mean(rel_errors)
        sigma_err = np.std(rel_errors, ddof=1)
        n = len(rel_errors)
        df = n - 1
        results[name] = {
            'relative_errors': rel_errors,
            'mu_err': mu_err,
            'sigma_err': sigma_err,
            'df': df
        }
        print(f"    平均誤差 (%): {mu_err:.3f}, 誤差標準差 (%): {sigma_err:.3f}")

    # 設定統一的 X 軸範圍
    overall_min = np.min(all_relative_errors)
    overall_max = np.max(all_relative_errors)
    max_sigma = max(res['sigma_err'] for res in results.values()) if results else 1
    x_min = overall_min - 4 * max_sigma
    x_max = overall_max + 4 * max_sigma
    x_min = np.floor(x_min * 10) / 10
    x_max = np.ceil(x_max * 10) / 10
    print(f"  {group_name} 通用 X 軸範圍設定為: {x_min:.1f}% 到 {x_max:.1f}%")

    # 為每個儀器指定顏色
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments_in_group)))
    instrument_colors = {name: colors[i] for i, name in enumerate(results.keys())}

    # 圖 A：疊加 t 分佈曲線
    print(f"  正在生成圖 A ({group_name})：疊加 t-分佈曲線")
    plt.figure(figsize=(12, 7))
    ax_t = plt.gca()
    x_curve = np.linspace(x_min, x_max, 1000)

    for name, res in results.items():
        y_curve_t = stats.t.pdf(x_curve, df=res['df'], loc=res['mu_err'], scale=res['sigma_err'])
        ax_t.plot(x_curve, y_curve_t, color=instrument_colors[name], linewidth=2,
                  label=f"{name} (平均誤差={res['mu_err']:.2f}%, σ={res['sigma_err']:.2f}%)")
        peak_t_y = stats.t.pdf(res['mu_err'], df=res['df'], loc=res['mu_err'], scale=res['sigma_err'])
        ax_t.plot(res['mu_err'], peak_t_y, 'k.', markersize=10)
        ax_t.text(res['mu_err'], peak_t_y, f" {res['mu_err']:.2f}%",
                  color='black', fontsize=9, ha='left', va='bottom')

    # 加入 0% 誤差基準線
    ax_t.axvline(0.0, color='b', linestyle=':', linewidth=2, label='目標體積 (0% 誤差)')

    ax_t.set_title(f'{group_name} 相對誤差分佈比較 (理論 t-分佈)', fontsize=16)
    ax_t.set_xlabel('與目標體積偏差 (%)', fontsize=12)
    ax_t.set_ylabel('機率密度', fontsize=12)
    ax_t.set_xlim(x_min, x_max)
    ax_t.legend(loc='best')
    ax_t.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    safe_group_name = group_name.replace(' ', '_').replace('/', '_')
    file_name_t = os.path.join(output_dir, f"overlay_t_error_{safe_group_name}.png")
    plt.savefig(file_name_t, dpi=300)
    plt.show()
    print(f"  圖 A ({group_name}) 已儲存至: {file_name_t}")

    # 圖 B：疊加 KDE 曲線
    print(f"  正在生成圖 B ({group_name})：疊加 KDE 曲線")
    plt.figure(figsize=(12, 7))
    ax_kde = plt.gca()

    for name, res in results.items():
        try:
            kde = gaussian_kde(res['relative_errors'])
            y_curve_kde = kde(x_curve)
            ax_kde.plot(x_curve, y_curve_kde, color=instrument_colors[name], linestyle='--', linewidth=2,
                        label=f"{name}")
            kde_peak_idx = np.argmax(y_curve_kde)
            kde_peak_x = x_curve[kde_peak_idx]
            kde_peak_y = y_curve_kde[kde_peak_idx]
            ax_kde.plot(kde_peak_x, kde_peak_y, 'k.', markersize=10)
            ax_kde.text(kde_peak_x, kde_peak_y, f" {kde_peak_x:.2f}%",
                        color='black', fontsize=9, ha='left', va='bottom')
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"    KDE 警告 ({name})：無法繪製曲線: {e}")
            ax_kde.plot([], [], color=instrument_colors[name], linestyle='--',
                        label=f"{name} (KDE 失敗)")

    ax_kde.axvline(0.0, color='b', linestyle=':', linewidth=2, label='目標體積 (0% 誤差)')
    ax_kde.set_title(f'{group_name} 相對誤差分佈比較 (樣本密度 KDE)', fontsize=16)
    ax_kde.set_xlabel('與目標體積偏差 (%)', fontsize=12)
    ax_kde.set_ylabel('機率密度', fontsize=12)
    ax_kde.set_xlim(x_min, x_max)
    ax_kde.legend(loc='best')
    ax_kde.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    file_name_kde = os.path.join(output_dir, f"overlay_kde_error_{safe_group_name}.png")
    plt.savefig(file_name_kde, dpi=300)
    plt.show()
    print(f"  圖 B ({group_name}) 已儲存至: {file_name_kde}")


def main():
    """
    主函數：定義所有實驗數據，按儀器類型分組，並為每組生成相對誤差疊圖。
    """
    set_chinese_font()
    all_experiments = [
        ("1mL刻度吸管", 1.0, 0.9968, np.array([0.7517, 0.9122, 0.9343, 0.9174, 1.0165, 1.0071])),
        ("10mL刻度吸管", 10.0, 0.9968, np.array([9.9094, 9.8970, 9.8806, 9.9107, 9.8025, 9.7799])),
        ("1mL滴定管", 1.0, 0.9968, np.array([1.1935, 1.0952, 1.0515, 0.9904, 1.0004, 1.0601])),
        ("10mL滴定管", 10.0, 0.9968, np.array([10.0740, 10.0338, 9.8277, 10.0358, 9.426, 10.1115])),
        ("1000uL微量吸管", 1.0, 0.9968, np.array([0.9933, 0.9678, 0.9998, 0.9914, 0.9845, 0.9823]))
    ]

    burette_experiments = [exp for exp in all_experiments if "滴定管" in exp[0]]
    pipette_experiments = [exp for exp in all_experiments if "吸管" in exp[0]]

    output_dir = "output_overlay_grouped_error"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立儲存資料夾: {output_dir}")

    generate_overlay_plots("滴定管組", burette_experiments, output_dir)
    generate_overlay_plots("吸管組", pipette_experiments, output_dir)
    print("\n--- 所有分組相對誤差疊圖已完成 ---")


if __name__ == "__main__":
    main()
