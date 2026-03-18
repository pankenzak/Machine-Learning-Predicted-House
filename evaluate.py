"""
Phan 3: Danh gia & Output (Evaluation & Output Analysis)
=========================================================
- Buoc 1: Goodness of Fit  (R², RMSE, SST/SSR/SSE)         → da co chart 04, 05
- Buoc 2: Phan tich Phan du (Q-Q Plot)                      → chart moi 08
- Buoc 3: Hypothesis Testing & P-value                      → chart moi 09, 10
- Buoc 4: Confidence Intervals                               → chart moi 11, 12
- Buoc 5: Summary Dashboard                                  → chart moi 13
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Thu muc luu anh Phan 3
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Output", "Evaluation-&-Output-Analysis")


def _save(fig, filename, title_explain):
    """Luu figure ra file PNG va dong."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] {filename}  — {title_explain}")
    return path


# ========================== BUOC 2 ==========================
def plot_qq(y_test, predictions):
    """Bieu do 08 — Q-Q Plot kiem tra phan du co tuan theo phan phoi chuan."""
    residuals = y_test.values - predictions if hasattr(y_test, 'values') else y_test - predictions

    fig, ax = plt.subplots(figsize=(8, 7))
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, color='steelblue', s=12, alpha=0.5, zorder=3)
    line_x = np.array([osm.min(), osm.max()])
    ax.plot(line_x, slope * line_x + intercept, color='red', lw=2,
            label=f'Duong ly thuyet (r={r:.4f})')

    ax.set_xlabel('Quantiles ly thuyet (Standard Normal)', fontsize=11)
    ax.set_ylabel('Quantiles thuc te (Phan du)', fontsize=11)
    ax.set_title(
        "BIEU DO 08 — Q-Q Plot (Normal Probability Plot)\n"
        "Kiem tra: Phan du co tuan theo Phan phoi Chuan N(0, σ²) khong?\n"
        "Cac diem sat duong do = phan du ~ Normal",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Ghi chu giai thich
    ax.text(0.02, 0.95,
            'Ly thuyet (Ch.4 MAS291):\n'
            'Neu phan du tuan theo phan phoi\n'
            'chuan → cac diem nam tren\n'
            'duong thang do.',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    return _save(fig, "08_qq_plot.png",
                 "Q-Q Plot — Kiem tra phan du ~ N(0, sigma^2)")


# ========================== BUOC 3 ==========================
def fit_ols(X_train, y_train, feature_names):
    """Train OLS model bang statsmodels de lay P-value, CI, F-test."""
    X_with_const = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_with_const).fit()
    return ols_model


def plot_pvalue_bar(ols_model, feature_names):
    """Bieu do 09 — P-value Bar Chart cho tung bien."""
    pvalues = ols_model.pvalues[1:]  # Bo constant
    names = feature_names

    # Sap xep theo P-value giam dan (de doc)
    sorted_idx = np.argsort(pvalues)[::-1]
    names_sorted = [names[i] for i in sorted_idx]
    pvalues_sorted = pvalues.iloc[sorted_idx] if hasattr(pvalues, 'iloc') else pvalues[sorted_idx]

    colors = ['steelblue' if p < 0.05 else 'tomato' for p in pvalues_sorted]

    fig, ax = plt.subplots(figsize=(11, max(6, len(names) * 0.4)))
    bars = ax.barh(names_sorted, pvalues_sorted, color=colors, edgecolor='white', height=0.6)
    ax.axvline(0.05, color='red', lw=2, ls='--', label='α = 0.05 (Nguong y nghia)')

    # Ghi P-value len bar
    for bar, pv in zip(bars, pvalues_sorted):
        txt = f'{pv:.4f}' if pv >= 0.001 else f'{pv:.2e}'
        offset = max(pv * 0.05, 0.002)
        ax.text(pv + offset, bar.get_y() + bar.get_height() / 2,
                txt, va='center', fontsize=8)

    ax.set_xlabel('P-value', fontsize=11)
    ax.set_title(
        "BIEU DO 09 — Kiem dinh Gia thuyet: P-value cho tung bien\n"
        "H₀: βⱼ = 0 (bien khong anh huong) | H₁: βⱼ ≠ 0\n"
        "Xanh = Bac bo H₀ (P < 0.05, CO y nghia) | Do = Khong bac bo H₀ (P ≥ 0.05)",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10, loc='lower right')

    # Ghi chu ly thuyet
    sig_count = sum(1 for p in pvalues_sorted if p < 0.05)
    nonsig_count = len(pvalues_sorted) - sig_count
    ax.text(0.98, 0.05,
            f'Ket qua: {sig_count} bien CO y nghia,\n{nonsig_count} bien KHONG y nghia\n'
            '(Chuong 9 MAS291: Hypothesis Testing)',
            transform=ax.transAxes, fontsize=9, ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    return _save(fig, "09_pvalue_bar.png",
                 "Kiem dinh Gia thuyet — P-value tung bien")


def plot_ols_summary(ols_model, feature_names):
    """Bieu do 10 — Bang tom tat OLS (t-stat, P-value, CI)."""
    # Lay du lieu tu model
    params = ols_model.params[1:]
    se = ols_model.bse[1:]
    tvalues = ols_model.tvalues[1:]
    pvalues = ols_model.pvalues[1:]
    ci = ols_model.conf_int(alpha=0.05).iloc[1:]

    # Tao DataFrame
    summary_df = pd.DataFrame({
        'β̂': [f'{v:.4f}' for v in params],
        'SE(β̂)': [f'{v:.4f}' for v in se],
        't-stat': [f'{v:.2f}' for v in tvalues],
        'P-value': [f'{v:.4f}' if v >= 0.001 else f'{v:.2e}' for v in pvalues],
        '95% CI Lower': [f'{v:.4f}' for v in ci.iloc[:, 0]],
        '95% CI Upper': [f'{v:.4f}' for v in ci.iloc[:, 1]],
        'Y nghia?': ['✓' if p < 0.05 else '✗' for p in pvalues],
    }, index=feature_names)

    # Ve bang
    fig, ax = plt.subplots(figsize=(14, max(4, len(feature_names) * 0.45 + 3)))
    ax.axis('off')

    # Tieu de
    fig.suptitle(
        "BIEU DO 10 — Bang Tom tat OLS Regression\n"
        f"R² = {ols_model.rsquared:.4f} | Adj. R² = {ols_model.rsquared_adj:.4f} | "
        f"F-stat = {ols_model.fvalue:.2f} | P(F) = {ols_model.f_pvalue:.2e}",
        fontsize=13, fontweight='bold', y=0.98
    )

    # Tao table
    cell_colors = []
    for p in pvalues:
        if p < 0.05:
            cell_colors.append(['#d4edda'] * 7)
        else:
            cell_colors.append(['#f8d7da'] * 7)

    table = ax.table(
        cellText=summary_df.values,
        rowLabels=summary_df.index,
        colLabels=summary_df.columns,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#343a40')
            cell.set_text_props(color='white', fontweight='bold')
        if col == -1:
            cell.set_text_props(fontweight='bold')

    # Ghi chu F-test
    conclusion = "Bac bo H₀ → Mo hinh CO y nghia thong ke" if ols_model.f_pvalue < 0.05 \
        else "Khong bac bo H₀ → Mo hinh KHONG y nghia"

    ax.text(0.5, 0.02,
            f'F-test toan bo mo hinh: F = {ols_model.fvalue:.2f}, P = {ols_model.f_pvalue:.2e} → {conclusion}\n'
            'Xanh = bien co y nghia (P < 0.05) | Do = bien khong y nghia (P ≥ 0.05)\n'
            '(Chuong 9 & 10 MAS291: t-test cho tung bien, F-test cho toan bo mo hinh)',
            transform=ax.transAxes, fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    return _save(fig, "10_ols_summary_table.png",
                 f"OLS Summary — R2={ols_model.rsquared:.4f}, F={ols_model.fvalue:.2f}")


# ========================== BUOC 4 ==========================
def plot_coefficient_ci(ols_model, feature_names):
    """Bieu do 11 — 95% Confidence Interval cho he so β."""
    ci = ols_model.conf_int(alpha=0.05).iloc[1:]
    betas = ols_model.params[1:]

    ci_lower = ci.iloc[:, 0]
    ci_upper = ci.iloc[:, 1]
    errors_low = betas - ci_lower
    errors_up = ci_upper - betas

    # Xac dinh mau: CI chua 0 = khong y nghia (do), khong chua 0 = y nghia (xanh)
    colors = []
    for lo, up in zip(ci_lower, ci_upper):
        if lo <= 0 <= up:
            colors.append('tomato')
        else:
            colors.append('steelblue')

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.45)))
    y_pos = range(len(feature_names))

    # Ve tung error bar rieng de co mau rieng
    for i, (beta, lo, up, color, name) in enumerate(zip(betas, errors_low, errors_up, colors, feature_names)):
        ax.errorbar(beta, i, xerr=[[lo], [up]],
                    fmt='o', color=color, ecolor=color, capsize=5, capthick=1.5,
                    elinewidth=1.5, markersize=7)

    ax.axvline(0, color='red', lw=2, ls='--', label='β = 0 (Khong anh huong)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel('Gia tri he so β (95% CI)', fontsize=11)
    ax.set_title(
        "BIEU DO 11 — Khoang Tin Cay 95% cho He so Hoi quy β\n"
        "Neu thanh CI cat duong do (β=0) → Bien KHONG co y nghia thong ke\n"
        "Xanh = CI khong chua 0 (y nghia) | Do = CI chua 0 (khong y nghia)",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)

    # Ghi chu ly thuyet
    ax.text(0.02, 0.02,
            'Ly thuyet (Ch.8 MAS291):\n'
            'CI = β̂ ± t(α/2,n-k-1) × SE(β̂)\n'
            'CI khong chua 0 ↔ P-value < 0.05',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    return _save(fig, "11_coefficient_ci.png",
                 "95% CI cho he so Beta — Kiem tra y nghia truc quan")


def plot_prediction_interval(ols_model, X_train, X_test, y_test, feature_names):
    """Bieu do 12 — 95% Prediction Interval Band."""
    X_test_const = sm.add_constant(X_test)

    try:
        prstd, ci_lower, ci_upper = wls_prediction_std(ols_model, exog=X_test_const, alpha=0.05)
    except Exception:
        # Fallback: tu tinh prediction interval
        predictions = ols_model.predict(X_test_const)
        residuals = y_test.values - predictions.values if hasattr(predictions, 'values') else y_test - predictions
        se = np.std(residuals)
        n = len(X_train)
        k = X_train.shape[1]
        t_crit = stats.t.ppf(0.975, df=n - k - 1)
        ci_lower = predictions - t_crit * se
        ci_upper = predictions + t_crit * se
        prstd = se

    predictions = ols_model.predict(X_test_const)

    # Chon 1 bien de ve (bien co tuong quan cao nhat voi price)
    # Uu tien 'baths' hoac 'bedrooms' vi de hieu
    plot_var = None
    for candidate in ['baths', 'bedrooms', 'area']:
        if candidate in feature_names:
            plot_var = candidate
            break
    if plot_var is None:
        plot_var = feature_names[0]

    x_vals = X_test[plot_var].values if hasattr(X_test, '__getitem__') else X_test[:, feature_names.index(plot_var)]
    y_actual = y_test.values if hasattr(y_test, 'values') else y_test
    y_pred = predictions.values if hasattr(predictions, 'values') else predictions
    ci_lo = ci_lower.values if hasattr(ci_lower, 'values') else ci_lower
    ci_up = ci_upper.values if hasattr(ci_upper, 'values') else ci_upper

    # Sap xep theo bien X
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    pred_sorted = y_pred[sort_idx]
    ci_lo_sorted = ci_lo[sort_idx]
    ci_up_sorted = ci_up[sort_idx]
    actual_sorted = y_actual[sort_idx]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(x_sorted, actual_sorted, alpha=0.2, s=8, color='gray', label='Du lieu thuc te', zorder=2)
    ax.plot(x_sorted, pred_sorted, color='royalblue', lw=1.5, label='Du bao (ŷ)', zorder=3)
    ax.fill_between(x_sorted, ci_lo_sorted, ci_up_sorted,
                    alpha=0.2, color='royalblue', label='95% Prediction Interval', zorder=1)

    ax.set_xlabel(f'{plot_var}', fontsize=11)
    ax.set_ylabel('Log-Gia nha', fontsize=11)
    ax.set_title(
        f"BIEU DO 12 — Khoang Du Bao 95% (Prediction Interval) theo '{plot_var}'\n"
        "Vung to xanh = 95% kha nang gia nha THAT nam trong khoang nay\n"
        "Prediction Interval rong hon Confidence Interval vi them σ² cua quan sat moi",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Ghi chu ly thuyet
    ax.text(0.02, 0.95,
            'Ly thuyet (Ch.8 MAS291):\n'
            'PI = ŷ₀ ± t(α/2,n-k-1) × SE(ŷ₀)\n'
            'PI luon > CI vi them phuong sai\n'
            'cua quan sat moi (σ²)',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    return _save(fig, "12_prediction_interval.png",
                 f"95% Prediction Interval theo {plot_var}")


# ========================== BUOC 5 ==========================
def plot_summary_dashboard(ols_model, y_test, predictions, feature_names):
    """Bieu do 13 — Summary Dashboard tong hop."""
    from sklearn.metrics import mean_squared_error

    r2 = ols_model.rsquared
    adj_r2 = ols_model.rsquared_adj
    y_actual = y_test.values if hasattr(y_test, 'values') else y_test
    rmse = np.sqrt(mean_squared_error(y_actual, predictions))
    f_stat = ols_model.fvalue
    f_pval = ols_model.f_pvalue
    aic = ols_model.aic
    bic = ols_model.bic
    pvalues = ols_model.pvalues[1:]
    sig_count = sum(1 for p in pvalues if p < 0.05)
    total_vars = len(pvalues)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#fafafa')

    # Style chung
    metric_style = dict(fontsize=28, fontweight='bold', ha='center', va='center')
    label_style = dict(fontsize=11, ha='center', va='center', color='#555')

    # 1. R²
    ax = axes[0, 0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                 boxstyle="round,pad=0.05", facecolor='#e3f2fd', edgecolor='#1976d2', lw=2))
    ax.text(0.5, 0.65, f'{r2:.4f}', color='#1565c0', **metric_style)
    ax.text(0.5, 0.4, f'({r2*100:.1f}% bien thien)', fontsize=12, ha='center', color='#1976d2')
    ax.text(0.5, 0.22, 'R² (Goodness of Fit)', **label_style)

    # 2. Adj. R²
    ax = axes[0, 1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                 boxstyle="round,pad=0.05", facecolor='#e8f5e9', edgecolor='#388e3c', lw=2))
    ax.text(0.5, 0.6, f'{adj_r2:.4f}', color='#2e7d32', **metric_style)
    ax.text(0.5, 0.25, 'Adjusted R²', **label_style)

    # 3. RMSE
    ax = axes[0, 2]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                 boxstyle="round,pad=0.05", facecolor='#fff3e0', edgecolor='#f57c00', lw=2))
    ax.text(0.5, 0.6, f'{rmse:.4f}', color='#e65100', **metric_style)
    ax.text(0.5, 0.25, 'RMSE (Sai so trung binh)', **label_style)

    # 4. F-statistic
    ax = axes[1, 0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                 boxstyle="round,pad=0.05", facecolor='#fce4ec', edgecolor='#c62828', lw=2))
    ax.text(0.5, 0.65, f'{f_stat:.1f}', color='#b71c1c', **metric_style)
    ax.text(0.5, 0.4, f'P = {f_pval:.2e}', fontsize=12, ha='center', color='#c62828')
    ax.text(0.5, 0.22, 'F-statistic (Mo hinh)', **label_style)

    # 5. Bien y nghia
    ax = axes[1, 1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                 boxstyle="round,pad=0.05", facecolor='#f3e5f5', edgecolor='#7b1fa2', lw=2))
    ax.text(0.5, 0.6, f'{sig_count} / {total_vars}', color='#6a1b9a', **metric_style)
    ax.text(0.5, 0.25, 'Bien co Y nghia (P < 0.05)', **label_style)

    # 6. AIC / BIC
    ax = axes[1, 2]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                 boxstyle="round,pad=0.05", facecolor='#e0f7fa', edgecolor='#00838f', lw=2))
    ax.text(0.5, 0.65, f'{aic:.0f}', color='#006064', fontsize=22, fontweight='bold', ha='center')
    ax.text(0.5, 0.45, f'BIC = {bic:.0f}', fontsize=14, ha='center', color='#00838f')
    ax.text(0.5, 0.22, 'AIC / BIC', **label_style)

    fig.suptitle(
        "BIEU DO 13 — TONG HOP KET QUA DANH GIA MO HINH\n"
        "Tat ca chi so tu Confidence Intervals, Hypothesis Testing, Goodness of Fit\n"
        "(Chuong 8, 9, 10 MAS291: Xac suat Thong ke)",
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    return _save(fig, "13_summary_dashboard.png",
                 f"Tong hop — R2={r2:.4f}, F={f_stat:.1f}, {sig_count}/{total_vars} bien y nghia")


# ========================== MAIN RUNNER ==========================
def run_evaluation(df_final, feature_names, model, X_train, X_test, y_train, y_test, predictions):
    """
    Chay toan bo Phan 3: Danh gia & Output Analysis.
    Tham so:
        df_final      : DataFrame da xu ly
        feature_names : list ten dac trung
        model         : sklearn LinearRegression (da train)
        X_train/X_test: train/test features
        y_train/y_test: train/test target
        predictions   : gia tri du bao tren test set
    """
    print(f"\n{'='*60}")
    print(f"  PHAN 3: DANH GIA & OUTPUT ANALYSIS")
    print(f"  Luu anh vao: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Buoc 2: Q-Q Plot
    print("[Buoc 2] Q-Q Plot — Kiem tra phan du ~ N(0, σ²)")
    plot_qq(y_test, predictions)

    # Buoc 3: Hypothesis Testing — can train OLS model
    print("\n[Buoc 3] Hypothesis Testing — P-value & F-test")
    print("  Training OLS model (statsmodels) de lay P-value...")
    ols_model = fit_ols(X_train, y_train, feature_names)

    print(f"  R² = {ols_model.rsquared:.4f}")
    print(f"  Adj. R² = {ols_model.rsquared_adj:.4f}")
    print(f"  F-statistic = {ols_model.fvalue:.2f}, P(F) = {ols_model.f_pvalue:.2e}")

    plot_pvalue_bar(ols_model, feature_names)
    plot_ols_summary(ols_model, feature_names)

    # Buoc 4: Confidence Intervals
    print("\n[Buoc 4] Confidence Intervals — 95% CI")
    plot_coefficient_ci(ols_model, feature_names)
    plot_prediction_interval(ols_model, X_train, X_test, y_test, feature_names)

    # Buoc 5: Summary Dashboard
    print("\n[Buoc 5] Summary Dashboard")
    ols_predictions = ols_model.predict(sm.add_constant(X_test))
    plot_summary_dashboard(ols_model, y_test, ols_predictions, feature_names)

    print(f"\n{'='*60}")
    print(f"  HOAN THANH! {6} chart da luu vao: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    return ols_model
