import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Thư mục lưu ảnh
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots_output")

def _save(fig, filename, title_explain):
    """Lưu figure ra file PNG và đóng."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] {filename}  — {title_explain}")
    return path


def plot_correlation_heatmap(df_final):
    """Biểu đồ 1 — Heatmap tương quan."""
    cor = df_final.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(13, 9))
    sns.heatmap(cor, annot=False, cmap='coolwarm', ax=ax,
                linewidths=0.3, vmin=-1, vmax=1, center=0)
    ax.set_title(
        "BIEU DO 1 — Ma tran Tuong quan (Correlation Matrix)\n"
        "Moi o the hien he so tuong quan Pearson r giua 2 bien.\n"
        "Mau do dam = tuong quan duong manh (r→1)  |  Mau xanh dam = tuong quan nghich (r→-1)",
        fontsize=12, fontweight='bold', pad=14
    )
    return _save(fig, "01_correlation_heatmap.png",
                 "Ma tran tuong quan — de chon bien dua vao |r|")


def plot_baseline_regression(df_final):
    """Biểu đồ 2 — Regression line baseline."""
    sample = df_final[['baths', 'price']].dropna().sample(n=400, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = sample['baths'].values
    y_vals = sample['price'].values
    b1 = np.cov(x_vals, y_vals)[0, 1] / np.var(x_vals)
    b0 = np.mean(y_vals) - b1 * np.mean(x_vals)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_line = b0 + b1 * x_line

    ax.scatter(x_vals, y_vals, color='tomato', alpha=0.5, s=25,
               label='Du lieu thuc te ($y_i$)', zorder=3)
    ax.plot(x_line, y_line, color='royalblue', linewidth=2.5,
            label=r'Duong hoi quy $\hat{y}=\hat{\beta}_0+\hat{\beta}_1 x$')

    # Vẽ phần dư mẫu
    for xi, yi in list(zip(x_vals[:20], y_vals[:20])):
        y_hat_i = b0 + b1 * xi
        ax.plot([xi, xi], [yi, y_hat_i], color='gray', lw=0.8, ls='--', alpha=0.6)

    ax.annotate(
        'Phan du $e_i = y_i - \\hat{y}_i$\n= khoang cach tu diem den duong thang',
        xy=(x_vals[5], y_vals[5]),
        xytext=(x_vals[5] + 0.6, y_vals[5] + 0.8),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, color='gray'
    )
    ax.set_xlabel('So phong tam (baths)', fontsize=11)
    ax.set_ylabel('Log-Gia nha', fontsize=11)
    ax.set_title(
        "BIEU DO 2 — Hoi quy Tuyen tinh Don (Simple Linear Regression)\n"
        r"$\hat{y}=\hat{\beta}_0+\hat{\beta}_1 x$ — Tim duong thang sao cho SSE = $\sum e_i^2$ la nho nhat (OLS)",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10)
    return _save(fig, "02_baseline_regression.png",
                 "Duong hoi quy + phan du — Phuong phap Binh phuong Toi thieu")


def plot_residuals(y_test, predictions):
    """Biểu đồ 3 — Phân phối phần dư."""
    residuals = y_test.values - predictions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    sns.histplot(residuals, kde=True, ax=axes[0], color='steelblue')
    axes[0].axvline(0, color='red', lw=1.5, ls='--', label='E(e) = 0')
    axes[0].set_title("Phan phoi Phan du (Residuals)\nKiem tra gia dinh: e ~ N(0, sigma^2)", fontweight='bold')
    axes[0].set_xlabel("Phan du $e_i = y_i - \\hat{y}_i$")
    axes[0].set_ylabel("Tan suat")
    axes[0].legend()

    # Q-Q-like scatter (sorted residuals vs index)
    axes[1].scatter(range(len(residuals[:200])), sorted(residuals)[:200],
                    color='steelblue', s=15, alpha=0.7)
    axes[1].axhline(0, color='red', lw=1.5, ls='--')
    axes[1].set_title("Phan du theo thu tu (Residual Plot)\nCac diem phan bo ngau nhien quanh 0 = mo hinh dat yeu cau", fontweight='bold')
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Phan du")

    fig.suptitle(
        "BIEU DO 3 — Phan tich Phan du\n"
        "Gia dinh cot loi: Sai so ngau nhien e phai tuan theo N(0, sigma^2) — Phan phoi chuan voi ky vong bang 0",
        fontsize=12, fontweight='bold', y=1.03
    )
    plt.tight_layout()
    return _save(fig, "03_residuals.png",
                 "Kiem tra phan du ~ N(0, sigma^2)")


def plot_predicted_vs_actual(y_test, predictions, r2):
    """Biểu đồ 4 — Predicted vs Actual."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_test, predictions, alpha=0.4, s=15, color='steelblue',
               label='Du bao vs. Thuc te')
    mn = min(y_test.min(), predictions.min())
    mx = max(y_test.max(), predictions.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Duong ly tuong (y=y_hat)')
    ax.set_xlabel('Log-Gia thuc te ($y_i$)', fontsize=11)
    ax.set_ylabel('Log-Gia du bao ($\\hat{y}_i$)', fontsize=11)
    ax.set_title(
        f"BIEU DO 4 — Predicted vs. Actual (Goodness of Fit)\n"
        f"He so xac dinh $R^2 = {r2:.4f}$ → Mo hinh giai thich {r2*100:.1f}% bien thien cua gia nha\n"
        "Cac diem cang sat duong cheo do 45 do thi R2 cang cao",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.text(0.05, 0.93, f'$R^2 = {r2:.4f}$', transform=ax.transAxes,
            fontsize=14, color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    return _save(fig, "04_predicted_vs_actual.png",
                 f"R2 = {r2:.4f} — Mo hinh giai thich {r2*100:.1f}% bien thien")


def plot_sse_decomposition(y_test, predictions):
    """Biểu đồ 5 — SST = SSR + SSE."""
    y_vals = y_test.values[:80]
    preds = predictions[:80]
    y_mean_val = y_test.mean()
    r2 = 1 - np.sum((y_test.values - predictions)**2) / np.sum((y_test.values - y_mean_val)**2)
    idx = range(len(y_vals))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # SST
    axes[0].scatter(idx, y_vals, color='tomato', s=18, zorder=3)
    axes[0].axhline(y_mean_val, color='gray', lw=2, ls='--')
    for i, yi in enumerate(y_vals):
        axes[0].plot([i, i], [yi, y_mean_val], color='tomato', lw=0.8, alpha=0.5)
    axes[0].set_title(f"SST — Tong bien thien\n$SST=\\sum(y_i-\\bar{{y}})^2$", fontweight='bold')
    axes[0].set_ylabel("Log-Gia nha")

    # SSR
    axes[1].scatter(idx, y_vals, color='tomato', s=18, zorder=3)
    axes[1].scatter(idx, preds, color='royalblue', s=18, marker='D', zorder=3)
    axes[1].axhline(y_mean_val, color='gray', lw=1.5, ls='--')
    for i in range(len(preds)):
        axes[1].plot([i, i], [preds[i], y_mean_val], color='royalblue', lw=0.8, alpha=0.6)
    axes[1].set_title(f"SSR — Bien thien duoc giai thich\n$SSR=\\sum(\\hat{{y}}_i-\\bar{{y}})^2$", fontweight='bold')

    # SSE
    axes[2].scatter(idx, y_vals, color='tomato', s=18, zorder=3)
    axes[2].scatter(idx, preds, color='royalblue', s=18, marker='D', zorder=3)
    for i in range(len(preds)):
        axes[2].plot([i, i], [y_vals[i], preds[i]], color='green', lw=0.8, alpha=0.7)
    axes[2].set_title(f"SSE — Phan du chua giai thich\n$SSE=\\sum(y_i-\\hat{{y}}_i)^2$", fontweight='bold')

    fig.suptitle(
        f"BIEU DO 5 — Phan tich Phuong sai: SST = SSR + SSE\n"
        f"$R^2 = SSR/SST = {r2:.4f}$ — {r2*100:.1f}% bien thien duoc giai thich boi mo hinh",
        fontsize=13, fontweight='bold', y=1.03
    )
    plt.tight_layout()
    return _save(fig, "05_sse_decomposition.png",
                 "SST = SSR + SSE — Giai thich R2 truc quan")


def plot_pipeline(ax_ext=None):
    """Biểu đồ 6 — ML Pipeline."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.2); ax.axis('off')
    fig.patch.set_facecolor('#f5f7fa')

    steps = [
        (0.7,  '1. Du lieu Tho\n(Load Data)',         '#AED6F1'),
        (2.4,  '2. Tien xu ly\n(Preprocess)',          '#A9DFBF'),
        (4.1,  '3. Chon dac trung\n(Feature Select)',  '#F9E79F'),
        (5.8,  '4. Huan luyen\n(model.fit)',           '#F1948A'),
        (7.5,  '5. Danh gia\n(R2, RMSE)',              '#D7BDE2'),
        (9.2,  '6. Du bao\n(predict)',                 '#FAD7A0'),
    ]
    for x, label, color in steps:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.62, 0.6), 1.24, 0.95,
            boxstyle="round,pad=0.07",
            facecolor=color, edgecolor='#555', linewidth=1.4
        ))
        ax.text(x, 1.07, label, ha='center', va='center',
                fontsize=9, fontweight='bold')

    # Mũi tên
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + 0.62
        x2 = steps[i+1][0] - 0.62
        ax.annotate('', xy=(x2, 1.07), xytext=(x1, 1.07),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    ax.text(4.95, 0.28,
            'Quy trinh Machine Learning — Ho quy Tuyen tinh Da bien (Multiple Linear Regression)',
            ha='center', fontsize=10, color='#333', style='italic')
    ax.set_title(
        "BIEU DO 6 — Quy trinh ML (Machine Learning Pipeline)\n"
        "Toan bo du an tuan thu 6 buoc: tu du lieu tho den du bao cuoi cung",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    return _save(fig, "06_ml_pipeline.png", "Quy trinh ML — 6 buoc tong the")


def plot_beta_coefficients(model, feature_names):
    """Biểu đồ 7 — Hệ số hồi quy β."""
    if not hasattr(model, 'coef_'):
        return None
    coefs = model.coef_
    colors = ['tomato' if c < 0 else 'steelblue' for c in coefs]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feature_names, coefs, color=colors, edgecolor='white', height=0.6)
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('Gia tri he so $\\hat{\\beta}$ (tac dong den Log-Gia nha)', fontsize=11)
    ax.set_title(
        "BIEU DO 7 — He so Hoi quy $\\hat{\\beta}$ (Feature Importance)\n"
        "Moi he so cho biet: neu bien do tang 1 don vi thi Log-Gia nha thay doi bao nhieu\n"
        "Xanh = tang gia | Do = giam gia",
        fontsize=12, fontweight='bold'
    )
    for bar, coef in zip(bars, coefs):
        offset = 0.003 if coef >= 0 else -0.003
        ax.text(coef + offset, bar.get_y() + bar.get_height() / 2,
                f'{coef:.3f}', va='center',
                ha='left' if coef >= 0 else 'right', fontsize=9)
    ax.legend(handles=[
        mpatches.Patch(color='steelblue', label='Tac dong duong (tang gia)'),
        mpatches.Patch(color='tomato',    label='Tac dong am (giam gia)'),
    ], fontsize=10)
    plt.tight_layout()
    return _save(fig, "07_beta_coefficients.png",
                 "He so Beta — Bien nao anh huong nhieu nhat den gia")


def plot_ml_explanation(df_final, feature_names, model, X_test, y_test, predictions):
    """Vẽ và lưu toàn bộ biểu đồ giải thích ML ra thư mục plots_output/."""
    print(f"\nLuu anh vao: {OUTPUT_DIR}")
    plot_baseline_regression(df_final)
    plot_residuals(y_test, predictions)
    plot_predicted_vs_actual(y_test, predictions, r2_val(y_test, predictions))
    plot_sse_decomposition(y_test, predictions)
    plot_pipeline()
    plot_beta_coefficients(model, feature_names)
    print(f"\nXong! Tat ca {6} anh da duoc luu vao thu muc: {OUTPUT_DIR}")


def r2_val(y_test, predictions):
    return 1 - np.sum((y_test.values - predictions)**2) / np.sum((y_test.values - y_test.mean())**2)
