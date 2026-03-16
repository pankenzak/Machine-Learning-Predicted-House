import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots_output")

def train_baseline_model(df_final, plot=False):
    """Train a simple Linear Regression model on 1 variable (area)."""
    X = df_final[['area']]
    y = df_final['price']
    
    # Optional plot for the presentation (Baseline)
    if plot:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sample = df_final[['area', 'price']].dropna().sample(n=800, random_state=42)
        ax.scatter(sample['area'], sample['price'], alpha=0.35, s=12, color='tomato', label='Du lieu thuc te')
        # draw regression line
        b1 = sample['area'].cov(sample['price']) / sample['area'].var()
        b0 = sample['price'].mean() - b1 * sample['area'].mean()
        xl = [sample['area'].min(), sample['area'].max()]
        ax.plot(xl, [b0 + b1*x for x in xl], color='royalblue', lw=2.5,
                label=r'Duong hoi quy $\hat{y}=\hat{\beta}_0+\hat{\beta}_1 x$')
        ax.set_xlabel('Dien tich (area - da xu ly)', fontsize=11)
        ax.set_ylabel('Log-Gia nha', fontsize=11)
        ax.set_title(
            'BIEU DO 2 — Hoi quy Tuyen tinh Don: Log-Gia vs. Dien tich\n'
            r'Tim duong thang bang OLS sao cho SSE = $\sum e_i^2$ la nho nhat',
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=10)
        path = os.path.join(OUTPUT_DIR, '02_baseline_regression.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('  [saved] 02_baseline_regression.png — Duong hoi quy baseline')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    predictions = lr.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return lr, predictions, y_test, rmse, r2

def train_model(df_final, feature_names, plot=False):
    """Train a Linear Regression model and evaluate it."""
    X = df_final[feature_names] 
    y = df_final['price'] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    predictions = lr.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    if plot:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        residuals = y_test - predictions
        r2 = r2_score(y_test, predictions)

        # 3. Residuals distribution
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.histplot(residuals, kde=True, ax=axes[0], color='steelblue')
        axes[0].axvline(0, color='red', lw=1.5, ls='--', label='E(e)=0')
        axes[0].set_title('Phan phoi Phan du\nKiem tra: e ~ N(0, sigma^2)', fontweight='bold')
        axes[0].set_xlabel('Phan du $e_i = y_i - \\hat{y}_i$')
        axes[0].legend()
        axes[1].scatter(range(len(sorted(residuals)[:200])), sorted(residuals)[:200],
                        color='steelblue', s=12, alpha=0.7)
        axes[1].axhline(0, color='red', lw=1.5, ls='--')
        axes[1].set_title('Phan du theo thu tu\nPhan bo ngau nhien quanh 0 = dat yeu cau', fontweight='bold')
        axes[1].set_xlabel('Index')
        axes[1].set_ylabel('Phan du')
        fig.suptitle(
            'BIEU DO 3 — Phan tich Phan du (Residual Analysis)\n'
            'Gia dinh: Sai so ngau nhien e phai tuan theo N(0, sigma^2)',
            fontsize=12, fontweight='bold', y=1.03
        )
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, '03_residuals.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('  [saved] 03_residuals.png — Kiem tra phan du')

        # 4. Predicted vs Actual
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(y_test, predictions, alpha=0.4, s=15, color='steelblue')
        mn = min(y_test.min(), predictions.min())
        mx = max(y_test.max(), predictions.max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Duong ly tuong (y=y_hat)')
        ax.set_xlabel('Log-Gia thuc te', fontsize=11)
        ax.set_ylabel('Log-Gia du bao', fontsize=11)
        ax.set_title(
            f'BIEU DO 4 — Predicted vs. Actual\n'
            f'R^2 = {r2:.4f} → Mo hinh giai thich {r2*100:.1f}% bien thien cua gia nha\n'
            'Cac diem sat duong cheo do 45 do = mo hinh chinh xac',
            fontsize=12, fontweight='bold'
        )
        ax.text(0.05, 0.92, f'R2 = {r2:.4f}', transform=ax.transAxes,
                fontsize=13, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.legend(fontsize=10)
        path = os.path.join(OUTPUT_DIR, '04_predicted_vs_actual.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [saved] 04_predicted_vs_actual.png — Predicted vs Actual (R2={r2:.4f})')
    
    return lr, predictions, y_test, rmse, r2
