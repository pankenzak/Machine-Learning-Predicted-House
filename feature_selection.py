import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots_output")

def select_features(df_final, threshold=0.2, plot=False):
    """Calculate correlation and select features above threshold."""
    # Ensure numeric correlation only
    cor = df_final.corr(numeric_only=True)
    
    if plot:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Chỉ lấy các cột quan trọng — loại bỏ các cột one-hot location_ cho heatmap sạch đẹp
        key_cols = [c for c in cor.columns 
                    if not c.startswith('location_') and not c.startswith('Area')]
        cor_small = cor.loc[key_cols, key_cols]
        
        fig, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(
            cor_small,
            annot=True,           # Hiện số r trong từng ô (giống hình mẫu)
            fmt='.2f',
            cmap='coolwarm',
            ax=ax,
            linewidths=0.5,
            vmin=-1, vmax=1, center=0,
            annot_kws={'size': 9}
        )
        ax.set_title(
            "BIEU DO 1 — Ma tran Tuong quan (Correlation Matrix)\n"
            "Moi o the hien he so r. Do dam = tuong quan manh. "
            "Chon bien co |r| > 0.2 voi 'price'.",
            fontsize=12, fontweight='bold', pad=14
        )
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        path = os.path.join(OUTPUT_DIR, "01_correlation_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [saved] 01_correlation_heatmap.png — Ma tran tuong quan (co so r)")
        

    cor_target = abs(cor["price"])
    relevant_features = cor_target[cor_target > threshold]
    
    names = [index for index, value in relevant_features.items()]
    
    if 'price' in names:
        names.remove('price')
        
    return names
