import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

def run_eda(df):
    """Run Exploratory Data Analysis and save plots."""
    from visualize import _save
    print("--- Data Info ---")
    print(df.info())
    print("\nShape:", df.shape)
    print("\nDescribe:\n", df.describe())
    print("\nNull Values:\n", df.isnull().sum())
    
    numeric_df = df.select_dtypes(include=[np.number])
    cols_to_keep = [col for col in numeric_df.columns if numeric_df[col].nunique() > 1]
    df_filtered_numeric = numeric_df[cols_to_keep]
    
    print("\n--- Saving EDA Plots to output folder ---")
    
    g = sns.pairplot(df_filtered_numeric, height=2.5, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20})
    _save(g.figure, "eda_01_pairplot.png", "BIEU DO EDA 1: Phan phoi tuong quan cap bien (Pairplot)")
    
    # Price distribution
    fig1 = plt.figure()
    sns.histplot(df['price'], kde=True)
    plt.title('Price Distribution')
    _save(fig1, "eda_02_price_dist.png", "BIEU DO EDA 2: Phan phoi tan suat gia nha")
    
    print("Skewness: %f" % df['price'].skew())
    print("Kurtosis: %f" % df['price'].kurt())
    
    # Price vs City
    fig3, ax = plt.subplots()
    ax.scatter(x = df['city'], y = df['price'])
    plt.ylabel('price', fontsize=13)
    plt.xlabel('city', fontsize=13)
    _save(fig3, "eda_03_price_city.png", "BIEU DO EDA 3: Price vs City")
    
    # Top 20 locations
    top_locations = df['location'].value_counts().nlargest(20).index
    df_filtered = df[df['location'].isin(top_locations)]
    fig4, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(x = df_filtered['location'], y = df_filtered['price'])
    plt.xticks(rotation=45)
    _save(fig4, "eda_04_price_location.png", "BIEU DO EDA 4: Price vs Top 20 Locations")
    
    # Baths vs Price
    fig5, ax = plt.subplots()
    ax.scatter(x = df['baths'], y = df['price'])
    plt.ylabel('price', fontsize=13)
    plt.xlabel('baths', fontsize=13)
    _save(fig5, "eda_05_price_baths.png", "BIEU DO EDA 5: Baths vs Price")
    
    # Bedrooms vs Price
    fig6, ax = plt.subplots()
    ax.scatter(x = df['bedrooms'], y = df['price'])
    plt.ylabel('price', fontsize=13)
    plt.xlabel('bedrooms', fontsize=13)
    _save(fig6, "eda_06_price_bedrooms.png", "BIEU DO EDA 6: Bedrooms vs Price")
    
    # Normal distribution fit
    fig7 = plt.figure()
    sns.histplot(df['price'], kde=True, stat="density", linewidth=0)
    (mu, sigma) = norm.fit(df['price'].dropna())
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend([r'Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    _save(fig7, "eda_07_price_normal_fit.png", "BIEU DO EDA 7: Phan phoi chua xu ly vs Bieu do hinh chuong")
    
    fig8 = plt.figure()
    res = stats.probplot(df['price'].dropna(), plot=plt)
    _save(fig8, "eda_08_qq_plot.png", "BIEU DO EDA 8: Q-Q Plot kiem tra tinh trang lech")

    # OUTLIER DETECTION USING IQR RULE (Boxplot)
    fig9, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Boxplot for Price
    sns.boxplot(y=df['price'], ax=axes[0], color='lightblue')
    axes[0].set_title('Boxplot of Price\n(Detecting extreme large outliers)')
    
    # Calculate Q1, Q3 and Annotate the Price plot
    Q1_p = df['price'].quantile(0.25)
    Q3_p = df['price'].quantile(0.75)
    IQR_p = Q3_p - Q1_p
    upper_bound_p = Q3_p + 1.5 * IQR_p
    axes[0].axhline(upper_bound_p, color='red', linestyle='--', label=f'Upper = Q3 + 1.5*IQR')
    axes[0].legend()
    # 2. Boxplot for Area (if the 'area' column has been converted to numeric)
    if 'area' in df.columns and pd.api.types.is_numeric_dtype(df['area']):
        sns.boxplot(y=df['area'], ax=axes[1], color='lightgreen')
        axes[1].set_title('Boxplot of Area\n(Detecting Area outliers)')
        
        Q1_a = df['area'].quantile(0.25)
        Q3_a = df['area'].quantile(0.75)
        IQR_a = Q3_a - Q1_a
        upper_bound_a = Q3_a + 1.5 * IQR_a
        axes[1].axhline(upper_bound_a, color='red', linestyle='--', label=f'Upper Bound')
        axes[1].legend()
    else:
        axes[1].set_title('Column "area" is not numeric yet\n(Run through preprocess.py first)')
        axes[1].axis('off')
    plt.suptitle("OUTLIER DETECTION USING 1.5 x IQR RULE (CHAPTER 6)", fontweight='bold', fontsize=14)
    plt.tight_layout()
    _save(fig9, "eda_09_boxplot_outliers.png", "BIEU DO EDA 9: Phan loai Outliers (Quy tac 1.5x IQR)")

    # STANDARDIZATION USING Z-SCORE
    from sklearn.preprocessing import StandardScaler
    
    # Take a sample focusing on 2 variables: 'baths' (small unit: 1-10) and 'price' (extremely large)
    df_sample = df[['baths', 'price']].dropna().copy()
    
    # Initialize Scaler and transform data to Z = (X - mean) / std
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_sample)
    df_scaled = pd.DataFrame(scaled_data, columns=['baths_scaled', 'price_scaled'])
    
    # Start drawing BEFORE and AFTER comparison
    fig10, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Figure 1: WITHOUT SCALING
    sns.kdeplot(df_sample['baths'], ax=axes[0], label='Baths (Unit 1-10)', fill=True)
    sns.kdeplot(df_sample['price'], ax=axes[0], label='Price (Millions)', fill=True) 
    axes[0].set_title('BEFORE STANDARDIZATION\nVariables have completely different scales, harder for the model to learn')
    axes[0].legend()
    
    # Figure 2: SCALED WITH Z-SCORE
    sns.kdeplot(df_scaled['baths_scaled'], ax=axes[1], label='Baths (Z-score)', fill=True)
    sns.kdeplot(df_scaled['price_scaled'], ax=axes[1], label='Price (Z-score)', fill=True)
    axes[1].set_title('AFTER Z-SCORE STANDARDIZATION\nAll transformed to N(0, 1): Scales overlap, ML model converges faster!')
    axes[1].legend()
    plt.suptitle(r"STANDARDIZATION USING Z-SCORE THEOREM: $Z = \frac{X - \mu}{\sigma}$ (CHAPTER 4)", fontweight='bold', fontsize=14)
    plt.tight_layout()
    _save(fig10, "eda_10_standardization.png", "BIEU DO EDA 10: Tieu chuan hoa Z-Score")