import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

def run_eda(df):
    """Run Exploratory Data Analysis and show plots."""
    print("--- Data Info ---")
    print(df.info())
    print("\nShape:", df.shape)
    print("\nDescribe:\n", df.describe())
    print("\nNull Values:\n", df.isnull().sum())
    
    numeric_df = df.select_dtypes(include=[np.number])
    cols_to_keep = [col for col in numeric_df.columns if numeric_df[col].nunique() > 1]
    df_filtered_numeric = numeric_df[cols_to_keep]
    
    sns.pairplot(df_filtered_numeric, height=2.5, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20})
    plt.show()
    
    # Price distribution
    sns.histplot(df['price'], kde=True)
    plt.title('Price Distribution')
    plt.show()
    
    print("Skewness: %f" % df['price'].skew())
    print("Kurtosis: %f" % df['price'].kurt())
    
    # Price vs City
    fig, ax = plt.subplots()
    ax.scatter(x = df['city'], y = df['price'])
    plt.ylabel('price', fontsize=13)
    plt.xlabel('city', fontsize=13)
    plt.show()
    
    # Top 20 locations
    top_locations = df['location'].value_counts().nlargest(20).index
    df_filtered = df[df['location'].isin(top_locations)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(x = df_filtered['location'], y = df_filtered['price'])
    plt.xticks(rotation=45)
    plt.show()
    
    # Baths vs Price
    fig, ax = plt.subplots()
    ax.scatter(x = df['baths'], y = df['price'])
    plt.ylabel('price', fontsize=13)
    plt.xlabel('baths', fontsize=13)
    plt.show()
    
    # Bedrooms vs Price
    fig, ax = plt.subplots()
    ax.scatter(x = df['bedrooms'], y = df['price'])
    plt.ylabel('price', fontsize=13)
    plt.xlabel('bedrooms', fontsize=13)
    plt.show()
    
    # Normal distribution fit
    sns.histplot(df['price'], kde=True, stat="density", linewidth=0)
    (mu, sigma) = norm.fit(df['price'].dropna())
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    plt.show()
    
    fig = plt.figure()
    res = stats.probplot(df['price'].dropna(), plot=plt)
    plt.show()
