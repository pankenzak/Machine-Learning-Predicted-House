import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def clean_area(x):
    """Convert Area to standard units (Marla/Kanal)."""
    if pd.isna(x): return 0.0
    try:
        parts = str(x).split()
        val = float(parts[0])
        unit = parts[1]
        if unit == 'Marla':
            return val * 252929.0
        elif unit == 'Kanal':
            return val * 505857.0
    except:
        pass
    return 0.0

def preprocess_data(df):
    """Clean data, log transform price, drop unnecessary columns, and one-hot encode."""
    df = df.copy()
    
    # 1. Clean Area
    df['area'] = df['area'].apply(clean_area)
    
    # 2. Drop unused columns
    cols_to_drop = ['property_id', 'location_id', 'page_url', 'province_name', 
                    'Area Type', 'Area Size', 'date_added', 'Area Category', 
                    'agency', 'agent', 'latitude', 'longitude']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 3. Log-transform price
    df["price"] = np.log1p(df["price"])
    
    # 4. One-hot Encoding for categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[cat_cols])
    
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols))
    
    # 5. Combine numeric and encoded categorical data
    df_numeric = df.select_dtypes(include=[np.number]).reset_index(drop=True)
    df_final = pd.concat([df_numeric, encoded_df], axis=1)
    
    return df_final
