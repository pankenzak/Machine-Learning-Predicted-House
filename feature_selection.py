import numpy as np
import pandas as pd

def select_features(df_final, threshold=0.2):
    """Calculate correlation and select features above threshold."""
    # Ensure numeric correlation only
    cor = df_final.corr(numeric_only=True)
    
    cor_target = abs(cor["price"])
    relevant_features = cor_target[cor_target > threshold]
    
    names = [index for index, value in relevant_features.items()]
    
    if 'price' in names:
        names.remove('price')
        
    return names
