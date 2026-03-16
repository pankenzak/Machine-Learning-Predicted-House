import pandas as pd

def load_dataset(data_path):
    """Load dataset from CSV."""
    df = pd.read_csv(data_path)
    return df
