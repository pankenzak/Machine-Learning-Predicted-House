import os
import argparse
from load_data import load_dataset
from preprocess import preprocess_data
from feature_selection import select_features
from train import train_model
from eda import run_eda

def main():
    parser = argparse.ArgumentParser(description="House Price Prediction")
    # Thay đổi đường dẫn mặc định nếu cần
    parser.add_argument("--data", type=str, default=r"C:\Users\nhatk\Desktop\dataset\House_Price_dataset.csv", help="Path to the dataset")
    parser.add_argument("--eda", action="store_true", help="Run Exploratory Data Analysis and show plots")
    args = parser.parse_args()

    data_path = args.data
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print("Loading data...")
    df = load_dataset(data_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.\n")
    
    if args.eda:
        print("--- Running EDA ---")
        run_eda(df)
        print("-------------------\n")
        
    print("Preprocessing data...")
    df_final = preprocess_data(df)
    print("Data preprocessed successfully.\n")
    
    print("Selecting features...")
    feature_names = select_features(df_final)
    print(f"Selected {len(feature_names)} features: {feature_names}\n")
    
    print("Training model...")
    model, predictions, y_test, rmse = train_model(df_final, feature_names)
    
    print("Train thành công!\n")
    print(f"Actual log-price of a house: {y_test.iloc[0]:.4f}")
    print(f"Model Predicted log-price Value: {predictions[0]:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")

if __name__ == "__main__":
    main()
