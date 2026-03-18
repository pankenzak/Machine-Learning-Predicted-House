import os
import argparse
from load_data import load_dataset
from preprocess import preprocess_data
from feature_selection import select_features
from train import train_model, train_baseline_model
from eda import run_eda
from visualize import plot_ml_explanation
from evaluate import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="House Price Prediction")
    # Thay đổi đường dẫn mặc định nếu cần
    parser.add_argument("--data", type=str, default=r"C:\Users\nhatk\Desktop\ML Predicted House\House_Price_dataset.csv", help="Path to the dataset")
    parser.add_argument("--eda", action="store_true", help="Run Exploratory Data Analysis and show plots")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Show/hide presentation plots")
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
    feature_names = select_features(df_final, plot=args.plots)
    print(f"Selected {len(feature_names)} features: {feature_names}\n")
    
    print("--- Training Baseline Model (1 Variable: area) ---")
    base_model, base_preds, base_y_test, base_rmse, base_r2 = train_baseline_model(df_final, plot=args.plots)
    print(f"Baseline - RMSE: {base_rmse:.4f}, R-squared: {base_r2:.4f}\n")
    
    print("--- Training Multi-variable Model ---")
    model, predictions, y_test, rmse, r2 = train_model(df_final, feature_names, plot=args.plots)
    
    print("Train thanh cong!\n")
    print(f"Actual log-price of a house: {y_test.iloc[0]:.4f}")
    print(f"Model Predicted log-price Value: {predictions[0]:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"He so R-squared: {r2:.4f}")
    
    if args.plots:
        print("\n--- Hien thi bieu do giai thich ML (educational visualizations) ---")
        X_test_df = df_final[feature_names].iloc[y_test.index]
        plot_ml_explanation(df_final, feature_names, model, X_test_df, y_test, predictions)

        # === PHAN 3: Danh gia & Output Analysis ===
        from sklearn.model_selection import train_test_split
        X_all = df_final[feature_names]
        y_all = df_final['price']
        X_train, X_test_eval, y_train, y_test_eval = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42
        )
        preds_eval = model.predict(X_test_eval)
        run_evaluation(
            df_final, feature_names, model,
            X_train, X_test_eval, y_train, y_test_eval, preds_eval
        )

if __name__ == "__main__":
    main()
