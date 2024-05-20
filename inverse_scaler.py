import joblib
import os

import pandas as pd

def main():
    # Paths
    scaler_path = 'feature_scaler_all-en.pkl'
    predictions_base_path = "results/gaze/all-en/"
    predictions_name = "preds-12-bert-base-cased-True-False-False.csv"
    predictions_path = predictions_base_path + predictions_name
    scaled_test_path = "scaled-test-all-en.csv"
    
    output_dir = "unscaled"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define Features
    features = ["n_fix", "first_fix_dur", "first_pass_dur",
                "total_fix_dur", "mean_fix_dur", "fix_prob",
                "n_refix", "reread_prob"
    ]
    
    # Load predictions
    predictions = pd.read_csv(predictions_path)
    
    # Load scaled test data
    scaled_test = pd.read_csv(scaled_test_path)
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Unscale predictions
    unscaled_predictions = list(scaler.inverse_transform(predictions[features]))
    df = pd.DataFrame(unscaled_predictions, columns=["n_fix", "first_fix_dur", "first_pass_dur",
                                                     "total_fix_dur", "mean_fix_dur", "fix_prob",
                                                     "n_refix", "reread_prob"])
    
    df.to_csv(f"{output_dir}/unscaled_{predictions_name}.csv")
    
    unscaled_test = list(scaler.inverse_transform(scaled_test[features]))
    df = pd.DataFrame(unscaled_test, columns=["n_fix", "first_fix_dur", "first_pass_dur",
                                              "total_fix_dur", "mean_fix_dur", "fix_prob",
                                              "n_refix", "reread_prob"])
    df.to_csv(f"{output_dir}/unscaled_{scaled_test_path}")
    
if __name__ == "__main__":
    main()