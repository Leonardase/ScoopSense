import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def predict():
    """
    Uses historical data to forecast volume of sales. Uses features to keep track of other variables to emphasize
    the importance of days and recent data. Creates predictions.csv which allows us to calculate run out dates for
    each flavor. Added progress bar to allow the user to see that the program is working.
    """
    # Load data
    print("[5/8] Loading data...")
    train = pd.read_csv('data/temp/trainNeg.csv', parse_dates=['Date'])
    test = pd.read_csv('data/temp/output.csv', parse_dates=['Date'])

    # Get flavor statistics
    print("[6/8] Analyzing flavors...")
    flavor_summary = summarize_flavor_columns(train)
    flavor_means = dict(zip(flavor_summary['Flavor'], flavor_summary['Mean (ignoring -1)']))

    # Add features
    print("[7/8] Adding features...")
    train = add_features(train)
    test = add_features(test)

    # Features to use
    base_features = [
        'Day_of_Year', 'Is_Weekend',
        'Day_of_Week_Sin', 'Day_of_Week_Cos',
        'Day_of_Year_Sin', 'Day_of_Year_Cos',
        'Total_Sales_Lag_1', 'Total_Sales_Lag_7', 'Total_Sales_Lag_14',
        'Total_Sales_Rolling_Mean_7', 'Total_Sales_Rolling_Std_7',
        'Total_Sales_Rolling_Mean_14', 'Total_Sales_Rolling_Std_14'
    ]

    non_flavor_cols = ['Date', 'Total_Sales']

    # Identify flavors
    flavors = [col for col in train.select_dtypes(include=[np.number]).columns
               if col not in base_features + non_flavor_cols]

    # Prepare outputs
    predictions = test[['Date']].copy()
    evaluation_results = []

    print(f"[8/8] Generating predictions...")
    print("-" * 60)
    bar_length = 40
    # Process each flavor
    for i, flavor in enumerate(flavors):
        # Simple ASCII progress bar
        progress = (i + 1) / len(flavors) * 100
        bar_length = 40
        filled = int(bar_length * progress / 100)

        bar = '#' * filled + '-' * (bar_length - filled)

        # Show flavor name while processing
        status_text = flavor[:20] if i < len(flavors) - 1 else "Complete!"

        print(f'\r[{bar}] {progress:.1f}% - {status_text:20s}', end='', flush=True)

        try:
            # Get the training data with -1 flags
            y_train = train[flavor].copy()

            zero_ratio = (y_train == -1).mean()
            # Creates a boolean mask for days without the -1 flag
            valid_mask = y_train != -1

            # Check if we have enough valid data
            valid_count = valid_mask.sum()
            if valid_count < 20:
                # Not enough data, use the mean as the prediction
                predictions[flavor] = flavor_means.get(flavor, 0)
                continue

            # Filter training data to only valid (non -1) days
            y_train_clean = y_train[valid_mask].copy()

            # Add sparse-demand features
            train_temp = add_sparse_features(train.copy(), flavor)
            test_temp = add_sparse_features(test.copy(), flavor)

            flavor_features = base_features + [
                f'{flavor}_days_since_sale',
                f'{flavor}_rolling_zeros'
            ]

            # Gets all training features
            X_train_full = train_temp[flavor_features].copy()
            X_test = test_temp[flavor_features].copy()

            # Filters to only the valid days to avoid training on the -1 values
            X_train_clean = X_train_full[valid_mask].copy()

            # Fill any remaining NaN in features
            X_train_clean = X_train_clean.fillna(0)
            X_test = X_test.fillna(0)

            # Handle NaN/inf in y_train
            if y_train_clean.isna().any():
                y_train_clean = y_train_clean.fillna(flavor_means.get(flavor, 0))
            y_train_clean = y_train_clean.replace([np.inf, -np.inf], flavor_means.get(flavor, 0))


            # Standard XGBoost for non-intermittent flavors
            try:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    min_child_weight=3,
                ).fit(X_train_clean, y_train_clean)

                test_pred = model.predict(X_test)

            except Exception as xgb_error:
                print(xgb_error)
                # Fallback to simple mean prediction
                test_pred = np.full(len(test), flavor_means.get(flavor, 0))

            # Apply mean floor and ensure non-negative
            test_pred = np.maximum(test_pred, flavor_means.get(flavor, 0))
            test_pred = np.round(test_pred.astype(float), 2)

            predictions[flavor] = test_pred

            # Evaluate
            y_test = test[flavor].fillna(0)
            metrics = {
                'Flavor': flavor,
                'RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
                'R2': r2_score(y_test, test_pred),
                'Zero_Ratio': zero_ratio,
                'Valid_Days_Count': valid_count,
                'Total_Days_Count': len(y_train),
                'MAE': mean_absolute_error(y_test, test_pred)
            }
            evaluation_results.append(metrics)

        except Exception as e:
            # Use the mean as a fallback
            print(e)
            predictions[flavor] = flavor_means.get(flavor, 0)

    # Final complete message
    print(f'\r[{"#" * bar_length}] 100.0% - Complete!', flush=True)
    print("" + "-" * 60)

    # Save results
    evaluation = pd.DataFrame(evaluation_results)
    predictions.to_csv('data/temp/predictions.csv', index=False)
    evaluation.to_csv('data/temp/model_evaluation.csv', index=False)

    return predictions


def add_sparse_features(df, flavor):
    """
    Adds features that are helpful for sparse data
    """
    # Create all new columns at once in a temporary DataFrame
    new_cols = pd.DataFrame({
        f'{flavor}_days_since_sale': df.groupby((df[flavor] > 0).cumsum()).cumcount(),
        f'{flavor}_rolling_zeros': df[flavor].rolling(7, min_periods=1).apply(lambda x: (x == 0).sum()),
        f'{flavor}_occurred': (df[flavor] > 0).astype(int)
    })

    # Concatenate all new columns at once
    return pd.concat([df, new_cols], axis=1)


def add_features(df):
    """
    Creates features that are used to train our model
    """
    # Ensure Total_Sales exists and is numeric
    if 'Total_Sales' not in df.columns:
        df['Total_Sales'] = 0

    # Convert to numeric, handling errors
    df['Total_Sales'] = pd.to_numeric(df['Total_Sales'], errors='coerce').fillna(0)

    # Cyclical encoding
    df['Day_of_Week_Sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 7)
    df['Day_of_Week_Cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 7)

    # Ensure Day_of_Year exists
    if 'Day_of_Year' not in df.columns:
        df['Day_of_Year'] = df['Date'].dt.dayofyear

    df['Day_of_Year_Sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365)
    df['Day_of_Year_Cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365)

    # Lag/rolling features with proper NaN handling
    for lag in [1, 7, 14]:
        df[f'Total_Sales_Lag_{lag}'] = df['Total_Sales'].shift(lag).fillna(0)

    for window in [7, 14]:
        df[f'Total_Sales_Rolling_Mean_{window}'] = df['Total_Sales'].rolling(window, min_periods=1).mean().fillna(0)
        df[f'Total_Sales_Rolling_Std_{window}'] = df['Total_Sales'].rolling(window, min_periods=1).std().fillna(0)

    return df


def summarize_flavor_columns(df):
    """
    Calculates the mean of each flavor's sales while ignoring the -1s
    """
    summary = []
    non_flavor_cols = ['Date', 'Day_of_Week', 'Total_Sales', 'Day_of_Year', 'Is_Weekend', 'Num_Available_Flavors']
    flavor_cols = [col for col in df.columns if col not in non_flavor_cols]

    for col in flavor_cols:
        # Mask -1 values
        valid_values = df[col][df[col] != -1]

        # Handle case where all values are -1 (mean would be NaN)
        if len(valid_values) == 0:
            mean = 0.0
        else:
            mean = valid_values.mean()

        num_neg_ones = (df[col] == -1).sum()

        summary.append({
            'Flavor': col,
            'Mean (ignoring -1)': round(mean, 2),
            '# of -1s': num_neg_ones
        })

    return pd.DataFrame(summary).sort_values('Mean (ignoring -1)', ascending=False)