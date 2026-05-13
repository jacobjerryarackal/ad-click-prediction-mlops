import pandas as pd

def run_eda(data_path: str = "data/dataset.csv", output_file: str = "eda_report.md") -> None:
    """
    Performs Exploratory Data Analysis (EDA) on the dataset.
    Generates a markdown report documenting target distribution and cardinality.
    
    Concepts Documented:
    - Target Distribution: Understanding class imbalance (how many clicks vs non-clicks).
      High imbalance requires metrics like Precision/Recall/LogLoss instead of Accuracy.
    - Cardinality: The number of unique values in a categorical feature. 
      High cardinality (many unique values) can lead to overfitting and requires special encoding.
    """
    print(f"Running EDA on {data_path}...")
    df = pd.read_csv(data_path)
    
    with open(output_file, "w") as f:
        f.write("# Exploratory Data Analysis Report\n\n")
        
        # 1. Target Distribution
        f.write("## 1. Target Distribution (`click`)\n")
        f.write("Class imbalance tells us if the dataset is skewed towards one class.\n\n")
        dist = df['click'].value_counts(normalize=True) * 100
        f.write(f"- **Non-clicks (0)**: {dist[0]:.2f}%\n")
        f.write(f"- **Clicks (1)**: {dist[1]:.2f}%\n\n")
        f.write("> **Insight**: A highly imbalanced dataset means a model predicting '0' all the time could have high accuracy. This is why we optimize for Precision and LogLoss.\n\n")
        
        # 2. Missing Values
        f.write("## 2. Missing Values\n")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            f.write("No missing values detected (Note: Categorical datasets often encode missing as '-1' or 'unknown').\n\n")
        else:
            f.write("Missing values per column:\n")
            for col, count in missing.items():
                f.write(f"- {col}: {count} ({(count/len(df))*100:.2f}%)\n")
            f.write("\n")
            
        # 3. Cardinality
        f.write("## 3. Feature Cardinality\n")
        f.write("Cardinality is the count of unique values. High cardinality features (e.g., `device_ip` with thousands of unique values) often require Target Encoding or Hashing, or should be dropped in the baseline to prevent overfitting.\n\n")
        f.write("| Feature | Unique Values | Type |\n")
        f.write("|---------|---------------|------|\n")
        
        for col in df.columns:
            if col == 'id': continue
            unique_count = df[col].nunique()
            
            # Simple heuristic for type
            col_type = "Categorical (Low/Mid)"
            if unique_count > 1000:
                col_type = "High Cardinality ID"
            if col in ['click', 'hour']: 
                col_type = "Context/Target"
                
            f.write(f"| {col} | {unique_count} | {col_type} |\n")
            
    print(f"EDA Report generated at {output_file}")

if __name__ == "__main__":
    run_eda()
