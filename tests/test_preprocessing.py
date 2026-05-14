import pandas as pd
import numpy as np
from core.advanced_preprocessing import get_advanced_preprocessor

def test_advanced_preprocessor_pipeline():
    """
    Tests that the ColumnTransformer correctly separates numeric, 
    low-cardinality categorical, and high-cardinality target-encoded features.
    """
    # 1. Arrange: Create mock data representing our 3 feature types
    mock_data = pd.DataFrame({
        "site_id": ["domain_A", "domain_B", "domain_A"], # High Cardinality (Target Encoded)
        "site_category": ["news", "sports", "news"],     # Low Cardinality (One-Hot Encoded)
        "C14": [1500, 2500, 3500]                        # Numeric (Scaled)
    })
    mock_labels = pd.Series([1, 0, 1])

    # 2. Act: Build and fit the preprocessor
    preprocessor = get_advanced_preprocessor(mock_data)
    transformed = preprocessor.fit_transform(mock_data, mock_labels)

    # 3. Assert: Verify the output format and dimensions
    assert isinstance(transformed, np.ndarray), "Output must be a numpy array."
    
    # Expected columns: 
    # C14 (1) + site_category_news (1) + site_category_sports (1) + site_id (1) = 4 columns
    expected_columns = 4
    assert transformed.shape[1] == expected_columns, f"Expected {expected_columns} columns, got {transformed.shape[1]}"