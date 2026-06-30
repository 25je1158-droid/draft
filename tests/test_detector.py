"""
Unit tests for anomaly detection module
"""
import pytest
import pandas as pd
import numpy as np
from detector.anomaly import calculate_features, train_detector, detect_anomalies


@pytest.fixture
def sample_data():
    """Create sample stock data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'Open': np.random.uniform(150, 160, 100),
        'High': np.random.uniform(160, 170, 100),
        'Low': np.random.uniform(140, 150, 100),
        'Close': np.random.uniform(150, 160, 100),
        'Volume': np.random.randint(1000000, 5000000, 100),
    }
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def sample_data_with_anomaly():
    """Create sample data with artificial anomalies"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'Open': np.random.uniform(150, 160, 100),
        'High': np.random.uniform(160, 170, 100),
        'Low': np.random.uniform(140, 150, 100),
        'Close': np.random.uniform(150, 160, 100),
        'Volume': np.random.randint(1000000, 5000000, 100),
    }
    df = pd.DataFrame(data, index=dates)
    
    # Inject anomalies: extreme spike at indices 20, 50, 80
    df.loc[df.index[20], 'Close'] = 500  # Price spike
    df.loc[df.index[50], 'Volume'] = 50000000  # Volume spike
    df.loc[df.index[80], 'Close'] = 50  # Price crash
    
    return df


class TestCalculateFeatures:
    """Tests for feature engineering"""
    
    def test_calculate_features_returns_dataframe(self, sample_data):
        """Test that calculate_features returns a DataFrame"""
        result = calculate_features(sample_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_calculate_features_removes_nan(self, sample_data):
        """Test that NaN values are removed"""
        result = calculate_features(sample_data)
        assert result.isna().sum().sum() == 0
    
    def test_calculate_features_adds_columns(self, sample_data):
        """Test that new feature columns are added"""
        result = calculate_features(sample_data)
        expected_cols = ['daily_spread', 'rolling_mean', 'rolling_std', 'z_score', 'price_change_pct']
        for col in expected_cols:
            assert col in result.columns
    
    def test_daily_spread_calculation(self, sample_data):
        """Test daily spread is High - Low (for matching indices)"""
        result = calculate_features(sample_data)
        # Compare only the indices that exist in result (after dropna removes first 19 rows)
        expected_spread = sample_data['High'] - sample_data['Low']
        expected_spread = expected_spread.loc[result.index]
        
        pd.testing.assert_series_equal(
            result['daily_spread'], 
            expected_spread,
            check_names=False
        )
    
    def test_calculate_features_with_missing_data(self):
        """Test handling of missing data"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        data = {
            'Open': np.random.uniform(150, 160, 50),
            'High': np.random.uniform(160, 170, 50),
            'Low': np.random.uniform(140, 150, 50),
            'Close': np.random.uniform(150, 160, 50),
            'Volume': np.random.randint(1000000, 5000000, 50),
        }
        df = pd.DataFrame(data, index=dates)
        df.loc[df.index[10], 'Close'] = np.nan
        df.loc[df.index[20], 'Volume'] = np.nan
        
        result = calculate_features(df)
        assert result.isna().sum().sum() == 0


class TestTrainDetector:
    """Tests for model training"""
    
    def test_train_detector_returns_model(self, sample_data):
        """Test that train_detector returns an IsolationForest model"""
        features_df = calculate_features(sample_data)
        model = train_detector(features_df)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'decision_function')
    
    def test_model_contamination_parameter(self, sample_data):
        """Test that contamination parameter is set correctly"""
        features_df = calculate_features(sample_data)
        model = train_detector(features_df)
        assert model.contamination == 0.03
    
    def test_model_random_state(self, sample_data):
        """Test that random state is set for reproducibility"""
        features_df = calculate_features(sample_data)
        model = train_detector(features_df)
        assert model.random_state == 42
    
    def test_model_reproducibility(self, sample_data):
        """Test that same data produces same predictions"""
        features_df = calculate_features(sample_data)
        model1 = train_detector(features_df)
        model2 = train_detector(features_df)
        
        features = features_df[['daily_spread', 'z_score', 'Volume', 'price_change_pct']]
        pred1 = model1.predict(features)
        pred2 = model2.predict(features)
        
        np.testing.assert_array_equal(pred1, pred2)


class TestDetectAnomalies:
    """Tests for anomaly detection"""
    
    def test_detect_anomalies_returns_dataframe(self, sample_data):
        """Test that detect_anomalies returns a DataFrame"""
        features_df = calculate_features(sample_data)
        model = train_detector(features_df)
        result = detect_anomalies(model, features_df)
        assert isinstance(result, pd.DataFrame)
    
    def test_detect_anomalies_adds_columns(self, sample_data):
        """Test that anomaly score and flag are added"""
        features_df = calculate_features(sample_data)
        model = train_detector(features_df)
        result = detect_anomalies(model, features_df)
        
        assert 'anomaly_score' in result.columns
        assert 'is_anomaly' in result.columns
    
    def test_anomaly_flag_is_boolean(self, sample_data):
        """Test that is_anomaly column contains only True/False"""
        features_df = calculate_features(sample_data)
        model = train_detector(features_df)
        result = detect_anomalies(model, features_df)
        
        assert result['is_anomaly'].dtype == bool
        assert set(result['is_anomaly'].unique()).issubset({True, False})
    
    def test_detects_artificial_anomalies(self, sample_data_with_anomaly):
        """Test that extreme values are detected as anomalies"""
        features_df = calculate_features(sample_data_with_anomaly)
        model = train_detector(features_df)
        result = detect_anomalies(model, features_df)
        
        # Should detect at least one anomaly
        assert result['is_anomaly'].sum() > 0
    
    def test_contamination_ratio(self, sample_data):
        """Test that detected anomalies respect contamination ratio"""
        features_df = calculate_features(sample_data)
        model = train_detector(features_df)
        result = detect_anomalies(model, features_df)
        
        anomaly_ratio = result['is_anomaly'].sum() / len(result)
        # Should be approximately 3% (allow some tolerance due to rounding)
        assert 0 <= anomaly_ratio <= 0.10  # Allow up to 10% for variance
    
    def test_decision_function_output(self, sample_data):
        """Test that decision function produces numeric scores"""
        features_df = calculate_features(sample_data)
        model = train_detector(features_df)
        result = detect_anomalies(model, features_df)
        
        assert pd.api.types.is_numeric_dtype(result['anomaly_score'])
        assert result['anomaly_score'].notna().all()


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_single_row_data(self):
        """Test behavior with single row of data"""
        dates = pd.date_range(start='2023-01-01', periods=1, freq='D')
        data = {
            'Open': [155.0],
            'High': [165.0],
            'Low': [145.0],
            'Close': [160.0],
            'Volume': [2000000],
        }
        df = pd.DataFrame(data, index=dates)
        
        # calculate_features should drop NaN from rolling calculations
        result = calculate_features(df)
        assert len(result) == 0  # All NaN due to rolling window
    
    def test_constant_prices(self):
        """Test with constant prices (no volatility) - should raise error"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = {
            'Open': [155.0] * 100,
            'High': [165.0] * 100,
            'Low': [145.0] * 100,
            'Close': [160.0] * 100,
            'Volume': [2000000] * 100,
        }
        df = pd.DataFrame(data, index=dates)
        
        features_df = calculate_features(df)
        
        # With constant prices and std=0, z_score becomes NaN
        # After dropna, we get empty dataframe, which causes ValueError
        with pytest.raises(ValueError, match="Insufficient data"):
            train_detector(features_df)
    
    def test_all_nan_column(self):
        """Test with a column that becomes all NaN after processing"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        data = {
            'Open': np.random.uniform(150, 160, 50),
            'High': np.random.uniform(160, 170, 50),
            'Low': np.random.uniform(140, 150, 50),
            'Close': np.random.uniform(150, 160, 50),
            'Volume': np.random.randint(1000000, 5000000, 50),
        }
        df = pd.DataFrame(data, index=dates)
        
        result = calculate_features(df)
        # Should still work even with NaN columns
        assert len(result) > 0
        assert result.isna().sum().sum() == 0


class TestDataIntegrity:
    """Tests to ensure data integrity during processing"""
    
    def test_no_data_loss_in_features(self, sample_data):
        """Test that rows aren't unexpectedly dropped"""
        original_len = len(sample_data)
        result = calculate_features(sample_data)
        
        # Due to rolling window and dropna, we lose first 19 rows (20-day window - 1)
        # This is expected behavior
        assert len(result) <= original_len
        assert len(result) >= original_len - 20
    
    def test_preserve_original_data(self, sample_data):
        """Test that original data is not modified"""
        original = sample_data.copy()
        calculate_features(sample_data)
        
        pd.testing.assert_frame_equal(sample_data, original)
    
    def test_feature_values_are_finite(self, sample_data):
        """Test that all feature values are finite (no inf or excessive nan)"""
        result = calculate_features(sample_data)
        
        for col in result.columns:
            assert np.isfinite(result[col]).all(), f"Non-finite values in {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
