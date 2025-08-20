import pytest
import numpy as np
import pandas as pd
from magica.core.data_processor import DataProcessor

class TestDataProcessor:
    def test_init_empty(self):
        """Test initialization without data."""
        dp = DataProcessor()
        assert dp.data is None
        assert len(dp) == 0
    
    def test_load_numpy_array(self):
        """Test loading numpy array."""
        data = np.array([1, 2, 3, 4, 5])
        dp = DataProcessor(data)
        assert len(dp) == 5
        np.testing.assert_array_equal(dp.get_data_array(), data)
    
    def test_load_list(self):
        """Test loading list."""
        data = [1, 2, 3, 4, 5]
        dp = DataProcessor(data)
        assert len(dp) == 5
    
    def test_basic_stats(self):
        """Test basic statistics calculation."""
        data = [1, 2, 3, 4, 5]
        dp = DataProcessor(data)
        stats = dp.get_basic_stats()
        
        assert stats['count'] == 5
        assert stats['mean'] == 3.0
        assert stats['min'] == 1
        assert stats['max'] == 5
    
    def test_repr(self):
        """Test string representation."""
        dp = DataProcessor()
        assert "no data loaded" in str(dp)
        
        dp.load_data([1, 2, 3])
        assert "length=3" in str(dp)