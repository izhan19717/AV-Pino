"""
Unit tests for CWRU dataset loader.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import scipy.io

from src.data.cwru_loader import CWRUDataLoader, CWRUDataSample, CWRUDataset


class TestCWRUDataLoader(unittest.TestCase):
    """Test cases for CWRU data loader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = CWRUDataLoader(data_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test loader initialization."""
        self.assertEqual(self.loader.data_dir, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))
        
    def test_cwru_data_sample_creation(self):
        """Test CWRUDataSample dataclass."""
        signal = np.random.randn(1000)
        sample = CWRUDataSample(
            signal=signal,
            fault_type='normal',
            fault_size=0.0,
            load=0,
            rpm=1797,
            sampling_rate=12000,
            filename='test.mat'
        )
        
        self.assertEqual(sample.fault_type, 'normal')
        self.assertEqual(sample.fault_size, 0.0)
        self.assertEqual(sample.load, 0)
        self.assertEqual(sample.rpm, 1797)
        self.assertEqual(sample.sampling_rate, 12000)
        np.testing.assert_array_equal(sample.signal, signal)
        
    def test_parse_mat_file(self):
        """Test parsing of .mat files."""
        # Create a mock .mat file
        test_signal = np.random.randn(2000)
        test_data = {'X097_DE_time': test_signal}
        test_file = os.path.join(self.temp_dir, 'test.mat')
        scipy.io.savemat(test_file, test_data)
        
        # Parse the file
        samples = self.loader.parse_mat_file(test_file, 'normal', 0.0, 0)
        
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.fault_type, 'normal')
        self.assertEqual(sample.fault_size, 0.0)
        self.assertEqual(sample.load, 0)
        self.assertEqual(sample.sampling_rate, 12000)
        np.testing.assert_array_equal(sample.signal, test_signal)
        
    def test_parse_mat_file_invalid(self):
        """Test parsing of invalid .mat files."""
        # Test with non-existent file
        samples = self.loader.parse_mat_file('nonexistent.mat', 'normal', 0.0, 0)
        self.assertEqual(len(samples), 0)
        
    @patch('urllib.request.urlretrieve')
    def test_download_file(self, mock_urlretrieve):
        """Test file downloading."""
        filename = 'test.mat'
        expected_path = os.path.join(self.temp_dir, filename)
        
        # Mock successful download
        mock_urlretrieve.return_value = None
        
        result_path = self.loader.download_file(filename)
        
        self.assertEqual(result_path, expected_path)
        mock_urlretrieve.assert_called_once()
        
    def test_download_file_existing(self):
        """Test downloading when file already exists."""
        filename = 'existing.mat'
        filepath = os.path.join(self.temp_dir, filename)
        
        # Create existing file
        with open(filepath, 'w') as f:
            f.write('test')
            
        with patch('urllib.request.urlretrieve') as mock_urlretrieve:
            result_path = self.loader.download_file(filename)
            
            self.assertEqual(result_path, filepath)
            mock_urlretrieve.assert_not_called()
            
    def test_get_samples_by_fault_type(self):
        """Test filtering samples by fault type."""
        samples = [
            CWRUDataSample(np.array([1]), 'normal', 0.0, 0, 1797, 12000, 'test1.mat'),
            CWRUDataSample(np.array([2]), 'inner_race', 0.007, 0, 1797, 12000, 'test2.mat'),
            CWRUDataSample(np.array([3]), 'normal', 0.0, 1, 1772, 48000, 'test3.mat')
        ]
        dataset = CWRUDataset(samples=samples, fault_types=['normal', 'inner_race'], metadata={})
        
        normal_samples = self.loader.get_samples_by_fault_type(dataset, 'normal')
        self.assertEqual(len(normal_samples), 2)
        self.assertTrue(all(s.fault_type == 'normal' for s in normal_samples))
        
        inner_race_samples = self.loader.get_samples_by_fault_type(dataset, 'inner_race')
        self.assertEqual(len(inner_race_samples), 1)
        self.assertEqual(inner_race_samples[0].fault_type, 'inner_race')
        
    def test_get_samples_by_load(self):
        """Test filtering samples by load condition."""
        samples = [
            CWRUDataSample(np.array([1]), 'normal', 0.0, 0, 1797, 12000, 'test1.mat'),
            CWRUDataSample(np.array([2]), 'inner_race', 0.007, 1, 1772, 48000, 'test2.mat'),
            CWRUDataSample(np.array([3]), 'normal', 0.0, 0, 1797, 12000, 'test3.mat')
        ]
        dataset = CWRUDataset(samples=samples, fault_types=['normal', 'inner_race'], metadata={})
        
        load_0_samples = self.loader.get_samples_by_load(dataset, 0)
        self.assertEqual(len(load_0_samples), 2)
        self.assertTrue(all(s.load == 0 for s in load_0_samples))
        
        load_1_samples = self.loader.get_samples_by_load(dataset, 1)
        self.assertEqual(len(load_1_samples), 1)
        self.assertEqual(load_1_samples[0].load, 1)


if __name__ == '__main__':
    unittest.main()