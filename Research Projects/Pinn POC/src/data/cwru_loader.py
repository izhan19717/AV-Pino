"""
CWRU Dataset Loader for bearing fault data processing.
"""

import os
import urllib.request
import numpy as np
import scipy.io
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import kagglehub
import glob

logger = logging.getLogger(__name__)

@dataclass
class CWRUDataSample:
    """Single CWRU dataset sample."""
    signal: np.ndarray
    fault_type: str
    fault_size: float
    load: int
    rpm: int
    sampling_rate: int
    filename: str

@dataclass
class CWRUDataset:
    """Complete CWRU dataset."""
    samples: List[CWRUDataSample]
    fault_types: List[str]
    metadata: Dict

class CWRUDataLoader:
    """
    Loader for Case Western Reserve University bearing fault dataset.
    
    Handles downloading, parsing, and organizing bearing fault data files
    with different fault types, sizes, and operating conditions.
    """
    
    # CWRU dataset file mappings based on actual CWRU website structure from screenshots
    CWRU_FILES = {
        'normal': {
            '0hp': ['Normal_0.mat', 'Normal_1.mat', 'Normal_2.mat', 'Normal_3.mat']
        },
        'inner_race': {
            '0hp': {
                '0.007': ['IR007_0.mat', 'IR007_1.mat', 'IR007_2.mat', 'IR007_3.mat'],
                '0.014': ['IR014_0.mat', 'IR014_1.mat', 'IR014_2.mat', 'IR014_3.mat'],
                '0.021': ['IR021_0.mat', 'IR021_1.mat', 'IR021_2.mat', 'IR021_3.mat']
            }
        },
        'outer_race': {
            '0hp': {
                '0.007': ['OR007@6_0.mat', 'OR007@6_1.mat', 'OR007@6_2.mat', 'OR007@6_3.mat'],
                '0.014': ['OR014@6_0.mat', 'OR014@6_1.mat', 'OR014@6_2.mat', 'OR014@6_3.mat'],
                '0.021': ['OR021@6_0.mat', 'OR021@6_1.mat', 'OR021@6_2.mat', 'OR021@6_3.mat']
            }
        },
        'ball': {
            '0hp': {
                '0.007': ['B007_0.mat', 'B007_1.mat', 'B007_2.mat', 'B007_3.mat'],
                '0.014': ['B014_0.mat', 'B014_1.mat', 'B014_2.mat', 'B014_3.mat'],
                '0.021': ['B021_0.mat', 'B021_1.mat', 'B021_2.mat', 'B021_3.mat']
            }
        }
    }
    
    # Updated URLs based on actual CWRU Bearing Data Center structure from screenshots
    BASE_URLS = {
        'normal': "https://engineering.case.edu/sites/default/files/bearingdatacenter/normal/",
        'inner_race': "https://engineering.case.edu/sites/default/files/bearingdatacenter/12k_drive_end_bearing_fault_data/",
        'outer_race': "https://engineering.case.edu/sites/default/files/bearingdatacenter/12k_drive_end_bearing_fault_data/",
        'ball': "https://engineering.case.edu/sites/default/files/bearingdatacenter/12k_drive_end_bearing_fault_data/"
    }
    
    def __init__(self, data_dir: str = "data/cwru"):
        """
        Initialize CWRU data loader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.samples = []
        self.kaggle_dataset_path = None
        os.makedirs(data_dir, exist_ok=True)
    
    def download_kaggle_dataset(self) -> str:
        """Download CWRU dataset from Kaggle."""
        try:
            logger.info("Downloading CWRU dataset from Kaggle...")
            path = kagglehub.dataset_download("brjapon/cwru-bearing-datasets")
            logger.info(f"Kaggle dataset downloaded to: {path}")
            self.kaggle_dataset_path = path
            return path
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset: {e}")
            raise
        
    def download_file(self, filename: str, fault_type: str = 'normal') -> str:
        """Download a single CWRU data file."""
        base_url = self.BASE_URLS.get(fault_type, self.BASE_URLS['normal'])
        url = f"{base_url}{filename}"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            logger.info(f"File {filename} already exists, skipping download")
            return filepath
            
        try:
            logger.info(f"Downloading {filename} from {fault_type} category...")
            urllib.request.urlretrieve(url, filepath)
            logger.info(f"Downloaded {filename} successfully")
            return filepath
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise
    
    def parse_mat_file(self, filepath: str, fault_type: str, fault_size: float, load: int) -> List[CWRUDataSample]:
        """Parse a .mat file and extract signal data."""
        try:
            mat_data = scipy.io.loadmat(filepath)
            samples = []
            
            # Extract signal data (CWRU files contain multiple variables)
            for key, value in mat_data.items():
                if not key.startswith('_') and isinstance(value, np.ndarray) and value.size > 1000:
                    # Assume this is signal data
                    signal = value.flatten()
                    
                    # Determine sampling rate and RPM from filename/load
                    sampling_rate = 12000 if load == 0 else 48000
                    rpm_map = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
                    rpm = rpm_map.get(load, 1797)
                    
                    sample = CWRUDataSample(
                        signal=signal,
                        fault_type=fault_type,
                        fault_size=fault_size,
                        load=load,
                        rpm=rpm,
                        sampling_rate=sampling_rate,
                        filename=os.path.basename(filepath)
                    )
                    samples.append(sample)
                    break  # Take first valid signal
                    
            return samples
            
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return []
    
    def load_dataset(self, download: bool = True) -> CWRUDataset:
        """
        Load complete CWRU dataset.
        
        Args:
            download: Whether to download missing files
            
        Returns:
            CWRUDataset object containing all samples
        """
        all_samples = []
        fault_types = set()
        
        if download:
            try:
                # Try to download from Kaggle first
                kaggle_path = self.download_kaggle_dataset()
                logger.info(f"Using Kaggle dataset from: {kaggle_path}")
                
                # Find all .mat files in the Kaggle dataset
                mat_files = glob.glob(os.path.join(kaggle_path, "**", "*.mat"), recursive=True)
                logger.info(f"Found {len(mat_files)} .mat files in Kaggle dataset")
                
                # Process each .mat file
                for mat_file in mat_files:
                    filename = os.path.basename(mat_file)
                    
                    # Determine fault type and parameters from filename
                    fault_type, fault_size, load = self._parse_filename(filename)
                    if fault_type:
                        fault_types.add(fault_type)
                        samples = self.parse_mat_file(mat_file, fault_type, fault_size, load)
                        all_samples.extend(samples)
                        logger.info(f"Processed {filename}: {fault_type}, size={fault_size}, load={load}")
                
            except Exception as e:
                logger.warning(f"Failed to load Kaggle dataset: {e}")
                logger.info("Falling back to synthetic data generation...")
                return self._create_synthetic_dataset_fallback()
        
        if not all_samples:
            logger.warning("No real data found, using synthetic data fallback")
            return self._create_synthetic_dataset_fallback()
        
        metadata = {
            'total_samples': len(all_samples),
            'fault_types': list(fault_types),
            'data_dir': self.kaggle_dataset_path or self.data_dir,
            'data_source': 'Kaggle CWRU Dataset'
        }
        
        logger.info(f"Loaded {len(all_samples)} samples from CWRU dataset")
        return CWRUDataset(samples=all_samples, fault_types=list(fault_types), metadata=metadata)
    
    def _parse_filename(self, filename: str) -> Tuple[str, float, int]:
        """Parse filename to extract fault type, size, and load."""
        filename = filename.lower()
        
        # Default values
        fault_type = 'unknown'
        fault_size = 0.0
        load = 0
        
        # Determine fault type
        if 'normal' in filename or 'baseline' in filename:
            fault_type = 'normal'
        elif 'ir' in filename or 'inner' in filename:
            fault_type = 'inner_race'
        elif 'or' in filename or 'outer' in filename:
            fault_type = 'outer_race'
        elif 'b' in filename or 'ball' in filename:
            fault_type = 'ball'
        
        # Extract fault size (0.007, 0.014, 0.021)
        if '007' in filename:
            fault_size = 0.007
        elif '014' in filename:
            fault_size = 0.014
        elif '021' in filename:
            fault_size = 0.021
        
        # Extract load (0, 1, 2, 3)
        if '_0' in filename or 'hp0' in filename:
            load = 0
        elif '_1' in filename or 'hp1' in filename:
            load = 1
        elif '_2' in filename or 'hp2' in filename:
            load = 2
        elif '_3' in filename or 'hp3' in filename:
            load = 3
        
        return fault_type, fault_size, load
    
    def _create_synthetic_dataset_fallback(self) -> CWRUDataset:
        """Create a synthetic dataset as fallback."""
        logger.info("Creating enhanced synthetic CWRU dataset...")
        
        # Create synthetic signals for each fault type
        synthetic_data = {}
        fault_types = ['normal', 'inner_race', 'outer_race', 'ball']
        
        for fault_type in fault_types:
            signals = self.create_synthetic_dataset(250)  # 250 samples per fault type
            synthetic_data[fault_type] = signals
        
        # Convert to CWRUDataSample objects
        all_samples = []
        for fault_type, signals in synthetic_data.items():
            for i, signal in enumerate(signals):
                sample = CWRUDataSample(
                    signal=signal,
                    fault_type=fault_type,
                    fault_size=0.007 if fault_type != 'normal' else 0.0,
                    load=0,
                    rpm=1797,
                    sampling_rate=12000,
                    filename=f"synthetic_{fault_type}_{i}.mat"
                )
                all_samples.append(sample)
        
        metadata = {
            'total_samples': len(all_samples),
            'fault_types': fault_types,
            'data_dir': self.data_dir,
            'data_source': 'Enhanced Synthetic CWRU Data'
        }
        
        logger.info(f"Created {len(all_samples)} synthetic samples")
        return CWRUDataset(samples=all_samples, fault_types=fault_types, metadata=metadata)
    
    def get_samples_by_fault_type(self, dataset: CWRUDataset, fault_type: str) -> List[CWRUDataSample]:
        """Get all samples of a specific fault type."""
        return [sample for sample in dataset.samples if sample.fault_type == fault_type]
    
    def get_samples_by_load(self, dataset: CWRUDataset, load: int) -> List[CWRUDataSample]:
        """Get all samples at a specific load condition."""
        return [sample for sample in dataset.samples if sample.load == load]
    
    def create_train_test_split(self, dataset: CWRUDataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dict, Dict]:
        """
        Create train/test split from CWRU dataset.
        
        Args:
            dataset: CWRU dataset
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        np.random.seed(random_state)
        
        train_data = {}
        test_data = {}
        
        for fault_type in dataset.fault_types:
            samples = self.get_samples_by_fault_type(dataset, fault_type)
            
            if samples:
                # Extract signals from samples
                signals = [sample.signal for sample in samples]
                
                # Shuffle and split
                indices = np.random.permutation(len(signals))
                split_idx = int(len(signals) * (1 - test_size))
                
                train_indices = indices[:split_idx]
                test_indices = indices[split_idx:]
                
                train_data[fault_type] = [signals[i] for i in train_indices]
                test_data[fault_type] = [signals[i] for i in test_indices]
                
                logger.info(f"{fault_type}: {len(train_data[fault_type])} train, {len(test_data[fault_type])} test samples")
        
        return train_data, test_data
    
    def create_synthetic_dataset(self, n_samples: int = 1000) -> List[np.ndarray]:
        """Create synthetic bearing data for testing when real data unavailable."""
        np.random.seed(42)
        synthetic_signals = []
        
        for i in range(n_samples):
            # Create synthetic bearing signal with multiple frequency components
            t = np.linspace(0, 1, 1024)
            
            # Base rotation frequency and harmonics
            f_rot = 30  # Hz
            signal = 0.5 * np.sin(2 * np.pi * f_rot * t)
            
            # Add bearing characteristic frequencies
            f_bpfo = f_rot * 3.5  # Ball pass frequency outer race
            f_bpfi = f_rot * 5.4  # Ball pass frequency inner race
            f_bsf = f_rot * 2.3   # Ball spin frequency
            
            signal += 0.3 * np.sin(2 * np.pi * f_bpfo * t)
            signal += 0.2 * np.sin(2 * np.pi * f_bpfi * t)
            signal += 0.1 * np.sin(2 * np.pi * f_bsf * t)
            
            # Add noise and random fault signatures
            signal += 0.1 * np.random.randn(len(t))
            
            # Add random impulses for fault simulation
            if np.random.random() > 0.5:
                impulse_times = np.random.choice(len(t), size=np.random.randint(1, 5), replace=False)
                for imp_time in impulse_times:
                    signal[imp_time] += np.random.uniform(0.5, 2.0)
            
            synthetic_signals.append(signal)
        
        logger.info(f"Created {len(synthetic_signals)} synthetic bearing signals")
        return synthetic_signals