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
    
    # CWRU dataset file mappings
    CWRU_FILES = {
        'normal': {
            '0hp': ['97.mat', '98.mat', '99.mat', '100.mat'],
            '1hp': ['156.mat', '157.mat', '158.mat', '159.mat'],
            '2hp': ['185.mat', '186.mat', '187.mat', '188.mat'],
            '3hp': ['222.mat', '223.mat', '224.mat', '225.mat']
        },
        'inner_race': {
            '0hp': {
                '0.007': ['105.mat', '106.mat', '107.mat', '108.mat'],
                '0.014': ['169.mat', '170.mat', '171.mat', '172.mat'],
                '0.021': ['209.mat', '210.mat', '211.mat', '212.mat']
            },
            '1hp': {
                '0.007': ['118.mat', '119.mat', '120.mat', '121.mat'],
                '0.014': ['130.mat', '131.mat', '132.mat', '133.mat'],
                '0.021': ['197.mat', '198.mat', '199.mat', '200.mat']
            }
        },
        'outer_race': {
            '0hp': {
                '0.007': ['144.mat', '145.mat', '146.mat', '147.mat'],
                '0.014': ['189.mat', '190.mat', '191.mat', '192.mat'],
                '0.021': ['234.mat', '235.mat', '236.mat', '237.mat']
            }
        },
        'ball': {
            '0hp': {
                '0.007': ['109.mat', '110.mat', '111.mat', '112.mat'],
                '0.014': ['173.mat', '174.mat', '175.mat', '176.mat'],
                '0.021': ['213.mat', '214.mat', '215.mat', '216.mat']
            }
        }
    }
    
    BASE_URL = "https://engineering.case.edu/sites/default/files/bearingdatacenter/"
    
    def __init__(self, data_dir: str = "data/cwru"):
        """
        Initialize CWRU data loader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.samples = []
        os.makedirs(data_dir, exist_ok=True)
        
    def download_file(self, filename: str) -> str:
        """Download a single CWRU data file."""
        url = f"{self.BASE_URL}{filename}"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            logger.info(f"File {filename} already exists, skipping download")
            return filepath
            
        try:
            logger.info(f"Downloading {filename}...")
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
        
        for fault_type, load_data in self.CWRU_FILES.items():
            fault_types.add(fault_type)
            
            if fault_type == 'normal':
                # Normal data structure
                for load_str, filenames in load_data.items():
                    load = int(load_str.replace('hp', ''))
                    
                    for filename in filenames:
                        if download:
                            filepath = self.download_file(filename)
                        else:
                            filepath = os.path.join(self.data_dir, filename)
                            
                        if os.path.exists(filepath):
                            samples = self.parse_mat_file(filepath, fault_type, 0.0, load)
                            all_samples.extend(samples)
            else:
                # Fault data structure
                for load_str, size_data in load_data.items():
                    load = int(load_str.replace('hp', ''))
                    
                    for fault_size_str, filenames in size_data.items():
                        fault_size = float(fault_size_str)
                        
                        for filename in filenames:
                            if download:
                                filepath = self.download_file(filename)
                            else:
                                filepath = os.path.join(self.data_dir, filename)
                                
                            if os.path.exists(filepath):
                                samples = self.parse_mat_file(filepath, fault_type, fault_size, load)
                                all_samples.extend(samples)
        
        metadata = {
            'total_samples': len(all_samples),
            'fault_types': list(fault_types),
            'data_dir': self.data_dir
        }
        
        logger.info(f"Loaded {len(all_samples)} samples from CWRU dataset")
        return CWRUDataset(samples=all_samples, fault_types=list(fault_types), metadata=metadata)
    
    def get_samples_by_fault_type(self, dataset: CWRUDataset, fault_type: str) -> List[CWRUDataSample]:
        """Get all samples of a specific fault type."""
        return [sample for sample in dataset.samples if sample.fault_type == fault_type]
    
    def get_samples_by_load(self, dataset: CWRUDataset, load: int) -> List[CWRUDataSample]:
        """Get all samples at a specific load condition."""
        return [sample for sample in dataset.samples if sample.load == load]