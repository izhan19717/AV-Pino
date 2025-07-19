"""
Data preprocessing and augmentation for neural operator training.

Implements normalization, windowing, sequence preparation, and data augmentation
strategies for motor fault diagnosis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    window_size: int = 1024
    overlap_ratio: float = 0.5
    normalization_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    sequence_length: int = 256
    target_sampling_rate: Optional[int] = None
    remove_dc: bool = True
    apply_filtering: bool = True
    filter_cutoff: Tuple[float, float] = (1.0, 1000.0)  # Hz

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    noise_std_ratio: float = 0.05
    time_shift_ratio: float = 0.1
    amplitude_scale_range: Tuple[float, float] = (0.8, 1.2)
    frequency_shift_range: Tuple[float, float] = (0.95, 1.05)
    synthetic_fault_ratio: float = 0.2
    enable_mixup: bool = True
    mixup_alpha: float = 0.2

class DataPreprocessor:
    """
    Data preprocessor for motor fault diagnosis neural operator training.
    
    Handles normalization, windowing, sequence preparation, and ensures
    data is properly formatted for neural operator architectures.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize data preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.scalers = {}
        self.fitted = False
        
    def fit(self, signals: List[np.ndarray], labels: Optional[List] = None) -> None:
        """
        Fit preprocessing parameters on training data.
        
        Args:
            signals: List of signal arrays
            labels: Optional labels for supervised fitting
        """
        logger.info("Fitting preprocessing parameters...")
        
        # Concatenate all signals for global statistics
        all_data = np.concatenate([sig.flatten() for sig in signals])
        
        # Fit normalization scaler
        if self.config.normalization_method == 'standard':
            self.scalers['signal'] = StandardScaler()
            self.scalers['signal'].fit(all_data.reshape(-1, 1))
        elif self.config.normalization_method == 'minmax':
            self.scalers['signal'] = MinMaxScaler()
            self.scalers['signal'].fit(all_data.reshape(-1, 1))
        elif self.config.normalization_method == 'robust':
            # For robust normalization, we don't need a scaler, just use global stats
            pass
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization_method}")
        
        # Compute global statistics for robust normalization
        self.global_stats = {
            'mean': np.mean(all_data),
            'std': np.std(all_data),
            'median': np.median(all_data),
            'mad': np.median(np.abs(all_data - np.median(all_data))),
            'percentile_1': np.percentile(all_data, 1),
            'percentile_99': np.percentile(all_data, 99)
        }
        
        self.fitted = True
        logger.info("Preprocessing parameters fitted successfully")
        
    def preprocess_signal(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Preprocess a single signal.
        
        Args:
            signal: Input signal array
            sampling_rate: Signal sampling rate
            
        Returns:
            Preprocessed signal
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before use")
            
        processed_signal = signal.copy()
        
        # Remove DC component
        if self.config.remove_dc:
            processed_signal = processed_signal - np.mean(processed_signal)
            
        # Apply filtering
        if self.config.apply_filtering:
            processed_signal = self._apply_bandpass_filter(
                processed_signal, sampling_rate, 
                self.config.filter_cutoff[0], self.config.filter_cutoff[1]
            )
            
        # Resample if needed
        if self.config.target_sampling_rate and self.config.target_sampling_rate != sampling_rate:
            processed_signal = self._resample_signal(
                processed_signal, sampling_rate, self.config.target_sampling_rate
            )
            
        # Normalize
        processed_signal = self._normalize_signal(processed_signal)
        
        return processed_signal
    
    def create_windows(self, signal: np.ndarray, 
                      labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create overlapping windows from signal.
        
        Args:
            signal: Input signal
            labels: Optional labels (same for all windows)
            
        Returns:
            Tuple of (windowed_signals, windowed_labels)
        """
        window_size = self.config.window_size
        overlap = int(window_size * self.config.overlap_ratio)
        step_size = window_size - overlap
        
        # Calculate number of windows
        n_windows = (len(signal) - window_size) // step_size + 1
        
        if n_windows <= 0:
            # Signal too short, pad or return single window
            if len(signal) < window_size:
                padded_signal = np.pad(signal, (0, window_size - len(signal)), mode='reflect')
                windows = padded_signal.reshape(1, -1)
            else:
                windows = signal[:window_size].reshape(1, -1)
            n_windows = 1
        else:
            windows = np.zeros((n_windows, window_size))
            for i in range(n_windows):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                windows[i] = signal[start_idx:end_idx]
        
        # Handle labels
        windowed_labels = None
        if labels is not None:
            if np.isscalar(labels):
                windowed_labels = np.full(n_windows, labels)
            else:
                # For sequence labels, take the most common label in each window
                windowed_labels = np.zeros(n_windows)
                for i in range(n_windows):
                    start_idx = i * step_size
                    end_idx = start_idx + window_size
                    window_labels = labels[start_idx:min(end_idx, len(labels))]
                    windowed_labels[i] = np.bincount(window_labels.astype(int)).argmax()
        
        return windows, windowed_labels
    
    def prepare_sequences(self, windows: np.ndarray, 
                         labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequences for neural operator training.
        
        Args:
            windows: Windowed signal data
            labels: Optional labels
            
        Returns:
            Tuple of (sequences, sequence_labels)
        """
        sequence_length = self.config.sequence_length
        
        if windows.shape[1] == sequence_length:
            # Already correct length
            return windows, labels
        elif windows.shape[1] > sequence_length:
            # Truncate or subsample
            indices = np.linspace(0, windows.shape[1]-1, sequence_length, dtype=int)
            sequences = windows[:, indices]
        else:
            # Interpolate to target length
            sequences = np.zeros((windows.shape[0], sequence_length))
            for i in range(windows.shape[0]):
                sequences[i] = np.interp(
                    np.linspace(0, 1, sequence_length),
                    np.linspace(0, 1, windows.shape[1]),
                    windows[i]
                )
        
        return sequences, labels
    
    def process_dataset(self, signals: List[np.ndarray], 
                       labels: List, 
                       sampling_rates: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process complete dataset.
        
        Args:
            signals: List of signal arrays
            labels: List of labels
            sampling_rates: List of sampling rates
            
        Returns:
            Tuple of (processed_sequences, processed_labels)
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before processing dataset")
            
        all_sequences = []
        all_labels = []
        
        for signal, label, sr in zip(signals, labels, sampling_rates):
            # Preprocess signal
            processed_signal = self.preprocess_signal(signal, sr)
            
            # Create windows
            windows, window_labels = self.create_windows(processed_signal, label)
            
            # Prepare sequences
            sequences, seq_labels = self.prepare_sequences(windows, window_labels)
            
            all_sequences.append(sequences)
            all_labels.extend(seq_labels if seq_labels is not None else [label] * len(sequences))
        
        # Concatenate all sequences
        final_sequences = np.vstack(all_sequences)
        final_labels = np.array(all_labels)
        
        logger.info(f"Processed dataset: {final_sequences.shape[0]} sequences of length {final_sequences.shape[1]}")
        
        return final_sequences, final_labels
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal using fitted scaler."""
        if self.config.normalization_method == 'robust':
            # Robust normalization using median and MAD
            median = self.global_stats['median']
            mad = self.global_stats['mad']
            return (signal - median) / (mad + 1e-8)
        else:
            # Use fitted scaler
            return self.scalers['signal'].transform(signal.reshape(-1, 1)).flatten()
    
    def _apply_bandpass_filter(self, signal: np.ndarray, sampling_rate: int,
                              low_cutoff: float, high_cutoff: float) -> np.ndarray:
        """Apply bandpass filter to signal."""
        from scipy import signal as scipy_signal
        
        nyquist = sampling_rate / 2
        low = low_cutoff / nyquist
        high = min(high_cutoff / nyquist, 0.99)  # Ensure below Nyquist
        
        if low >= high:
            logger.warning("Invalid filter cutoffs, skipping filtering")
            return signal
            
        try:
            sos = scipy_signal.butter(4, [low, high], btype='band', output='sos')
            filtered_signal = scipy_signal.sosfilt(sos, signal)
            return filtered_signal
        except Exception as e:
            logger.warning(f"Filtering failed: {e}, returning original signal")
            return signal
    
    def _resample_signal(self, signal: np.ndarray, 
                        original_rate: int, target_rate: int) -> np.ndarray:
        """Resample signal to target sampling rate."""
        from scipy import signal as scipy_signal
        
        if original_rate == target_rate:
            return signal
            
        # Calculate resampling ratio
        ratio = target_rate / original_rate
        new_length = int(len(signal) * ratio)
        
        # Use scipy's resample function
        resampled_signal = scipy_signal.resample(signal, new_length)
        
        return resampled_signal


class DataAugmentor:
    """
    Data augmentation for motor fault diagnosis.
    
    Implements various augmentation strategies including noise injection,
    time shifting, amplitude scaling, and synthetic fault generation.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize data augmentor.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.rng = np.random.RandomState(42)  # For reproducibility
        
    def augment_signal(self, signal: np.ndarray, label: Optional[int] = None) -> np.ndarray:
        """
        Apply augmentation to a single signal.
        
        Args:
            signal: Input signal
            label: Optional label for label-aware augmentation
            
        Returns:
            Augmented signal
        """
        augmented = signal.copy()
        
        # Add noise
        if self.config.noise_std_ratio > 0:
            noise_std = np.std(signal) * self.config.noise_std_ratio
            noise = self.rng.normal(0, noise_std, len(signal))
            augmented += noise
        
        # Time shifting
        if self.config.time_shift_ratio > 0:
            max_shift = int(len(signal) * self.config.time_shift_ratio)
            shift = self.rng.randint(-max_shift, max_shift + 1)
            augmented = np.roll(augmented, shift)
        
        # Amplitude scaling
        scale_min, scale_max = self.config.amplitude_scale_range
        scale_factor = self.rng.uniform(scale_min, scale_max)
        augmented *= scale_factor
        
        # Frequency domain augmentation
        if self.config.frequency_shift_range != (1.0, 1.0):
            augmented = self._frequency_shift(augmented)
        
        return augmented
    
    def generate_synthetic_faults(self, normal_signals: List[np.ndarray],
                                fault_types: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Generate synthetic fault scenarios.
        
        Args:
            normal_signals: List of normal operation signals
            fault_types: List of target fault types to generate
            
        Returns:
            Tuple of (synthetic_signals, synthetic_labels)
        """
        synthetic_signals = []
        synthetic_labels = []
        
        for fault_type in fault_types:
            for normal_signal in normal_signals:
                # Generate synthetic fault based on type
                if fault_type == 'bearing_fault':
                    synthetic_signal = self._generate_bearing_fault(normal_signal)
                elif fault_type == 'unbalance':
                    synthetic_signal = self._generate_unbalance_fault(normal_signal)
                elif fault_type == 'misalignment':
                    synthetic_signal = self._generate_misalignment_fault(normal_signal)
                else:
                    # Generic fault injection
                    synthetic_signal = self._generate_generic_fault(normal_signal)
                
                synthetic_signals.append(synthetic_signal)
                synthetic_labels.append(fault_type)
        
        return synthetic_signals, synthetic_labels
    
    def mixup_augmentation(self, signals: np.ndarray, labels: np.ndarray,
                          alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation.
        
        Args:
            signals: Input signals
            labels: Input labels
            alpha: Mixup parameter (uses config if None)
            
        Returns:
            Tuple of (mixed_signals, mixed_labels)
        """
        if alpha is None:
            alpha = self.config.mixup_alpha
            
        n_samples = len(signals)
        indices = self.rng.permutation(n_samples)
        
        # Sample mixing coefficients
        lam = self.rng.beta(alpha, alpha, n_samples)
        
        mixed_signals = np.zeros_like(signals)
        mixed_labels = np.zeros_like(labels)
        
        for i in range(n_samples):
            j = indices[i]
            mixed_signals[i] = lam[i] * signals[i] + (1 - lam[i]) * signals[j]
            # For classification, we might need to handle mixed labels differently
            # Here we use the dominant label
            mixed_labels[i] = labels[i] if lam[i] > 0.5 else labels[j]
        
        return mixed_signals, mixed_labels
    
    def augment_dataset(self, signals: np.ndarray, labels: np.ndarray,
                       augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment entire dataset.
        
        Args:
            signals: Input signals
            labels: Input labels
            augmentation_factor: How many augmented versions per original
            
        Returns:
            Tuple of (augmented_signals, augmented_labels)
        """
        augmented_signals = [signals]  # Include original
        augmented_labels = [labels]
        
        for _ in range(augmentation_factor - 1):
            batch_augmented = np.zeros_like(signals)
            for i, (signal, label) in enumerate(zip(signals, labels)):
                batch_augmented[i] = self.augment_signal(signal, label)
            
            augmented_signals.append(batch_augmented)
            augmented_labels.append(labels.copy())
        
        # Apply mixup if enabled
        if self.config.enable_mixup:
            mixed_signals, mixed_labels = self.mixup_augmentation(signals, labels)
            augmented_signals.append(mixed_signals)
            augmented_labels.append(mixed_labels)
        
        # Concatenate all augmented data
        final_signals = np.vstack(augmented_signals)
        final_labels = np.concatenate(augmented_labels)
        
        logger.info(f"Dataset augmented: {len(signals)} -> {len(final_signals)} samples")
        
        return final_signals, final_labels
    
    def _frequency_shift(self, signal: np.ndarray) -> np.ndarray:
        """Apply frequency domain shifting."""
        # Simple frequency shifting using interpolation
        shift_min, shift_max = self.config.frequency_shift_range
        shift_factor = self.rng.uniform(shift_min, shift_max)
        
        # Create new time axis
        original_indices = np.arange(len(signal))
        new_indices = original_indices / shift_factor
        
        # Interpolate
        shifted_signal = np.interp(original_indices, new_indices, signal, 
                                 left=signal[0], right=signal[-1])
        
        return shifted_signal
    
    def _generate_bearing_fault(self, normal_signal: np.ndarray) -> np.ndarray:
        """Generate synthetic bearing fault signal."""
        # Add periodic impulses characteristic of bearing faults
        fault_signal = normal_signal.copy()
        
        # Bearing fault frequencies (simplified)
        sampling_rate = 12000  # Assume default
        bpfo_freq = 105  # Ball pass frequency outer race
        impulse_period = int(sampling_rate / bpfo_freq)
        
        # Add impulses
        for i in range(0, len(fault_signal), impulse_period):
            if i < len(fault_signal):
                # Create decaying impulse
                impulse_length = min(50, len(fault_signal) - i)
                impulse = np.exp(-np.arange(impulse_length) / 10) * self.rng.normal(0, 0.5)
                fault_signal[i:i+impulse_length] += impulse
        
        return fault_signal
    
    def _generate_unbalance_fault(self, normal_signal: np.ndarray) -> np.ndarray:
        """Generate synthetic unbalance fault signal."""
        # Add 1x running speed component
        fault_signal = normal_signal.copy()
        
        # Assume 30 Hz running speed
        sampling_rate = 12000
        running_freq = 30
        time = np.arange(len(fault_signal)) / sampling_rate
        
        # Add unbalance component
        unbalance_amplitude = np.std(normal_signal) * 0.3
        unbalance_component = unbalance_amplitude * np.sin(2 * np.pi * running_freq * time)
        fault_signal += unbalance_component
        
        return fault_signal
    
    def _generate_misalignment_fault(self, normal_signal: np.ndarray) -> np.ndarray:
        """Generate synthetic misalignment fault signal."""
        # Add 2x running speed component
        fault_signal = normal_signal.copy()
        
        # Assume 30 Hz running speed
        sampling_rate = 12000
        running_freq = 30
        time = np.arange(len(fault_signal)) / sampling_rate
        
        # Add misalignment component (2x running speed)
        misalignment_amplitude = np.std(normal_signal) * 0.2
        misalignment_component = misalignment_amplitude * np.sin(2 * np.pi * 2 * running_freq * time)
        fault_signal += misalignment_component
        
        return fault_signal
    
    def _generate_generic_fault(self, normal_signal: np.ndarray) -> np.ndarray:
        """Generate generic fault by adding random disturbances."""
        fault_signal = normal_signal.copy()
        
        # Add random spikes
        n_spikes = self.rng.randint(1, 5)
        spike_positions = self.rng.choice(len(fault_signal), n_spikes, replace=False)
        spike_amplitude = np.std(normal_signal) * 2
        
        for pos in spike_positions:
            fault_signal[pos] += self.rng.normal(0, spike_amplitude)
        
        return fault_signal