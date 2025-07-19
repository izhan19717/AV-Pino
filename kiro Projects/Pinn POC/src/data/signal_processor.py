"""
Signal processing utilities for time-domain, frequency-domain, 
and time-frequency feature extraction.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    Signal processor for extracting features from vibration signals.
    
    Provides time-domain, frequency-domain, and time-frequency analysis
    capabilities for motor fault diagnosis applications.
    """
    
    def __init__(self, sampling_rate: int = 12000):
        """
        Initialize signal processor.
        
        Args:
            sampling_rate: Default sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
    def extract_time_domain_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain statistical features.
        
        Args:
            signal_data: Input signal array
            
        Returns:
            Dictionary of time-domain features
        """
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['variance'] = np.var(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['peak'] = np.max(np.abs(signal_data))
        features['peak_to_peak'] = np.ptp(signal_data)
        features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] > 0 else 0
        
        # Shape factors
        features['skewness'] = self._calculate_skewness(signal_data)
        features['kurtosis'] = self._calculate_kurtosis(signal_data)
        
        # Energy-based features
        features['energy'] = np.sum(signal_data**2)
        features['power'] = features['energy'] / len(signal_data)
        
        # Zero crossing rate
        features['zero_crossing_rate'] = self._zero_crossing_rate(signal_data)
        
        return features
    
    def extract_frequency_domain_features(self, signal_data: np.ndarray, 
                                        sampling_rate: Optional[int] = None) -> Dict[str, float]:
        """
        Extract frequency-domain features using FFT.
        
        Args:
            signal_data: Input signal array
            sampling_rate: Sampling rate (uses default if None)
            
        Returns:
            Dictionary of frequency-domain features
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
            
        # Compute FFT
        fft_values = fft(signal_data)
        fft_magnitude = np.abs(fft_values)
        frequencies = fftfreq(len(signal_data), 1/sampling_rate)
        
        # Only consider positive frequencies
        positive_freq_idx = frequencies > 0
        fft_magnitude = fft_magnitude[positive_freq_idx]
        frequencies = frequencies[positive_freq_idx]
        
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = np.sum(frequencies * fft_magnitude) / np.sum(fft_magnitude)
        
        # Spectral spread
        features['spectral_spread'] = np.sqrt(
            np.sum(((frequencies - features['spectral_centroid'])**2) * fft_magnitude) / 
            np.sum(fft_magnitude)
        )
        
        # Spectral rolloff (95% of energy)
        cumulative_energy = np.cumsum(fft_magnitude**2)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0]
        features['spectral_rolloff'] = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else frequencies[-1]
        
        # Peak frequency
        peak_idx = np.argmax(fft_magnitude)
        features['peak_frequency'] = frequencies[peak_idx]
        features['peak_magnitude'] = fft_magnitude[peak_idx]
        
        # Frequency bands energy
        features.update(self._frequency_band_energy(frequencies, fft_magnitude))
        
        return features
    
    def extract_time_frequency_features(self, signal_data: np.ndarray,
                                      sampling_rate: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Extract time-frequency features using Short-Time Fourier Transform.
        
        Args:
            signal_data: Input signal array
            sampling_rate: Sampling rate (uses default if None)
            
        Returns:
            Dictionary containing STFT results
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
            
        # Compute STFT
        window_length = min(256, len(signal_data) // 4)
        overlap = window_length // 2
        
        frequencies, times, stft_matrix = signal.stft(
            signal_data, 
            fs=sampling_rate,
            window='hann',
            nperseg=window_length,
            noverlap=overlap
        )
        
        stft_magnitude = np.abs(stft_matrix)
        
        features = {
            'stft_frequencies': frequencies,
            'stft_times': times,
            'stft_magnitude': stft_magnitude,
            'spectrogram': stft_magnitude**2
        }
        
        return features
    
    def compute_envelope_spectrum(self, signal_data: np.ndarray,
                                sampling_rate: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute envelope spectrum for bearing fault detection.
        
        Args:
            signal_data: Input signal array
            sampling_rate: Sampling rate (uses default if None)
            
        Returns:
            Tuple of (frequencies, envelope_spectrum)
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
            
        # High-pass filter to remove low-frequency components
        sos = signal.butter(4, 1000, btype='high', fs=sampling_rate, output='sos')
        filtered_signal = signal.sosfilt(sos, signal_data)
        
        # Compute envelope using Hilbert transform
        analytic_signal = signal.hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)
        
        # Compute FFT of envelope
        envelope_fft = fft(envelope)
        envelope_magnitude = np.abs(envelope_fft)
        frequencies = fftfreq(len(envelope), 1/sampling_rate)
        
        # Only positive frequencies
        positive_idx = frequencies > 0
        return frequencies[positive_idx], envelope_magnitude[positive_idx]
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of signal."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of signal."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3
    
    def _zero_crossing_rate(self, data: np.ndarray) -> float:
        """Calculate zero crossing rate."""
        zero_crossings = np.where(np.diff(np.signbit(data)))[0]
        return len(zero_crossings) / len(data)
    
    def _frequency_band_energy(self, frequencies: np.ndarray, 
                             magnitude: np.ndarray) -> Dict[str, float]:
        """Calculate energy in different frequency bands."""
        bands = {
            'low_freq_energy': (0, 1000),
            'mid_freq_energy': (1000, 5000),
            'high_freq_energy': (5000, np.inf)
        }
        
        band_energies = {}
        total_energy = np.sum(magnitude**2)
        
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (frequencies >= low_freq) & (frequencies < high_freq)
            band_energy = np.sum(magnitude[band_mask]**2)
            band_energies[band_name] = band_energy / total_energy if total_energy > 0 else 0
            
        return band_energies