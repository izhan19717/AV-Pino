"""
Physics-based feature extraction for motor fault diagnosis.

Computes electromagnetic, thermal, and mechanical features from raw signals
based on underlying physical principles.
"""

import numpy as np
from scipy import signal, integrate
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PhysicsFeatures:
    """Container for physics-based features."""
    electromagnetic: Dict[str, float]
    thermal: Dict[str, float]
    mechanical: Dict[str, float]
    coupling_terms: Dict[str, float]

class PhysicsFeatureExtractor:
    """
    Extract physics-informed features from motor signals.
    
    Computes features based on electromagnetic field theory, heat transfer,
    and mechanical vibration principles for motor fault diagnosis.
    """
    
    def __init__(self, sampling_rate: int = 12000, motor_params: Optional[Dict] = None):
        """
        Initialize physics feature extractor.
        
        Args:
            sampling_rate: Signal sampling rate in Hz
            motor_params: Motor physical parameters (optional)
        """
        self.sampling_rate = sampling_rate
        self.motor_params = motor_params or self._default_motor_params()
        
    def _default_motor_params(self) -> Dict:
        """Default motor parameters for feature extraction."""
        return {
            'pole_pairs': 2,
            'rotor_inertia': 0.01,  # kg⋅m²
            'stator_resistance': 1.5,  # Ω
            'rotor_resistance': 1.2,  # Ω
            'mutual_inductance': 0.3,  # H
            'bearing_stiffness': 1e8,  # N/m
            'damping_coefficient': 100,  # N⋅s/m
            'thermal_capacity': 500,  # J/K
            'thermal_conductivity': 50,  # W/(m⋅K)
        }
    
    def extract_all_features(self, vibration_signal: np.ndarray,
                           current_signal: Optional[np.ndarray] = None,
                           voltage_signal: Optional[np.ndarray] = None,
                           temperature_signal: Optional[np.ndarray] = None) -> PhysicsFeatures:
        """
        Extract all physics-based features from motor signals.
        
        Args:
            vibration_signal: Vibration measurement array
            current_signal: Motor current array (optional)
            voltage_signal: Motor voltage array (optional)
            temperature_signal: Temperature measurement array (optional)
            
        Returns:
            PhysicsFeatures object containing all extracted features
        """
        # Extract electromagnetic features
        em_features = self.extract_electromagnetic_features(
            vibration_signal, current_signal, voltage_signal
        )
        
        # Extract thermal features
        thermal_features = self.extract_thermal_features(
            vibration_signal, temperature_signal
        )
        
        # Extract mechanical features
        mechanical_features = self.extract_mechanical_features(vibration_signal)
        
        # Compute coupling terms
        coupling_features = self.compute_coupling_terms(
            vibration_signal, current_signal, temperature_signal
        )
        
        return PhysicsFeatures(
            electromagnetic=em_features,
            thermal=thermal_features,
            mechanical=mechanical_features,
            coupling_terms=coupling_features
        )
    
    def extract_electromagnetic_features(self, vibration_signal: np.ndarray,
                                       current_signal: Optional[np.ndarray] = None,
                                       voltage_signal: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract electromagnetic field-based features.
        
        Based on Maxwell's equations and motor electromagnetic theory.
        """
        features = {}
        
        # Magnetic flux density variations (from vibration)
        features['flux_density_variation'] = np.std(vibration_signal)
        features['flux_density_peak'] = np.max(np.abs(vibration_signal))
        
        # Electromagnetic force characteristics
        em_force = self._compute_electromagnetic_force(vibration_signal)
        features['em_force_rms'] = np.sqrt(np.mean(em_force**2))
        features['em_force_peak'] = np.max(np.abs(em_force))
        
        # Air gap variations (related to eccentricity faults)
        air_gap_variations = self._estimate_air_gap_variations(vibration_signal)
        features['air_gap_variation_std'] = np.std(air_gap_variations)
        features['air_gap_asymmetry'] = self._compute_air_gap_asymmetry(air_gap_variations)
        
        if current_signal is not None:
            # Current signature analysis
            features.update(self._current_signature_features(current_signal))
            
        if voltage_signal is not None and current_signal is not None:
            # Power factor and impedance features
            features.update(self._power_impedance_features(voltage_signal, current_signal))
            
        # Frequency domain electromagnetic features
        features.update(self._electromagnetic_frequency_features(vibration_signal))
        
        return features
    
    def extract_thermal_features(self, vibration_signal: np.ndarray,
                               temperature_signal: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract thermal dynamics features.
        
        Based on heat transfer equations and thermal modeling.
        """
        features = {}
        
        # Heat generation estimation from vibration (friction losses)
        heat_generation = self._estimate_heat_generation(vibration_signal)
        features['heat_generation_rate'] = np.mean(heat_generation)
        features['heat_generation_peak'] = np.max(heat_generation)
        features['heat_generation_std'] = np.std(heat_generation)
        
        # Thermal time constants
        thermal_response = self._compute_thermal_response(vibration_signal)
        features['thermal_time_constant'] = self._estimate_time_constant(thermal_response)
        
        # Temperature gradient estimation
        temp_gradient = self._estimate_temperature_gradient(vibration_signal)
        features['temperature_gradient_rms'] = np.sqrt(np.mean(temp_gradient**2))
        
        if temperature_signal is not None:
            # Direct temperature analysis
            features['temperature_mean'] = np.mean(temperature_signal)
            features['temperature_std'] = np.std(temperature_signal)
            features['temperature_trend'] = self._compute_temperature_trend(temperature_signal)
            
            # Thermal diffusivity estimation
            features['thermal_diffusivity'] = self._estimate_thermal_diffusivity(
                temperature_signal, vibration_signal
            )
        
        # Thermal stability indicators
        features['thermal_stability'] = self._compute_thermal_stability(vibration_signal)
        
        return features
    
    def extract_mechanical_features(self, vibration_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract mechanical dynamics features.
        
        Based on structural dynamics and vibration theory.
        """
        features = {}
        
        # Modal analysis features
        modal_features = self._modal_analysis(vibration_signal)
        features.update(modal_features)
        
        # Bearing fault indicators
        bearing_features = self._bearing_fault_indicators(vibration_signal)
        features.update(bearing_features)
        
        # Shaft dynamics
        shaft_features = self._shaft_dynamics_features(vibration_signal)
        features.update(shaft_features)
        
        # Structural stiffness estimation
        features['structural_stiffness'] = self._estimate_structural_stiffness(vibration_signal)
        
        # Damping characteristics
        features['damping_ratio'] = self._estimate_damping_ratio(vibration_signal)
        
        # Resonance detection
        resonance_features = self._detect_resonances(vibration_signal)
        features.update(resonance_features)
        
        return features
    
    def compute_coupling_terms(self, vibration_signal: np.ndarray,
                             current_signal: Optional[np.ndarray] = None,
                             temperature_signal: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute multi-physics coupling terms.
        
        Represents interactions between electromagnetic, thermal, and mechanical domains.
        """
        features = {}
        
        # Electromagnetic-mechanical coupling
        em_mech_coupling = self._electromagnetic_mechanical_coupling(vibration_signal)
        features['em_mech_coupling_strength'] = np.mean(np.abs(em_mech_coupling))
        features['em_mech_coupling_phase'] = np.angle(np.mean(em_mech_coupling))
        
        # Thermal-mechanical coupling
        thermal_mech_coupling = self._thermal_mechanical_coupling(vibration_signal)
        # Ensure arrays have same length for correlation
        min_len = min(len(vibration_signal), len(thermal_mech_coupling))
        if min_len > 1:
            features['thermal_mech_coupling'] = np.corrcoef(
                vibration_signal[:min_len], thermal_mech_coupling[:min_len]
            )[0, 1]
        else:
            features['thermal_mech_coupling'] = 0.0
        
        if current_signal is not None:
            # Electromagnetic-thermal coupling via current
            features['em_thermal_coupling'] = self._electromagnetic_thermal_coupling(
                vibration_signal, current_signal
            )
        
        # Energy transfer between domains
        features['energy_transfer_efficiency'] = self._compute_energy_transfer_efficiency(
            vibration_signal
        )
        
        return features
    
    def _compute_electromagnetic_force(self, vibration_signal: np.ndarray) -> np.ndarray:
        """Compute electromagnetic force from vibration signal."""
        # Simplified model: F_em ∝ d²x/dt² (Newton's second law)
        acceleration = np.gradient(np.gradient(vibration_signal))
        return self.motor_params['rotor_inertia'] * acceleration
    
    def _estimate_air_gap_variations(self, vibration_signal: np.ndarray) -> np.ndarray:
        """Estimate air gap variations from vibration."""
        # Air gap variations are related to radial vibrations
        # Apply high-pass filter to remove DC component
        sos = signal.butter(4, 10, btype='high', fs=self.sampling_rate, output='sos')
        return signal.sosfilt(sos, vibration_signal)
    
    def _compute_air_gap_asymmetry(self, air_gap_variations: np.ndarray) -> float:
        """Compute air gap asymmetry indicator."""
        # Asymmetry can be measured by the skewness of air gap variations
        mean_val = np.mean(air_gap_variations)
        std_val = np.std(air_gap_variations)
        if std_val == 0:
            return 0.0
        return np.mean(((air_gap_variations - mean_val) / std_val) ** 3)
    
    def _current_signature_features(self, current_signal: np.ndarray) -> Dict[str, float]:
        """Extract motor current signature analysis features."""
        features = {}
        
        # Current harmonics analysis
        fft_current = np.fft.fft(current_signal)
        freqs = np.fft.fftfreq(len(current_signal), 1/self.sampling_rate)
        
        # Find fundamental frequency (typically 50/60 Hz)
        fundamental_idx = np.argmax(np.abs(fft_current[1:len(fft_current)//2])) + 1
        fundamental_freq = freqs[fundamental_idx]
        
        features['fundamental_frequency'] = fundamental_freq
        features['fundamental_amplitude'] = np.abs(fft_current[fundamental_idx])
        
        # Harmonic distortion
        harmonics = []
        for h in range(2, 6):  # 2nd to 5th harmonics
            harmonic_idx = int(h * fundamental_idx)
            if harmonic_idx < len(fft_current)//2:
                harmonics.append(np.abs(fft_current[harmonic_idx]))
        
        features['total_harmonic_distortion'] = np.sqrt(np.sum(np.array(harmonics)**2)) / features['fundamental_amplitude']
        
        return features
    
    def _power_impedance_features(self, voltage_signal: np.ndarray, 
                                current_signal: np.ndarray) -> Dict[str, float]:
        """Extract power and impedance features."""
        features = {}
        
        # Instantaneous power
        power = voltage_signal * current_signal
        features['average_power'] = np.mean(power)
        features['power_factor'] = np.mean(power) / (np.sqrt(np.mean(voltage_signal**2)) * 
                                                    np.sqrt(np.mean(current_signal**2)))
        
        # Impedance estimation (simplified)
        features['impedance_magnitude'] = np.sqrt(np.mean(voltage_signal**2)) / np.sqrt(np.mean(current_signal**2))
        
        return features
    
    def _electromagnetic_frequency_features(self, vibration_signal: np.ndarray) -> Dict[str, float]:
        """Extract electromagnetic frequency domain features."""
        features = {}
        
        # Compute power spectral density
        freqs, psd = signal.welch(vibration_signal, fs=self.sampling_rate)
        
        # Electromagnetic characteristic frequencies
        pole_pass_freq = self.motor_params['pole_pairs'] * 50  # Assuming 50 Hz supply
        features['pole_pass_frequency_amplitude'] = self._get_frequency_amplitude(freqs, psd, pole_pass_freq)
        
        # Slot harmonics (simplified estimation)
        slot_freq = 12 * pole_pass_freq  # Typical slot number
        features['slot_harmonic_amplitude'] = self._get_frequency_amplitude(freqs, psd, slot_freq)
        
        return features
    
    def _estimate_heat_generation(self, vibration_signal: np.ndarray) -> np.ndarray:
        """Estimate heat generation from vibration (friction losses)."""
        # Heat generation ∝ velocity² (friction losses)
        velocity = np.gradient(vibration_signal) * self.sampling_rate
        return self.motor_params['damping_coefficient'] * velocity**2
    
    def _compute_thermal_response(self, vibration_signal: np.ndarray) -> np.ndarray:
        """Compute thermal response based on vibration."""
        heat_gen = self._estimate_heat_generation(vibration_signal)
        
        # Simple thermal model: C*dT/dt = Q_gen - Q_loss
        # Assuming Q_loss ∝ T, we get first-order response
        thermal_time_const = self.motor_params['thermal_capacity'] / self.motor_params['thermal_conductivity']
        
        # Simulate thermal response using exponential filter
        alpha = 1 / (1 + thermal_time_const * self.sampling_rate)
        thermal_response = np.zeros_like(heat_gen)
        thermal_response[0] = heat_gen[0]
        
        for i in range(1, len(heat_gen)):
            thermal_response[i] = alpha * heat_gen[i] + (1 - alpha) * thermal_response[i-1]
            
        return thermal_response
    
    def _estimate_time_constant(self, thermal_response: np.ndarray) -> float:
        """Estimate thermal time constant from response."""
        # Find time to reach 63% of final value
        final_value = thermal_response[-1]
        target_value = 0.63 * final_value
        
        idx = np.argmax(thermal_response >= target_value)
        return idx / self.sampling_rate if idx > 0 else 1.0
    
    def _estimate_temperature_gradient(self, vibration_signal: np.ndarray) -> np.ndarray:
        """Estimate temperature gradient from vibration."""
        # Temperature gradient related to heat flux
        heat_gen = self._estimate_heat_generation(vibration_signal)
        return np.gradient(heat_gen) / self.motor_params['thermal_conductivity']
    
    def _compute_temperature_trend(self, temperature_signal: np.ndarray) -> float:
        """Compute temperature trend (slope)."""
        time_vector = np.arange(len(temperature_signal)) / self.sampling_rate
        return np.polyfit(time_vector, temperature_signal, 1)[0]
    
    def _estimate_thermal_diffusivity(self, temperature_signal: np.ndarray,
                                    vibration_signal: np.ndarray) -> float:
        """Estimate thermal diffusivity from temperature and vibration."""
        # Simplified estimation based on correlation
        temp_gradient = np.gradient(temperature_signal)
        heat_gen = self._estimate_heat_generation(vibration_signal)
        
        # Ensure arrays have compatible lengths
        min_len = min(len(temp_gradient), len(heat_gen))
        if min_len > 1:
            correlation = np.corrcoef(temp_gradient[:min_len], heat_gen[:min_len])[0, 1]
        else:
            correlation = 0.0
            
        return abs(correlation) * self.motor_params['thermal_conductivity']
    
    def _compute_thermal_stability(self, vibration_signal: np.ndarray) -> float:
        """Compute thermal stability indicator."""
        thermal_response = self._compute_thermal_response(vibration_signal)
        return 1.0 / (1.0 + np.std(thermal_response))
    
    def _modal_analysis(self, vibration_signal: np.ndarray) -> Dict[str, float]:
        """Perform modal analysis on vibration signal."""
        features = {}
        
        # Compute power spectral density
        freqs, psd = signal.welch(vibration_signal, fs=self.sampling_rate)
        
        # Find dominant modes (peaks in PSD)
        peaks, properties = signal.find_peaks(psd, height=np.max(psd)*0.1, distance=10)
        
        if len(peaks) > 0:
            # First mode (dominant frequency)
            features['first_mode_frequency'] = freqs[peaks[0]]
            features['first_mode_amplitude'] = psd[peaks[0]]
            
            # Modal damping estimation (simplified)
            features['modal_damping'] = self._estimate_modal_damping(freqs, psd, peaks[0])
        else:
            features['first_mode_frequency'] = 0.0
            features['first_mode_amplitude'] = 0.0
            features['modal_damping'] = 0.0
            
        return features
    
    def _bearing_fault_indicators(self, vibration_signal: np.ndarray) -> Dict[str, float]:
        """Compute bearing fault indicators."""
        features = {}
        
        # Envelope analysis for bearing faults
        analytic_signal = signal.hilbert(vibration_signal)
        envelope = np.abs(analytic_signal)
        
        # Envelope spectrum
        envelope_fft = np.fft.fft(envelope)
        envelope_freqs = np.fft.fftfreq(len(envelope), 1/self.sampling_rate)
        
        # Bearing characteristic frequencies (simplified)
        shaft_freq = 30.0  # Hz, typical
        ball_pass_freq_outer = 3.5 * shaft_freq
        ball_pass_freq_inner = 5.4 * shaft_freq
        
        features['bpfo_amplitude'] = self._get_frequency_amplitude(
            envelope_freqs, np.abs(envelope_fft), ball_pass_freq_outer
        )
        features['bpfi_amplitude'] = self._get_frequency_amplitude(
            envelope_freqs, np.abs(envelope_fft), ball_pass_freq_inner
        )
        
        # Crest factor for impulsive faults
        features['envelope_crest_factor'] = np.max(envelope) / np.sqrt(np.mean(envelope**2))
        
        return features
    
    def _shaft_dynamics_features(self, vibration_signal: np.ndarray) -> Dict[str, float]:
        """Extract shaft dynamics features."""
        features = {}
        
        # Shaft unbalance indicators
        features['shaft_unbalance'] = self._compute_shaft_unbalance(vibration_signal)
        
        # Shaft misalignment
        features['shaft_misalignment'] = self._compute_shaft_misalignment(vibration_signal)
        
        return features
    
    def _estimate_structural_stiffness(self, vibration_signal: np.ndarray) -> float:
        """Estimate structural stiffness from vibration."""
        # Stiffness estimation from natural frequency
        freqs, psd = signal.welch(vibration_signal, fs=self.sampling_rate)
        dominant_freq_idx = np.argmax(psd)
        dominant_freq = freqs[dominant_freq_idx]
        
        # k = (2πf)²m (simplified single DOF system)
        return (2 * np.pi * dominant_freq)**2 * self.motor_params['rotor_inertia']
    
    def _estimate_damping_ratio(self, vibration_signal: np.ndarray) -> float:
        """Estimate damping ratio from vibration."""
        # Simplified estimation using logarithmic decrement
        peaks, _ = signal.find_peaks(vibration_signal)
        if len(peaks) < 2:
            return 0.1  # Default value
            
        # Find consecutive peaks
        peak_values = vibration_signal[peaks]
        if len(peak_values) < 2:
            return 0.1
            
        # Logarithmic decrement
        delta = np.log(peak_values[0] / peak_values[1]) if peak_values[1] != 0 else 0
        return delta / np.sqrt(4 * np.pi**2 + delta**2)
    
    def _detect_resonances(self, vibration_signal: np.ndarray) -> Dict[str, float]:
        """Detect resonance frequencies."""
        features = {}
        
        freqs, psd = signal.welch(vibration_signal, fs=self.sampling_rate)
        peaks, properties = signal.find_peaks(psd, height=np.max(psd)*0.2, distance=20)
        
        features['num_resonances'] = len(peaks)
        if len(peaks) > 0:
            features['primary_resonance_freq'] = freqs[peaks[0]]
            features['primary_resonance_amplitude'] = psd[peaks[0]]
        else:
            features['primary_resonance_freq'] = 0.0
            features['primary_resonance_amplitude'] = 0.0
            
        return features
    
    def _electromagnetic_mechanical_coupling(self, vibration_signal: np.ndarray) -> np.ndarray:
        """Compute electromagnetic-mechanical coupling."""
        # Coupling through electromagnetic force
        em_force = self._compute_electromagnetic_force(vibration_signal)
        
        # Phase relationship between force and displacement
        analytic_vibration = signal.hilbert(vibration_signal)
        analytic_force = signal.hilbert(em_force)
        
        return analytic_vibration * np.conj(analytic_force)
    
    def _thermal_mechanical_coupling(self, vibration_signal: np.ndarray) -> np.ndarray:
        """Compute thermal-mechanical coupling."""
        # Thermal expansion effects on mechanical response
        heat_gen = self._estimate_heat_generation(vibration_signal)
        thermal_expansion = integrate.cumtrapz(heat_gen, initial=0) / self.motor_params['thermal_capacity']
        
        return thermal_expansion
    
    def _electromagnetic_thermal_coupling(self, vibration_signal: np.ndarray,
                                        current_signal: np.ndarray) -> float:
        """Compute electromagnetic-thermal coupling."""
        # Joule heating from current
        joule_heating = self.motor_params['stator_resistance'] * current_signal**2
        
        # Mechanical losses from vibration
        mechanical_losses = self._estimate_heat_generation(vibration_signal)
        
        # Correlation between electromagnetic and mechanical heating
        return np.corrcoef(joule_heating, mechanical_losses)[0, 1]
    
    def _compute_energy_transfer_efficiency(self, vibration_signal: np.ndarray) -> float:
        """Compute energy transfer efficiency between domains."""
        # Kinetic energy
        velocity = np.gradient(vibration_signal) * self.sampling_rate
        kinetic_energy = 0.5 * self.motor_params['rotor_inertia'] * velocity**2
        
        # Potential energy (elastic)
        potential_energy = 0.5 * self.motor_params['bearing_stiffness'] * vibration_signal**2
        
        # Total mechanical energy
        total_energy = kinetic_energy + potential_energy
        
        # Energy dissipation
        dissipated_energy = self._estimate_heat_generation(vibration_signal)
        
        # Efficiency as ratio of useful energy to total energy
        return np.mean(total_energy) / (np.mean(total_energy) + np.mean(dissipated_energy))
    
    def _get_frequency_amplitude(self, freqs: np.ndarray, spectrum: np.ndarray, 
                               target_freq: float, tolerance: float = 2.0) -> float:
        """Get amplitude at specific frequency with tolerance."""
        freq_mask = np.abs(freqs - target_freq) <= tolerance
        if np.any(freq_mask):
            return np.max(spectrum[freq_mask])
        return 0.0
    
    def _estimate_modal_damping(self, freqs: np.ndarray, psd: np.ndarray, peak_idx: int) -> float:
        """Estimate modal damping from PSD peak."""
        # Half-power bandwidth method
        peak_value = psd[peak_idx]
        half_power = peak_value / 2
        
        # Find frequencies at half power
        left_idx = peak_idx
        right_idx = peak_idx
        
        while left_idx > 0 and psd[left_idx] > half_power:
            left_idx -= 1
        while right_idx < len(psd)-1 and psd[right_idx] > half_power:
            right_idx += 1
            
        if right_idx > left_idx:
            bandwidth = freqs[right_idx] - freqs[left_idx]
            natural_freq = freqs[peak_idx]
            return bandwidth / (2 * natural_freq) if natural_freq > 0 else 0.1
        
        return 0.1  # Default damping ratio
    
    def _compute_shaft_unbalance(self, vibration_signal: np.ndarray) -> float:
        """Compute shaft unbalance indicator."""
        # Unbalance typically shows up at 1x running speed
        freqs, psd = signal.welch(vibration_signal, fs=self.sampling_rate)
        
        # Assume running speed around 30 Hz (1800 RPM)
        running_speed = 30.0
        unbalance_amplitude = self._get_frequency_amplitude(freqs, psd, running_speed)
        
        return unbalance_amplitude
    
    def _compute_shaft_misalignment(self, vibration_signal: np.ndarray) -> float:
        """Compute shaft misalignment indicator."""
        # Misalignment typically shows up at 2x running speed
        freqs, psd = signal.welch(vibration_signal, fs=self.sampling_rate)
        
        # 2x running speed
        misalignment_freq = 60.0
        misalignment_amplitude = self._get_frequency_amplitude(freqs, psd, misalignment_freq)
        
        return misalignment_amplitude