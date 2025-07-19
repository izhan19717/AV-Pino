"""
Visualization tools for physics-based features and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from .feature_extractor import PhysicsFeatures
import seaborn as sns

class PhysicsVisualizer:
    """
    Visualization tools for physics-based motor fault diagnosis features.
    
    Provides comprehensive plotting capabilities for electromagnetic, thermal,
    and mechanical features extracted from motor signals.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize physics visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        
    def plot_physics_features_overview(self, features: PhysicsFeatures, 
                                     title: str = "Physics Features Overview") -> plt.Figure:
        """
        Create overview plot of all physics features.
        
        Args:
            features: PhysicsFeatures object
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Electromagnetic features
        self._plot_feature_dict(features.electromagnetic, axes[0, 0], 
                               "Electromagnetic Features", color='blue')
        
        # Thermal features
        self._plot_feature_dict(features.thermal, axes[0, 1], 
                               "Thermal Features", color='red')
        
        # Mechanical features
        self._plot_feature_dict(features.mechanical, axes[1, 0], 
                               "Mechanical Features", color='green')
        
        # Coupling features
        self._plot_feature_dict(features.coupling_terms, axes[1, 1], 
                               "Multi-Physics Coupling", color='purple')
        
        plt.tight_layout()
        return fig
    
    def plot_electromagnetic_analysis(self, vibration_signal: np.ndarray,
                                    current_signal: Optional[np.ndarray] = None,
                                    sampling_rate: int = 12000,
                                    title: str = "Electromagnetic Analysis") -> plt.Figure:
        """
        Create detailed electromagnetic analysis plots.
        
        Args:
            vibration_signal: Vibration signal array
            current_signal: Motor current signal (optional)
            sampling_rate: Sampling rate in Hz
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Time domain vibration
        time = np.arange(len(vibration_signal)) / sampling_rate
        axes[0, 0].plot(time, vibration_signal, 'b-', linewidth=0.8)
        axes[0, 0].set_title('Vibration Signal')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Vibration spectrum
        freqs, psd = self._compute_psd(vibration_signal, sampling_rate)
        axes[0, 1].semilogy(freqs, psd, 'b-', linewidth=1)
        axes[0, 1].set_title('Vibration Spectrum')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('PSD')
        axes[0, 1].grid(True, alpha=0.3)
        
        if current_signal is not None:
            # Current signal
            axes[1, 0].plot(time[:len(current_signal)], current_signal, 'r-', linewidth=0.8)
            axes[1, 0].set_title('Motor Current')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Current (A)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Current spectrum
            current_freqs, current_psd = self._compute_psd(current_signal, sampling_rate)
            axes[1, 1].semilogy(current_freqs, current_psd, 'r-', linewidth=1)
            axes[1, 1].set_title('Current Spectrum')
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('PSD')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Air gap variations
            air_gap = self._estimate_air_gap_variations(vibration_signal, sampling_rate)
            axes[1, 0].plot(time[:len(air_gap)], air_gap, 'g-', linewidth=0.8)
            axes[1, 0].set_title('Air Gap Variations')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Variation')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Electromagnetic force
            em_force = self._compute_electromagnetic_force(vibration_signal, sampling_rate)
            axes[1, 1].plot(time[:len(em_force)], em_force, 'm-', linewidth=0.8)
            axes[1, 1].set_title('Electromagnetic Force')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Force (N)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_thermal_analysis(self, vibration_signal: np.ndarray,
                            temperature_signal: Optional[np.ndarray] = None,
                            sampling_rate: int = 12000,
                            title: str = "Thermal Analysis") -> plt.Figure:
        """
        Create detailed thermal analysis plots.
        
        Args:
            vibration_signal: Vibration signal array
            temperature_signal: Temperature signal (optional)
            sampling_rate: Sampling rate in Hz
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        time = np.arange(len(vibration_signal)) / sampling_rate
        
        # Heat generation estimation
        heat_gen = self._estimate_heat_generation(vibration_signal, sampling_rate)
        axes[0, 0].plot(time[:len(heat_gen)], heat_gen, 'r-', linewidth=1)
        axes[0, 0].set_title('Heat Generation Rate')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Heat Rate (W)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Thermal response
        thermal_response = self._compute_thermal_response(vibration_signal, sampling_rate)
        axes[0, 1].plot(time[:len(thermal_response)], thermal_response, 'orange', linewidth=1)
        axes[0, 1].set_title('Thermal Response')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Temperature Rise (K)')
        axes[0, 1].grid(True, alpha=0.3)
        
        if temperature_signal is not None:
            # Actual temperature
            temp_time = np.arange(len(temperature_signal)) / sampling_rate
            axes[1, 0].plot(temp_time, temperature_signal, 'red', linewidth=1)
            axes[1, 0].set_title('Measured Temperature')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Temperature (°C)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Temperature gradient
            temp_gradient = np.gradient(temperature_signal)
            axes[1, 1].plot(temp_time[:-1], temp_gradient[:-1], 'darkred', linewidth=1)
            axes[1, 1].set_title('Temperature Gradient')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('dT/dt (°C/s)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Temperature gradient estimation
            temp_gradient = self._estimate_temperature_gradient(vibration_signal, sampling_rate)
            axes[1, 0].plot(time[:len(temp_gradient)], temp_gradient, 'darkred', linewidth=1)
            axes[1, 0].set_title('Estimated Temperature Gradient')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Gradient (K/m)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Thermal stability
            stability = self._compute_thermal_stability_evolution(vibration_signal, sampling_rate)
            axes[1, 1].plot(time[:len(stability)], stability, 'brown', linewidth=1)
            axes[1, 1].set_title('Thermal Stability')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Stability Index')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_mechanical_analysis(self, vibration_signal: np.ndarray,
                               sampling_rate: int = 12000,
                               title: str = "Mechanical Analysis") -> plt.Figure:
        """
        Create detailed mechanical analysis plots.
        
        Args:
            vibration_signal: Vibration signal array
            sampling_rate: Sampling rate in Hz
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        time = np.arange(len(vibration_signal)) / sampling_rate
        
        # Original vibration signal
        axes[0, 0].plot(time, vibration_signal, 'g-', linewidth=0.8)
        axes[0, 0].set_title('Vibration Signal')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Displacement')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Envelope analysis
        envelope = self._compute_envelope(vibration_signal)
        axes[0, 1].plot(time, vibration_signal, 'g-', alpha=0.5, linewidth=0.5, label='Signal')
        axes[0, 1].plot(time, envelope, 'r-', linewidth=1.5, label='Envelope')
        axes[0, 1].plot(time, -envelope, 'r-', linewidth=1.5)
        axes[0, 1].set_title('Envelope Analysis')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Frequency spectrum with bearing fault frequencies
        freqs, psd = self._compute_psd(vibration_signal, sampling_rate)
        axes[1, 0].semilogy(freqs, psd, 'g-', linewidth=1)
        
        # Mark characteristic frequencies
        shaft_freq = 30.0  # Hz
        bpfo = 3.5 * shaft_freq
        bpfi = 5.4 * shaft_freq
        
        axes[1, 0].axvline(x=bpfo, color='red', linestyle='--', alpha=0.7, label=f'BPFO: {bpfo:.1f} Hz')
        axes[1, 0].axvline(x=bpfi, color='blue', linestyle='--', alpha=0.7, label=f'BPFI: {bpfi:.1f} Hz')
        axes[1, 0].set_title('Vibration Spectrum')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('PSD')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Envelope spectrum
        envelope_freqs, envelope_psd = self._compute_psd(envelope, sampling_rate)
        axes[1, 1].semilogy(envelope_freqs, envelope_psd, 'r-', linewidth=1)
        axes[1, 1].axvline(x=bpfo, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].axvline(x=bpfi, color='blue', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Envelope Spectrum')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('PSD')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_coupling_analysis(self, vibration_signal: np.ndarray,
                             current_signal: Optional[np.ndarray] = None,
                             temperature_signal: Optional[np.ndarray] = None,
                             sampling_rate: int = 12000,
                             title: str = "Multi-Physics Coupling Analysis") -> plt.Figure:
        """
        Create multi-physics coupling analysis plots.
        
        Args:
            vibration_signal: Vibration signal array
            current_signal: Motor current signal (optional)
            temperature_signal: Temperature signal (optional)
            sampling_rate: Sampling rate in Hz
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        time = np.arange(len(vibration_signal)) / sampling_rate
        
        # Electromagnetic-mechanical coupling
        em_mech_coupling = self._compute_em_mech_coupling(vibration_signal, sampling_rate)
        axes[0, 0].plot(time[:len(em_mech_coupling)], np.real(em_mech_coupling), 'purple', linewidth=1)
        axes[0, 0].set_title('EM-Mechanical Coupling (Real)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Coupling Strength')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Thermal-mechanical coupling
        thermal_mech_coupling = self._compute_thermal_mech_coupling(vibration_signal, sampling_rate)
        axes[0, 1].plot(time[:len(thermal_mech_coupling)], thermal_mech_coupling, 'orange', linewidth=1)
        axes[0, 1].set_title('Thermal-Mechanical Coupling')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Thermal Expansion')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy distribution
        kinetic_energy = self._compute_kinetic_energy(vibration_signal, sampling_rate)
        potential_energy = self._compute_potential_energy(vibration_signal)
        
        axes[1, 0].plot(time[:len(kinetic_energy)], kinetic_energy, 'blue', linewidth=1, label='Kinetic')
        axes[1, 0].plot(time[:len(potential_energy)], potential_energy, 'red', linewidth=1, label='Potential')
        axes[1, 0].set_title('Energy Distribution')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Energy (J)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Phase relationships
        if current_signal is not None:
            # Cross-correlation between vibration and current
            correlation = np.correlate(vibration_signal, current_signal[:len(vibration_signal)], mode='same')
            lags = np.arange(-len(correlation)//2, len(correlation)//2) / sampling_rate
            axes[1, 1].plot(lags, correlation, 'green', linewidth=1)
            axes[1, 1].set_title('Vibration-Current Cross-Correlation')
            axes[1, 1].set_xlabel('Lag (s)')
            axes[1, 1].set_ylabel('Correlation')
        else:
            # Phase portrait (velocity vs displacement)
            velocity = np.gradient(vibration_signal) * sampling_rate
            axes[1, 1].plot(vibration_signal[:-1], velocity[:-1], 'green', alpha=0.7, linewidth=0.5)
            axes[1, 1].set_title('Phase Portrait')
            axes[1, 1].set_xlabel('Displacement')
            axes[1, 1].set_ylabel('Velocity')
        
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_comparison(self, features_list: List[PhysicsFeatures],
                              labels: List[str],
                              title: str = "Physics Features Comparison") -> plt.Figure:
        """
        Compare physics features across multiple samples.
        
        Args:
            features_list: List of PhysicsFeatures objects
            labels: Labels for each feature set
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        # Combine all features into a single dictionary for each sample
        all_features = []
        for features in features_list:
            combined = {}
            combined.update({f"EM_{k}": v for k, v in features.electromagnetic.items()})
            combined.update({f"Thermal_{k}": v for k, v in features.thermal.items()})
            combined.update({f"Mech_{k}": v for k, v in features.mechanical.items()})
            combined.update({f"Coupling_{k}": v for k, v in features.coupling_terms.items()})
            all_features.append(combined)
        
        # Create comparison plots
        feature_names = list(all_features[0].keys())
        n_features = len(feature_names)
        
        # Select top features for visualization (limit to avoid clutter)
        max_features = 16
        if n_features > max_features:
            # Select features with highest variance across samples
            variances = []
            for feature_name in feature_names:
                values = [features[feature_name] for features in all_features]
                variances.append(np.var(values))
            
            top_indices = np.argsort(variances)[-max_features:]
            selected_features = [feature_names[i] for i in top_indices]
        else:
            selected_features = feature_names
        
        n_cols = 4
        n_rows = (len(selected_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        fig.suptitle(title, fontsize=16)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature_name in enumerate(selected_features):
            row = i // n_cols
            col = i % n_cols
            
            values = [features[feature_name] for features in all_features]
            
            axes[row, col].bar(labels, values, alpha=0.7)
            axes[row, col].set_title(feature_name.replace('_', ' '), fontsize=10)
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(selected_features), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _plot_feature_dict(self, feature_dict: Dict[str, float], ax: plt.Axes,
                          title: str, color: str = 'blue') -> None:
        """Plot features from dictionary as bar chart."""
        if not feature_dict:
            ax.text(0.5, 0.5, 'No features available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
            
        names = list(feature_dict.keys())
        values = list(feature_dict.values())
        
        # Limit number of features shown
        if len(names) > 10:
            # Show top 10 by absolute value
            sorted_indices = np.argsort([abs(v) for v in values])[-10:]
            names = [names[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
        
        bars = ax.bar(range(len(names)), values, color=color, alpha=0.7)
        ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([name.replace('_', '\n') for name in names], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2e}' if abs(value) < 0.01 else f'{value:.2f}',
                   ha='center', va='bottom', fontsize=7)
    
    def _compute_psd(self, signal: np.ndarray, sampling_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density."""
        from scipy import signal as scipy_signal
        freqs, psd = scipy_signal.welch(signal, fs=sampling_rate, nperseg=min(1024, len(signal)//4))
        return freqs, psd
    
    def _estimate_air_gap_variations(self, vibration_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Estimate air gap variations from vibration."""
        from scipy import signal as scipy_signal
        sos = scipy_signal.butter(4, 10, btype='high', fs=sampling_rate, output='sos')
        return scipy_signal.sosfilt(sos, vibration_signal)
    
    def _compute_electromagnetic_force(self, vibration_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Compute electromagnetic force from vibration signal."""
        acceleration = np.gradient(np.gradient(vibration_signal)) * sampling_rate**2
        rotor_inertia = 0.01  # kg⋅m²
        return rotor_inertia * acceleration
    
    def _estimate_heat_generation(self, vibration_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Estimate heat generation from vibration."""
        velocity = np.gradient(vibration_signal) * sampling_rate
        damping_coefficient = 100  # N⋅s/m
        return damping_coefficient * velocity**2
    
    def _compute_thermal_response(self, vibration_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Compute thermal response."""
        heat_gen = self._estimate_heat_generation(vibration_signal, sampling_rate)
        thermal_time_const = 5.0  # seconds
        alpha = 1 / (1 + thermal_time_const * sampling_rate)
        
        thermal_response = np.zeros_like(heat_gen)
        thermal_response[0] = heat_gen[0]
        
        for i in range(1, len(heat_gen)):
            thermal_response[i] = alpha * heat_gen[i] + (1 - alpha) * thermal_response[i-1]
            
        return thermal_response
    
    def _estimate_temperature_gradient(self, vibration_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Estimate temperature gradient."""
        heat_gen = self._estimate_heat_generation(vibration_signal, sampling_rate)
        thermal_conductivity = 50  # W/(m⋅K)
        return np.gradient(heat_gen) / thermal_conductivity
    
    def _compute_thermal_stability_evolution(self, vibration_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Compute thermal stability evolution."""
        window_size = min(1000, len(vibration_signal) // 10)
        stability = np.zeros(len(vibration_signal) - window_size + 1)
        
        for i in range(len(stability)):
            window = vibration_signal[i:i+window_size]
            thermal_response = self._compute_thermal_response(window, sampling_rate)
            stability[i] = 1.0 / (1.0 + np.std(thermal_response))
            
        return stability
    
    def _compute_envelope(self, signal: np.ndarray) -> np.ndarray:
        """Compute signal envelope using Hilbert transform."""
        from scipy import signal as scipy_signal
        analytic_signal = scipy_signal.hilbert(signal)
        return np.abs(analytic_signal)
    
    def _compute_em_mech_coupling(self, vibration_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Compute electromagnetic-mechanical coupling."""
        from scipy import signal as scipy_signal
        em_force = self._compute_electromagnetic_force(vibration_signal, sampling_rate)
        
        analytic_vibration = scipy_signal.hilbert(vibration_signal)
        analytic_force = scipy_signal.hilbert(em_force)
        
        return analytic_vibration * np.conj(analytic_force)
    
    def _compute_thermal_mech_coupling(self, vibration_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Compute thermal-mechanical coupling."""
        from scipy import integrate
        heat_gen = self._estimate_heat_generation(vibration_signal, sampling_rate)
        thermal_capacity = 500  # J/K
        thermal_expansion = integrate.cumtrapz(heat_gen, initial=0) / thermal_capacity
        return thermal_expansion
    
    def _compute_kinetic_energy(self, vibration_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Compute kinetic energy."""
        velocity = np.gradient(vibration_signal) * sampling_rate
        rotor_inertia = 0.01  # kg⋅m²
        return 0.5 * rotor_inertia * velocity**2
    
    def _compute_potential_energy(self, vibration_signal: np.ndarray) -> np.ndarray:
        """Compute potential energy."""
        bearing_stiffness = 1e8  # N/m
        return 0.5 * bearing_stiffness * vibration_signal**2