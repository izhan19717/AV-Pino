"""
Physics consistency analysis plots showing PDE residuals and constraint violations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd

@dataclass
class PhysicsResiduals:
    """Container for physics constraint residuals."""
    maxwell_residual: np.ndarray
    heat_equation_residual: np.ndarray
    structural_dynamics_residual: np.ndarray
    coupling_residual: np.ndarray
    timestamps: np.ndarray
    constraint_weights: Optional[Dict[str, float]] = None

@dataclass
class PDEConstraintViolations:
    """Container for PDE constraint violation analysis."""
    violation_magnitudes: Dict[str, np.ndarray]
    violation_frequencies: Dict[str, np.ndarray]
    critical_violations: Dict[str, List[int]]  # Indices of critical violations
    conservation_errors: Dict[str, np.ndarray]

class PhysicsAnalysisVisualizer:
    """
    Visualization tools for physics consistency analysis and PDE residuals.
    
    Provides comprehensive plotting capabilities for analyzing physics constraint
    satisfaction, PDE residuals, conservation law violations, and multi-physics
    coupling consistency.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10), style: str = 'seaborn-v0_8'):
        """
        Initialize physics analysis visualizer.
        
        Args:
            figsize: Default figure size for plots
            style: Matplotlib style to use
        """
        self.figsize = figsize
        plt.style.use(style)
        self.constraint_colors = {
            'Maxwell': '#FF6B6B',
            'Heat': '#4ECDC4', 
            'Structural': '#45B7D1',
            'Coupling': '#96CEB4',
            'Conservation': '#FFEAA7'
        }
        
    def plot_pde_residuals_overview(self,
                                  residuals: PhysicsResiduals,
                                  title: str = "PDE Residuals Overview") -> plt.Figure:
        """
        Plot overview of all PDE residuals over time.
        
        Args:
            residuals: PhysicsResiduals object containing residual data
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Maxwell equations residual
        axes[0, 0].plot(residuals.timestamps, residuals.maxwell_residual, 
                       color=self.constraint_colors['Maxwell'], linewidth=1.5)
        axes[0, 0].fill_between(residuals.timestamps, 0, residuals.maxwell_residual,
                               alpha=0.3, color=self.constraint_colors['Maxwell'])
        axes[0, 0].set_title('Maxwell Equations Residual')
        axes[0, 0].set_ylabel('Residual Magnitude')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Heat equation residual
        axes[0, 1].plot(residuals.timestamps, residuals.heat_equation_residual,
                       color=self.constraint_colors['Heat'], linewidth=1.5)
        axes[0, 1].fill_between(residuals.timestamps, 0, residuals.heat_equation_residual,
                               alpha=0.3, color=self.constraint_colors['Heat'])
        axes[0, 1].set_title('Heat Equation Residual')
        axes[0, 1].set_ylabel('Residual Magnitude')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Structural dynamics residual
        axes[1, 0].plot(residuals.timestamps, residuals.structural_dynamics_residual,
                       color=self.constraint_colors['Structural'], linewidth=1.5)
        axes[1, 0].fill_between(residuals.timestamps, 0, residuals.structural_dynamics_residual,
                               alpha=0.3, color=self.constraint_colors['Structural'])
        axes[1, 0].set_title('Structural Dynamics Residual')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Residual Magnitude')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Multi-physics coupling residual
        axes[1, 1].plot(residuals.timestamps, residuals.coupling_residual,
                       color=self.constraint_colors['Coupling'], linewidth=1.5)
        axes[1, 1].fill_between(residuals.timestamps, 0, residuals.coupling_residual,
                               alpha=0.3, color=self.constraint_colors['Coupling'])
        axes[1, 1].set_title('Multi-Physics Coupling Residual')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Residual Magnitude')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def plot_constraint_violation_analysis(self,
                                         violations: PDEConstraintViolations,
                                         title: str = "Constraint Violation Analysis") -> plt.Figure:
        """
        Plot detailed analysis of constraint violations.
        
        Args:
            violations: PDEConstraintViolations object
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16)
        
        constraint_names = list(violations.violation_magnitudes.keys())
        
        # Violation magnitude distributions
        for i, constraint in enumerate(constraint_names[:3]):
            row = 0
            col = i
            
            magnitudes = violations.violation_magnitudes[constraint]
            axes[row, col].hist(magnitudes, bins=30, alpha=0.7, 
                              color=self.constraint_colors.get(constraint, '#808080'),
                              edgecolor='black')
            axes[row, col].axvline(np.mean(magnitudes), color='red', linestyle='--',
                                 linewidth=2, label=f'Mean: {np.mean(magnitudes):.2e}')
            axes[row, col].axvline(np.median(magnitudes), color='green', linestyle='--',
                                 linewidth=2, label=f'Median: {np.median(magnitudes):.2e}')
            axes[row, col].set_title(f'{constraint} Violation Magnitudes')
            axes[row, col].set_xlabel('Violation Magnitude')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_yscale('log')
        
        # Violation frequency analysis
        for i, constraint in enumerate(constraint_names[:3]):
            row = 1
            col = i
            
            frequencies = violations.violation_frequencies[constraint]
            time_points = np.arange(len(frequencies))
            
            axes[row, col].plot(time_points, frequencies, 
                              color=self.constraint_colors.get(constraint, '#808080'),
                              linewidth=1.5)
            
            # Mark critical violations
            if constraint in violations.critical_violations:
                critical_indices = violations.critical_violations[constraint]
                if critical_indices:
                    axes[row, col].scatter(critical_indices, 
                                         [frequencies[idx] for idx in critical_indices],
                                         color='red', s=50, marker='x', linewidth=3,
                                         label=f'Critical ({len(critical_indices)})')
            
            axes[row, col].set_title(f'{constraint} Violation Frequency')
            axes[row, col].set_xlabel('Time Step')
            axes[row, col].set_ylabel('Violation Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_conservation_law_analysis(self,
                                     violations: PDEConstraintViolations,
                                     title: str = "Conservation Law Analysis") -> plt.Figure:
        """
        Plot analysis of conservation law violations.
        
        Args:
            violations: PDEConstraintViolations object
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        conservation_laws = list(violations.conservation_errors.keys())
        
        if len(conservation_laws) >= 1:
            # Energy conservation
            energy_errors = violations.conservation_errors.get('Energy', np.array([]))
            if len(energy_errors) > 0:
                time_points = np.arange(len(energy_errors))
                axes[0, 0].plot(time_points, energy_errors, 'r-', linewidth=1.5)
                axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[0, 0].fill_between(time_points, 0, energy_errors, alpha=0.3, color='red')
                axes[0, 0].set_title('Energy Conservation Error')
                axes[0, 0].set_ylabel('Energy Error')
                axes[0, 0].grid(True, alpha=0.3)
        
        if len(conservation_laws) >= 2:
            # Momentum conservation
            momentum_errors = violations.conservation_errors.get('Momentum', np.array([]))
            if len(momentum_errors) > 0:
                time_points = np.arange(len(momentum_errors))
                axes[0, 1].plot(time_points, momentum_errors, 'b-', linewidth=1.5)
                axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[0, 1].fill_between(time_points, 0, momentum_errors, alpha=0.3, color='blue')
                axes[0, 1].set_title('Momentum Conservation Error')
                axes[0, 1].set_ylabel('Momentum Error')
                axes[0, 1].grid(True, alpha=0.3)
        
        if len(conservation_laws) >= 3:
            # Charge conservation
            charge_errors = violations.conservation_errors.get('Charge', np.array([]))
            if len(charge_errors) > 0:
                time_points = np.arange(len(charge_errors))
                axes[1, 0].plot(time_points, charge_errors, 'g-', linewidth=1.5)
                axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[1, 0].fill_between(time_points, 0, charge_errors, alpha=0.3, color='green')
                axes[1, 0].set_title('Charge Conservation Error')
                axes[1, 0].set_xlabel('Time Step')
                axes[1, 0].set_ylabel('Charge Error')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Overall conservation summary
        if conservation_laws:
            all_errors = []
            labels = []
            colors = []
            
            for law in conservation_laws:
                errors = violations.conservation_errors[law]
                if len(errors) > 0:
                    all_errors.append(np.abs(errors))
                    labels.append(law)
                    colors.append(self.constraint_colors.get('Conservation', '#FFEAA7'))
            
            if all_errors:
                bp = axes[1, 1].boxplot(all_errors, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                axes[1, 1].set_title('Conservation Error Summary')
                axes[1, 1].set_xlabel('Conservation Law')
                axes[1, 1].set_ylabel('Absolute Error')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True, alpha=0.3)
        
        # Fill empty subplots with informative text
        for i in range(2):
            for j in range(2):
                if not axes[i, j].has_data():
                    axes[i, j].text(0.5, 0.5, 'No data available\nfor this conservation law',
                                   ha='center', va='center', transform=axes[i, j].transAxes,
                                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        return fig
    
    def plot_physics_consistency_evolution(self,
                                         residuals: PhysicsResiduals,
                                         window_size: int = 100,
                                         title: str = "Physics Consistency Evolution") -> plt.Figure:
        """
        Plot evolution of physics consistency over time using moving averages.
        
        Args:
            residuals: PhysicsResiduals object
            window_size: Window size for moving average
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        fig.suptitle(title, fontsize=16)
        
        # Compute moving averages
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Individual constraint evolution
        constraints = {
            'Maxwell': residuals.maxwell_residual,
            'Heat': residuals.heat_equation_residual,
            'Structural': residuals.structural_dynamics_residual,
            'Coupling': residuals.coupling_residual
        }
        
        for name, residual in constraints.items():
            if len(residual) >= window_size:
                smoothed = moving_average(residual, window_size)
                time_smoothed = residuals.timestamps[:len(smoothed)]
                
                axes[0].plot(time_smoothed, smoothed, 
                           color=self.constraint_colors.get(name, '#808080'),
                           linewidth=2, label=name)
        
        axes[0].set_title('Individual Constraint Residuals (Smoothed)')
        axes[0].set_ylabel('Residual Magnitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Combined physics consistency metric
        combined_residual = (residuals.maxwell_residual + 
                           residuals.heat_equation_residual +
                           residuals.structural_dynamics_residual +
                           residuals.coupling_residual) / 4
        
        if len(combined_residual) >= window_size:
            combined_smoothed = moving_average(combined_residual, window_size)
            time_combined = residuals.timestamps[:len(combined_smoothed)]
            
            axes[1].plot(time_combined, combined_smoothed, 'purple', linewidth=2)
            axes[1].fill_between(time_combined, 0, combined_smoothed, alpha=0.3, color='purple')
        
        axes[1].set_title('Combined Physics Consistency Metric')
        axes[1].set_ylabel('Combined Residual')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        # Physics consistency score (inverse of residual)
        consistency_score = 1.0 / (1.0 + combined_residual)
        
        if len(consistency_score) >= window_size:
            score_smoothed = moving_average(consistency_score, window_size)
            time_score = residuals.timestamps[:len(score_smoothed)]
            
            axes[2].plot(time_score, score_smoothed, 'green', linewidth=2)
            axes[2].fill_between(time_score, 0, score_smoothed, alpha=0.3, color='green')
            axes[2].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='High Consistency')
            axes[2].axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Medium Consistency')
        
        axes[2].set_title('Physics Consistency Score')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Consistency Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def plot_multiphysics_coupling_analysis(self,
                                          electromagnetic_field: np.ndarray,
                                          thermal_field: np.ndarray,
                                          mechanical_displacement: np.ndarray,
                                          timestamps: np.ndarray,
                                          title: str = "Multi-Physics Coupling Analysis") -> plt.Figure:
        """
        Plot analysis of multi-physics coupling interactions.
        
        Args:
            electromagnetic_field: Electromagnetic field values
            thermal_field: Thermal field values
            mechanical_displacement: Mechanical displacement values
            timestamps: Time points
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16)
        
        # Individual field evolution
        axes[0, 0].plot(timestamps, electromagnetic_field, 'r-', linewidth=1.5, label='EM Field')
        axes[0, 0].set_title('Electromagnetic Field')
        axes[0, 0].set_ylabel('Field Strength')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(timestamps, thermal_field, 'orange', linewidth=1.5, label='Thermal')
        axes[0, 1].set_title('Thermal Field')
        axes[0, 1].set_ylabel('Temperature')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(timestamps, mechanical_displacement, 'b-', linewidth=1.5, label='Mechanical')
        axes[0, 2].set_title('Mechanical Displacement')
        axes[0, 2].set_ylabel('Displacement')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Cross-correlation analysis
        # EM-Thermal coupling
        em_thermal_corr = np.correlate(electromagnetic_field, thermal_field, mode='same')
        lags = np.arange(-len(em_thermal_corr)//2, len(em_thermal_corr)//2)
        axes[1, 0].plot(lags, em_thermal_corr, 'purple', linewidth=1.5)
        axes[1, 0].set_title('EM-Thermal Cross-Correlation')
        axes[1, 0].set_xlabel('Lag')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Thermal-Mechanical coupling
        thermal_mech_corr = np.correlate(thermal_field, mechanical_displacement, mode='same')
        axes[1, 1].plot(lags, thermal_mech_corr, 'green', linewidth=1.5)
        axes[1, 1].set_title('Thermal-Mechanical Cross-Correlation')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # EM-Mechanical coupling
        em_mech_corr = np.correlate(electromagnetic_field, mechanical_displacement, mode='same')
        axes[1, 2].plot(lags, em_mech_corr, 'brown', linewidth=1.5)
        axes[1, 2].set_title('EM-Mechanical Cross-Correlation')
        axes[1, 2].set_xlabel('Lag')
        axes[1, 2].set_ylabel('Correlation')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_pde_residual_statistics(self,
                                   residuals: PhysicsResiduals,
                                   title: str = "PDE Residual Statistics") -> plt.Figure:
        """
        Plot statistical analysis of PDE residuals.
        
        Args:
            residuals: PhysicsResiduals object
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Collect all residuals
        all_residuals = {
            'Maxwell': residuals.maxwell_residual,
            'Heat': residuals.heat_equation_residual,
            'Structural': residuals.structural_dynamics_residual,
            'Coupling': residuals.coupling_residual
        }
        
        # Box plot comparison
        residual_data = [residual for residual in all_residuals.values()]
        residual_labels = list(all_residuals.keys())
        
        bp = axes[0, 0].boxplot(residual_data, labels=residual_labels, patch_artist=True)
        for patch, label in zip(bp['boxes'], residual_labels):
            patch.set_facecolor(self.constraint_colors.get(label, '#808080'))
            patch.set_alpha(0.7)
        
        axes[0, 0].set_title('Residual Distribution Comparison')
        axes[0, 0].set_ylabel('Residual Magnitude')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residual correlation matrix
        residual_matrix = np.column_stack([residual for residual in all_residuals.values()])
        correlation_matrix = np.corrcoef(residual_matrix.T)
        
        im = axes[0, 1].imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 1].set_xticks(range(len(residual_labels)))
        axes[0, 1].set_yticks(range(len(residual_labels)))
        axes[0, 1].set_xticklabels(residual_labels, rotation=45)
        axes[0, 1].set_yticklabels(residual_labels)
        axes[0, 1].set_title('Residual Correlation Matrix')
        
        # Add correlation values
        for i in range(len(residual_labels)):
            for j in range(len(residual_labels)):
                text = axes[0, 1].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                     ha="center", va="center", 
                                     color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black")
        
        plt.colorbar(im, ax=axes[0, 1])
        
        # Residual magnitude over time (log scale)
        for name, residual in all_residuals.items():
            axes[1, 0].semilogy(residuals.timestamps, residual,
                              color=self.constraint_colors.get(name, '#808080'),
                              linewidth=1.5, label=name, alpha=0.8)
        
        axes[1, 0].set_title('Residual Evolution (Log Scale)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Residual Magnitude (log)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistical summary table
        stats_data = []
        for name, residual in all_residuals.items():
            stats_data.append([
                name,
                f'{np.mean(residual):.2e}',
                f'{np.std(residual):.2e}',
                f'{np.min(residual):.2e}',
                f'{np.max(residual):.2e}',
                f'{np.median(residual):.2e}'
            ])
        
        table = axes[1, 1].table(cellText=stats_data,
                               colLabels=['Constraint', 'Mean', 'Std', 'Min', 'Max', 'Median'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Statistical Summary')
        
        plt.tight_layout()
        return fig