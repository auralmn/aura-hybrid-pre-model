import unittest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal
    def setUp(self):
        self.output_dir = "tests/artifacts"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_visualize_place_cells(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create place cells
        place_cells = []
        for i in range(20):
            center = np.random.uniform(-5, 5, 2)
            radius = np.random.uniform(0.8, 1.5)
            place_cells.append(PlaceCell(center, radius))
        
        # Visualize receptive fields
        for cell in place_cells:
            circle = patches.Circle(cell.center, cell.radius, 
                                   fill=True, alpha=0.3, color='blue')
            ax.add_patch(circle)
            ax.plot(cell.center[0], cell.center[1], 'b.', markersize=8)
        
        # Test trajectory
        trajectory = np.linspace(-5, 5, 100)
        path = np.column_stack([trajectory, np.sin(trajectory)])
        ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='Trajectory')
        
        # Compute firing rates along trajectory
        for i, point in enumerate(path):
            total_firing = sum(cell.compute_firing_rate(point) for cell in place_cells)
            if total_firing > 5:
                ax.plot(point[0], point[1], 'ro', markersize=4, alpha=0.5)
        
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Place Cell Receptive Fields and Activity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.output_dir, 'place_cells_visualization.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved place cells visualization to {output_path}")
        self.assertTrue(os.path.exists(output_path))
    
    def test_visualize_grid_cells(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Different grid cell scales
        spacings = [1.0, 1.5, 2.0, 2.5]
        
        for idx, (ax, spacing) in enumerate(zip(axes.flat, spacings)):
            grid_cell = GridCell(spacing=spacing, orientation=0.0)
            
            # Create 2D grid of locations
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            
            # Compute firing rates
            Z = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    location = np.array([X[j, i], Y[j, i]])
                    Z[j, i] = grid_cell.compute_firing_rate(location)
            
            # Plot hexagonal pattern
            im = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'Grid Cell (spacing={spacing:.1f})')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax, label='Firing Rate (Hz)')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'grid_cells_visualization.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved grid cells visualization to {output_path}")
        self.assertTrue(os.path.exists(output_path))
    
    def test_visualize_time_cells(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create time cells with different preferred intervals
        intervals = np.logspace(0, 2, 10)
        time_cells = [TimeCell(interval, width=interval*0.3) for interval in intervals]
        
        # Test timeline
        time_points = np.linspace(0, 100, 1000)
        last_event = 0.0
        
        # Plot firing rates over time
        for i, cell in enumerate(time_cells):
            rates = [cell.compute_firing_rate(t, last_event) for t in time_points]
            ax.plot(time_points, rates, label=f'{cell.preferred_interval:.1f}s', alpha=0.7)
        
        ax.set_xlabel('Time Since Last Event (s)')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title('Time Cell Sequential Coding')
        ax.legend(title='Preferred Interval', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'time_cells_visualization.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved time cells visualization to {output_path}")
        self.assertTrue(os.path.exists(output_path))
    
    def test_visualize_episodic_memories(self):
        hippo = HippocampalFormation(n_place_cells=50, n_time_cells=20, n_grid_cells=30)
        
        # Create memories at different locations
        np.random.seed(42)
        locations = []
        for i in range(15):
            loc = np.random.uniform(-8, 8, 2)
            locations.append(loc)
            hippo.update_spatial_state(loc)
            features = np.random.randn(64)
            hippo.create_episodic_memory(f"mem_{i}", f"event_{i}", features)
            time.sleep(0.01)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot memory locations
        locations = np.array(locations)
        scatter = ax.scatter(locations[:, 0], locations[:, 1], 
                           c=range(len(locations)), cmap='coolwarm', 
                           s=200, alpha=0.7, edgecolors='black', linewidths=2)
        
        # Add memory IDs
        for i, loc in enumerate(locations):
            ax.annotate(f'M{i}', (loc[0], loc[1]), 
                       ha='center', va='center', color='white', fontweight='bold')
        
        # Draw cognitive map connections
        for (mem1, mem2), distance in hippo.cognitive_map.items():
            idx1 = int(mem1.split('_')[1])
            idx2 = int(mem2.split('_')[1])
            if distance < 5.0:
                ax.plot([locations[idx1, 0], locations[idx2, 0]],
                       [locations[idx1, 1], locations[idx2, 1]],
                       'gray', alpha=0.2, linewidth=1)
        
        plt.colorbar(scatter, ax=ax, label='Memory Sequence')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Episodic Memory Spatial Distribution with Cognitive Map')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        output_path = os.path.join(self.output_dir, 'episodic_memories_visualization.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved episodic memories visualization to {output_path}")
        self.assertTrue(os.path.exists(output_path))
    
    def test_visualize_memory_interpolation(self):
        interpolator = TemporalMemoryInterpolator()
        
        # Create two different signals
        t_points = np.linspace(0, 2*np.pi, 128)
        M0 = np.sin(t_points)
        M1 = np.sin(3 * t_points) * 0.5
        
        # Interpolate using different modes
        modes = ['linear', 'fourier', 'hilbert', 'hamiltonian']
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(len(modes), len(t_values), figure=fig)
        
        for mode_idx, mode in enumerate(modes):
            for t_idx, t_val in enumerate(t_values):
                ax = fig.add_subplot(gs[mode_idx, t_idx])
                
                if t_val == 0.0:
                    result = M0
                elif t_val == 1.0:
                    result = M1
                else:
                    result = interpolator.interpolate(M0, M1, t_val, mode=mode)
                
                ax.plot(t_points, result, 'b-', linewidth=1.5)
                ax.plot(t_points, M0, 'g--', alpha=0.3, linewidth=1)
                ax.plot(t_points, M1, 'r--', alpha=0.3, linewidth=1)
                
                if t_idx == 0:
                    ax.set_ylabel(mode.capitalize(), fontweight='bold')
                if mode_idx == 0:
                    ax.set_title(f't={t_val:.2f}')
                if mode_idx == len(modes) - 1:
                    ax.set_xlabel('Time')
                
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-1.5, 1.5)
        
        plt.suptitle('Temporal Memory Interpolation Modes', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'memory_interpolation_visualization.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved memory interpolation visualization to {output_path}")
        self.assertTrue(os.path.exists(output_path))
    
    def test_visualize_hippocampal_system(self):
        hippo = HippocampalFormation(n_place_cells=30, n_time_cells=15, n_grid_cells=20)
        
        # Create a trajectory with events
        trajectory_points = 20
        trajectory = []
        for i in range(trajectory_points):
            angle = 2 * np.pi * i / trajectory_points
            loc = np.array([5 * np.cos(angle), 5 * np.sin(angle)])
            trajectory.append(loc)
            hippo.update_spatial_state(loc, dt=0.1)
            
            if i % 4 == 0:
                features = np.random.randn(32)
                hippo.create_episodic_memory(f"mem_{i}", f"event_{i}", features)
        
        trajectory = np.array(trajectory)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Spatial context (place cells)
        ax1 = fig.add_subplot(gs[0, 0])
        spatial_ctx = hippo.get_spatial_context()
        place_activity = spatial_ctx['place_cells']
        ax1.bar(range(len(place_activity)), place_activity, color='blue', alpha=0.7)
        ax1.set_xlabel('Place Cell Index')
        ax1.set_ylabel('Firing Rate (Hz)')
        ax1.set_title('Place Cell Population Activity')
        ax1.grid(True, alpha=0.3)
        
        # 2. Grid cell activity
        ax2 = fig.add_subplot(gs[0, 1])
        grid_activity = spatial_ctx['grid_cells']
        ax2.bar(range(len(grid_activity)), grid_activity, color='green', alpha=0.7)
        ax2.set_xlabel('Grid Cell Index')
        ax2.set_ylabel('Firing Rate (Hz)')
        ax2.set_title('Grid Cell Population Activity')
        ax2.grid(True, alpha=0.3)
        
        # 3. Temporal context (time cells)
        ax3 = fig.add_subplot(gs[0, 2])
        temporal_ctx = hippo.get_temporal_context()
        time_activity = temporal_ctx['time_cells']
        ax3.bar(range(len(time_activity)), time_activity, color='red', alpha=0.7)
        ax3.set_xlabel('Time Cell Index')
        ax3.set_ylabel('Firing Rate (Hz)')
        ax3.set_title('Time Cell Population Activity')
        ax3.grid(True, alpha=0.3)
        
        # 4. Trajectory and memories
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
        
        memory_locs = []
        for mem_id, memory in hippo.episodic_memories.items():
            loc = memory.spatial_location.coordinates
            memory_locs.append(loc)
            ax4.plot(loc[0], loc[1], 'ro', markersize=15, alpha=0.7)
            ax4.annotate(mem_id, (loc[0], loc[1]), fontsize=8, ha='center', va='center')
        
        ax4.set_xlabel('X Position')
        ax4.set_ylabel('Y Position')
        ax4.set_title('Spatial Trajectory with Episodic Memories')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        # 5. Theta rhythm
        ax5 = fig.add_subplot(gs[2, 0])
        theta_phases = np.linspace(0, 4*np.pi, 100)
        theta_signal = np.sin(theta_phases)
        ax5.plot(theta_phases, theta_signal, 'purple', linewidth=2)
        ax5.axvline(spatial_ctx['theta_phase'], color='red', linestyle='--', 
                   linewidth=2, label=f'Current: {spatial_ctx["theta_phase"]:.2f}')
        ax5.set_xlabel('Phase (rad)')
        ax5.set_ylabel('Amplitude')
        ax5.set_title(f'Theta Rhythm (8 Hz)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Memory statistics
        ax6 = fig.add_subplot(gs[2, 1:])
        stats_text = f"""
        Hippocampal Formation Statistics:
        
        Place Cells: {len(hippo.place_cells)}
        Time Cells: {len(hippo.time_cells)}
        Grid Cells: {len(hippo.grid_cells)}
        
        Episodic Memories: {len(hippo.episodic_memories)}
        Spatial Locations: {len(hippo.spatial_locations)}
        Temporal Events: {len(hippo.temporal_events)}
        
        Cognitive Map Edges: {len(hippo.cognitive_map)}
        Temporal Map Edges: {len(hippo.temporal_map)}
        
        Current Location: [{spatial_ctx['current_location'][0]:.2f}, {spatial_ctx['current_location'][1]:.2f}]
        Theta Phase: {spatial_ctx['theta_phase']:.2f} rad
        """
        ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        ax6.axis('off')
        
        plt.suptitle('Hippocampal Formation System Overview', fontsize=16, fontweight='bold')
        
        output_path = os.path.join(self.output_dir, 'hippocampal_system_overview.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved hippocampal system overview to {output_path}")
        self.assertTrue(os.path.exists(output_path))


if __name__ == '__main__':
    print("\nGenerating Hippocampal Formation Visualizations")
    print("=" * 60)
    unittest.main(verbosity=2)
