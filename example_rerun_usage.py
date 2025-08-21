#!/usr/bin/env python3
"""
Example usage of the RerunVisualizer for SpaTrackerV2 results.
This script demonstrates how to use the new rerun-based visualizer.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from rerun_visualizer import RerunVisualizer


def example_basic_usage():
    """Example of basic usage of the RerunVisualizer."""
    print("=== Basic Usage Example ===")
    
    # Example path - replace with your actual results.npz file
    npz_path = "assets/example0/results/result.npz"
    
    if not os.path.exists(npz_path):
        print(f"Example file not found: {npz_path}")
        print("Please run inference.py first to generate results, or update the path.")
        return
    
    try:
        # Create visualizer
        print(f"Loading results from: {npz_path}")
        visualizer = RerunVisualizer(npz_path, fps=10)
        
        # Visualize all frames
        print("Starting visualization...")
        visualizer.visualize_all_frames()
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error: {e}")


def example_interactive_usage():
    """Example of interactive usage of the RerunVisualizer."""
    print("=== Interactive Usage Example ===")
    
    # Example path - replace with your actual results.npz file
    npz_path = "assets/example0/results/result.npz"
    
    if not os.path.exists(npz_path):
        print(f"Example file not found: {npz_path}")
        print("Please run inference.py first to generate results, or update the path.")
        return
    
    try:
        # Create visualizer
        print(f"Loading results from: {npz_path}")
        visualizer = RerunVisualizer(npz_path, fps=10)
        
        # Start interactive visualization
        print("Starting interactive visualization...")
        print("Use the rerun viewer to navigate through frames.")
        print("Press Ctrl+C to exit.")
        
        visualizer.interactive_visualization()
        
    except KeyboardInterrupt:
        print("\nInteractive visualization stopped by user.")
    except Exception as e:
        print(f"Error: {e}")


def example_single_frame():
    """Example of visualizing a single frame."""
    print("=== Single Frame Example ===")
    
    # Example path - replace with your actual results.npz file
    npz_path = "assets/example0/results/result.npz"
    
    if not os.path.exists(npz_path):
        print(f"Example file not found: {npz_path}")
        print("Please run inference.py first to generate results, or update the path.")
        return
    
    try:
        # Create visualizer
        print(f"Loading results from: {npz_path}")
        visualizer = RerunVisualizer(npz_path, fps=10)
        
        # Visualize specific frame (e.g., frame 0)
        frame_idx = 0
        print(f"Visualizing frame {frame_idx}...")
        visualizer.visualize_frame(frame_idx)
        
        print("Single frame visualization complete!")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function to run examples."""
    print("SpaTrackerV2 Rerun Visualizer Examples")
    print("=" * 40)
    
    # Check if rerun is available
    try:
        import rerun as rr
        print("✓ Rerun library is available")
    except ImportError:
        print("✗ Rerun library not found. Please install it with:")
        print("  pip install rerun-sdk")
        return
    
    print("\nAvailable examples:")
    print("1. Basic visualization (all frames)")
    print("2. Interactive visualization")
    print("3. Single frame visualization")
    
    choice = input("\nEnter your choice (1-3) or press Enter to run all: ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_interactive_usage()
    elif choice == "3":
        example_single_frame()
    else:
        print("\nRunning all examples...")
        print("\n" + "="*50)
        example_basic_usage()
        print("\n" + "="*50)
        example_single_frame()
        print("\n" + "="*50)
        example_interactive_usage()


if __name__ == "__main__":
    main()
