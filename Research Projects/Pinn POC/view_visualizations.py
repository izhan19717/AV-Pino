#!/usr/bin/env python3
"""
Simple visualization viewer for AV-PINO real data results.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import os

def view_visualizations():
    """Display all generated visualizations."""
    viz_dir = Path("real_data_outputs/visualizations")
    
    if not viz_dir.exists():
        print("❌ Visualizations not found. Please run create_real_data_visualizations.py first.")
        return
    
    # Get all PNG files
    png_files = list(viz_dir.glob("*.png"))
    
    if not png_files:
        print("❌ No visualization files found.")
        return
    
    print(f"🎨 Found {len(png_files)} visualizations")
    
    # Sort files
    png_files.sort()
    
    # Display each visualization
    for i, png_file in enumerate(png_files, 1):
        print(f"\n📊 Displaying: {png_file.name}")
        
        # Load and display image
        img = mpimg.imread(png_file)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"AV-PINO Visualization {i}/{len(png_files)}: {png_file.stem}", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # Wait for user input to continue
        if i < len(png_files):
            input(f"Press Enter to view next visualization ({i+1}/{len(png_files)})...")
    
    print("\n🎉 All visualizations displayed!")
    print(f"📁 Visualization files saved in: {viz_dir}")

def open_visualization_folder():
    """Open the visualization folder in file explorer."""
    viz_dir = Path("real_data_outputs/visualizations")
    
    if viz_dir.exists():
        if os.name == 'nt':  # Windows
            os.startfile(viz_dir)
        elif os.name == 'posix':  # macOS and Linux
            os.system(f'open "{viz_dir}"' if os.uname().sysname == 'Darwin' else f'xdg-open "{viz_dir}"')
        print(f"📁 Opened visualization folder: {viz_dir}")
    else:
        print("❌ Visualization folder not found.")

if __name__ == "__main__":
    print("🎨 AV-PINO Visualization Viewer")
    print("=" * 40)
    
    choice = input("Choose an option:\n1. View visualizations in Python\n2. Open folder in file explorer\n3. Both\nEnter choice (1/2/3): ")
    
    if choice in ['1', '3']:
        view_visualizations()
    
    if choice in ['2', '3']:
        open_visualization_folder()