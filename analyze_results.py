"""
Analyze YOLO Training Results
Reads and displays key metrics from training
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_training_results(results_dir='runs/detect/yolo_custom_train'):
    """
    Analyze and display training results from YOLO training.
    
    Args:
        results_dir: Path to training results directory
    """
    results_path = Path(results_dir)
    csv_path = results_path / 'results.csv'
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return
    
    # Read results
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Remove whitespace from column names
    
    print("="*60)
    print("YOLO TRAINING RESULTS ANALYSIS")
    print("="*60)
    
    # Best epoch
    best_epoch = df['metrics/mAP50-95(B)'].idxmax()
    print(f"\nBest Epoch: {best_epoch + 1}")
    print(f"Best mAP50-95: {df['metrics/mAP50-95(B)'].iloc[best_epoch]:.4f}")
    
    # Final metrics
    print("\n" + "-"*60)
    print("FINAL METRICS (Last Epoch)")
    print("-"*60)
    last_row = df.iloc[-1]
    
    metrics = {
        'Precision': 'metrics/precision(B)',
        'Recall': 'metrics/recall(B)',
        'mAP50': 'metrics/mAP50(B)',
        'mAP50-95': 'metrics/mAP50-95(B)'
    }
    
    for name, col in metrics.items():
        if col in df.columns:
            print(f"{name:15s}: {last_row[col]:.4f}")
    
    # Loss values
    print("\n" + "-"*60)
    print("FINAL LOSSES")
    print("-"*60)
    
    losses = {
        'Box Loss': 'train/box_loss',
        'Class Loss': 'train/cls_loss',
        'DFL Loss': 'train/dfl_loss'
    }
    
    for name, col in losses.items():
        if col in df.columns:
            print(f"{name:15s}: {last_row[col]:.4f}")
    
    # Improvement over training
    print("\n" + "-"*60)
    print("IMPROVEMENT OVER TRAINING")
    print("-"*60)
    
    first_row = df.iloc[0]
    improvements = {
        'mAP50-95': ('metrics/mAP50-95(B)', True),
        'Precision': ('metrics/precision(B)', True),
        'Box Loss': ('train/box_loss', False)
    }
    
    for name, (col, higher_better) in improvements.items():
        if col in df.columns:
            first_val = first_row[col]
            last_val = last_row[col]
            change = last_val - first_val
            pct_change = (change / first_val * 100) if first_val != 0 else 0
            
            arrow = "↑" if (change > 0 and higher_better) or (change < 0 and not higher_better) else "↓"
            print(f"{name:15s}: {first_val:.4f} → {last_val:.4f} ({arrow} {abs(pct_change):.1f}%)")
    
    # Create visualization
    print("\n" + "-"*60)
    print("GENERATING PLOTS...")
    print("-"*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: mAP over epochs
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['metrics/mAP50(B)'], label='mAP50', linewidth=2)
    ax1.plot(df.index, df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mAP')
    ax1.set_title('Mean Average Precision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision & Recall
    ax2 = axes[0, 1]
    ax2.plot(df.index, df['metrics/precision(B)'], label='Precision', linewidth=2)
    ax2.plot(df.index, df['metrics/recall(B)'], label='Recall', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Precision & Recall')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Losses
    ax3 = axes[1, 0]
    ax3.plot(df.index, df['train/box_loss'], label='Box Loss', linewidth=2)
    ax3.plot(df.index, df['train/cls_loss'], label='Class Loss', linewidth=2)
    ax3.plot(df.index, df['train/dfl_loss'], label='DFL Loss', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Losses')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    ax4 = axes[1, 1]
    if 'lr/pg0' in df.columns:
        ax4.plot(df.index, df['lr/pg0'], linewidth=2, color='purple')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = results_path / 'analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Analysis plot saved to: {output_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    analyze_training_results()
