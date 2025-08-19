
## src/visualizer.py
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self):
        pass
        
    def plot_shap_summary(self, shap_values, X, output_path):
        """Create SHAP summary plot."""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1], X, show=False)
        plt.savefig(output_path)
        plt.close()
        
    def plot_class_distribution(self, y, output_path):
        """Plot class distribution."""
        plt.figure(figsize=(8, 6))
        sns.countplot(x=y)
        plt.title('Class Distribution')
        plt.savefig(output_path)
        plt.close()
