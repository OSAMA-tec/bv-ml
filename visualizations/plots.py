import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class HousePricePlotter:
    def __init__(self):
        self.plt_style = {
            'figsize': (12, 8),
            'title_fontsize': 16,
            'label_fontsize': 12
        }
        # Ensure the output directory exists
        self.output_dir = os.path.join('visualizations', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_feature_importance(self, features, importance):
        """Plot feature importance from model."""
        try:
            plt.figure(figsize=self.plt_style['figsize'])
            sns.barplot(x=importance, y=features)
            plt.title('Feature Importance', fontsize=self.plt_style['title_fontsize'])
            plt.xlabel('Importance', fontsize=self.plt_style['label_fontsize'])
            plt.ylabel('Features', fontsize=self.plt_style['label_fontsize'])
            plt.tight_layout()
            
            output_path = os.path.join(self.output_dir, 'feature_importance.png')
            plt.savefig(output_path)
            plt.close()
            print(f"Feature importance plot saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error in feature importance plotting: {str(e)}")
            raise

    def plot_future_predictions(self, predictions: dict, current_price: float):
        """Plot predicted prices over years."""
        try:
            years = list(predictions.keys())
            prices = list(predictions.values())

            plt.figure(figsize=self.plt_style['figsize'])
            plt.plot(years, prices, marker='o', linewidth=2, markersize=8)
            
            # Add value labels
            for i, price in enumerate(prices):
                plt.text(i, price, f'₨{price:,.0f}', ha='center', va='bottom')

            plt.title('Predicted House Price Trend', fontsize=self.plt_style['title_fontsize'])
            plt.xlabel('Year', fontsize=self.plt_style['label_fontsize'])
            plt.ylabel('Price (₨)', fontsize=self.plt_style['label_fontsize'])
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₨{x:,.0f}'))
            plt.tight_layout()
            
            output_path = os.path.join(self.output_dir, 'price_prediction_trend.png')
            plt.savefig(output_path)
            plt.close()
            print(f"Price prediction trend plot saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error in prediction trend plotting: {str(e)}")
            raise

    def plot_price_distribution(self, prices: pd.Series):
        """Plot distribution of house prices."""
        try:
            plt.figure(figsize=self.plt_style['figsize'])
            sns.histplot(data=prices, bins=30, kde=True)
            plt.title('Distribution of House Prices', fontsize=self.plt_style['title_fontsize'])
            plt.xlabel('Price (₨)', fontsize=self.plt_style['label_fontsize'])
            plt.ylabel('Count', fontsize=self.plt_style['label_fontsize'])
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₨{x:,.0f}'))
            plt.tight_layout()
            
            output_path = os.path.join(self.output_dir, 'price_distribution.png')
            plt.savefig(output_path)
            plt.close()
            print(f"Price distribution plot saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error in price distribution plotting: {str(e)}")
            raise

    def plot_prediction_matrix(self, y_true, y_pred, model_name, metrics_by_range=None, accuracy=None):
        """Plot a binned prediction matrix for regression results with metrics."""
        try:
            fig = plt.figure(figsize=(15, 10))
            
            # Create price ranges for binning
            price_ranges = [0, 1e7, 2e7, 5e7, 1e8, float('inf')]
            labels = ['0-10M', '10M-20M', '20M-50M', '50M-100M', '100M+']
            
            # Bin the true and predicted values
            y_true_binned = pd.cut(y_true, bins=price_ranges, labels=labels)
            y_pred_binned = pd.cut(y_pred, bins=price_ranges, labels=labels)
            
            # Create confusion matrix
            conf_matrix = pd.crosstab(
                y_true_binned, 
                y_pred_binned, 
                normalize='index'
            )
            
            # Plot heatmap
            ax = plt.subplot(1, 2, 1)
            sns.heatmap(
                conf_matrix, 
                annot=True, 
                fmt='.2%', 
                cmap='YlOrRd',
                square=True
            )
            
            plt.title(f'Prediction Matrix - {model_name}', fontsize=self.plt_style['title_fontsize'])
            plt.xlabel('Predicted Price Range', fontsize=self.plt_style['label_fontsize'])
            plt.ylabel('Actual Price Range', fontsize=self.plt_style['label_fontsize'])
            
            # Add metrics table
            if metrics_by_range and accuracy:
                ax = plt.subplot(1, 2, 2)
                ax.axis('off')
                
                # Prepare metrics data
                table_data = []
                columns = ['Price Range', 'Precision', 'Recall', 'F1-Score']
                table_data.append(columns)
                
                for price_range, metrics in metrics_by_range.items():
                    row = [
                        price_range,
                        f"{metrics['Precision']:.3f}",
                        f"{metrics['Recall']:.3f}",
                        f"{metrics['F1-Score']:.3f}"
                    ]
                    table_data.append(row)
                
                # Add overall accuracy
                table_data.append(['Overall Accuracy', f"{accuracy:.3f}", "", ""])
                
                # Create table
                table = ax.table(
                    cellText=table_data,
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.2]
                )
                
                # Style the table
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                
                # Add title
                plt.title('Performance Metrics', fontsize=self.plt_style['title_fontsize'])
            
            plt.tight_layout()
            
            output_path = os.path.join(self.output_dir, f'prediction_matrix_{model_name.lower().replace(" ", "_")}.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Prediction matrix plot with metrics saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error in prediction matrix plotting: {str(e)}")
            raise 