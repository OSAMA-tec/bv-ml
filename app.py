import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import logging
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import xgboost as xgb
from lightgbm import LGBMRegressor
import joblib
import os
from difflib import get_close_matches
from visualizations.plots import HousePricePlotter
from flask import Flask, render_template, request, jsonify

# Set up logging and styling
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
plt.style.use('seaborn-v0_8')

class HousePricePredictor:
    def __init__(self):
        self.models = {
            'Ridge (L2)': Ridge(
                alpha=1.0, 
                random_state=42
            ),
            'Lasso (L1)': Lasso(
                alpha=0.001,  
                random_state=42
            ),
            'Elastic Net (L1 + L2)': ElasticNet(
                alpha=0.001, 
                l1_ratio=0.5,  
                random_state=42
            ),
            'Random Forest (Bagging)': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt', 
                bootstrap=True,       
                oob_score=True,    
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.01,  
                max_depth=5,         
                min_samples_split=15,  
                min_samples_leaf=8,   
                subsample=0.8,        
                max_features=0.7,     
                random_state=42
            ),
            'XGBoost (Boosting)': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.01,   
                max_depth=5,       
                reg_alpha=0.2,      
                reg_lambda=1.5,      
                subsample=0.7      
            )
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.location_avg_price = None
        self.city_avg_price = None
        self.feature_columns = None

        self.plt_style = {
            'figsize': (12, 8),
            'title_fontsize': 16,
            'label_fontsize': 12
        }
        
        self.yearly_growth_rate = 0.05

        self.plotter = HousePricePlotter()

    def load_and_preprocess_data(self, file_paths: Dict[str, str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess data from Google Sheets and local files."""
        try:
            processed_dfs = []
            
            sheets_url = "https://docs.google.com/spreadsheets/d/1cSSzgqgzZGlALGV4r63Ff_zYyfSYMlMum4FPLlVnSZM/edit?gid=0#gid=0"
            csv_export_url = sheets_url.replace('/edit?gid=0#gid=0', '/export?format=csv&gid=0')
            
            print("\nLoading data from Google Sheets...")
            df_sheets = pd.read_csv(csv_export_url)
            df_mix = pd.read_csv(file_paths['mix'])
            
            for df in [df_sheets, df_mix]:
                df.columns = [col.strip() for col in df.columns]
                
                # Rename columns
                df = df.rename(columns={
                    'City': 'city',
                    'Location': 'location',
                    'Price': 'price',
                    'Bedrooms': 'bedrooms',
                    'Baths': 'baths',
                    'Size': 'size'
                })
                
                # Convert numeric columns
                df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
                df['size'] = pd.to_numeric(df['size'].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
                df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
                df['baths'] = pd.to_numeric(df['baths'], errors='coerce')
                
                # Basic cleaning
                df = df.dropna(subset=['price', 'size', 'bedrooms', 'baths'])
                df = df[(df['price'] > 0) & (df['size'] > 0) & 
                       (df['bedrooms'] > 0) & (df['baths'] > 0)]
                
                # Add derived features
                df['price_per_sqft'] = df['price'] / df['size']
                
                processed_dfs.append(df)
            
            # Combine datasets
            final_df = pd.concat(processed_dfs, ignore_index=True)
            
            # Remove outliers
            for col in ['price', 'size', 'price_per_sqft']:
                Q1 = final_df[col].quantile(0.01)
                Q3 = final_df[col].quantile(0.99)
                final_df = final_df[(final_df[col] >= Q1) & (final_df[col] <= Q3)]
            
            # Select final features
            self.feature_columns = ['city', 'location', 'bedrooms', 'baths', 'size', 'price_per_sqft']
            X = final_df[self.feature_columns]
            y = final_df['price']
            
            # Encode categorical variables
            for col in ['city', 'location']:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error in data loading and preprocessing: {str(e)}")
            raise

    def train_and_evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train and evaluate multiple models."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            results = {}
            best_r2 = -float('inf')

            for name, model in self.models.items():
                logger.info(f"Training {name}...")
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                results[name] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'CV_R2_mean': cv_mean,
                    'CV_R2_std': cv_std
                }

                if r2 > best_r2:
                    best_r2 = r2
                    self.best_model = model
                    self.best_model_name = name

            logger.info(f"Best performing model: {self.best_model_name}")
            return results

        except Exception as e:
            logger.error(f"Error in model training and evaluation: {str(e)}")
            raise

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model."""
        try:
            if self.best_model is None:
                raise ValueError("Model has not been trained yet")
            
            features_scaled = self.scaler.transform(features)
            
            if self.best_model_name == 'Deep Learning':
                predictions = self.best_model.predict(features_scaled, verbose=0).flatten()
            else:
                predictions = self.best_model.predict(features_scaled)
            
            return predictions

        except Exception as e:
            logger.error(f"Error in making predictions: {str(e)}")
            raise

    def analyze_feature_importance(self, X: pd.DataFrame) -> None:
        """Analyze and plot feature importance."""
        try:
            if not isinstance(self.best_model, (RandomForestRegressor, GradientBoostingRegressor)):
                logger.info("Feature importance analysis only available for tree-based models")
                return

            importance = self.best_model.feature_importances_
            features = X.columns
            self.plotter.plot_feature_importance(features, importance)

        except Exception as e:
            logger.error(f"Error in feature importance analysis: {str(e)}")
            raise

    def predict_future_prices(self, features: pd.DataFrame, years: int = 2) -> Dict[str, float]:
        """Predict house prices for future years."""
        try:
            current_year = 2024
            predictions = {}
            
            current_price = self.predict(features)[0]
            predictions[str(current_year)] = current_price
            
            for i in range(1, years + 1):
                future_year = current_year + i
                future_price = current_price * (1 + self.yearly_growth_rate) ** i
                predictions[str(future_year)] = future_price
            
            return predictions
        except Exception as e:
            logger.error(f"Error in future price prediction: {str(e)}")
            raise

    def plot_price_distribution(self, y: pd.Series) -> None:
        """Plot the distribution of house prices."""
        try:
            self.plotter.plot_price_distribution(y)
        except Exception as e:
            logger.error(f"Error in plotting price distribution: {str(e)}")
            raise

    def _calculate_growth_rate(self, df: pd.DataFrame) -> float:
        """Calculate average yearly price growth rate from historical data."""
        yearly_avg_prices = df.groupby('year')['price'].mean()
        total_years = yearly_avg_prices.index.max() - yearly_avg_prices.index.min()
        total_growth = (yearly_avg_prices.iloc[-1] / yearly_avg_prices.iloc[0]) - 1
        yearly_growth = (1 + total_growth) ** (1/total_years) - 1
        return yearly_growth

    def _convert_area_to_sqft(self, area_str: str) -> float:
        """Convert various area formats to square feet."""
        try:
            if pd.isna(area_str):
                return None
            
            area_str = str(area_str).lower().strip()
            
            value_str = ''.join(c for c in area_str if c.isdigit() or c == '.' or c == ',')
            value_str = value_str.replace(',', '')
            
            try:
                value = float(value_str)
            except ValueError:
                return None
            
            if 'marla' in area_str:
                return value * 272.25  # 1 Marla = 272.25 sq ft
            elif 'kanal' in area_str:
                return value * 5445  # 1 Kanal = 20 Marla = 5445 sq ft
            elif 'sq yd' in area_str or 'square yard' in area_str:
                return value * 9  # 1 Square Yard = 9 sq ft
            elif 'sq ft' in area_str or 'sqft' in area_str or 'square feet' in area_str:
                return value
            else:
                return value
                
        except Exception as e:
            logger.warning(f"Error converting area: {area_str}. Error: {str(e)}")
            return None

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for better prediction."""
        try:
            if 'price' in df.columns and 'size' in df.columns:
                df['price_per_sqft'] = df['price'] / df['size'].replace(0, np.nan)
                df = df.replace([np.inf, -np.inf], np.nan)
            
            if 'baths' in df.columns and 'bedrooms' in df.columns:
                df['bath_bed_ratio'] = df['baths'] / df['bedrooms'].replace(0, np.nan)
                df = df.replace([np.inf, -np.inf], np.nan)
            
            if 'year' in df.columns:
                df['years_since_2019'] = df['year'] - 2019
            
            if 'price' in df.columns:
                if 'location' in df.columns:
                    self.location_avg_price = df.groupby('location')['price'].mean()
                    df['location_avg_price'] = df['location'].map(self.location_avg_price)
                
                if 'city' in df.columns:
                    self.city_avg_price = df.groupby('city')['price'].mean()
                    df['city_avg_price'] = df['city'].map(self.city_avg_price)
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error in adding derived features: {str(e)}")
            raise

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        try:
            for column in ['price', 'size', 'bedrooms', 'baths']:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            return df
        except Exception as e:
            logger.warning(f"Error removing outliers: {str(e)}")
            return df

    def save_model(self, output_dir='saved_model'):
        """Save the trained model and necessary data"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            model_data = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'location_avg_price': self.location_avg_price,
                'city_avg_price': self.city_avg_price,
                'feature_columns': self.feature_columns,
                'yearly_growth_rate': self.yearly_growth_rate
            }
            
            joblib.dump(model_data, os.path.join(output_dir, 'model.joblib'))  
            logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def find_closest_location(query, available_locations):
    """Find the closest matching location from available locations."""
    matches = get_close_matches(query, available_locations, n=1, cutoff=0.6)
    return matches[0] if matches else available_locations[0]

def main():
    try:
        os.makedirs('visualizations/output', exist_ok=True)

        predictor = HousePricePredictor()

        file_paths = {
            'mix': 'zameen_mix.csv'  # Keep historical data
        }
        
        print("Loading data from Google Sheets and historical dataset")
        
        X, y = predictor.load_and_preprocess_data(file_paths)
        
        if X is None or y is None:
            print("Error: Could not load data. Please check the Google Sheets URL and CSV file.")
            return
            
        predictor.feature_columns = X.columns
        
        results = predictor.train_and_evaluate_models(X, y)
        
        predictor.save_model()
        
        print("\nModel Performance Metrics:")
        print("=" * 50)
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"RMSE: {metrics['RMSE']:,.2f}")
            print(f"MAE: {metrics['MAE']:,.2f}")
            print(f"R² Score: {metrics['R2']:.4f}")
            print(f"Cross-validation R² Score: {metrics['CV_R2_mean']:.4f} (±{metrics['CV_R2_std']*2:.4f})")
        
        sample_property = {
            'city': 'Islamabad',
            'location': 'DHA Defence Phase 2, DHA Defence',
            'bedrooms': 4,
            'baths': 3,
            'size': 2000,
            'price_per_sqft': 15000
        }
        
        sample_df = pd.DataFrame([sample_property])[predictor.feature_columns]
        
        for col in ['city', 'location']:
            try:
                sample_df[col] = predictor.label_encoders[col].transform(sample_df[col])
            except ValueError as e:
                print(f"\nError with {col}: {str(e)}")
                print(f"Available {col}s in training data:", predictor.label_encoders[col].classes_)
                raise
        
        try:
            predictions = predictor.predict_future_prices(sample_df, years=2)
            
            print(f"\nPrediction for Sample Property:")
            print("=" * 50)
            print(f"Location: {sample_property['city']}, {sample_property['location']}")
            print(f"Specifications: {sample_property['bedrooms']} bed, {sample_property['baths']} bath, {sample_property['size']} sq ft")
            print("-" * 50)
            print("\nPrice Predictions:")
            print("-" * 30)
            
            for year, price in predictions.items():
                if year == '2024':
                    print(f"Current Price ({year}): ₨{price:,.2f}")
                    print(f"                  (${price/282:,.2f})")  # Assuming 1 USD = 282 PKR
                    current_price = price
                else:
                    print(f"\nPredicted Price ({year}): ₨{price:,.2f}")
                    print(f"                     (${price/282:,.2f})")
                    print(f"Expected Growth: {((price/current_price - 1) * 100):,.1f}%")
            
            print("-" * 30)
            print(f"Price per sq ft: ₨{predictions['2024']/sample_property['size']:,.2f}")
            
            predictor.plotter.plot_future_predictions(predictions, predictions['2024'])
            
        except Exception as e:
            logger.error(f"Error in making predictions: {str(e)}")
            raise
        
        print("\nVisualization files generated in:", os.path.abspath('visualizations/output'))
        for plot_file in ['feature_importance.png', 'price_prediction_trend.png', 'price_distribution.png']:
            plot_path = os.path.join('visualizations/output', plot_file)
            if os.path.exists(plot_path):
                print(f"- {plot_file}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()