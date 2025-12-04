"""
Stock Price Prediction - Main Application
Complete ML Pipeline: Data -> Features -> Models -> Predictions -> Visualization
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

# ML Libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Input, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Data & APIs
import yfinance as yf
from bs4 import BeautifulSoup
import requests

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    """Configuration Settings"""
    STOCK_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    LOOKBACK_DAYS = 60
    PREDICTION_PERIODS = {
        'next_day': 1,
        'next_week': 5,
        'next_month': 20
    }
    TRAIN_TEST_SPLIT = 0.8
    EPOCHS = 100
    BATCH_SIZE = 32
    RANDOM_STATE = 42
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    RESULTS_DIR = BASE_DIR / 'results'
    
    def __init__(self):
        self.DATA_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA FETCHER
# ============================================================================
class DataFetcher:
    """Fetch historical stock data"""
    
    def __init__(self, symbol, lookback_days=60):
        self.symbol = symbol
        self.lookback_days = lookback_days
        
    def fetch_data(self):
        """Fetch OHLCV data from Yahoo Finance"""
        end_date = datetime.now()
        # Add buffer for technical indicators
        start_date = end_date - timedelta(days=self.lookback_days + 100)
        
        try:
            data = yf.download(self.symbol, start=start_date, end=end_date, 
                             progress=False, interval='1d')
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date').reset_index(drop=True)
            print(f"✓ Fetched {len(data)} days of data for {self.symbol}")
            return data
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return None

# ============================================================================
# FEATURE ENGINEERING - TECHNICAL INDICATORS
# ============================================================================
class TechnicalIndicators:
    """Calculate technical indicators from OHLCV data"""
    
    @staticmethod
    def add_indicators(df):
        """Add comprehensive technical indicators"""
        df = df.copy()
        
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Change'] = df['Close'] - df['Open']
        
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
        
        # RSI - Relative Strength Index
        df['RSI'] = TechnicalIndicators._calculate_rsi(df['Close'])
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR - Average True Range
        df['ATR'] = TechnicalIndicators._calculate_atr(df)
        
        # Stochastic Oscillator
        df['K%'], df['D%'] = TechnicalIndicators._calculate_stochastic(df['Close'])
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        
        # Rate of Change
        df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
        
        # CCI - Commodity Channel Index
        df['CCI'] = TechnicalIndicators._calculate_cci(df)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_atr(df, period=14):
        """Calculate Average True Range"""
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift()),
                abs(df['Low'] - df['Close'].shift())
            )
        )
        return df['TR'].rolling(window=period).mean()
    
    @staticmethod
    def _calculate_stochastic(prices, period=14):
        """Calculate Stochastic Oscillator"""
        low_min = prices.rolling(window=period).min()
        high_max = prices.rolling(window=period).max()
        k_percent = 100 * (prices - low_min) / (high_max - low_min)
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent
    
    @staticmethod
    def _calculate_cci(df, period=20):
        """Calculate Commodity Channel Index"""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return cci

# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================
class SentimentAnalyzer:
    """Analyze financial news sentiment"""
    
    @staticmethod
    def generate_sentiment_features(symbol, lookback_days=30):
        """Generate sentiment score from news (simulated)"""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=lookback_days, freq='D')
        
        # Simulate sentiment scores (in real scenario, use NewsAPI + VADER/BERT)
        sentiment_scores = np.random.uniform(-1, 1, lookback_days)
        volatility = np.random.uniform(0.01, 0.05, lookback_days)
        
        df_sentiment = pd.DataFrame({
            'Date': dates,
            'Sentiment_Score': sentiment_scores,
            'News_Sentiment_MA': pd.Series(sentiment_scores).rolling(5).mean(),
            'Market_Volatility_Proxy': volatility
        })
        
        return df_sentiment

# ============================================================================
# DATA PREPROCESSOR
# ============================================================================
class DataPreprocessor:
    """Prepare data for ML models"""
    
    def __init__(self, lookback_window=60):
        self.lookback_window = lookback_window
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data(self, df, target_col='Close', prediction_days=1):
        """Prepare sequences for modeling"""
        data = df[[target_col]].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.lookback_window - prediction_days + 1):
            X.append(scaled_data[i:i + self.lookback_window])
            y.append(scaled_data[i + self.lookback_window + prediction_days - 1])
        
        return np.array(X), np.array(y)
    
    def prepare_multifeature_data(self, df, feature_cols, target_col='Close', prediction_days=1):
        """Prepare data with multiple features"""
        data = df[feature_cols].values
        scaler_multi = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler_multi.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.lookback_window - prediction_days + 1):
            X.append(scaled_data[i:i + self.lookback_window])
            y.append(df[target_col].values[i + self.lookback_window + prediction_days - 1])
        
        return np.array(X), np.array(y), scaler_multi

# ============================================================================
# MODELS
# ============================================================================
class ModelFactory:
    """Create all ML/DL models"""
    
    @staticmethod
    def create_linear_regression():
        return LinearRegression()
    
    @staticmethod
    def create_ridge(alpha=1.0):
        return Ridge(alpha=alpha)
    
    @staticmethod
    def create_lasso(alpha=0.1):
        return Lasso(alpha=alpha, max_iter=10000)
    
    @staticmethod
    def create_knn(n_neighbors=5):
        return KNeighborsRegressor(n_neighbors=n_neighbors)
    
    @staticmethod
    def create_random_forest(n_estimators=100, max_depth=10):
        return RandomForestRegressor(n_estimators=n_estimators, 
                                    max_depth=max_depth, random_state=42)
    
    @staticmethod
    def create_xgboost(n_estimators=100, max_depth=6):
        return xgb.XGBRegressor(n_estimators=n_estimators, 
                               max_depth=max_depth, 
                               learning_rate=0.1, 
                               random_state=42)
    
    @staticmethod
    def create_adaboost(n_estimators=100, learning_rate=0.1):
        return AdaBoostRegressor(n_estimators=n_estimators, 
                                learning_rate=learning_rate, 
                                random_state=42)
    
    @staticmethod
    def create_lstm(input_shape, output_dim=1):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_dim)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    @staticmethod
    def create_gru(input_shape, output_dim=1):
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_dim)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    @staticmethod
    def create_rnn(input_shape, output_dim=1):
        model = Sequential([
            keras.layers.SimpleRNN(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            keras.layers.SimpleRNN(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_dim)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

# ============================================================================
# MODEL EVALUATOR
# ============================================================================
class ModelEvaluator:
    """Evaluate all models"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate(self, y_true, y_pred, model_name):
        """Calculate evaluation metrics"""
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        self.results[model_name] = {
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        return self.results[model_name]
    
    def get_comparison_df(self):
        """Return results as DataFrame"""
        return pd.DataFrame(self.results).T

# ============================================================================
# PIPELINE
# ============================================================================
class StockPredictionPipeline:
    """Complete ML Pipeline"""
    
    def __init__(self, symbol='AAPL', lookback_days=60):
        self.symbol = symbol
        self.config = Config()
        self.fetcher = DataFetcher(symbol, lookback_days)
        self.preprocessor = DataPreprocessor(lookback_days)
        self.evaluator = ModelEvaluator()
        self.models = {}
        self.predictions = {}
        
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        print(f"\n{'='*60}")
        print(f"Stock Price Prediction Pipeline: {self.symbol}")
        print(f"{'='*60}\n")
        
        # 1. Fetch Data
        print("1. Fetching data...")
        raw_data = self.fetcher.fetch_data()
        if raw_data is None:
            return None
        
        # 2. Add Technical Indicators
        print("2. Adding technical indicators...")
        df = TechnicalIndicators.add_indicators(raw_data)
        print(f"   Features: {len(df.columns)} indicators added")
        
        # 3. Add Sentiment
        print("3. Adding sentiment features...")
        df_sentiment = SentimentAnalyzer.generate_sentiment_features(self.symbol, len(df))
        df = df.merge(df_sentiment, left_on='Date', right_on='Date', how='left')
        df = df.dropna()
        
        # 4. Prepare Data
        print("4. Preparing data...")
        feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
        X, y, scaler = self.preprocessor.prepare_multifeature_data(
            df, feature_cols, 'Close', prediction_days=1
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
        
        # 5. Flatten for traditional ML
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # 6. Train Models
        print("5. Training models...")
        
        # Traditional ML
        self._train_traditional_models(X_train_flat, X_test_flat, y_train, y_test)
        
        # Ensemble
        self._train_ensemble_models(X_train_flat, X_test_flat, y_train, y_test)
        
        # Deep Learning
        self._train_deep_learning_models(X_train, X_test, y_train, y_test)
        
        # 7. Comparison
        print("\n6. Model Comparison:")
        comparison_df = self.evaluator.get_comparison_df()
        print(comparison_df.round(4))
        
        return {
            'data': df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'models': self.models,
            'results': self.evaluator.results,
            'comparison': comparison_df
        }
    
    def _train_traditional_models(self, X_train, X_test, y_train, y_test):
        """Train traditional ML models"""
        print("   - Linear Regression...", end=' ')
        model = ModelFactory.create_linear_regression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluator.evaluate(y_test, y_pred, 'Linear Regression')
        self.models['Linear Regression'] = model
        print("✓")
        
        print("   - Ridge...", end=' ')
        model = ModelFactory.create_ridge()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluator.evaluate(y_test, y_pred, 'Ridge')
        self.models['Ridge'] = model
        print("✓")
        
        print("   - Lasso...", end=' ')
        model = ModelFactory.create_lasso()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluator.evaluate(y_test, y_pred, 'Lasso')
        self.models['Lasso'] = model
        print("✓")
        
        print("   - KNN...", end=' ')
        model = ModelFactory.create_knn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluator.evaluate(y_test, y_pred, 'KNN')
        self.models['KNN'] = model
        print("✓")
    
    def _train_ensemble_models(self, X_train, X_test, y_train, y_test):
        """Train ensemble models"""
        print("   - Random Forest...", end=' ')
        model = ModelFactory.create_random_forest()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluator.evaluate(y_test, y_pred, 'Random Forest')
        self.models['Random Forest'] = model
        print("✓")
        
        print("   - XGBoost...", end=' ')
        model = ModelFactory.create_xgboost()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluator.evaluate(y_test, y_pred, 'XGBoost')
        self.models['XGBoost'] = model
        print("✓")
        
        print("   - AdaBoost...", end=' ')
        model = ModelFactory.create_adaboost()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluator.evaluate(y_test, y_pred, 'AdaBoost')
        self.models['AdaBoost'] = model
        print("✓")
    
    def _train_deep_learning_models(self, X_train, X_test, y_train, y_test):
        """Train deep learning models"""
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        print("   - LSTM...", end=' ')
        model = ModelFactory.create_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=50, batch_size=32, 
                 validation_split=0.2, callbacks=[early_stop], verbose=0)
        y_pred = model.predict(X_test, verbose=0).flatten()
        self.evaluator.evaluate(y_test, y_pred, 'LSTM')
        self.models['LSTM'] = model
        print("✓")
        
        print("   - GRU...", end=' ')
        model = ModelFactory.create_gru(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=50, batch_size=32, 
                 validation_split=0.2, callbacks=[early_stop], verbose=0)
        y_pred = model.predict(X_test, verbose=0).flatten()
        self.evaluator.evaluate(y_test, y_pred, 'GRU')
        self.models['GRU'] = model
        print("✓")
        
        print("   - RNN...", end=' ')
        model = ModelFactory.create_rnn(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=50, batch_size=32, 
                 validation_split=0.2, callbacks=[early_stop], verbose=0)
        y_pred = model.predict(X_test, verbose=0).flatten()
        self.evaluator.evaluate(y_test, y_pred, 'RNN')
        self.models['RNN'] = model
        print("✓")

# ============================================================================
# VISUALIZATION
# ============================================================================
class Visualizer:
    """Create visualizations"""
    
    @staticmethod
    def plot_model_comparison(results_df, metric='R2'):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # R2 Score
        ax = axes[0, 0]
        results_df['R2'].sort_values(ascending=False).plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('R² Score Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('R² Score')
        
        # MAE
        ax = axes[0, 1]
        results_df['MAE'].sort_values(ascending=True).plot(kind='barh', ax=ax, color='coral')
        ax.set_title('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
        ax.set_xlabel('MAE')
        
        # RMSE
        ax = axes[1, 0]
        results_df['RMSE'].sort_values(ascending=True).plot(kind='barh', ax=ax, color='lightgreen')
        ax.set_title('RMSE (Root Mean Squared Error)', fontsize=12, fontweight='bold')
        ax.set_xlabel('RMSE')
        
        # MAPE
        ax = axes[1, 1]
        results_df['MAPE'].sort_values(ascending=True).plot(kind='barh', ax=ax, color='plum')
        ax.set_title('MAPE (Mean Absolute % Error)', fontsize=12, fontweight='bold')
        ax.set_xlabel('MAPE (%)')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_predictions(y_test, y_pred_dict):
        """Plot actual vs predicted"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (model_name, y_pred) in enumerate(list(y_pred_dict.items())[:4]):
            ax = axes[idx]
            ax.plot(y_test, label='Actual', linewidth=2, alpha=0.7)
            ax.plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
            ax.set_title(f'{model_name} Predictions', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_technical_indicators(df):
        """Plot technical indicators"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Price & Moving Averages
        ax = axes[0]
        ax.plot(df['Date'], df['Close'], label='Close Price', linewidth=2)
        ax.plot(df['Date'], df['MA_20'], label='MA 20', alpha=0.7)
        ax.plot(df['Date'], df['MA_50'], label='MA 50', alpha=0.7)
        ax.set_title('Price and Moving Averages', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RSI
        ax = axes[1]
        ax.plot(df['Date'], df['RSI'], label='RSI(14)', linewidth=2, color='orange')
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        ax.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        ax.set_title('RSI Indicator', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MACD
        ax = axes[2]
        ax.plot(df['Date'], df['MACD'], label='MACD', linewidth=2)
        ax.plot(df['Date'], df['MACD_Signal'], label='Signal', linewidth=2)
        ax.bar(df['Date'], df['MACD_Diff'], label='Histogram', alpha=0.3)
        ax.set_title('MACD Indicator', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Run pipeline
    pipeline = StockPredictionPipeline(symbol='AAPL', lookback_days=60)
    results = pipeline.run_full_pipeline()
    
    if results:
        # Visualizations
        print("\n7. Generating visualizations...")
        
        fig1 = Visualizer.plot_model_comparison(results['comparison'])
        fig1.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✓ Model comparison plot saved")
        
        fig2 = Visualizer.plot_technical_indicators(results['data'])
        fig2.savefig('technical_indicators.png', dpi=300, bbox_inches='tight')
        print("   ✓ Technical indicators plot saved")
        
        # Save results
        results['comparison'].to_csv('model_results.csv')
        print("   ✓ Results saved to CSV")
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
