"""
Stock Price Prediction - Streamlit Web Application
Interactive Dashboard for Model Predictions and Analysis
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
from stock_prediction import ModelFactory

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 20px;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("⚙️ Configuration")
stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="e.g., AAPL, GOOGL, MSFT")
lookback_days = st.sidebar.slider("Lookback Period (Days)", 30, 3650, 60)
prediction_period = st.sidebar.selectbox(
    "Prediction Period",
    ["Next Day", "Next Week", "Next Month"],
    help="Select prediction timeframe"
)

period_map = {"Next Day": 1, "Next Week": 5, "Next Month": 20}
pred_days = period_map[prediction_period]

# Model selection
selected_models = st.sidebar.multiselect(
    "Select Models",
    ["Linear Regression", "Ridge", "Lasso", "KNN", "Random Forest", 
     "XGBoost", "AdaBoost", "LSTM", "GRU", "RNN"],
    default=["Linear Regression", "XGBoost", "LSTM"]
)

# ============================================================================
# HEADER
# ============================================================================
st.title("📈 Stock Price Prediction System")
st.markdown("---")
st.write(f"**Symbol:** {stock_symbol} | **Lookback:** {lookback_days} days | **Prediction:** {prediction_period}")

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_stock_data(symbol, lookback):
    end_date = datetime.now()
    # Add buffer for technical indicators (need extra data for rolling averages)
    start_date = end_date - timedelta(days=lookback + 100)
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data

@st.cache_data
def add_technical_indicators(df):
    df = df.copy()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Mid'] - (bb_std * 2)
    
    return df.dropna()

# Load data
try:
    data = load_stock_data(stock_symbol, lookback_days)
    data = add_technical_indicators(data)
    
    if len(data) == 0:
        st.error("❌ No data found for this symbol")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
    with col2:
        change = data['Close'].iloc[-1] - data['Close'].iloc[0]
        st.metric("Period Change", f"${change:.2f}")
    with col3:
        pct_change = (change / data['Close'].iloc[0]) * 100
        st.metric("% Change", f"{pct_change:.2f}%")
    with col4:
        st.metric("Volatility", f"{data['Close'].std():.2f}")
    
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

st.markdown("---")

# ============================================================================
# TAB LAYOUT
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Price Chart", "🔧 Technical Indicators", "🤖 Model Predictions", 
     "📈 Model Comparison", "📋 Data"]
)

# ============================================================================
# TAB 1: PRICE CHART
# ============================================================================
with tab1:
    st.subheader("📊 Historical Price Chart")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Close'],
        mode='lines', name='Close Price',
        line=dict(color='steelblue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['MA_20'],
        mode='lines', name='MA 20',
        line=dict(color='orange', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['MA_50'],
        mode='lines', name='MA 50',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f"{stock_symbol} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: TECHNICAL INDICATORS
# ============================================================================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 RSI Indicator")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=data['Date'], y=data['RSI'],
            mode='lines', name='RSI(14)',
            line=dict(color='purple')
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(title="RSI(14)", height=400)
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        st.subheader("📊 MACD Indicator")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=data['Date'], y=data['MACD'],
            mode='lines', name='MACD',
            line=dict(color='blue')
        ))
        fig_macd.add_trace(go.Scatter(
            x=data['Date'], y=data['MACD_Signal'],
            mode='lines', name='Signal',
            line=dict(color='red')
        ))
        fig_macd.update_layout(title="MACD", height=400)
        st.plotly_chart(fig_macd, use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📊 Bollinger Bands")
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(
            x=data['Date'], y=data['Close'],
            mode='lines', name='Close',
            line=dict(color='blue')
        ))
        fig_bb.add_trace(go.Scatter(
            x=data['Date'], y=data['BB_Upper'],
            mode='lines', name='BB Upper',
            line=dict(color='gray', dash='dash')
        ))
        fig_bb.add_trace(go.Scatter(
            x=data['Date'], y=data['BB_Lower'],
            mode='lines', name='BB Lower',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        ))
        fig_bb.update_layout(title="Bollinger Bands", height=400)
        st.plotly_chart(fig_bb, use_container_width=True)
    
    with col4:
        st.subheader("📊 Volume")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=data['Date'], y=data['Volume'],
            name='Volume',
            marker_color='steelblue'
        ))
        fig_vol.update_layout(title="Trading Volume", height=400)
        st.plotly_chart(fig_vol, use_container_width=True)

# ============================================================================
# TAB 3: MODEL PREDICTIONS
# ============================================================================
with tab3:
    st.subheader("🤖 Model-based Predictions")
    
    # Prepare data
    feature_cols = ['Close', 'Volume']
    X = data[feature_cols].values
    y = data[['Close']].values  # Keep as 2D array for scaler
    
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # Create sequences
    window = 20
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - window - pred_days + 1):
        X_seq.append(X_scaled[i:i+window])
        y_seq.append(y_scaled[i+window+pred_days-1])
    
    X_seq = np.array(X_seq).reshape(len(X_seq), -1)
    y_seq = np.array(y_seq).flatten() # Flatten for 1D array expected by some models, but keep scaler for inverse
    
    if len(X_seq) == 0:
        st.warning("⚠️ Not enough data for prediction")
    else:
        # Train test split
        split = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]
        
        # Keep original scale y_test for metrics/plotting (optional, but better to inverse transform predictions)
        # Actually, we should inverse transform everything at the end.
        
        predictions = {}
        metrics = {}
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("### Training models...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            model_list = [
                ("Linear Regression", LinearRegression()),
                ("Ridge", Ridge(alpha=1.0)),
                ("Lasso", Lasso(alpha=0.1, max_iter=10000)),
                ("KNN", KNeighborsRegressor(n_neighbors=5)),
                ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=10)),
                ("XGBoost", xgb.XGBRegressor(n_estimators=100, max_depth=6)),
                ("AdaBoost", AdaBoostRegressor(n_estimators=100, learning_rate=0.1))
            ]
            
            # Add DL models if selected
            # Note: DL models need to be created with input shape, so we do it inside the loop or just before
            
            for idx, name in enumerate(selected_models):
                status_text.text(f"Training {name}...")
                
                y_pred_scaled = None
                
                # Handle DL Models
                if name in ["LSTM", "GRU", "RNN"]:
                    n_features = X.shape[1]
                    
                    X_train_dl = X_train.reshape((X_train.shape[0], window, n_features))
                    X_test_dl = X_test.reshape((X_test.shape[0], window, n_features))
                    
                    if name == "LSTM":
                        model = ModelFactory.create_lstm(input_shape=(window, n_features))
                    elif name == "GRU":
                        model = ModelFactory.create_gru(input_shape=(window, n_features))
                    elif name == "RNN":
                        model = ModelFactory.create_rnn(input_shape=(window, n_features))
                    
                    # Train DL model
                    model.fit(X_train_dl, y_train, epochs=20, batch_size=32, verbose=0)
                    y_pred_scaled = model.predict(X_test_dl, verbose=0).flatten()
                    
                else:
                    # ML Models
                    model_instance = None
                    for m_name, m_inst in model_list:
                        if m_name == name:
                            model_instance = m_inst
                            break
                    
                    if model_instance is not None:
                        model = model_instance
                        model.fit(X_train, y_train)
                        y_pred_scaled = model.predict(X_test)
                    else:
                        continue

                # Inverse transform predictions
                if y_pred_scaled is not None:
                    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    predictions[name] = y_pred
                    
                    # Inverse transform y_test for metrics (do this once or for each iteration if we want to be safe)
                    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    
                    r2 = r2_score(y_test_original, y_pred)
                    mae = mean_absolute_error(y_test_original, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
                    
                    metrics[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                
                progress_bar.progress((idx + 1) / len(selected_models))
            
            status_text.text("✓ Training completed!")
            
            # Plot predictions
            fig_pred = go.Figure()
            
            # Use original scale y_test for plotting actuals
            if len(y_test) > 0:
                 y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                 fig_pred.add_trace(go.Scatter(
                    y=y_test_original, mode='lines', name='Actual',
                    line=dict(color='black', width=2)
                ))
            
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
            for i, (name, y_pred) in enumerate(predictions.items()):
                fig_pred.add_trace(go.Scatter(
                    y=y_pred, mode='lines', name=name,
                    line=dict(color=colors[i % len(colors)], dash='dash')
                ))
            
            fig_pred.update_layout(
                title=f"Model Predictions - {prediction_period}",
                xaxis_title="Test Sample",
                yaxis_title="Price ($)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col2:
            st.write("### Metrics")
            metrics_df = pd.DataFrame(metrics).T
            st.dataframe(metrics_df.round(4), use_container_width=True)

# ============================================================================
# TAB 4: MODEL COMPARISON
# ============================================================================
with tab4:
    st.subheader("📈 Model Performance Comparison")
    
    if metrics:
        metrics_df = pd.DataFrame(metrics).T
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### R² Score Comparison")
            fig_r2 = px.bar(
                metrics_df['R2'].sort_values(ascending=False),
                labels={'value': 'R² Score', 'index': 'Model'},
                color=metrics_df['R2'].sort_values(ascending=False),
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            st.write("### RMSE Comparison")
            fig_rmse = px.bar(
                metrics_df['RMSE'].sort_values(ascending=True),
                labels={'value': 'RMSE', 'index': 'Model'},
                color=metrics_df['RMSE'].sort_values(ascending=True),
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("### MAE Comparison")
            fig_mae = px.bar(
                metrics_df['MAE'].sort_values(ascending=True),
                labels={'value': 'MAE', 'index': 'Model'},
                color=metrics_df['MAE'].sort_values(ascending=True),
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with col4:
            st.write("### Summary Statistics")
            st.dataframe(metrics_df.describe().round(4))

# ============================================================================
# TAB 5: DATA
# ============================================================================
with tab5:
    st.subheader("📋 Raw Data")
    st.dataframe(data.tail(20), use_container_width=True)
    
    # Download button
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f"{stock_symbol}_data.csv",
        mime="text/csv"
    )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>Stock Price Prediction System | Powered by ML & Deep Learning</p>
    <p>⚠️ Disclaimer: This tool is for educational purposes only. Always consult financial advisors before making investment decisions.</p>
</div>
""", unsafe_allow_html=True)
