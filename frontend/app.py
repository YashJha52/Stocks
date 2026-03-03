import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_loader import DataLoader
from src.feature_engineering import StatisticalFeatures
from src.models import StockPredictor
from src.report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="Stock Predictor & Financial Analyst",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'report' not in st.session_state:
    st.session_state.report = None

# Main title
st.markdown('<h1 class="main-header">📈 Stock Predictor & Financial Analyst</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## Search Company")
ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()
search_button = st.sidebar.button("Analyze")

# Main content
if search_button or ticker:
    # Load data
    data_loader = DataLoader()
    
    try:
        # Get company info
        company_info = data_loader.get_company_info(ticker)
        
        # Display company info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Company Name", company_info.get('longName', 'N/A'))
        with col2:
            st.metric("Sector", company_info.get('sector', 'N/A'))
        with col3:
            st.metric("Current Price", f"${company_info.get('currentPrice', 0):.2f}")
        
        # Load stock data
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.DateOffset(days=365)).strftime('%Y-%m-%d')
        
        stock_data = data_loader.load_stock_data(ticker, start_date, end_date)
        
        if not stock_data.empty:
            # Display stock chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name=ticker
            ))
            fig.update_layout(
                title=f'{ticker} Stock Price',
                yaxis_title='Price ($)',
                xaxis_title='Date'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Make predictions
            if st.button("Generate Predictions"):
                with st.spinner("Analyzing data and generating predictions..."):
                    # Compute features
                    fe = StatisticalFeatures()
                    features = fe.compute_features(stock_data)
                    features['Close'] = stock_data['Close']
                    features = features.dropna()
                    
                    # Train models
                    predictor = StockPredictor()
                    X = features.drop('Close', axis=1)
                    y = features['Close']
                    predictor.train_models(X, y)
                    
                    # Make predictions
                    last_features = X.iloc[-1:].copy()
                    predictions = predictor.predict(last_features)
                    
                    # Store predictions
                    st.session_state.predictions = predictions
                    
                    # Generate report
                    report_gen = ReportGenerator()
                    financial_data = data_loader.get_financial_statements(ticker)
                    report = report_gen.generate_financial_report(
                        ticker, company_info, financial_data, predictions
                    )
                    st.session_state.report = report
            
            # Display predictions
            if st.session_state.predictions is not None:
                st.markdown('<h2 class="sub-header">Stock Price Predictions</h2>', unsafe_allow_html=True)
                
                pred_df = st.session_state.predictions
                st.dataframe(pred_df)
                
                # Plot predictions
                fig = go.Figure()
                for col in pred_df.columns:
                    fig.add_trace(go.Scatter(
                        x=pred_df.index,
                        y=pred_df[col],
                        mode='lines+markers',
                        name=col.upper()
                    ))
                fig.update_layout(
                    title='Stock Price Predictions',
                    xaxis_title='Date',
                    yaxis_title='Predicted Price ($)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display report
            if st.session_state.report:
                st.markdown('<h2 class="sub-header">Financial Analysis Report</h2>', unsafe_allow_html=True)
                st.markdown(st.session_state.report)
        
    except Exception as e:
        st.error(f"Error analyzing {ticker}: {str(e)}")
        st.info("Please check the ticker symbol and try again.")

# Footer
st.markdown("---")
st.markdown("© 2023 Stock Predictor & Financial Analyst. Data provided by Yahoo Finance.")