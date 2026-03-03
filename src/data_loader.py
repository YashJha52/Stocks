import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup

class DataLoader:
    def load_stock_data(self, ticker, start_date, end_date):
        """Load historical stock data"""
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        return data
    
    def get_company_info(self, ticker):
        """Get company information"""
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    
    def get_financial_statements(self, ticker):
        """Get financial statements"""
        stock = yf.Ticker(ticker)
        financials = {
            'income_statement': stock.financials,
            'balance_sheet': stock.balance_sheet,
            'cash_flow': stock.cashflow
        }
        return financials