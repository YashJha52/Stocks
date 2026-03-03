import pandas as pd
import numpy as np

class ReportGenerator:
    def generate_financial_report(self, ticker, company_info, financial_data, predictions):
        """Generate comprehensive financial report"""
        
        report = f"""
# Financial Analysis Report: {ticker}

## Company Overview
**Name**: {company_info.get('longName', 'N/A')}
**Sector**: {company_info.get('sector', 'N/A')}
**Industry**: {company_info.get('industry', 'N/A')}
**Market Cap**: ${company_info.get('marketCap', 0):,.0f}
**Current Price**: ${company_info.get('currentPrice', 0):.2f}

## Financial Metrics
"""
        
        # Add key metrics
        if 'metrics' in financial_data:
            metrics = financial_data['metrics']
            report += f"""
- **P/E Ratio**: {metrics.get('peRatio', 'N/A')}
- **P/B Ratio**: {metrics.get('priceToBook', 'N/A')}
- **Debt-to-Equity**: {metrics.get('debtToEquity', 'N/A')}
- **ROE**: {metrics.get('returnOnEquity', 'N/A')}%
- **Profit Margin**: {metrics.get('profitMargins', 'N/A')}%
"""
        
        # Add prediction analysis
        if not predictions.empty:
            current_price = predictions.iloc[0].mean()
            future_price = predictions.iloc[-1].mean()
            price_change = ((future_price - current_price) / current_price) * 100
            
            report += f"""
## Stock Price Prediction
**Current Price**: ${current_price:.2f}
**Predicted Price (30 days)**: ${future_price:.2f}
**Expected Change**: {price_change:+.2f}%

## Investment Recommendation
"""
            
            if price_change > 5:
                report += "**BUY** - The model predicts significant upside potential.\n"
            elif price_change < -5:
                report += "**SELL** - The model predicts potential decline.\n"
            else:
                report += "**HOLD** - The model predicts stable price movement.\n"
        
        report += """
## Risk Factors
- Market volatility and economic conditions
- Company-specific risks and competition
- Model limitations and prediction uncertainty

## Disclaimer
This report is generated using automated analysis and should not be considered as financial advice. 
Please consult with a qualified financial advisor before making investment decisions.
"""
        
        return report