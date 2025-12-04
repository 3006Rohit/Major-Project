"""
Stock Price Prediction - HTML Dashboard
Complete visualization dashboard with interactive charts
"""

import json
import base64
from pathlib import Path

def generate_html_dashboard():
    """Generate interactive HTML dashboard"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Price Prediction Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            
            .header h1 {
                color: #667eea;
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .header p {
                color: #666;
                font-size: 1.1em;
            }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
            }
            
            .metric-card h3 {
                color: #666;
                font-size: 0.9em;
                margin-bottom: 10px;
                text-transform: uppercase;
            }
            
            .metric-card .value {
                font-size: 1.8em;
                font-weight: bold;
                color: #333;
            }
            
            .metric-card .subtext {
                color: #999;
                font-size: 0.85em;
                margin-top: 5px;
            }
            
            .chart-container {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            
            .chart-container h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.5em;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }
            
            .chart-row {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .comparison-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            
            .comparison-table th {
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
            }
            
            .comparison-table td {
                padding: 12px;
                border-bottom: 1px solid #eee;
            }
            
            .comparison-table tr:hover {
                background: #f5f5f5;
            }
            
            .best {
                background: #d4edda;
                font-weight: bold;
            }
            
            .worst {
                background: #f8d7da;
                font-weight: bold;
            }
            
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            
            .tab-btn {
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
                transition: all 0.3s;
            }
            
            .tab-btn:hover {
                background: #764ba2;
                transform: translateY(-2px);
            }
            
            .tab-btn.active {
                background: #764ba2;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            
            .tab-content {
                display: none;
            }
            
            .tab-content.active {
                display: block;
            }
            
            .footer {
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: #666;
                margin-top: 30px;
            }
            
            .alert {
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
            }
            
            .alert-info {
                background: #d1ecf1;
                color: #0c5460;
                border-left: 4px solid #0c5460;
            }
            
            .alert-warning {
                background: #fff3cd;
                color: #856404;
                border-left: 4px solid #856404;
            }
            
            @media (max-width: 768px) {
                .chart-row {
                    grid-template-columns: 1fr;
                }
                
                .metrics-grid {
                    grid-template-columns: 1fr;
                }
                
                .header h1 {
                    font-size: 1.8em;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="header">
                <h1>📈 Stock Price Prediction Dashboard</h1>
                <p>Machine Learning Models Comparison: Next Day, Week & Month Predictions</p>
            </div>
            
            <!-- Alert -->
            <div class="alert alert-warning">
                <strong>⚠️ Disclaimer:</strong> This tool is for educational purposes only. 
                Past performance does not guarantee future results. Always consult with financial advisors 
                before making investment decisions.
            </div>
            
            <!-- Metrics -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Current Price</h3>
                    <div class="value">$<span id="current-price">0.00</span></div>
                    <div class="subtext">AAPL (as of today)</div>
                </div>
                <div class="metric-card">
                    <h3>Best Model (R²)</h3>
                    <div class="value"><span id="best-model">XGBoost</span></div>
                    <div class="subtext"><span id="best-r2">0.00</span></div>
                </div>
                <div class="metric-card">
                    <h3>Lowest MAE</h3>
                    <div class="value">$<span id="lowest-mae">0.00</span></div>
                    <div class="subtext">Mean Absolute Error</div>
                </div>
                <div class="metric-card">
                    <h3>Lowest RMSE</h3>
                    <div class="value">$<span id="lowest-rmse">0.00</span></div>
                    <div class="subtext">Root Mean Squared Error</div>
                </div>
            </div>
            
            <!-- Tabs -->
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('predictions')">📊 Predictions</button>
                <button class="tab-btn" onclick="switchTab('comparison')">📈 Model Comparison</button>
                <button class="tab-btn" onclick="switchTab('technical')">🔧 Technical Indicators</button>
                <button class="tab-btn" onclick="switchTab('performance')">📋 Performance Metrics</button>
            </div>
            
            <!-- Tab: Predictions -->
            <div id="predictions" class="tab-content active">
                <div class="chart-container">
                    <h2>Model Predictions Comparison</h2>
                    <div class="chart-row">
                        <div id="chart-next-day"></div>
                    </div>
                    <div class="chart-row">
                        <div id="chart-next-week"></div>
                        <div id="chart-next-month"></div>
                    </div>
                </div>
            </div>
            
            <!-- Tab: Comparison -->
            <div id="comparison" class="tab-content">
                <div class="chart-container">
                    <h2>R² Score Comparison</h2>
                    <div class="chart-row">
                        <div id="chart-r2"></div>
                    </div>
                    
                    <h2>Error Metrics Comparison</h2>
                    <div class="chart-row">
                        <div id="chart-mae"></div>
                        <div id="chart-rmse"></div>
                    </div>
                </div>
            </div>
            
            <!-- Tab: Technical -->
            <div id="technical" class="tab-content">
                <div class="chart-container">
                    <h2>Technical Indicators</h2>
                    <div class="chart-row">
                        <div id="chart-rsi"></div>
                    </div>
                    <div class="chart-row">
                        <div id="chart-macd"></div>
                        <div id="chart-bollinger"></div>
                    </div>
                </div>
            </div>
            
            <!-- Tab: Performance -->
            <div id="performance" class="tab-content">
                <div class="chart-container">
                    <h2>Detailed Performance Metrics</h2>
                    <table class="comparison-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>R² Score</th>
                                <th>MAE</th>
                                <th>RMSE</th>
                                <th>MAPE (%)</th>
                            </tr>
                        </thead>
                        <tbody id="performance-table">
                            <!-- Will be populated by JavaScript -->
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="footer">
                <p>Stock Price Prediction System | ML & Deep Learning Models</p>
                <p style="font-size: 0.9em; margin-top: 10px;">
                    Models: Linear Regression, Ridge, Lasso, KNN, Random Forest, XGBoost, AdaBoost, LSTM, GRU, RNN
                </p>
            </div>
        </div>
        
        <script>
            // Sample data structure for models
            const models = [
                { name: 'Linear Regression', r2: 0.78, mae: 2.5, rmse: 3.2, mape: 4.5 },
                { name: 'Ridge', r2: 0.79, mae: 2.4, rmse: 3.1, mape: 4.3 },
                { name: 'Lasso', r2: 0.75, mae: 2.8, rmse: 3.5, mape: 4.9 },
                { name: 'KNN', r2: 0.81, mae: 2.2, rmse: 2.9, mape: 3.9 },
                { name: 'Random Forest', r2: 0.85, mae: 1.9, rmse: 2.5, mape: 3.3 },
                { name: 'XGBoost', r2: 0.87, mae: 1.7, rmse: 2.3, mape: 3.0 },
                { name: 'AdaBoost', r2: 0.83, mae: 2.1, rmse: 2.7, mape: 3.6 },
                { name: 'LSTM', r2: 0.84, mae: 2.0, rmse: 2.6, mape: 3.4 },
                { name: 'GRU', r2: 0.86, mae: 1.8, rmse: 2.4, mape: 3.1 },
                { name: 'RNN', r2: 0.82, mae: 2.3, rmse: 2.8, mape: 3.8 }
            ];
            
            // Initialize metrics
            function initMetrics() {
                const best = models.reduce((prev, current) => 
                    (prev.r2 > current.r2) ? prev : current
                );
                const lowestMAE = models.reduce((prev, current) => 
                    (prev.mae < current.mae) ? prev : current
                );
                const lowestRMSE = models.reduce((prev, current) => 
                    (prev.rmse < current.rmse) ? prev : current
                );
                
                document.getElementById('best-model').textContent = best.name;
                document.getElementById('best-r2').textContent = best.r2.toFixed(4);
                document.getElementById('lowest-mae').textContent = lowestMAE.mae.toFixed(2);
                document.getElementById('lowest-rmse').textContent = lowestRMSE.rmse.toFixed(2);
            }
            
            // Tab switching
            function switchTab(tabName) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                document.querySelectorAll('.tab-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                
                // Show selected tab
                document.getElementById(tabName).classList.add('active');
                event.target.classList.add('active');
            }
            
            // Create R² comparison chart
            function createR2Chart() {
                const names = models.map(m => m.name);
                const r2_scores = models.map(m => m.r2);
                
                const trace = {
                    x: names,
                    y: r2_scores,
                    type: 'bar',
                    marker: { color: r2_scores, colorscale: 'Viridis' }
                };
                
                const layout = {
                    title: 'R² Score by Model',
                    xaxis: { title: 'Model' },
                    yaxis: { title: 'R² Score' },
                    height: 400
                };
                
                Plotly.newPlot('chart-r2', [trace], layout);
            }
            
            // Create MAE comparison chart
            function createMAEChart() {
                const names = models.map(m => m.name);
                const mae = models.map(m => m.mae);
                
                const trace = {
                    x: names,
                    y: mae,
                    type: 'bar',
                    marker: { color: 'lightblue' }
                };
                
                const layout = {
                    title: 'MAE by Model',
                    xaxis: { title: 'Model' },
                    yaxis: { title: 'MAE ($)' },
                    height: 400
                };
                
                Plotly.newPlot('chart-mae', [trace], layout);
            }
            
            // Create RMSE comparison chart
            function createRMSEChart() {
                const names = models.map(m => m.name);
                const rmse = models.map(m => m.rmse);
                
                const trace = {
                    x: names,
                    y: rmse,
                    type: 'bar',
                    marker: { color: 'lightgreen' }
                };
                
                const layout = {
                    title: 'RMSE by Model',
                    xaxis: { title: 'Model' },
                    yaxis: { title: 'RMSE ($)' },
                    height: 400
                };
                
                Plotly.newPlot('chart-rmse', [trace], layout);
            }
            
            // Create predictions chart
            function createPredictionsChart() {
                const x = Array.from({length: 20}, (_, i) => i);
                const actual = Array.from({length: 20}, () => 
                    150 + Math.random() * 10 - 5
                );
                
                const traces = [
                    {
                        x: x,
                        y: actual,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Actual Price',
                        line: { color: 'black', width: 2 }
                    }
                ];
                
                models.slice(0, 4).forEach((model, idx) => {
                    const colors = ['red', 'blue', 'green', 'orange'];
                    const predicted = Array.from({length: 20}, () => 
                        148 + Math.random() * 8 - 4
                    );
                    
                    traces.push({
                        x: x,
                        y: predicted,
                        type: 'scatter',
                        mode: 'lines',
                        name: model.name,
                        line: { color: colors[idx], dash: 'dash' }
                    });
                });
                
                const layout = {
                    title: 'Model Predictions (Next Day)',
                    xaxis: { title: 'Days' },
                    yaxis: { title: 'Price ($)' },
                    height: 400,
                    hovermode: 'x unified'
                };
                
                Plotly.newPlot('chart-next-day', traces, layout);
            }
            
            // Create performance table
            function createPerformanceTable() {
                const tbody = document.getElementById('performance-table');
                
                models.forEach(model => {
                    const row = document.createElement('tr');
                    
                    // Highlight best/worst
                    const isBestR2 = model.r2 === Math.max(...models.map(m => m.r2));
                    const isWorstR2 = model.r2 === Math.min(...models.map(m => m.r2));
                    
                    row.innerHTML = `
                        <td><strong>${model.name}</strong></td>
                        <td class="${isBestR2 ? 'best' : isWorstR2 ? 'worst' : ''}">${model.r2.toFixed(4)}</td>
                        <td>${model.mae.toFixed(2)}</td>
                        <td>${model.rmse.toFixed(2)}</td>
                        <td>${model.mape.toFixed(2)}%</td>
                    `;
                    
                    tbody.appendChild(row);
                });
            }
            
            // Initialize on load
            window.addEventListener('DOMContentLoaded', () => {
                initMetrics();
                createR2Chart();
                createMAEChart();
                createRMSEChart();
                createPredictionsChart();
                createPerformanceTable();
            });
        </script>
    </body>
    </html>
    """
    
    return html_content

if __name__ == "__main__":
    html = generate_html_dashboard()
    
    with open('dashboard.html', 'w') as f:
        f.write(html)
    
    print("✓ Dashboard generated: dashboard.html")
