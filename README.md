# Stock Price Prediction with LSTM and Attention Mechanism

A deep learning project that predicts stock prices using LSTM neural networks with multi-head attention mechanisms. The model incorporates technical indicators and employs Monte Carlo dropout for uncertainty estimation, along with walk-forward validation for robust performance evaluation.

# About

This project implements an advanced stock price prediction system that combines LSTM (Long Short-Term Memory) networks with attention mechanisms to forecast future stock prices. Unlike traditional prediction models, this implementation includes uncertainty quantification through Monte Carlo dropout and validates performance using walk-forward testing with online learning capabilities.

The system fetches historical stock data, engineers relevant technical features (RSI, MACD, returns), and trains a sophisticated neural network architecture that can learn temporal patterns in price movements. The model is designed to prevent data leakage through proper train-test splitting and includes robust backtesting capabilities to evaluate trading strategy performance.

 # Features

- **LSTM with Multi-Head Attention**: Captures temporal dependencies and focuses on relevant time steps
- **Technical Indicators**: Integrates RSI, MACD, returns, and volume data
- **Uncertainty Quantification**: Monte Carlo dropout for prediction confidence intervals
- **Walk-Forward Validation**: Time-series cross-validation with online learning
- **Backtesting Framework**: Evaluates trading strategy with Sharpe ratio and maximum drawdown
- **Data Leakage Prevention**: Proper splitting before scaling to maintain integrity
- **Regularization**: Dropout and early stopping to prevent overfitting

# Technical Architecture

# Model Components
- **Input Layer**: Sequence length of 60 time steps with 5 features
- **LSTM Layers**: Two stacked LSTM layers (50 and 25 units) with dropout
- **Attention Mechanism**: 2-head multi-head attention for feature importance
- **Regularization**: Layer normalization and dropout (0.2)
- **Output**: Single neuron for price prediction

# Features Used
1. Close price
2. Returns (percentage change)
3. RSI (Relative Strength Index, 14-period)
4. MACD (Moving Average Convergence Divergence)
5. Trading volume

# Installation

# Requirements
```bash
pip install yfinance tensorflow pandas numpy matplotlib scikit-learn
```

# Dependencies
- Python 3.7+
- TensorFlow 2.x
- yfinance
- pandas
- numpy
- matplotlib
- scikit-learn

# Usage

# Basic Usage
```python
# Initialize the predictor
predictor = StockPredictor("AAPL", seq_length=60, epochs=50)

# Fetch historical data
predictor.fetch_data(start='2010-01-01', end='2023-12-31')

# Train the model
history = predictor.train()

# Make predictions with uncertainty
pred_mean, pred_std = predictor.predict_with_uncertainty()

# Evaluate backtesting performance
sharpe, max_drawdown = predictor.backtest(pred_mean)

# Visualize results
predictor.plot_results()
```

### Walk-Forward Validation
```python
# Run walk-forward validation with online learning
results, avg_results = online_learning_walk_forward(
    predictor, 
    n_splits=5, 
    update_rate=0.2
)
```

## Model Training

The model uses several advanced training techniques:

- **Loss Function**: Huber loss (robust to outliers)
- **Optimizer**: Adam with learning rate of 0.0001
- **Callbacks**:
  - Early stopping (patience=15)
  - Learning rate reduction on plateau
- **Batch Size**: 32
- **Validation Split**: 80/20 train-test split

# Performance Metrics

The system evaluates predictions using:

- **RMSE** (Root Mean Squared Error): Prediction accuracy in dollar terms
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (Coefficient of Determination): Variance explained by the model
- **Sharpe Ratio**: Risk-adjusted return of the trading strategy
- **Maximum Drawdown**: Largest peak-to-trough decline

# Project Status

**This project is currently unfinished and under development.**

# Current Implementation
- Data fetching and preprocessing
-  Feature engineering (RSI, MACD, returns)
-  LSTM model with attention mechanism
-  Monte Carlo dropout for uncertainty
- Backtesting framework
-  Walk-forward validation
-  Visualization capabilities

# Known Issues
- Model performance shows negative R² in some splits (indicating poor generalization)
- Sharpe ratios are negative (strategy underperforms buy-and-hold)
- High volatility in predictions across validation splits

# To-Do / Future Improvements
-  Hyperparameter tuning (sequence length, LSTM units, attention heads)
-  Add more technical indicators (Bollinger Bands, Stochastic Oscillator)
-  Implement ensemble methods (multiple models)
-  Add sentiment analysis from news/social media
-  Improve feature engineering
-  Experiment with different architectures (GRU, Transformer)
-  Add risk management (stop-loss, position sizing)
-  Implement paper trading simulation
-  Create live prediction dashboard
-  Add multiple asset support
-  Optimize for lower transaction costs

# Methodology

# Data Pipeline
1. Fetch historical OHLCV data using yfinance
2. Calculate technical indicators
3. Split into train/test sets (80/20)
4. Scale features independently for train and test
5. Create sequences of 60 time steps

# Training Strategy
1. Build LSTM model with attention
2. Train on historical sequences
3. Use early stopping to prevent overfitting
4. Validate on held-out test set

# Evaluation
1. Generate predictions with uncertainty bounds
2. Backtest trading strategy
3. Calculate performance metrics
4. Visualize predictions vs actual prices

# Limitations and Disclaimers

**Important**: This is an experimental project for educational purposes.

- Stock markets are influenced by countless factors not captured in price data alone
- Past performance does not guarantee future results
- The model currently shows poor performance metrics
- **Do not use this for real trading decisions**
- Transaction costs and slippage are simplified
- Market microstructure effects are not modeled

# Contributing

This is a personal learning project, but suggestions and improvements are welcome! Areas that could use attention:

- Model architecture improvements
- Additional feature engineering
- Better hyperparameter tuning
- Enhanced visualization
- Documentation improvements

# License

MIT License - feel free to use this code for learning and experimentation.

# Acknowledgments

- Data provided by Yahoo Finance via yfinance
- Built with TensorFlow and Keras
- Inspired by research in financial time series forecasting

# References

- LSTM networks for sequence prediction
- Multi-head attention mechanisms
- Monte Carlo dropout for uncertainty quantification
- Walk-forward analysis in time series
- Technical analysis indicators (RSI, MACD)



**Note**: This project demonstrates machine learning techniques applied to financial data. It is intended for educational purposes and should not be used for actual trading without significant improvements and risk management strategies.
