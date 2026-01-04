📈 Stochastic Volatility Price Forecasting (GBM + OU) – NIFTY 50

Overview

This project explores a stochastic volatility–based approach to price forecasting for the NIFTY 50 index.
Instead of assuming constant volatility, the model combines Geometric Brownian Motion (GBM) for prices with an Ornstein–Uhlenbeck (OU) process to capture mean-reverting volatility behavior observed in real markets.
The goal is not short-term prediction, but probabilistic forecasting and volatility dynamics analysis.

Standard GBM models assume volatility is constant, which is rarely true in financial markets.
In practice, volatility tends to cluster and revert toward a long-term average.
This project extends GBM by explicitly modeling volatility as a stochastic, mean-reverting process, making simulations more realistic while keeping the model interpretable.

Data

Asset: NIFTY 50 Index
Frequency: Daily
Period: 2007 – 2026
Source: Yahoo Finance (yfinance)
Only closing prices are used after basic validation and cleaning.

Approach

Log Returns
Prices are converted to log returns, which are more stable and additive over time:

rt = log(Pt/Pt-1)


Stationarity Check

The Augmented Dickey–Fuller (ADF) test is applied to:
Price levels (non-stationary)
Log returns (stationary)
This confirms that modeling is done on statistically valid series and avoids spurious regressions.


GBM Calibration

The GBM parameters are estimated directly from historical log returns:
Drift (μ): Average daily return
Volatility (σ): Standard deviation of returns
These define the baseline price dynamics.


Realized Volatility

To capture time-varying volatility, a 20-day rolling standard deviation of log returns is used as realized volatility, roughly corresponding to one trading month.


OU Model for Volatility

Realized volatility is modeled using an Ornstein–Uhlenbeck process, allowing volatility to fluctuate while slowly reverting toward a long-term mean.
From this model:
Mean reversion speed
Long-run volatility level
Volatility shock persistence (half-life)
are estimated.


Monte Carlo Simulation

Volatility paths are simulated using the OU process
Price paths are simulated using GBM with time-varying volatility
5,000 Monte Carlo simulations are generated over a 1-year horizon (252 trading days)
This produces both:
Individual realistic future scenarios
Confidence bands reflecting forecast uncertainty


Results

The model outputs:
Simulated future price paths
90% confidence intervals
Volatility forecasts
QQ plot of log returns to assess tail behavior
Forecast data exported as CSV for further analysis
