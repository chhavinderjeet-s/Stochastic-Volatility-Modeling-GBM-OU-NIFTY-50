import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 1. DOWNLOAD DAILY NIFTY50 DATA (2007–2026)
# =========================================================

ticker = "^NSEI"  # NIFTY 50 index
start_date = "2007-01-01"
end_date = "2026-01-01"

df = yf.download(ticker, start=start_date, end=end_date, progress=False)

if df.empty:
    raise ValueError("No data downloaded from yfinance. Check ticker or dates.")

df = df[["Close"]].dropna().reset_index()
df.columns = ["date", "price"]

prices = df["price"].values.astype(float)

print(f"Downloaded {len(df)} daily observations")

# =========================================================
# 2. DAILY LOG RETURNS
# =========================================================

log_returns = np.log(prices[1:] / prices[:-1])

# =========================================================
# 3. STATIONARITY CHECK (ADF)
# =========================================================

def adf_test(series, name):
    stat, pvalue, *_ = adfuller(series)
    print(f"{name:<12} | ADF Stat: {stat:>8.4f} | p-value: {pvalue:.4f}")

print("\nADF TEST RESULTS")
adf_test(prices, "Price")
adf_test(log_returns, "Returns")

# =========================================================
# 4. GBM PARAMETERS (DAILY)
# =========================================================

mu = np.mean(log_returns)
sigma_hist = np.std(log_returns)

print("\nGBM PARAMETERS (DAILY)")
print(f"Drift (mu): {mu:.6f}")
print(f"Volatility (sigma): {sigma_hist:.6f}")
plt.figure(figsize=(10, 5))
plt.hist(log_returns, bins=50, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma_hist)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Histogram of Daily Log Returns with Normal PDF')
plt.xlabel('Log Return')
plt.ylabel('Density')
plt.grid(True)
plt.show()
# =========================================================
# 5. REALIZED VOLATILITY (ROLLING)
# =========================================================

vol_window = 20  # ~1 trading month
realized_vol = (
    pd.Series(log_returns)
    .rolling(vol_window)
    .std()
    .dropna()
    .values
)

# =========================================================
# 6. ORNSTEIN–UHLENBECK MODEL FOR VOLATILITY
# =========================================================

def estimate_ou(series):
    x = series[:-1]
    y = series[1:]
    theta = -np.polyfit(x, y - x, 1)[0]
    mu = np.mean(series)
    sigma = np.std(y - x)
    return max(theta, 1e-4), mu, sigma

theta_v, mu_v, sigma_v = estimate_ou(realized_vol)
half_life = np.log(2) / theta_v

print("\nOU VOLATILITY PARAMETERS")
print(f"Theta: {theta_v:.4f}")
print(f"Long-run mean: {mu_v:.4f}")
print(f"Sigma: {sigma_v:.4f}")
print(f"Half-life (days): {half_life:.2f}")

# =========================================================
# 7. SIMULATE OU VOLATILITY
# =========================================================

def simulate_ou_vol(v0, theta, mu, sigma, steps):
    vol = np.zeros(steps)
    vol[0] = v0
    for t in range(1, steps):
        vol[t] = (
            vol[t-1]
            + theta * (mu - vol[t-1])
            + sigma * np.random.normal()
        )
        vol[t] = max(vol[t], 1e-4)
    return vol

# =========================================================
# 8. GBM WITH STOCHASTIC VOLATILITY
# =========================================================

def simulate_gbm_sv(S0, mu, vol_path, steps, n_sims=5000):
    paths = np.zeros((n_sims, steps + 1))
    paths[:, 0] = S0

    for t in range(1, steps + 1):
        z = np.random.normal(size=n_sims)
        sigma_t = vol_path[t-1]
        paths[:, t] = paths[:, t-1] * np.exp(
            (mu - 0.5 * sigma_t**2) + sigma_t * z
        )
    return paths

# =========================================================
# 9. RUN 1-YEAR FORECAST (252 TRADING DAYS)
# =========================================================

np.random.seed(42)

forecast_days = 252
S0 = prices[-1]
v0 = realized_vol[-1]

vol_forecast = simulate_ou_vol(v0, theta_v, mu_v, sigma_v, forecast_days)
price_paths = simulate_gbm_sv(S0, mu, vol_forecast, forecast_days)

# Pick ONE realistic path
np.random.seed(7)  # reproducible
random_idx = np.random.randint(0, price_paths.shape[0])
realistic_path = price_paths[random_idx, 1:]


p05 = np.percentile(price_paths[:, 1:], 5, axis=0)
p95 = np.percentile(price_paths[:, 1:], 95, axis=0)

future_dates = pd.bdate_range(
    start=df["date"].iloc[-1] + pd.Timedelta(days=1),
    periods=forecast_days
)

# =========================================================
# 10. SAVE FORECAST
# =========================================================

forecast_df = pd.DataFrame({
    "date": future_dates,
    "simulated_price": realistic_path,
    "p05": p05,
    "p95": p95
})


forecast_df.to_csv("gbm_ou_nifty50_daily_forecast.csv", index=False)

# =========================================================
# 11. PLOT PRICE FORECAST
# =========================================================

plt.figure(figsize=(15, 7))
plt.plot(df["date"], prices, label="Historical Price", linewidth=2)
plt.plot(
    future_dates,
    realistic_path,
    color="orange",
    linewidth=2.2,
    label="One Simulated Future Path"
)
plt.fill_between(future_dates, p05, p95, alpha=0.3, label="90% CI")
plt.axvline(df["date"].iloc[-1], linestyle="--", color="black")

plt.title("NIFTY50 – Simulated Daily Price Path (GBM + Mean-Reverting Volatility)")
plt.xlabel("Date")
plt.ylabel("Index Level")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("nifty50_gbm_ou_forecast.png", dpi=300)
plt.show()



# =========================================================
# 12. QQ PLOT (TAIL RISK)
# =========================================================

plt.figure(figsize=(6, 6))
stats.probplot(log_returns, dist="norm", plot=plt)
plt.title("QQ Plot of Daily NIFTY50 Log-Returns")
plt.grid(alpha=0.3)
plt.savefig("nifty50_qq_plot.png", dpi=300)
plt.show()


# =========================================================
# 13. SUMMARY
# =========================================================

print("\nFORECAST SUMMARY")
print("=" * 60)
print(f"Current Price: {S0:.2f}")
print(f"Expected Price (1Y): {mean_forecast[-1]:.2f}")
print(f"Expected Change: {(mean_forecast[-1]/S0 - 1)*100:+.2f}%")
print("=" * 60)
