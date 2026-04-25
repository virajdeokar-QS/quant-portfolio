"""
PROJECT 1: India Equity Factor Backtester
==========================================
Momentum + Low Volatility Factor Model | 30-Stock Universe | 2019-2024
Author: Viraj | Quant Strategy Interview Project

NOTE: All price data is generated internally with a realistic
      covariance structure — no external CSV file required.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

print("=" * 62)
print("  PROJECT 1: INDIA FACTOR BACKTESTER")
print("  Universe: 30 Stocks | Period: 2019–2024")
print("=" * 62)

# ─────────────────────────────────────────────────────
# STEP 0: GENERATE SIMULATED PRICE DATA INTERNALLY
# Realistic India equity market simulation:
#   - 30 stocks across 5 sectors
#   - Correlated via a factor model (market + sector)
#   - Daily returns with fat tails (t-distribution)
#   - ~1500 trading days (2019–2024)
# ─────────────────────────────────────────────────────
print("\n[0/6] Generating simulated price data...")

np.random.seed(42)

TICKERS = [
    # Financials
    'HDFCBANK','ICICIBANK','KOTAKBANK','AXISBANK','SBIN',
    # IT
    'TCS','INFY','WIPRO','HCLTECH','TECHM',
    # Consumer
    'HINDUNILVR','ITC','NESTLEIND','BRITANNIA','DABUR',
    # Energy & Industrials
    'RELIANCE','ONGC','POWERGRID','NTPC','COALINDIA',
    # Auto & Pharma
    'MARUTI','TATAMOTORS','BAJAJ-AUTO','SUNPHARMA','DRREDDY',
    # Metals & Others
    'TATASTEEL','JSWSTEEL','HINDALCO','ULTRACEMCO','ASIANPAINT'
]

N_STOCKS  = len(TICKERS)           # 30
N_DAYS    = 1500                   # ~6 years of trading days
START     = pd.Timestamp('2019-01-01')
DATES     = pd.bdate_range(START, periods=N_DAYS)

# Sector assignments (0–5)
SECTORS = np.array([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2,
                    3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5])

# --- Factor loadings ---
mkt_beta   = np.random.uniform(0.6, 1.4, N_STOCKS)   # market beta
sect_beta  = np.random.uniform(0.3, 0.8, N_STOCKS)   # sector beta
idio_vol   = np.random.uniform(0.008, 0.022, N_STOCKS)  # idiosyncratic vol

# --- Daily return simulation ---
mkt_ret   = np.random.standard_t(df=5, size=N_DAYS) * 0.008   # ~8% ann vol market
sect_rets = np.random.standard_t(df=5, size=(6, N_DAYS)) * 0.005

daily_rets = np.zeros((N_DAYS, N_STOCKS))
for d in range(N_DAYS):
    for s in range(N_STOCKS):
        daily_rets[d, s] = (mkt_beta[s]  * mkt_ret[d] +
                            sect_beta[s] * sect_rets[SECTORS[s], d] +
                            idio_vol[s]  * np.random.standard_t(df=6))

# Add a gentle upward drift (~12% ann return for the universe)
daily_rets += 0.00045

# Convert to price series (base = 100)
prices = pd.DataFrame(
    100 * np.cumprod(1 + daily_rets, axis=0),
    index=DATES,
    columns=TICKERS
)

print(f"    {prices.shape[1]} stocks × {prices.shape[0]} trading days generated")

# ─────────────────────────────────────────────────────
# STEP 1: MONTHLY RETURNS
# ─────────────────────────────────────────────────────
print("\n[1/6] Computing monthly returns...")

monthly_prices  = prices.resample('ME').last()
monthly_returns = monthly_prices.pct_change().dropna()
print(f"    {monthly_returns.shape[0]} monthly observations per stock")

# ─────────────────────────────────────────────────────
# STEP 2: BUILD FACTOR SIGNALS
#
# MOMENTUM SIGNAL (12-1 Month):
#   At each rebalance date T, look at returns from
#   T-13 to T-2 (12 months, SKIPPING last month).
#   Why skip last month? Short-term reversal.
#   High score = strong past winner = go LONG.
#
# LOW VOLATILITY SIGNAL:
#   At each rebalance date T, compute std of past
#   12 monthly returns. Lower std = better signal.
#   Negative sign so high score = low vol = LONG.
# ─────────────────────────────────────────────────────
print("\n[2/6] Building factor signals...")

N = len(monthly_returns)
momentum_signals = pd.DataFrame(np.nan, index=monthly_returns.index,
                                columns=monthly_returns.columns)
vol_signals      = pd.DataFrame(np.nan, index=monthly_returns.index,
                                columns=monthly_returns.columns)

for i in range(13, N):
    window = monthly_returns.iloc[i-13 : i-1]   # 12 months, skip last
    momentum_signals.iloc[i] = window.sum()
    vol_signals.iloc[i]      = window.std()

signal_idx  = momentum_signals.dropna(how='all').index
fwd_returns = monthly_returns.shift(-1).loc[signal_idx].dropna(how='all')
common_idx  = signal_idx.intersection(fwd_returns.index)

mom_sig = momentum_signals.loc[common_idx]
vol_sig = vol_signals.loc[common_idx]
fwd_ret = fwd_returns.loc[common_idx]

print(f"    Signals ready: {len(common_idx)} rebalance dates")

# ─────────────────────────────────────────────────────
# STEP 3: PORTFOLIO CONSTRUCTION
#
# Each month:
#  1. Rank all stocks by signal (1=worst, N=best)
#  2. LONG: top quintile (top 20%)
#  3. SHORT: bottom quintile (bottom 20%)
#  4. Equal weight within each leg
#  5. Record next-month return of L/S portfolio
# ─────────────────────────────────────────────────────
print("\n[3/6] Constructing long-short portfolios...")

def factor_backtest(signal_df, fwd_df, factor_name="Factor"):
    ls_rets = []
    dates   = []

    for date in signal_df.index:
        sig = signal_df.loc[date].dropna()
        fwd = fwd_df.loc[date, sig.index].dropna()
        sig = sig.loc[fwd.index]
        if len(sig) < 8:
            continue

        pct_rank   = sig.rank(pct=True)
        long_mask  = pct_rank >= 0.80
        short_mask = pct_rank <= 0.20

        long_ret  = fwd[long_mask].mean()
        short_ret = fwd[short_mask].mean()
        ls_rets.append(long_ret - short_ret)
        dates.append(date)

    return pd.Series(ls_rets, index=dates, name=factor_name)

momentum_ls = factor_backtest(mom_sig,  fwd_ret, "Momentum L/S")
lowvol_ls   = factor_backtest(-vol_sig, fwd_ret, "Low Vol L/S")
combined_ls = (momentum_ls + lowvol_ls) / 2
combined_ls.name = "Combined L/S"

print(f"    Factor return series: {len(momentum_ls)} monthly observations")

# ─────────────────────────────────────────────────────
# STEP 4: PERFORMANCE METRICS
# ─────────────────────────────────────────────────────
print("\n[4/6] Computing performance metrics...")

def performance_metrics(r, name):
    r = r.dropna()
    ann_ret  = r.mean() * 12
    ann_vol  = r.std()  * np.sqrt(12)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    cum      = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd       = (cum - roll_max) / roll_max
    max_dd   = dd.min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0
    hit_rate = (r > 0).mean()
    skew     = r.skew()
    kurt     = r.kurtosis()
    return {
        "Strategy"        : name,
        "Ann. Return"     : f"{ann_ret*100:+.1f}%",
        "Ann. Volatility" : f"{ann_vol*100:.1f}%",
        "Sharpe Ratio"    : f"{sharpe:.2f}",
        "Max Drawdown"    : f"{max_dd*100:.1f}%",
        "Calmar Ratio"    : f"{calmar:.2f}",
        "Hit Rate"        : f"{hit_rate*100:.1f}%",
        "Skewness"        : f"{skew:.2f}",
        "Excess Kurtosis" : f"{kurt:.2f}",
        "# Months"        : len(r)
    }, cum, dd

mom_m,  mom_cum,  mom_dd  = performance_metrics(momentum_ls, "Momentum L/S")
vol_m,  vol_cum,  vol_dd  = performance_metrics(lowvol_ls,   "Low Vol L/S")
comb_m, comb_cum, comb_dd = performance_metrics(combined_ls, "Combined L/S")

print("\n" + "─"*65)
print(f"  {'Metric':<22} {'Momentum':>12} {'Low Vol':>12} {'Combined':>12}")
print("─"*65)
for k in ["Ann. Return","Ann. Volatility","Sharpe Ratio","Max Drawdown",
          "Calmar Ratio","Hit Rate","Skewness","Excess Kurtosis","# Months"]:
    print(f"  {k:<22} {mom_m[k]:>12} {vol_m[k]:>12} {comb_m[k]:>12}")
print("─"*65)

# ─────────────────────────────────────────────────────
# STEP 5: INFORMATION COEFFICIENT
# IC = Spearman rank correlation between signal & fwd return
# Mean IC > 0.05 is considered a useful signal in practice
# ─────────────────────────────────────────────────────
ics, ic_dates = [], []
for date in mom_sig.index:
    sig = mom_sig.loc[date].dropna()
    fwd = fwd_ret.loc[date, sig.index].dropna()
    sig = sig.loc[fwd.index]
    if len(sig) < 5: continue
    ic = sig.rank().corr(fwd.rank())
    ics.append(ic)
    ic_dates.append(date)

ic_series = pd.Series(ics, index=ic_dates)
ic_roll   = ic_series.rolling(6).mean()

print(f"\n  Momentum Factor IC: Mean = {ic_series.mean():.4f} | ICIR = {ic_series.mean()/ic_series.std():.2f}")

# ─────────────────────────────────────────────────────
# STEP 6: VISUALISATION — 6-panel professional chart
# ─────────────────────────────────────────────────────
print("\n[5/6] Generating charts...")

fig = plt.figure(figsize=(18, 14), facecolor='#080D18')
fig.suptitle(
    'PROJECT 1  ·  India Equity Factor Backtester  ·  30-Stock Universe  ·  2019–2024',
    fontsize=15, fontweight='bold', color='white', y=0.985, fontfamily='monospace'
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

C = dict(mom='#00D4FF', vol='#FFB800', comb='#00FF88', neg='#FF4466',
         grid='#1A2535', text='#8BA3BC', bg='#0D1520')

def ax_style(ax, title):
    ax.set_facecolor(C['bg'])
    ax.tick_params(colors=C['text'], labelsize=8)
    ax.set_title(title, color='white', fontsize=9.5, fontweight='bold', pad=7)
    ax.grid(True, color=C['grid'], lw=0.5, alpha=0.8)
    for sp in ax.spines.values(): sp.set_edgecolor(C['grid'])

# ── 1. Cumulative Returns (wide)
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(mom_cum,  color=C['mom'],  lw=2,   label='Momentum L/S', alpha=0.9)
ax1.plot(vol_cum,  color=C['vol'],  lw=2,   label='Low Vol L/S',  alpha=0.9)
ax1.plot(comb_cum, color=C['comb'], lw=2.5, label='Combined L/S', alpha=1.0)
ax1.axhline(1, color='white', lw=0.6, ls='--', alpha=0.3)
ax1.set_ylabel('Growth of ₹1', color=C['text'], fontsize=8)
ax1.fill_between(comb_cum.index, 1, comb_cum, where=comb_cum >= 1,
                 alpha=0.08, color=C['comb'])
ax1.fill_between(comb_cum.index, 1, comb_cum, where=comb_cum < 1,
                 alpha=0.08, color=C['neg'])
ax1.legend(fontsize=8, facecolor=C['bg'], labelcolor='white', framealpha=0.9,
           edgecolor=C['grid'])
ax_style(ax1, '📈  Cumulative Factor Returns — Long/Short Portfolio')

# ── 2. Metrics Table
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(C['bg'])
ax2.axis('off')
rows = [['Metric','Mom','Comb']]
for k in ["Ann. Return","Sharpe Ratio","Max Drawdown","Hit Rate","Calmar Ratio"]:
    rows.append([k, mom_m[k], comb_m[k]])
tbl = ax2.table(cellText=rows[1:], colLabels=rows[0],
                cellLoc='center', loc='center', bbox=[0,0.05,1,0.9])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor('#162030' if r == 0 else (C['bg'] if r % 2 else '#111B28'))
    cell.set_text_props(color='white' if r == 0 else C['text'])
    cell.set_edgecolor(C['grid'])
ax2.set_title('📊  Key Metrics', color='white', fontsize=9.5, fontweight='bold', pad=7)

# ── 3. Return Distribution
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(momentum_ls*100, bins=22, color=C['mom'], alpha=0.85,
         edgecolor='#080D18', lw=0.5)
ax3.axvline(0, color='white', lw=1.2, ls='--', alpha=0.5)
ax3.axvline(momentum_ls.mean()*100, color=C['comb'], lw=1.5,
            ls='-', alpha=0.9, label=f"Mean={momentum_ls.mean()*100:.2f}%")
ax3.set_xlabel('Monthly Return (%)', color=C['text'], fontsize=8)
ax3.set_ylabel('Frequency', color=C['text'], fontsize=8)
ax3.legend(fontsize=7, facecolor=C['bg'], labelcolor='white')
ax_style(ax3, '📊  Momentum Return Distribution')

# ── 4. Drawdown
ax4 = fig.add_subplot(gs[1, 1])
ax4.fill_between(mom_dd.index,  mom_dd*100,  0, color=C['mom'],  alpha=0.5, label='Momentum')
ax4.fill_between(comb_dd.index, comb_dd*100, 0, color=C['comb'], alpha=0.5, label='Combined')
ax4.set_ylabel('Drawdown (%)', color=C['text'], fontsize=8)
ax4.legend(fontsize=7, facecolor=C['bg'], labelcolor='white')
ax_style(ax4, '📉  Drawdown Profile')

# ── 5. Rolling 12M Sharpe
ax5 = fig.add_subplot(gs[1, 2])
rs_mom  = (momentum_ls.rolling(12).mean() / momentum_ls.rolling(12).std()) * np.sqrt(12)
rs_comb = (combined_ls.rolling(12).mean() / combined_ls.rolling(12).std()) * np.sqrt(12)
ax5.plot(rs_mom.index,  rs_mom,  color=C['mom'],  lw=1.5, label='Momentum', alpha=0.9)
ax5.plot(rs_comb.index, rs_comb, color=C['comb'], lw=1.8, label='Combined',  alpha=0.9)
ax5.axhline(0,   color='white',  lw=0.6, ls='--', alpha=0.3)
ax5.axhline(0.5, color=C['vol'], lw=0.8, ls=':',  alpha=0.7, label='SR=0.5')
ax5.axhline(1.0, color=C['comb'],lw=0.8, ls=':',  alpha=0.5, label='SR=1.0')
ax5.set_ylabel('Rolling Sharpe', color=C['text'], fontsize=8)
ax5.legend(fontsize=7, facecolor=C['bg'], labelcolor='white')
ax_style(ax5, '📈  Rolling 12M Sharpe Ratio')

# ── 6. Monthly Returns Heatmap
ax6 = fig.add_subplot(gs[2, :2])
mom_indexed = momentum_ls.copy()
mom_indexed.index = pd.to_datetime(mom_indexed.index)
pivot = mom_indexed.groupby([mom_indexed.index.year,
                             mom_indexed.index.month]).mean().unstack() * 100
pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                 'Jul','Aug','Sep','Oct','Nov','Dec'][:pivot.shape[1]]
im = ax6.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-8, vmax=8)
ax6.set_xticks(range(len(pivot.columns)))
ax6.set_xticklabels(pivot.columns, color=C['text'], fontsize=8)
ax6.set_yticks(range(len(pivot.index)))
ax6.set_yticklabels(pivot.index.astype(str), color=C['text'], fontsize=8)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax6.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=7, color='white' if abs(v) > 4 else '#222',
                    fontweight='bold')
plt.colorbar(im, ax=ax6, fraction=0.015, pad=0.02).ax.tick_params(
    colors=C['text'], labelsize=7)
ax_style(ax6, '🗓️  Momentum Monthly Returns Heatmap (%)')

# ── 7. Information Coefficient
ax7 = fig.add_subplot(gs[2, 2])
bar_colors = [C['comb'] if v >= 0 else C['neg'] for v in ic_series]
ax7.bar(range(len(ic_series)), ic_series.values, color=bar_colors,
        alpha=0.65, width=0.8)
ax7.plot(range(len(ic_roll)), ic_roll.values,
         color=C['mom'], lw=2, label='6M Rolling IC')
ax7.axhline(0, color='white', lw=0.6, ls='--', alpha=0.3)
ax7.axhline(ic_series.mean(), color=C['vol'], lw=1.2, ls='--', alpha=0.8,
            label=f'Mean IC = {ic_series.mean():.3f}')
ax7.set_ylabel('IC (Spearman ρ)', color=C['text'], fontsize=8)
ax7.legend(fontsize=7, facecolor=C['bg'], labelcolor='white')
ax_style(ax7, f'🎯  Factor IC | ICIR = {ic_series.mean()/ic_series.std():.2f}')

fig.text(0.5, 0.005,
    'Long-short equal-weight quintile portfolios · Monthly rebalancing · No transaction costs · Simulated data with realistic covariance structure',
    ha='center', fontsize=7.5, color=C['text'], style='italic')

out_png = 'project1_factor_backtest.png'
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='#080D18')
plt.close()
print(f"  Chart saved → {out_png}")

# ── Optional: save outputs for downstream use
combined_ls.to_csv('p1_combined_returns.csv', header=True)
momentum_ls.to_csv('p1_momentum_returns.csv', header=True)
pd.DataFrame([mom_m, vol_m, comb_m]).to_csv('p1_metrics.csv', index=False)

print("\n" + "="*62)
print("  ✅  PROJECT 1 COMPLETE")
print("  Outputs: project1_factor_backtest.png")
print("           p1_combined_returns.csv")
print("           p1_momentum_returns.csv")
print("           p1_metrics.csv")
print("="*62)
