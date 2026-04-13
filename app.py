
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from datetime import date, timedelta
import math

st.set_page_config(page_title="Stock Analysis App", layout="wide")


st.sidebar.header("Settings")

ticker_input = st.sidebar.text_input("Enter tickers (2-5, comma separated)", value="AAPL, MSFT, GOOGL")

start_date = st.sidebar.date_input("Start date", value=date.today() - timedelta(days=365*3))
end_date = st.sidebar.date_input("End date", value=date.today())

with st.sidebar.expander("About / Methodology"):
    st.write("""
    This app downloads stock price data from Yahoo Finance and lets you
    compare multiple stocks side by side.
    
    **Assumptions:**
    - Returns are simple (arithmetic) daily returns
    - Annualized return = mean daily return x 252
    - Annualized volatility = daily std x sqrt(252)
    - 252 trading days per year
    - Data source: Yahoo Finance (adjusted closing prices)
    """)


tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
tickers = list(dict.fromkeys(tickers))  # remove duplicates

if len(tickers) < 2:
    st.error("Please enter at least 2 tickers.")
    st.stop()

if len(tickers) > 5:
    st.error("Please enter no more than 5 tickers.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if (end_date - start_date).days < 365:
    st.error("Please select a date range of at least 1 year.")
    st.stop()


@st.cache_data(ttl=3600)
def download_data(tickers_str, start, end):
    tickers_list = tickers_str.split(",")
    all_tickers = tickers_list + ["^GSPC"]

    prices = {}
    failed = []

    for t in all_tickers:
        try:
            raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)

            if raw.empty or len(raw) < 20:
                failed.append(t)
                continue

            # flatten MultiIndex columns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0] for col in raw.columns]

            if "Close" in raw.columns:
                prices[t] = raw["Close"].dropna()
            elif "Adj Close" in raw.columns:
                prices[t] = raw["Adj Close"].dropna()
            else:
                failed.append(t)

        except Exception:
            failed.append(t)

    return prices, [f for f in failed if f != "^GSPC"]


with st.spinner("Downloading data..."):
    prices_dict, failed_tickers = download_data(
        ",".join(tickers), str(start_date), str(end_date)
    )

if failed_tickers:
    st.error(f"Could not download data for: {', '.join(failed_tickers)}. Please check the ticker symbols.")
    st.stop()

if len([t for t in tickers if t in prices_dict]) < 2:
    st.error("Not enough valid tickers. Please try different symbols.")
    st.stop()


price_df = pd.DataFrame({t: prices_dict[t] for t in tickers if t in prices_dict})
price_df = price_df.dropna()


for t in list(price_df.columns):
    missing = prices_dict[t].isna().mean()
    if missing > 0.05:
        st.warning(f"{t} has too much missing data ({missing:.0%}) and was removed.")
        price_df = price_df.drop(columns=[t])

tickers = list(price_df.columns)

if len(tickers) < 2:
    st.error("Not enough tickers with valid data.")
    st.stop()


returns = price_df.pct_change().dropna()


ew_returns = returns.mean(axis=1)


sp500 = None
if "^GSPC" in prices_dict:
    sp500_prices = prices_dict["^GSPC"].reindex(returns.index)
    sp500 = sp500_prices.pct_change().dropna()


tab1, tab2, tab3, tab4 = st.tabs([
    "Price & Returns",
    "Risk & Distribution",
    "Correlation & Diversification",
    "Summary Statistics"
])


with tab1:
    st.header("Price & Return Analysis")

    # price chart
    st.subheader("Adjusted Closing Prices")

    show_tickers = st.multiselect("Select stocks to show", tickers, default=tickers)

    if show_tickers:
        fig = go.Figure()
        for t in show_tickers:
            fig.add_trace(go.Scatter(x=price_df.index, y=price_df[t], name=t, mode="lines"))
        fig.update_layout(title="Closing Prices", xaxis_title="Date", yaxis_title="Price (USD)",
                          template="plotly_white", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one stock to display.")

    st.divider()

    # cumulative wealth index
    st.subheader("Cumulative Wealth Index - $10,000 Investment")

    wealth = (1 + returns).cumprod() * 10000
    ew_wealth = (1 + ew_returns).cumprod() * 10000

    fig2 = go.Figure()
    for t in tickers:
        fig2.add_trace(go.Scatter(x=wealth.index, y=wealth[t], name=t, mode="lines"))

    fig2.add_trace(go.Scatter(x=ew_wealth.index, y=ew_wealth,
                              name="Equal-Weight Portfolio",
                              line=dict(dash="dash", color="black", width=2), mode="lines"))

    if sp500 is not None:
        sp500_wealth = (1 + sp500).cumprod() * 10000
        fig2.add_trace(go.Scatter(x=sp500_wealth.index, y=sp500_wealth,
                                  name="S&P 500",
                                  line=dict(dash="dot", color="grey", width=2), mode="lines"))

    fig2.update_layout(title="Growth of $10,000", xaxis_title="Date", yaxis_title="Value (USD)",
                       template="plotly_white", height=480, hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)



with tab2:
    st.header("Risk & Distribution Analysis")

    # rolling volatility
    vol_window = st.select_slider("Rolling window (days)", options=[10, 20, 30, 60, 90, 120], value=30)

    st.subheader(f"Rolling Annualized Volatility ({vol_window}-day window)")

    rolling_vol = returns.rolling(vol_window).std() * math.sqrt(252)

    fig3 = go.Figure()
    for t in tickers:
        fig3.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[t], name=t, mode="lines"))
    fig3.update_layout(title=f"Rolling {vol_window}-Day Volatility",
                       xaxis_title="Date", yaxis_title="Annualized Volatility",
                       yaxis_tickformat=".0%", template="plotly_white", height=420)
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()


    st.subheader("Return Distribution")

    selected_stock = st.selectbox("Select a stock", tickers)
    stock_returns = returns[selected_stock].dropna()

    plot_type = st.radio("Plot type", ["Histogram", "Q-Q Plot"], horizontal=True)

    if plot_type == "Histogram":
        mu, sigma = stats.norm.fit(stock_returns)
        x = np.linspace(stock_returns.min(), stock_returns.max(), 200)

        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(x=stock_returns, nbinsx=60, histnorm="probability density",
                                    name="Returns", marker_color="steelblue", opacity=0.7))
        fig4.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, mu, sigma),
                                  name="Normal Fit", mode="lines",
                                  line=dict(color="red", width=2)))
        fig4.update_layout(title=f"{selected_stock} Daily Return Distribution",
                           xaxis_title="Daily Return", yaxis_title="Density",
                           template="plotly_white", height=400)
        st.plotly_chart(fig4, use_container_width=True)

    else:
        (theoretical_q, sample_q), (slope, intercept, _) = stats.probplot(stock_returns)
        ref_line = np.array(theoretical_q) * slope + intercept

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=theoretical_q, y=sample_q, mode="markers",
                                  name="Data", marker=dict(color="steelblue", size=4, opacity=0.6)))
        fig5.add_trace(go.Scatter(x=theoretical_q, y=ref_line, mode="lines",
                                  name="Normal Reference", line=dict(color="red", width=2)))
        fig5.update_layout(title=f"{selected_stock} Q-Q Plot",
                           xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles",
                           template="plotly_white", height=400)
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Points deviating from the red line indicate the returns are not normally distributed.")

    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(stock_returns)
    if jb_p < 0.05:
        st.error(f"Jarque-Bera test: statistic = {jb_stat:.2f}, p-value = {jb_p:.2e} — Rejects normality (p < 0.05)")
    else:
        st.success(f"Jarque-Bera test: statistic = {jb_stat:.2f}, p-value = {jb_p:.4f} — Fails to reject normality (p >= 0.05)")

    st.divider()

    # box plot
    st.subheader("Box Plot - Daily Returns")

    fig6 = go.Figure()
    for t in tickers:
        fig6.add_trace(go.Box(y=returns[t].dropna(), name=t, boxmean="sd"))
    fig6.update_layout(title="Daily Return Distributions",
                       yaxis_title="Daily Return", yaxis_tickformat=".1%",
                       template="plotly_white", height=420)
    st.plotly_chart(fig6, use_container_width=True)



with tab3:
    st.header("Correlation & Diversification")

    # correlation heatmap
    st.subheader("Correlation Heatmap")

    corr_matrix = returns.corr()

    fig7 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}"
    ))
    fig7.update_layout(title="Pairwise Correlation of Daily Returns",
                       template="plotly_white", height=400)
    st.plotly_chart(fig7, use_container_width=True)

    st.divider()

    # scatter plot
    st.subheader("Scatter Plot")

    col1, col2 = st.columns(2)
    stock_a = col1.selectbox("Stock A", tickers, index=0, key="scatter_a")
    stock_b = col2.selectbox("Stock B", tickers, index=min(1, len(tickers)-1), key="scatter_b")

    if stock_a != stock_b:
        fig8 = px.scatter(x=returns[stock_a], y=returns[stock_b],
                          labels={"x": f"{stock_a} Return", "y": f"{stock_b} Return"},
                          title=f"{stock_a} vs {stock_b} Daily Returns",
                          trendline="ols", opacity=0.5)
        fig8.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("Please select two different stocks.")

    st.divider()

    # rolling correlation
    st.subheader("Rolling Correlation")

    corr_window = st.select_slider("Window (days)", options=[20, 30, 60, 90, 120], value=60)

    col3, col4 = st.columns(2)
    rc_a = col3.selectbox("Stock A", tickers, index=0, key="rc_a")
    rc_b = col4.selectbox("Stock B", tickers, index=min(1, len(tickers)-1), key="rc_b")

    if rc_a != rc_b:
        rolling_corr = returns[rc_a].rolling(corr_window).corr(returns[rc_b])

        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr,
                                  name=f"{rc_a}/{rc_b}", mode="lines",
                                  line=dict(color="darkorange")))
        fig9.add_hline(y=0, line_dash="dash", line_color="grey")
        fig9.update_layout(title=f"Rolling {corr_window}-Day Correlation: {rc_a} vs {rc_b}",
                           xaxis_title="Date", yaxis_title="Correlation",
                           yaxis=dict(range=[-1.1, 1.1]),
                           template="plotly_white", height=380)
        st.plotly_chart(fig9, use_container_width=True)
    else:
        st.info("Please select two different stocks.")

    st.divider()

    # two-asset portfolio explorer
    st.subheader("Two-Asset Portfolio Explorer")

    st.info("""
    When you combine two stocks into a portfolio, the overall volatility is usually
    lower than either stock on its own. This is the diversification effect.
    The chart below shows how portfolio volatility changes as you adjust the weights.
    The effect is stronger when the two stocks have lower correlation.
    """)

    col5, col6 = st.columns(2)
    port_a = col5.selectbox("Stock A", tickers, index=0, key="port_a")
    port_b = col6.selectbox("Stock B", tickers, index=min(1, len(tickers)-1), key="port_b")

    if port_a == port_b:
        st.warning("Please select two different stocks.")
    else:
        weight_a = st.slider(f"Weight on {port_a} (%)", 0, 100, 50)
        weight_b = 100 - weight_a
        wa = weight_a / 100

        # annualized stats
        ret_a = returns[port_a].mean() * 252
        ret_b = returns[port_b].mean() * 252
        vol_a = returns[port_a].std() * math.sqrt(252)
        vol_b = returns[port_b].std() * math.sqrt(252)
        cov_ab = returns[[port_a, port_b]].cov().iloc[0, 1] * 252

        # portfolio stats at current weight
        port_return = wa * ret_a + (1 - wa) * ret_b
        port_variance = wa**2 * vol_a**2 + (1-wa)**2 * vol_b**2 + 2 * wa * (1-wa) * cov_ab
        port_vol = math.sqrt(max(port_variance, 0))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{port_a} weight", f"{weight_a}%")
        m2.metric(f"{port_b} weight", f"{weight_b}%")
        m3.metric("Portfolio Return", f"{port_return:.2%}")
        m4.metric("Portfolio Volatility", f"{port_vol:.2%}")

        # full volatility curve across all weights
        all_weights = np.linspace(0, 1, 201)
        all_vols = []
        for w in all_weights:
            var = w**2 * vol_a**2 + (1-w)**2 * vol_b**2 + 2*w*(1-w)*cov_ab
            all_vols.append(math.sqrt(max(var, 0)))

        fig10 = go.Figure()
        fig10.add_trace(go.Scatter(x=all_weights*100, y=all_vols,
                                   mode="lines", name="Portfolio Volatility",
                                   line=dict(color="steelblue", width=2.5)))
        fig10.add_trace(go.Scatter(x=[weight_a], y=[port_vol],
                                   mode="markers", name="Current",
                                   marker=dict(color="red", size=12)))
        fig10.add_hline(y=vol_a, line_dash="dot", line_color="orange",
                        annotation_text=f"{port_a} ({vol_a:.1%})")
        fig10.add_hline(y=vol_b, line_dash="dot", line_color="green",
                        annotation_text=f"{port_b} ({vol_b:.1%})")
        fig10.update_layout(title=f"Portfolio Volatility vs Weight on {port_a}",
                            xaxis_title=f"Weight on {port_a} (%)",
                            yaxis_title="Annualized Volatility",
                            yaxis_tickformat=".1%",
                            template="plotly_white", height=450)
        st.plotly_chart(fig10, use_container_width=True)

        st.caption("The curve dipping below both dashed lines shows that combining the two stocks reduces risk.")



with tab4:
    st.header("Summary Statistics")

    all_series = {t: returns[t].dropna() for t in tickers}
    if sp500 is not None:
        all_series["S&P 500"] = sp500.dropna()

    stats_rows = {}
    for name, s in all_series.items():
        stats_rows[name] = {
            "Annualized Return":    f"{s.mean() * 252:.2%}",
            "Annualized Volatility":f"{s.std() * math.sqrt(252):.2%}",
            "Skewness":             f"{s.skew():.3f}",
            "Kurtosis":             f"{s.kurtosis():.3f}",
            "Min Daily Return":     f"{s.min():.2%}",
            "Max Daily Return":     f"{s.max():.2%}",
        }

    st.dataframe(pd.DataFrame(stats_rows).T, use_container_width=True)

    st.caption("Annualized return = mean daily return x 252. Annualized volatility = daily std x sqrt(252). Kurtosis shown is excess kurtosis (normal = 0).")