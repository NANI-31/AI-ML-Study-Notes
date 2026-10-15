import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import time

st.set_page_config(
    page_title="13.3 API-Based Applications",
    page_icon="🌐",
    layout="wide"
)

st.title("13.3 API-Based Applications in Streamlit")
st.caption("Designing, structuring, and deploying real-world API-driven applications with resilient client logic, local caching, and rich visualization.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### The API-Driven Application Pattern in Streamlit

Building production-ready Streamlit apps around external APIs requires separating concerns across three tiers:
1. **Network & Client Layer**: Manages HTTP connection pooling (`requests.Session`), query params, timeouts, headers, and rate limits.
2. **Caching & Transformation Layer**: Decouples API calls from Streamlit widget reruns using `@st.cache_data(ttl=...)`, parses raw JSON into typed DataFrames or domain objects, and handles fallback mock data if endpoints fail.
3. **Presentation Layer**: Renders interactive dashboards using `st.metric`, `st.dataframe`, `st.line_chart`, `st.map`, and modern containers (`border=True`).

""")

st.graphviz_chart(
    r'''
    digraph APIApp {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        ui [label="Streamlit User Interface\nst.selectbox('Select Model') & st.button('Infer')", fillcolor="#DBEAFE", color="#1D4ED8"];
        client [label="API Client Layer\n• Bearer Token Headers\n• Payload Serialization\n• Timeout & Exception Handlers", fillcolor="#FEF3C7", color="#D97706"];
        endpoint [label="Remote AI Endpoint\nCloud Inference Service", fillcolor="#D1FAE5", color="#059669"];

        ui -> client [label="User interaction"];
        client -> endpoint [label="HTTP POST request"];
        endpoint -> ui [label="JSON response -> st.metric() / chart"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. Weather Dashboard",
    "2. Currency Exchange",
    "3. News Aggregator",
    "4. Stock Viewer",
    "5. Geocoding Tool",
    "6. Public Transit",
    "7. Product Catalog",
    "8. External API Monitor"
])

# =============================================================================
# 1. WEATHER DASHBOARD
# =============================================================================
with tabs[0]:
    st.header("1. Weather Dashboard Application")
    st.markdown(r"""
    A live meteorological dashboard utilizing open weather endpoints (e.g., Open-Meteo). 
    Demonstrates **geographical parameter binding**, **hourly time-series parsing**, and **multi-metric KPI rendering**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import numpy as np
import datetime

@st.cache_data(ttl=600)
def fetch_weather_forecast(city_name: str, lat: float, lon: float):
    # Simulated resilient API response for reproducible execution
    np.random.seed(int(abs(lat * 100)) % 1000)
    base_temp = 22.0 if lat < 30 else 14.0
    times = pd.date_range(datetime.datetime.now(), periods=24, freq="h")
    temps = base_temp + 5 * np.sin(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 0.8, 24)
    humidity = np.clip(60 + 20 * np.cos(np.linspace(0, 2 * np.pi, 24)), 20, 95)
    
    return {
        "city": city_name,
        "current_temp": round(float(temps[0]), 1),
        "temp_delta": round(float(temps[0] - temps[-1]), 1),
        "current_humidity": int(humidity[0]),
        "wind_speed": round(float(np.random.uniform(5, 25)), 1),
        "hourly_df": pd.DataFrame({
            "Time": times,
            "Temperature (°C)": temps,
            "Humidity (%)": humidity
        }).set_index("Time")
    }

selected_city = st.selectbox("Select Station", ["San Francisco", "Tokyo", "London", "Bengaluru"])
coords = {
    "San Francisco": (37.77, -122.41),
    "Tokyo": (35.67, 139.65),
    "London": (51.50, -0.12),
    "Bengaluru": (12.97, 77.59)
}
lat, lon = coords[selected_city]
data = fetch_weather_forecast(selected_city, lat, lon)

c1, c2, c3 = st.columns(3)
c1.metric("Temperature", f"{data['current_temp']} °C", f"{data['temp_delta']} °C vs 24h")
c2.metric("Humidity", f"{data['current_humidity']} %")
c3.metric("Wind Speed", f"{data['wind_speed']} km/h")

st.line_chart(data["hourly_df"]["Temperature (°C)"], height=220)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            @st.cache_data(ttl=600)
            def fetch_weather_forecast(city_name: str, lat: float, lon: float):
                np.random.seed(int(abs(lat * 100)) % 1000)
                base_temp = 22.0 if lat < 30 else 14.0
                times = pd.date_range(datetime.datetime.now(), periods=24, freq="h")
                temps = base_temp + 5 * np.sin(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 0.8, 24)
                humidity = np.clip(60 + 20 * np.cos(np.linspace(0, 2 * np.pi, 24)), 20, 95)
                return {
                    "city": city_name,
                    "current_temp": round(float(temps[0]), 1),
                    "temp_delta": round(float(temps[0] - temps[-1]), 1),
                    "current_humidity": int(humidity[0]),
                    "wind_speed": round(float(np.random.uniform(5, 25)), 1),
                    "hourly_df": pd.DataFrame({
                        "Time": times,
                        "Temperature (°C)": temps,
                        "Humidity (%)": humidity
                    }).set_index("Time")
                }

            city_options = ["San Francisco", "Tokyo", "London", "Bengaluru"]
            selected_city = st.selectbox("Select Station", city_options, key="demo_weather_city")
            coords = {
                "San Francisco": (37.77, -122.41),
                "Tokyo": (35.67, 139.65),
                "London": (51.50, -0.12),
                "Bengaluru": (12.97, 77.59)
            }
            lat, lon = coords[selected_city]
            w_data = fetch_weather_forecast(selected_city, lat, lon)

            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("Temperature", f"{w_data['current_temp']} °C", f"{w_data['temp_delta']} °C vs 24h")
            wc2.metric("Humidity", f"{w_data['current_humidity']} %")
            wc3.metric("Wind Speed", f"{w_data['wind_speed']} km/h")

            st.caption("24-Hour Temperature Trajectory")
            st.line_chart(w_data["hourly_df"]["Temperature (°C)"], height=200)

# =============================================================================
# 2. CURRENCY EXCHANGE
# =============================================================================
with tabs[1]:
    st.header("2. Currency Information Dashboard")
    st.markdown(r"""
    A financial conversion dashboard querying foreign exchange rates.
    Illustrates **bidirectional rate lookup**, **matrix normalization**, and **cross-currency conversion math**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

@st.cache_data(ttl=900)
def get_exchange_rates(base_currency: str):
    # Standard rates relative to USD
    rates_usd = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.15, "JPY": 154.20}
    base_factor = rates_usd.get(base_currency, 1.0)
    # Re-base all rates against selected currency
    return {k: round(v / base_factor, 4) for k, v in rates_usd.items()}

col_a, col_b = st.columns(2)
with col_a:
    base = st.selectbox("From Currency", ["USD", "EUR", "GBP", "INR", "JPY"], index=0)
with col_b:
    target = st.selectbox("To Currency", ["USD", "EUR", "GBP", "INR", "JPY"], index=3)

amount = st.number_input("Amount", min_value=1.0, value=100.0, step=10.0)
rates = get_exchange_rates(base)
converted_amount = amount * rates[target]

st.metric(
    label=f"Exchange Result ({base} -> {target})",
    value=f"{converted_amount:,.2f} {target}",
    delta=f"1 {base} = {rates[target]} {target}"
)

# Render full matrix
df_rates = pd.DataFrame(list(rates.items()), columns=["Currency", "Conversion Rate"])
st.dataframe(df_rates, width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            @st.cache_data(ttl=900)
            def get_exchange_rates(base_currency: str):
                rates_usd = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.15, "JPY": 154.20}
                base_factor = rates_usd.get(base_currency, 1.0)
                return {k: round(v / base_factor, 4) for k, v in rates_usd.items()}

            ca, cb = st.columns(2)
            with ca:
                base = st.selectbox("From Currency", ["USD", "EUR", "GBP", "INR", "JPY"], index=0, key="fx_from")
            with cb:
                target = st.selectbox("To Currency", ["USD", "EUR", "GBP", "INR", "JPY"], index=3, key="fx_to")

            amount = st.number_input("Amount", min_value=1.0, value=100.0, step=10.0, key="fx_amount")
            rates = get_exchange_rates(base)
            converted = amount * rates[target]

            st.metric(
                label=f"Exchange Result ({base} → {target})",
                value=f"{converted:,.2f} {target}",
                delta=f"1 {base} = {rates[target]} {target}"
            )

            df_rates = pd.DataFrame(list(rates.items()), columns=["Target Currency", f"Rate (per 1 {base})"])
            st.dataframe(df_rates, width="stretch", hide_index=True)

# =============================================================================
# 3. NEWS DASHBOARD
# =============================================================================
with tabs[2]:
    st.header("3. News Dashboard & Sentiment Feeds")
    st.markdown(r"""
    A live news aggregation feed demonstrating **category filtering**, **text keyword searching**, 
    **sentiment categorization**, and **collapsible article cards**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

@st.cache_data(ttl=600)
def fetch_news_feed(category: str):
    articles = [
        {"title": "Open-Weights Model Breaks MMLU Benchmark", "cat": "AI/ML", "source": "ArXiv", "score": "Bullish", "hours_ago": 2},
        {"title": "Quantum Error Correction Reaches Threshold", "cat": "Science", "source": "Nature", "score": "Neutral", "hours_ago": 5},
        {"title": "Global Cloud Providers Expand Sovereign Regions", "cat": "Cloud", "source": "TechDaily", "score": "Bullish", "hours_ago": 7},
        {"title": "Semiconductor Supply Chain Tightens for High-HBM Chips", "cat": "Hardware", "source": "Reuters", "score": "Bearish", "hours_ago": 11},
        {"title": "Open-Source Agentic Framework Crosses 40k Stars", "cat": "AI/ML", "source": "GitHub Trending", "score": "Bullish", "hours_ago": 14}
    ]
    if category != "All":
        articles = [a for a in articles if a["cat"] == category]
    return articles

cat = st.selectbox("Category Filter", ["All", "AI/ML", "Science", "Cloud", "Hardware"])
query = st.text_input("Filter Headline Keyword", "")
items = fetch_news_feed(cat)

if query:
    items = [i for i in items if query.lower() in i["title"].lower()]

st.caption(f"Showing {len(items)} matching headlines")
for item in items:
    with st.container(border=True):
        sc1, sc2 = st.columns([4, 1])
        sc1.markdown(f"**{item['title']}**")
        sc1.caption(f"Source: {item['source']} • {item['hours_ago']}h ago • Category: `{item['cat']}`")
        badge_color = ":green[Bullish]" if item["score"] == "Bullish" else ":orange[Neutral]"
        sc2.markdown(badge_color)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            @st.cache_data(ttl=600)
            def fetch_news_feed(category: str):
                articles = [
                    {"title": "Open-Weights Model Breaks MMLU Benchmark", "cat": "AI/ML", "source": "ArXiv", "score": "Bullish", "hours_ago": 2},
                    {"title": "Quantum Error Correction Reaches Threshold", "cat": "Science", "source": "Nature", "score": "Neutral", "hours_ago": 5},
                    {"title": "Global Cloud Providers Expand Sovereign Regions", "cat": "Cloud", "source": "TechDaily", "score": "Bullish", "hours_ago": 7},
                    {"title": "Semiconductor Supply Chain Tightens for High-HBM Chips", "cat": "Hardware", "source": "Reuters", "score": "Bearish", "hours_ago": 11},
                    {"title": "Open-Source Agentic Framework Crosses 40k Stars", "cat": "AI/ML", "source": "GitHub Trending", "score": "Bullish", "hours_ago": 14}
                ]
                if category != "All":
                    articles = [a for a in articles if a["cat"] == category]
                return articles

            cat = st.selectbox("Category Filter", ["All", "AI/ML", "Science", "Cloud", "Hardware"], key="news_cat")
            query = st.text_input("Filter Headline Keyword", "", key="news_query")
            items = fetch_news_feed(cat)

            if query:
                items = [i for i in items if query.lower() in i["title"].lower()]

            st.caption(f"Showing {len(items)} matching headlines")
            for item in items:
                with st.container(border=True):
                    sc1, sc2 = st.columns([4, 1])
                    sc1.markdown(f"**{item['title']}**")
                    sc1.caption(f"Source: {item['source']} • {item['hours_ago']}h ago • Category: `{item['cat']}`")
                    if item["score"] == "Bullish":
                        sc2.success("Bullish")
                    elif item["score"] == "Bearish":
                        sc2.error("Bearish")
                    else:
                        sc2.warning("Neutral")

# =============================================================================
# 4. STOCK DATA VIEWER
# =============================================================================
with tabs[3]:
    st.header("4. Stock & Equity Data Viewer")
    st.markdown(r"""
    A market telemetry application consuming financial tick/candle APIs.
    Demonstrates **rolling moving average calculation**, **volume histogram alignment**, and **price delta badge formatting**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import numpy as np
import datetime

@st.cache_data(ttl=300)
def fetch_stock_history(ticker: str, days: int = 30):
    np.random.seed(abs(hash(ticker)) % 10000)
    dates = pd.date_range(datetime.datetime.now() - datetime.timedelta(days=days), periods=days, freq="D")
    base_price = {"NVDA": 125.0, "MSFT": 420.0, "GOOGL": 178.0, "AAPL": 220.0}.get(ticker, 100.0)
    returns = np.random.normal(0.001, 0.02, days)
    price_series = base_price * np.exp(np.cumsum(returns))
    volume = np.random.randint(15000000, 75000000, days)
    
    df = pd.DataFrame({
        "Date": dates,
        "Close": np.round(price_series, 2),
        "SMA_7": np.round(pd.Series(price_series).rolling(7, min_periods=1).mean(), 2),
        "Volume": volume
    }).set_index("Date")
    return df

ticker = st.selectbox("Select Equity Ticker", ["NVDA", "MSFT", "GOOGL", "AAPL"])
df_hist = fetch_stock_history(ticker, days=30)

current_p = df_hist["Close"].iloc[-1]
prev_p = df_hist["Close"].iloc[-2]
delta = round(current_p - prev_p, 2)
pct_delta = round((delta / prev_p) * 100, 2)

st.metric(f"{ticker} Current Close", f"${current_p:,.2f}", f"{delta:+.2f} ({pct_delta:+.2f}%)")
st.line_chart(df_hist[["Close", "SMA_7"]], height=240)
st.caption("Bar chart of daily trade volume:")
st.bar_chart(df_hist["Volume"], height=140)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            @st.cache_data(ttl=300)
            def fetch_stock_history(ticker: str, days: int = 30):
                np.random.seed(abs(hash(ticker)) % 10000)
                dates = pd.date_range(datetime.datetime.now() - datetime.timedelta(days=days), periods=days, freq="D")
                base_price = {"NVDA": 125.0, "MSFT": 420.0, "GOOGL": 178.0, "AAPL": 220.0}.get(ticker, 100.0)
                returns = np.random.normal(0.001, 0.02, days)
                price_series = base_price * np.exp(np.cumsum(returns))
                volume = np.random.randint(15000000, 75000000, days)
                return pd.DataFrame({
                    "Date": dates,
                    "Close": np.round(price_series, 2),
                    "SMA_7": np.round(pd.Series(price_series).rolling(7, min_periods=1).mean(), 2),
                    "Volume": volume
                }).set_index("Date")

            tk = st.selectbox("Select Equity Ticker", ["NVDA", "MSFT", "GOOGL", "AAPL"], key="stock_ticker")
            df_hist = fetch_stock_history(tk, days=30)
            curr_p = df_hist["Close"].iloc[-1]
            prev_p = df_hist["Close"].iloc[-2]
            d_val = round(curr_p - prev_p, 2)
            pct_val = round((d_val / prev_p) * 100, 2)

            st.metric(f"{tk} Current Close", f"${curr_p:,.2f}", f"{d_val:+.2f} ({pct_val:+.2f}%)")
            st.caption("30-Day Historical Close vs. 7-Day SMA")
            st.line_chart(df_hist[["Close", "SMA_7"]], height=180)
            st.caption("Daily Volume (Shares)")
            st.bar_chart(df_hist["Volume"], height=120)

# =============================================================================
# 5. GEOCODING APPLICATION
# =============================================================================
with tabs[4]:
    st.header("5. Geocoding & Mapping Tool")
    st.markdown(r"""
    A spatial location resolution application transforming textual addresses into latitude/longitude pairs 
    and projecting them onto `st.map()`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

@st.cache_data(ttl=3600)
def resolve_geocoding(location_query: str):
    # Simulated spatial resolution directory
    geo_db = {
        "Empire State Building": {"lat": 40.7484, "lon": -73.9857, "type": "Monuments"},
        "Eiffel Tower": {"lat": 48.8584, "lon": 2.2945, "type": "Monuments"},
        "Sydney Opera House": {"lat": -33.8568, "lon": 151.2153, "type": "Arts"},
        "Taj Mahal": {"lat": 27.1751, "lon": 78.0421, "type": "Monuments"},
        "Golden Gate Bridge": {"lat": 37.8199, "lon": -122.4783, "type": "Infrastructure"}
    }
    match = geo_db.get(location_query)
    if not match:
        return None
    return pd.DataFrame([{
        "lat": match["lat"],
        "lon": match["lon"],
        "name": location_query,
        "type": match["type"]
    }])

loc = st.selectbox(
    "Select Landmark to Geocode",
    ["Empire State Building", "Eiffel Tower", "Sydney Opera House", "Taj Mahal", "Golden Gate Bridge"]
)

geo_df = resolve_geocoding(loc)
if geo_df is not None:
    c1, c2 = st.columns(2)
    c1.metric("Latitude", f"{geo_df['lat'].iloc[0]:.4f}")
    c2.metric("Longitude", f"{geo_df['lon'].iloc[0]:.4f}")
    st.map(geo_df, zoom=12, height=250)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            @st.cache_data(ttl=3600)
            def resolve_geocoding(location_query: str):
                geo_db = {
                    "Empire State Building": {"lat": 40.7484, "lon": -73.9857, "type": "Monuments"},
                    "Eiffel Tower": {"lat": 48.8584, "lon": 2.2945, "type": "Monuments"},
                    "Sydney Opera House": {"lat": -33.8568, "lon": 151.2153, "type": "Arts"},
                    "Taj Mahal": {"lat": 27.1751, "lon": 78.0421, "type": "Monuments"},
                    "Golden Gate Bridge": {"lat": 37.8199, "lon": -122.4783, "type": "Infrastructure"}
                }
                match = geo_db.get(location_query)
                if not match:
                    return None
                return pd.DataFrame([{
                    "latitude": match["lat"],
                    "longitude": match["lon"],
                    "name": location_query,
                    "type": match["type"]
                }])

            landmark = st.selectbox(
                "Select Landmark to Geocode",
                ["Empire State Building", "Eiffel Tower", "Sydney Opera House", "Taj Mahal", "Golden Gate Bridge"],
                key="geo_select"
            )
            g_df = resolve_geocoding(landmark)
            if g_df is not None:
                gc1, gc2 = st.columns(2)
                gc1.metric("Latitude", f"{g_df['latitude'].iloc[0]:.4f}")
                gc2.metric("Longitude", f"{g_df['longitude'].iloc[0]:.4f}")
                st.map(g_df, zoom=11, height=220)

# =============================================================================
# 6. PUBLIC TRANSIT
# =============================================================================
with tabs[5]:
    st.header("6. Public Transport Information Tool")
    st.markdown(r"""
    A GTFS-inspired real-time commuter transit monitor showing **arrival countdowns**, 
    **line health status badges**, and **delay alerts**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

@st.cache_data(ttl=30)
def fetch_transit_status(station_id: str):
    # Simulated realtime GTFS arrival feed
    return [
        {"Line": "Red Line (Northbound)", "Dest": "Downtown Metro", "ETA_min": 3, "Status": "On Time", "Occupancy": "Normal"},
        {"Line": "Blue Line (Airport)", "Dest": "Terminal 2", "ETA_min": 7, "Status": "Delayed (+4m)", "Occupancy": "Crowded"},
        {"Line": "Express 101", "Dest": "Tech Campus", "ETA_min": 12, "Status": "On Time", "Occupancy": "Light"},
        {"Line": "Red Line (Southbound)", "Dest": "University South", "ETA_min": 15, "Status": "On Time", "Occupancy": "Normal"}
    ]

st_id = st.selectbox("Select Station Platform", ["Central Terminal - Bay 4", "Silicon Valley Transit Center"])
arrivals = fetch_transit_status(st_id)

st.subheader(f"Upcoming Departures — {st_id}")
for bus in arrivals:
    with st.container(border=True):
        col_l, col_m, col_r = st.columns([3, 2, 2])
        col_l.markdown(f"**{bus['Line']}** → {bus['Dest']}")
        col_m.metric("ETA", f"{bus['ETA_min']} min")
        if "Delayed" in bus["Status"]:
            col_r.error(f"{bus['Status']}")
        else:
            col_r.success(f"{bus['Status']}")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            @st.cache_data(ttl=30)
            def fetch_transit_status(station_id: str):
                return [
                    {"Line": "Red Line (Northbound)", "Dest": "Downtown Metro", "ETA_min": 3, "Status": "On Time", "Occupancy": "Normal"},
                    {"Line": "Blue Line (Airport)", "Dest": "Terminal 2", "ETA_min": 7, "Status": "Delayed (+4m)", "Occupancy": "Crowded"},
                    {"Line": "Express 101", "Dest": "Tech Campus", "ETA_min": 12, "Status": "On Time", "Occupancy": "Light"},
                    {"Line": "Red Line (Southbound)", "Dest": "University South", "ETA_min": 15, "Status": "On Time", "Occupancy": "Normal"}
                ]

            st_id = st.selectbox("Select Station Platform", ["Central Terminal - Bay 4", "Silicon Valley Transit Center"], key="transit_st")
            arrivals = fetch_transit_status(st_id)

            st.caption(f"Upcoming Live Departures for {st_id}")
            for bus in arrivals:
                with st.container(border=True):
                    col_l, col_m, col_r = st.columns([3, 2, 2])
                    col_l.markdown(f"**{bus['Line']}**  \n`Dest:` {bus['Dest']}")
                    col_m.metric("ETA", f"{bus['ETA_min']} min")
                    if "Delayed" in bus["Status"]:
                        col_r.error(bus["Status"])
                    else:
                        col_r.success(bus["Status"])

# =============================================================================
# 7. PRODUCT SEARCH
# =============================================================================
with tabs[6]:
    st.header("7. Product Search & Catalog Interface")
    st.markdown(r"""
    An e-commerce product search interface consuming catalog APIs with **multi-attribute filtering**, 
    **price sliders**, **in-stock badges**, and **rating star displays**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

@st.cache_data(ttl=600)
def search_catalog_api(max_price: float, category: str):
    inventory = [
        {"id": "SKU-101", "title": "Ultra-Low Latency GPU Server", "cat": "Compute", "price": 4500, "rating": 4.9, "stock": 8},
        {"id": "SKU-102", "title": "Developer Workstation 64GB DDR5", "cat": "Hardware", "price": 2400, "rating": 4.7, "stock": 15},
        {"id": "SKU-103", "title": "Mechanical Ergonomic Keyboard", "cat": "Peripherals", "price": 180, "rating": 4.8, "stock": 42},
        {"id": "SKU-104", "title": "4K Ultra-Wide Curved Monitor", "cat": "Displays", "price": 890, "rating": 4.6, "stock": 19},
        {"id": "SKU-105", "title": "Enterprise Vector DB Appliance", "cat": "Compute", "price": 7800, "rating": 5.0, "stock": 3}
    ]
    filtered = [p for p in inventory if p["price"] <= max_price]
    if category != "All":
        filtered = [p for p in filtered if p["cat"] == category]
    return filtered

c1, c2 = st.columns(2)
with c1:
    cat_filter = st.selectbox("Category", ["All", "Compute", "Hardware", "Peripherals", "Displays"])
with c2:
    price_cap = st.slider("Max Budget ($)", min_value=100, max_value=10000, value=5000, step=200)

products = search_catalog_api(price_cap, cat_filter)
st.caption(f"Found {len(products)} matching inventory items")

for p in products:
    with st.container(border=True):
        p_c1, p_c2, p_c3 = st.columns([3, 1, 1])
        p_c1.markdown(f"**{p['title']}** (`{p['id']}`)")
        p_c1.caption(f"Category: {p['cat']} • Rating: {'★' * int(p['rating'])} ({p['rating']})")
        p_c2.metric("Price", f"${p['price']:,}")
        p_c3.metric("Stock", f"{p['stock']} units")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            @st.cache_data(ttl=600)
            def search_catalog_api(max_price: float, category: str):
                inventory = [
                    {"id": "SKU-101", "title": "Ultra-Low Latency GPU Server", "cat": "Compute", "price": 4500, "rating": 4.9, "stock": 8},
                    {"id": "SKU-102", "title": "Developer Workstation 64GB DDR5", "cat": "Hardware", "price": 2400, "rating": 4.7, "stock": 15},
                    {"id": "SKU-103", "title": "Mechanical Ergonomic Keyboard", "cat": "Peripherals", "price": 180, "rating": 4.8, "stock": 42},
                    {"id": "SKU-104", "title": "4K Ultra-Wide Curved Monitor", "cat": "Displays", "price": 890, "rating": 4.6, "stock": 19},
                    {"id": "SKU-105", "title": "Enterprise Vector DB Appliance", "cat": "Compute", "price": 7800, "rating": 5.0, "stock": 3}
                ]
                f_list = [p for p in inventory if p["price"] <= max_price]
                if category != "All":
                    f_list = [p for p in f_list if p["cat"] == category]
                return f_list

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                cat_filter = st.selectbox("Category", ["All", "Compute", "Hardware", "Peripherals", "Displays"], key="prod_cat")
            with p_col2:
                price_cap = st.slider("Max Budget ($)", min_value=100, max_value=10000, value=5000, step=200, key="prod_price")

            items_found = search_catalog_api(price_cap, cat_filter)
            st.caption(f"Found {len(items_found)} matching items")

            for item in items_found:
                with st.container(border=True):
                    pc1, pc2, pc3 = st.columns([3, 1, 1])
                    pc1.markdown(f"**{item['title']}** (`{item['id']}`)")
                    pc1.caption(f"Category: {item['cat']} • Rating: {'★' * int(item['rating'])} ({item['rating']})")
                    pc2.metric("Price", f"${item['price']:,}")
                    pc3.metric("Stock", f"{item['stock']} units")

# =============================================================================
# 8. EXTERNAL API MONITOR
# =============================================================================
with tabs[7]:
    st.header("8. External API & Health Monitoring Dashboard")
    st.markdown(r"""
    A production-grade telemetry console monitoring external API endpoints.
    Illustrates **HTTP status code probing**, **latency quantile benchmarks**, and **operational uptime badges**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import numpy as np

def simulate_probe_external_apis():
    services = [
        {"Service": "LLM Gateway (v1/chat)", "URL": "https://api.openai.com/v1", "Status": 200, "Latency_ms": 285, "Uptime": "99.98%"},
        {"Service": "Embedding Cluster", "URL": "https://api.cohere.ai/v1/embed", "Status": 200, "Latency_ms": 112, "Uptime": "99.95%"},
        {"Service": "Weather Sensor API", "URL": "https://api.open-meteo.com/v1", "Status": 200, "Latency_ms": 140, "Uptime": "99.91%"},
        {"Service": "Payment Webhook Relay", "URL": "https://api.stripe.com/v1", "Status": 200, "Latency_ms": 94, "Uptime": "99.99%"},
        {"Service": "Legacy Billing Sync", "URL": "https://erp.internal/api/v2", "Status": 503, "Latency_ms": 1820, "Uptime": "94.20%"}
    ]
    return pd.DataFrame(services)

df_status = simulate_probe_external_apis()

m1, m2, m3 = st.columns(3)
active_count = len(df_status[df_status["Status"] == 200])
m1.metric("Healthy Services", f"{active_count} / {len(df_status)}")
m2.metric("Average Latency", f"{df_status['Latency_ms'].mean():.0f} ms")
m3.metric("Fleet Status", "Degraded" if active_count < len(df_status) else "Operational")

# Highlight error rows in dataframe
st.dataframe(
    df_status.style.apply(
        lambda row: ['background-color: #551111' if row['Status'] != 200 else '' for _ in row],
        axis=1
    ),
    width="stretch",
    hide_index=True
)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            def simulate_probe_external_apis():
                return pd.DataFrame([
                    {"Service": "LLM Gateway (v1/chat)", "URL": "https://api.openai.com/v1", "Status": 200, "Latency_ms": 285, "Uptime": "99.98%"},
                    {"Service": "Embedding Cluster", "URL": "https://api.cohere.ai/v1/embed", "Status": 200, "Latency_ms": 112, "Uptime": "99.95%"},
                    {"Service": "Weather Sensor API", "URL": "https://api.open-meteo.com/v1", "Status": 200, "Latency_ms": 140, "Uptime": "99.91%"},
                    {"Service": "Payment Webhook Relay", "URL": "https://api.stripe.com/v1", "Status": 200, "Latency_ms": 94, "Uptime": "99.99%"},
                    {"Service": "Legacy Billing Sync", "URL": "https://erp.internal/api/v2", "Status": 503, "Latency_ms": 1820, "Uptime": "94.20%"}
                ])

            df_status = simulate_probe_external_apis()

            m1, m2, m3 = st.columns(3)
            active_count = len(df_status[df_status["Status"] == 200])
            m1.metric("Healthy Services", f"{active_count} / {len(df_status)}")
            m2.metric("Mean Latency", f"{df_status['Latency_ms'].mean():.0f} ms")
            m3.metric("Fleet Status", "Degraded" if active_count < len(df_status) else "Operational", delta="-1 Node Degraded", delta_color="inverse")

            st.dataframe(
                df_status,
                width="stretch",
                hide_index=True
            )

# -----------------------------------------------------------------------------
# ARCHITECTURAL RULES MATRIX
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Production Rules for API-Based Streamlit Applications

| Challenge | Anti-Pattern | Recommended Production Pattern |
| :--- | :--- | :--- |
| **Frequent Reruns** | Calling `requests.get()` inside top-level script on every slider or selectbox change. | Wrap calls in `@st.cache_data(ttl=...)` with domain-specific TTLs (e.g. 15s for transit, 5m for weather, 24h for geocoding). |
| **Cold Starts & Stalls** | Unbounded HTTP requests hanging user sessions indefinitely. | Always supply explicit tuple timeouts: `timeout=(3.05, 10.0)` for connection & socket read phases. |
| **Endpoint Outages** | Crashing the Streamlit app with uncaught tracebacks (`ConnectionError`, `HTTPError 500`). | Trap exceptions gracefully and display contextual `st.warning()` or render fallback cached data. |
| **API Quotas** | Repeatedly triggering expensive proprietary LLM / financial APIs on UI interactions. | Use `st.form` or explicit trigger buttons (`if st.button('Execute Query'):`) to batch user inputs before network dispatch. |
| **State Synchronization** | Re-requesting data when navigating between multiple pages. | Store fetched records in `st.session_state` or rely on centralized `@st.cache_data` functions shared across pages. |
""")
