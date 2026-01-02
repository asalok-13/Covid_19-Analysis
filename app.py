
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

st.set_page_config(page_title="COVID-19 Global Dashboard", layout="wide")

@st.cache_data
def load_data():
    cases = pd.read_csv(
        "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
    )
    vaccines = pd.read_csv(
        "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    )
    populations = pd.read_csv(
        "https://raw.githubusercontent.com/owid/covid-19-data/master/scripts/input/un/population_latest.csv"
    )
    cases['Date'] = pd.to_datetime(cases['Date'])
    vaccines['date'] = pd.to_datetime(vaccines['date'])
    return cases, vaccines, populations

df, vax, pop = load_data()

st.title("🌍 COVID-19 Global Dashboard (Advanced Analytics)")

latest_date = df['Date'].max()
latest = df[df['Date'] == latest_date]

# Merge population for per-million calculation
latest = latest.merge(pop, left_on="Country", right_on="entity", how="left")
latest['deaths_per_million'] = (latest['Deaths'] / latest['population']) * 1_000_000

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("🦠 Total Confirmed", f"{latest['Confirmed'].sum():,}")
c2.metric("⚰️ Total Deaths", f"{latest['Deaths'].sum():,}")
c3.metric("💊 Total Recovered", f"{latest['Recovered'].sum():,}")
c4.metric("📊 Avg Deaths / Million", f"{latest['deaths_per_million'].mean():.2f}")

st.divider()

# Confirmed Cases Map
fig_map_cases = px.choropleth(
    latest,
    locations="Country",
    locationmode="country names",
    color="Confirmed",
    color_continuous_scale="Reds",
    title="🌍 Global Confirmed COVID-19 Cases"
)
st.plotly_chart(fig_map_cases, use_container_width=True)

# Vaccination % Map
vax_latest = vax.sort_values("date").groupby("location").last().reset_index()
vax_latest['vaccination_percent'] = (
    vax_latest['people_vaccinated'] / vax_latest['people_vaccinated'].max()
) * 100

fig_vax_map = px.choropleth(
    vax_latest,
    locations="location",
    locationmode="country names",
    color="vaccination_percent",
    color_continuous_scale="Greens",
    title="💉 Vaccination Coverage % (Relative)"
)
st.plotly_chart(fig_vax_map, use_container_width=True)

# Top 10 Countries
st.subheader("🏆 Top 10 Countries by Confirmed Cases")
top10 = latest.sort_values("Confirmed", ascending=False).head(10)
st.dataframe(top10[['Country', 'Confirmed', 'Deaths', 'Recovered', 'deaths_per_million']])

# Country-wise Section
st.subheader("📌 Country-wise Analysis")
country = st.selectbox("Select Country", sorted(df['Country'].unique()))
country_data = df[df['Country'] == country]

fig_trend = px.line(
    country_data,
    x="Date",
    y=["Confirmed", "Deaths", "Recovered"],
    title=f"📈 COVID-19 Trends – {country}"
)
st.plotly_chart(fig_trend, use_container_width=True)

# Vaccination trend
vax_country = vax[vax['location'] == country]
if not vax_country.empty:
    fig_vax = px.line(
        vax_country,
        x="date",
        y="total_vaccinations",
        title=f"💉 Vaccination Progress – {country}"
    )
    st.plotly_chart(fig_vax, use_container_width=True)

# Forecast
if st.checkbox("🔮 Show 30-Day Forecast"):
    forecast_df = country_data[['Date', 'Confirmed']]
    forecast_df.columns = ['ds', 'y']

    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig_forecast = px.line(
        forecast,
        x='ds',
        y='yhat',
        title=f"📊 30-Day COVID Forecast – {country}"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
