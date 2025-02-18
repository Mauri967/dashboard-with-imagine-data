#Once debugged, to run the code you must ask IDLE for the following instruction:
#"streamlit run .[code name].py --server.port 8888" or whatever port you want
# frameworks and libraries
#to run in browser
import streamlit as st
#for bar chart, time series and map
import plotly.express as px
#for sankey chart
import plotly.graph_objects as go
#for data manipulation
import pandas as pd
import os
import warnings
#for forecasting time series
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# page set up
st.set_page_config(page_title="All Tomorrows: an imaginary dashboard", page_icon=":signal_strength:", layout="wide")
st.title(" :signal_strength: All Tomorrows: an imaginary dashboard")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

# Database load
try:
    dir_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(dir_path, "base_de_datos_imaginaria.csv")
except NameError:
    csv_path = r"C:\Users\raro9\OneDrive\Documentos\Documentos Personales\LinkedIn\dashboard para publicar en python\base_de_datos_imaginaria.csv"

df = pd.read_csv(csv_path, encoding="utf-8")

# turn 'Year' to datetime
df['Year'] = pd.to_datetime(df['Year'], format="%m/%d/%Y")

# filters
def crear_filtro_estado(df, key):
    estados = df["State"].dropna().unique()
    return st.multiselect("Pick your State", estados, key=key)

def crear_filtro_producto(df, key):
    productos = df["Products"].dropna().unique()
    return st.multiselect("Pick the Product", productos, key=key)

def crear_filtro_anio(df, key):
    anios = sorted(df['Year'].dt.year.unique())
    return st.slider('Select a range of years', min_value=min(anios), max_value=max(anios), value=(min(anios), max(anios)), key=key)

def crear_filtro_concepto(df, key):
    conceptos = ["Select All"] + list(df["Concept"].unique())
    return st.selectbox("Pick your Concept", conceptos, key=key)

def crear_filtro_metodo_pronostico(key):
    return st.selectbox("Select Forecast Method (just for time series)", ["Linear regression", "Exponential Smoothing", "ARIMA"], key=key)

# filters side bar
with st.sidebar:
    st.subheader("Filters")
    concepto_seleccionado = crear_filtro_concepto(df, "concepto")
    productos_seleccionados = crear_filtro_producto(df, "producto")
    estados_seleccionados = crear_filtro_estado(df, "estado")
    anio_inicio, anio_fin = crear_filtro_anio(df, "anio")
    metodo_pronostico = crear_filtro_metodo_pronostico("metodo_timeseries")

filtered_df = df.copy()
if concepto_seleccionado != "Select All":
    filtered_df = filtered_df[filtered_df["Concept"] == concepto_seleccionado]
if productos_seleccionados:
    filtered_df = filtered_df[filtered_df["Products"].isin(productos_seleccionados)]
if estados_seleccionados:
    filtered_df = filtered_df[filtered_df["State"].isin(estados_seleccionados)]
filtered_df = filtered_df[(filtered_df['Year'].dt.year >= anio_inicio) & (filtered_df['Year'].dt.year <= anio_fin)]

#if filters is empty
if filtered_df.empty:
    st.warning("No data available for the selected filters.")

if not filtered_df.empty:
    # bar chart
    concept_df = filtered_df.groupby(by=["Concept"], as_index=False)["Import"].sum()
    fig = px.bar(concept_df, x="Concept", y="Import",
                 text=['${:,.2f}'.format(x) for x in concept_df["Import"]],
                 template="seaborn", color="Concept",
                 color_discrete_sequence=["#003785", "#1465bb", "#2196f3", "#81c9fa"],
                 height=320)
fig.update_layout(xaxis_title=None, yaxis_title=None, showlegend=False)

# Time Series Chart
filtered_df["month_year"] = filtered_df["Year"].dt.to_period("M")
data = filtered_df.groupby("month_year")["Import"].sum().reset_index()
data["month_year"] = data["month_year"].dt.to_timestamp()
data = data.set_index("month_year")

future = pd.date_range(start=data.index[-1], periods=60, freq='MS')
future_df = pd.DataFrame(index=future)

#forecasting
if metodo_pronostico == "Linear regression":
        model = LinearRegression()
        x = data.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        y = data["Import"]
        model.fit(x, y)
        future_df["Import"] = model.predict(future_df.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1))
elif metodo_pronostico == "Exponential Smoothing":
        model = ExponentialSmoothing(data["Import"], trend="add", seasonal="add", seasonal_periods=12)
        model_fit = model.fit()
        future_df["Import"] = model_fit.forecast(steps=60)
elif metodo_pronostico == "ARIMA":
        model = ARIMA(data["Import"], order=(5, 2, 0))
        model_fit = model.fit()
        future_df["Import"] = model_fit.forecast(steps=60)

forecast_df = pd.concat([data, future_df])
fig2 = px.line(forecast_df, x=forecast_df.index, y="Import", labels={"Import": "Amount"},
                height=320, template="gridon", title="Time Series")
fig2.update_layout(xaxis_title=None, yaxis_title=None, title_x=0.4)

# Sankey chart
sankey_data = pd.read_csv("base_de_datos_imaginaria_sankey.csv")
unique_labels = list(set(sankey_data["source"].unique().tolist() + sankey_data["target"].unique().tolist()))
label_to_index = {label: index for index, label in enumerate(unique_labels)}
source = [label_to_index[row["source"]] for _, row in sankey_data.iterrows()]
target = [label_to_index[row["target"]] for _, row in sankey_data.iterrows()]

# HEX for nodes and links
blue_green_hex = "#1e88e3"
blue_light_green_hex = "#9ae0fc"
light_red_hex = "#ffcccc"
blue_hex = "#2196f3"
light_blue_hex = "#a3d1ff"
blue_dark = "#0f3e8d"

# node colours
node_colors = []
for label in unique_labels:
    if label in ["Cost of Sales", "Total Expense", "Financial Cost(income)", "Taxes (21%)"]:
        node_colors.append(blue_dark)
    elif label in ["Sales", "Gross profit", "Operating profit", "Income before taxes", "Net Income"]:
        node_colors.append(blue_green_hex)
    elif label in ["Kappa", "Phi", "Lambda", "Rho"]:
        node_colors.append(blue_hex)
    else:
        node_colors.append("blue")

# links colours
link_colors = []
for i in range(len(source)):
    source_label = unique_labels[source[i]]
    target_label = unique_labels[target[i]]
    link_label = f"{source_label} - {target_label}" 

    if link_label in ["Sales - Gross profit", "Gross profit - Operating profit", "Operating profit - Income before taxes", "Income before taxes - Net Income"]:
        link_colors.append(blue_light_green_hex)
    elif link_label in ["Sales - Cost of Sales", "Gross profit - Total Expense", "Operating profit - Financial Cost(income)", "Income before taxes - Taxes (21%)"]:
        link_colors.append(light_blue_hex)
    elif link_label in ["Kappa - Sales", "Phi - Sales", "Lambda - Sales", "Rho - Sales"]:
        link_colors.append(light_blue_hex)
    else:
        link_colors.append("blue")

fig_sankey = go.Figure(data=[go.Sankey(node=dict(pad=15, line=dict(color="black", width=0.5),
            label=unique_labels, color=node_colors),link=dict(source=source, 
            target=target, value=sankey_data["value"], color=link_colors))])
fig_sankey.update_layout(title_text="Profit and Loss [2025]: Sankey", height=320,title_x=0.25)

# map choropleth
data_mapa = pd.read_csv("base_de_datos_imaginaria_state.csv")
fig_mapa = px.choropleth(data_mapa, locations="State", locationmode="USA-states",
color="Import", color_continuous_scale="blues", scope="usa",title="Import by State [2025]", height=320)
fig_mapa.update_layout(coloraxis_showscale=False,geo_bgcolor="rgba(0,0,0,0)",title_x=0.3)

#grid 2x2
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        if not filtered_df.empty:
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.plotly_chart(fig_sankey, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        if not filtered_df.empty:
            st.plotly_chart(fig2, use_container_width=True)
    with col4:
        st.plotly_chart(fig_mapa, use_container_width=True)

# download data
csv = df.to_csv(index = False).encode('utf-8')
st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")