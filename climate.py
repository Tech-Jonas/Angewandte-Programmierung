import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Lade Daten
temp_file_path = r"C:\Users\jonas\SB_AngewandteProgrammierung\Land_and_Ocean_complete.txt"
ghg_file_path = r"C:\Users\jonas\SB_AngewandteProgrammierung\Emissions.csv"
erf_file_path = r"C:\Users\jonas\SB_AngewandteProgrammierung\ERF_best_aggregates_1750-2023.csv"
ghg_update_file_path = r"C:\Users\jonas\SB_AngewandteProgrammierung\update_2019-2023.csv"

# relevante Spalten der Temperaturanomalien extrahieren
with open(temp_file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Extrahiere nur die relevanten Zeilen (die mit Jahreszahlen beginnen)
temp_data = []
for line in lines:
    parts = line.strip().split()
    if len(parts) >= 3 and parts[0].isdigit():
        year, month, anomaly = int(parts[0]), int(parts[1]), float(parts[2])
        temp_data.append([year, month, anomaly])

# Erstelle ein DataFrame für die Temperaturdaten
df_temp = pd.DataFrame(temp_data, columns=["Year", "Month", "Anomaly"])

# Aggregiere Temperaturdaten auf Jahresbasis (Mittelwert der monatlichen Anomalien pro Jahr)
df_temp_yearly = df_temp.groupby("Year")["Anomaly"].mean().reset_index()

# Der Mittelwert der Jahre 1850 bis 1900 wird berechnet, da das Pariser Klimaabkommen diesen Zeitraum
# als Basis für das 1.5 Grad Ziel nimmt.
avg_1850_until_1900 = df_temp_yearly.loc[(df_temp_yearly["Year"] >= 1850) & (df_temp_yearly["Year"] <= 1900), "Anomaly"].mean()
df_temp_yearly["Adjusted Anomaly"] = df_temp_yearly["Anomaly"] - avg_1850_until_1900
df_temp["Adjusted Anomaly"] = df_temp["Anomaly"] - avg_1850_until_1900

# Aggregierte Temperaturdaten auf Monatsbasis
# nur zur Visualisierung
df_monthly_anomaly_yearly = df_temp.groupby(["Year", "Month"])["Adjusted Anomaly"].mean().unstack(level=0)

# Lade die THG-Emissionsdaten, die zweite csv-Datei (ghg_update) enthält ein Update der letzten 5 Jahre
df_ghg = pd.read_csv(ghg_file_path, usecols=["YYYY", "CO2", "CH4", "N2O"])
df_ghg_update = pd.read_csv(ghg_update_file_path, usecols=["YYYY", "CO2", "CH4", "N2O"])

# Wähle die relevanten Spalten aus den THG-Daten
df_ghg = df_ghg.rename(columns={"YYYY": "Year"})
df_ghg_update = df_ghg_update.rename(columns={"YYYY": "Year"})

# Entferne Zeilen ohne CO2-Daten
df_ghg = df_ghg.dropna(subset="CO2")
df_ghg = df_ghg.iloc[:-1] # letzte Zeile ist das Jahr 2019, dies wird nun geupdated
df_ghg = pd.concat([df_ghg, df_ghg_update])

# Konvertiere Jahr auf Integer
df_ghg["Year"] = df_ghg["Year"].astype(int)

# Mergen der Temperatur- und Emissionsdaten basierend auf dem Jahr
df_merged = pd.merge(df_temp_yearly, df_ghg, on="Year", how="left")

# Lade die ERF-Daten (Effective Radiative Forcing)
df_erf = pd.read_csv(erf_file_path)

# Wichtige Spalten auswählen (Jahr, Gesamt-ERF, anthropogener Einfluss)
df_erf = df_erf.rename(columns={"Unnamed: 0": "Year"})
df_erf["Year"] = df_erf["Year"].astype(int)

# Relevante ERF-Spalten für die Analyse
df_erf_selected = df_erf[["Year", "total", "anthro", "solar", "volcanic", "aerosol"]]

# Verknüpfen der ERF-Daten mit den Temperaturdaten
df_merged = pd.merge(df_merged, df_erf_selected, on="Year", how="left")

# Erstellen eines Prognose-Datensatzes für zukünftige Jahre (2025-2100)
future_years = np.arange(2025, 2101)
df_future = pd.DataFrame({"Year": future_years})

month_name = {1: "Jan",
              2: "Feb",
              3: "Mar",
              4: "Apr",
              5: "May",
              6: "Jun",
              7: "Jul",
              8: "Aug",
              9: "Sep",
              10: "Oct",
              11: "Nov",
              12: "Dec"}

monthly_reference_temps_array = np.array([
    12.23, 12.44, 13.06, 13.97, 14.95, 15.67, 15.95, 15.79,
    15.19, 14.26, 13.24, 12.49
])

# Erwartete Temperaturanstiege auf Basis der RCP-Szenarien
expected_temp_increase = {
    'RCP 2.6 Anomaly': np.linspace(df_merged['Adjusted Anomaly'].max(), 1.8, len(future_years)),  # Unter 2°C Ziel
    'RCP 4.5 Anomaly': np.linspace(df_merged['Adjusted Anomaly'].max(), 2.6, len(future_years)),  # Mittleres Szenario
    'RCP 8.5 Anomaly': np.linspace(df_merged['Adjusted Anomaly'].max(), 4.8, len(future_years))  # Starkes Anstiegsszenario
}

# Erwartete CO2-Emissionen auf Basis der RCP-Szenarien
rcp_values = {
    'RCP 2.6 CO2': np.linspace(df_merged['CO2'].max(), 490, len(future_years)),
    'RCP 4.5 CO2': np.linspace(df_merged['CO2'].max(), 650, len(future_years)),
    'RCP 8.5 CO2': np.linspace(df_merged['CO2'].max(), 1370, len(future_years))
}

# Füge die vorhergesagten Temperaturanstiege der Forecast-Tabelle hinzu
for scenario, anomaly in expected_temp_increase.items():
    df_future[scenario] = anomaly

# Füge die vorhergesagten CO2-Emissionen der Forecast-Tabelle hinzu
for scenario, co2 in rcp_values.items():
    df_future[scenario] = co2

# Exponentielle Glättung auf historische CO₂-Werte anwenden
# um CO2-Emissionen vorherzusagen
# Annahme auf Basis der EDA
co2_model = ExponentialSmoothing(df_merged["CO2"].dropna(), trend="additive", seasonal=None)
co2_fit = co2_model.fit()

# Zukunftsprognose für CO₂
future_co2_hw = co2_fit.forecast(len(future_years))

df_future["Forecast CO2"] = future_co2_hw.values

# nun werden Prognosen anhand verschiedener Modelle und Variablen erstellt
# zunächst wird die Zeit als Trendvariable berechnet, linear sowie quadratisch
df_merged["Trend"] = df_merged["Year"] - df_merged["Year"].min()
df_merged["Trend_sq"] = df_merged["Trend"] ** 2

# es gibt eine hohe Multikollinearität zwischen der CO2-Emission und den anderen THG-Emissionen (hoher VIF-Faktor, >1000)
# daher wird nur der CO2-Emissionswert als Variable aufgenommen
df_LR = df_merged.copy(deep=True).dropna()

# Lineare Regression auf die exponentiell geglätteten CO2-Emissionen
X_multi = df_LR[["Trend_sq", "CO2"]].values
y_multi = df_LR["Adjusted Anomaly"].values

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

y_multi_pred = model_multi.predict(X_multi)

# Metriken berechnen
metrics_LR = {
    "R²-Score": round(r2_score(y_multi, y_multi_pred), 4),
    "MAE": round(mean_absolute_error(y_multi, y_multi_pred), 4),
    "RMSE": round(np.sqrt(mean_squared_error(y_multi, y_multi_pred)), 4)
    }

# Trendvariable Zeit
future_trend = future_years - df_merged["Year"].min()
future_trend_sq = future_trend ** 2

# Zukunftsprognose berechnen
future_X_multi_forecast = np.column_stack((future_trend_sq, future_co2_hw))
future_anomalies = model_multi.predict(future_X_multi_forecast)

# Ergebnisse in DataFrame speichern
df_future["MR_Predicted_Anomaly"] = future_anomalies

# ARIMAX
# p = 3, q = 1, d = 2
arimax_model = ARIMA(y_multi,
                     order=(3, 1, 2),
                     exog=X_multi,
                     enforce_invertibility=False)

arimax_fit = arimax_model.fit()

arimax_forecast = arimax_fit.forecast(steps=len(future_years), 
                                      exog=future_X_multi_forecast)

# Ergebnisse im DataFrame speichern
df_future["ARIMAX_Predicted_Anomaly"] = arimax_forecast

# Jahr finden, in dem der 20-jährige Mittelwert erstmals 1.5°C überschreitet
threshold_df = pd.concat([df_merged[["Year", "Adjusted Anomaly"]], df_future[["Year", "MR_Predicted_Anomaly", "ARIMAX_Predicted_Anomaly", "RCP 4.5 Anomaly"]]])
threshold_df.loc[threshold_df["Year"] < 2025, ["MR_Predicted_Anomaly", "ARIMAX_Predicted_Anomaly", "RCP 4.5 Anomaly"]] = threshold_df.loc[threshold_df["Year"] < 2025, "Adjusted Anomaly"]

# Berechne, wann die Modelle / die Szenarien, das 1.5 Grad Ziel im 20-Jahres-Mittel überschreiten
# 20-jähriger gleitender Mittelwert berechnen
threshold_df["20yr_MA_MR"] = threshold_df["MR_Predicted_Anomaly"].rolling(window=20, min_periods=1).mean()
threshold_df["20yr_MA_ARIMAX"] = threshold_df["ARIMAX_Predicted_Anomaly"].rolling(window=20, min_periods=1).mean()
threshold_df["20yr_MA_RCP_4.5"] = threshold_df["RCP 4.5 Anomaly"].rolling(window=20, min_periods=1).mean()

year_threshold_MR = threshold_df[threshold_df["20yr_MA_MR"] > 1.5]["Year"].min()
year_threshold_ARIMAX = threshold_df[threshold_df["20yr_MA_ARIMAX"] > 1.5]["Year"].min()
year_threshold_RCP = threshold_df[threshold_df["20yr_MA_RCP_4.5"] > 1.5]["Year"].min()

# Streamlit-Konfiguration
st.set_page_config(page_title="Klimawandel Dashboard", layout="wide")

# CSS, um den oberen Abstand zu reduzieren
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Klimawandel Dashboard")

# Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Wähle eine Seite:", [
    "Klima: Seit einem Jahr über 1,5-Grad-Marke",
    "Entwicklung der Anomalien und CO2-Emissionen",
    "Entwicklung von Methan und Distickstoffoxid",
    "Monatliche Temperaturen",
    "Vorhersage der Temperaturanomalien"
])

# Seite 1
if page == "Klima: Seit einem Jahr über 1,5-Grad-Marke":
    st.header("Klima: Seit einem Jahr über 1,5-Grad-Marke")

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(df_merged["Year"], df_merged["Adjusted Anomaly"], color="black", marker="o", markersize=2, linestyle="-", linewidth=1, alpha=0.7, label="Temperaturanomalie")
    plt.plot(df_merged["Year"], df_merged["Adjusted Anomaly"].rolling(window=20).mean(), color="red", linewidth=2, label="Gleitender Mittelwert (20 Jahre)")
    plt.axvspan(1850, 1900, color='gray', alpha=0.2, label="vorindustrielle Zeit")
    plt.axvline(1988, color='purple', linestyle='-', alpha=0.7, label='IPCC Gründung (1988)')
    plt.axvline(2015, color='darkgreen', linestyle='-', alpha=0.7, label='Klimaschutzabkommen (2015)')
    plt.xlabel("Jahr")
    plt.ylabel("Temperaturanomalie (°C)")
    plt.title("Historische Entwicklung der globalen Temperaturanomalien (1850–2024)")
    plt.ylim(-0.6, 1.8)
    plt.yticks(np.arange(-0.4, 1.8, 0.2))
    plt.xlim(1850, 2030)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)

elif page == "Entwicklung der Anomalien und CO2-Emissionen":
    st.header("Entwicklung der Anomalien und CO2-Emissionen")
    st.write("Wähle ein RCP Szenario aus. Die RCP-Szenarien legen bestimmte Szenarien von Treibhausgaskonzentrationen fest. Daraus berechnen Klimamodelle einerseits die Klimaänderung und andererseits die Emissionen (einschließlich aller Rückkopplungen des Kohlenstoffkreislaufs), die erforderlich sind, um diese Konzentrationen hervorzurufen. ")

    # Zeitreihenplots für Temperaturanomalien und Treibhausgasemissionen
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Jahr")
    ax1.set_ylabel("Temperaturanomalie (°C)", color="black")
    ax1.plot(df_temp_yearly["Year"], df_temp_yearly["Adjusted Anomaly"], color="tab:red", label="Temperaturanomalie")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("CO2-Konzentration (Gigatonnen)", color="black")
    ax2.plot(df_merged["Year"], df_merged["CO2"], color="tab:blue", label="CO2-Emissionen")
    ax2.plot(df_ghg["Year"], df_ghg["CO2"], color="tab:blue", linestyle="dashed", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    button1, button2, button3 = st.columns([1, 1, 1])

    with button1:

        if st.button("RCP niedriges Szenario"):
     
            ax1.plot(df_future["Year"], df_future["RCP 2.6 Anomaly"], color="tab:red", linestyle="dashed", linewidth=2)
            ax2.plot(df_future["Year"], df_future["RCP 2.6 CO2"], color="tab:blue", linestyle="dashed", linewidth=2)

    with button2:
        
        if st.button("RCP mittleres Szenario"):

            ax1.plot(df_future["Year"], df_future["RCP 4.5 Anomaly"], color="tab:red", linestyle="dashed", linewidth=2)
            ax2.plot(df_future["Year"], df_future["RCP 4.5 CO2"], color="tab:blue", linestyle="dashed", linewidth=2)

    with button3:

        if st.button("RCP sehr hohes Szenario"):

            ax1.plot(df_future["Year"], df_future["RCP 8.5 Anomaly"], color="tab:red", linestyle="dashed", linewidth=2)
            ax2.plot(df_future["Year"], df_future["RCP 8.5 CO2"], color="tab:blue", linestyle="dashed", linewidth=2)

    # Legenden kombinieren
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Gemeinsame Legende unterhalb des Plots
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.23, 0.88), ncol=1)

    plt.title("Entwicklung der Temperaturanomalien und CO₂-Emissionen")
    plt.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)

elif page == "Entwicklung von Methan und Distickstoffoxid":
    st.header("Entwicklung von Methan und Distickstoffoxid")
    st.write("Treibhausgase sind Gase in der Atmosphäre, die die Wärmerückstrahlung von der Erdoberfläche in das All verhindern. Die natürliche Treibhausgaskonzentration in der Atmosphäre sorgt dafür, dass auf der Erde statt eisiger Weltraumkälte eine durchschnittliche Temperatur von 15°C herrscht.")

    # Zeitreihenplot für CH4 und N2O
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df_merged["Year"], df_merged["CH4"], label="Methan (CH₄)", color="green")
    ax.plot(df_merged["Year"], df_merged["N2O"], label="Distickstoffoxid (N₂O)", color="purple")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Konzentration (ppm)")
    ax.legend()

    plt.title("Entwicklung von CH₄ und N₂O über die Jahre")
    plt.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)

elif page == "Monatliche Temperaturen":
    st.header("Monatliche Temperaturen")

    # Durchschnittliche Temperatur pro Jahr und Monat berechnen, um Duplikate zu vermeiden

    color_scheme = {
        (2024, 2024): "darkred",
        (2023, 2023): "red",
        (2022, 2022): "orange",
        (2011, 2021): "green",
        (1981, 2010): "lightgreen",
        (1951, 1980): "cyan",
        (1921, 1950): "blue",
        (1891, 1920): "darkblue",
        (1850, 1890): "navy",
    }

    df_monthly_temps_yearly = df_monthly_anomaly_yearly.add(monthly_reference_temps_array, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Alle Jahre durchgehen und passende Farbe wählen
    for col in df_monthly_temps_yearly.columns:
        color = "gray"  # Standardfarbe für nicht markierte Jahre

        if col < 2022:
            linewidth = 0.3  # Dünnere Linien
            alpha = 0.01  # Standardtransparenz für nicht markierte Jahre

        else:
            linewidth = 1.5    # Dickere Linien
            alpha = 1

        # Suche die richtige Farbe basierend auf den definierten Intervallen
        for (start, end), c in color_scheme.items():
            if start <= col <= end:
                color = c
                alpha = 1  # Volle Deckkraft für markierte Zeiträume
                break

        plt.plot(df_monthly_temps_yearly.index, df_monthly_temps_yearly[col], color=color, alpha=alpha, linewidth=linewidth)

    plt.xlabel("Monat")
    plt.ylabel("Durchschnittstemperatur (°C)")
    plt.title("Durchschnittliche globale Temperatur pro Monat")
    plt.xticks(np.arange(1, 13, 1), labels=[month_name[i] for i in range(1, 13)])  # X-Achse mit allen Monaten (1-12)
    plt.xlim(1, 12)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Aktualisierte Legende (2022, 2023, 2024 als einzelne Jahre, nicht als Zeiträume)
    legend_handles = [
        plt.Line2D([0], [0], color=color_scheme[(2024, 2024)], lw=2, label="2024"),
        plt.Line2D([0], [0], color=color_scheme[(2023, 2023)], lw=2, label="2023"),
        plt.Line2D([0], [0], color=color_scheme[(2022, 2022)], lw=2, label="2022")
    ]

    # Füge die restlichen Zeiträume zur Legende hinzu
    for (start, end), c in color_scheme.items():
        if start < 2022:  # Nur für ältere Zeiträume
            legend_handles.append(plt.Line2D([0], [0], color=c, lw=1.5, label=f"{start}-{end}"))

    plt.legend(handles=legend_handles, loc="lower center")

    st.pyplot(fig)

elif page == "Vorhersage der Temperaturanomalien":
    st.header("Vorhersage der Temperaturanomalien")
    st.write("1,5-Grad-Ziel (auch 1,5-Grad-Grenze) nennt man das Klimaziel, den menschengemachten globalen Temperaturanstieg durch den Treibhauseffekt im 20-Jahresmittel auf 1,5 Grad Celsius zu begrenzen, gerechnet vom Beginn der Industrialisierung bis zum Jahr 2100. Als vorindustriell wird der Mittelwert der Jahre 1850 bis 1900 verwendet.")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Visualisierung der Prognosen und wann das 1.5 Grad Ziel erreicht wird
    ax.scatter(df_merged["Year"], df_merged["Adjusted Anomaly"], s=10, label="Beobachtete Werte", alpha=0.5)

    ax.plot(df_future["Year"], df_future["MR_Predicted_Anomaly"], color="blue", linestyle="dashed", linewidth=2, label="Multiple Regression")
    ax.axvline(year_threshold_MR, color="blue", linestyle="--", zorder=1)

    ax.plot(df_future["Year"], df_future["ARIMAX_Predicted_Anomaly"], color="red", linestyle="dashed", linewidth=2, label="ARIMAX")
    ax.axvline(year_threshold_ARIMAX, color="red", linestyle="--", zorder=1)
    
    ax.plot(df_future["Year"], df_future["RCP 4.5 Anomaly"], color="green", linestyle="dashed", linewidth=2, label="RCP 4.5 Szenario")
    ax.axvline(year_threshold_RCP, color="green", linestyle="--", zorder=1)

    plt.xlabel("Jahr")
    plt.ylabel("Temperaturanomalie (°C)")
    plt.ylim(bottom=-0.4)
    plt.xticks(np.arange(1850, 2125, 25))
    plt.title("Prognose der globalen Temperaturanomalien bis 2100")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    st.pyplot(fig)

    st.write("Die Modelle überschreiten in den folgenden Jahren erstmals das 1.5 Grad Ziel im 20-Jahres-Mittel:")
    st.write(f"RCP 4.5 Szenario: {year_threshold_RCP}")
    st.write(f"Multiple Lineare Regression: {year_threshold_MR}")
    st.write(f"ARIMAX: {year_threshold_ARIMAX}")