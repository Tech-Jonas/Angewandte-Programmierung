# Sonstige Beteiligung Angewandte Programmierung
## 🌍 Klima Streamlit-Dashboard

### 📌 Projektübersicht: Temperatur- und Klimaanalyse  
Dieses Projekt analysiert globale Temperaturdaten von 1850 bis 2024 und erstellt eine Prognose für die zukünftige Erwärmung basierend auf verschiedenen Klimafaktoren. Es kombiniert statistische Modelle mit exogenen Variablen (z. B. CO₂-Konzentrationen oder Treibhausgas-Emissionen) zur Vorhersage des Zeitpunkts, an dem die 1.5°C-Schwelle überschritten wird.

### 📊 Datenquellen  
Die verwendeten Daten stammen aus verschiedenen Quellen:

| Daten | Quelle |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| **Temperaturdaten** | [Berkeley Earth - Land & Ozean Temperaturen](https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_complete.txt) |
| **Treibhausgaskonzentrationen** | [Climate Indicator - Treibhausgaskonzentrationen](https://github.com/ClimateIndicator/data/blob/v2024.05.29b/data/greenhouse_gas_emissions/greenhouse_gas_emissions_co2eq.csv) |
| **Effective Radiative Forcing (ERF, Strahlungsantriebe natürlicher und anthropogener Faktoren)** | [Climate Indicator - ERF Timeseries](https://github.com/ClimateIndicator/forcing-timeseries/blob/main/output/ERF_best_aggregates_1750-2023.csv) |
| **THG-Emissionen** | [Climate Indicator - THG-Emissionen](https://github.com/ClimateIndicator/forcing-timeseries/blob/main/data/ghg_concentrations/ar6_updated/ipcc_ar6_wg1.csv) |


### 🏗 Methodik und Modellierung

#### Datenaufbereitung
- Berechnung der Temperaturanomalien pro Monat und Jahr
- Erstellung von monatlichen und jährlichen Mittelwerten
- Verknüpfung mit THG-Konzentrationen

#### Explorative Datenanalyse (EDA)
- Visualisierung der historischen Temperaturtrends
- Untersuchung von monatlichen Temperaturverläufen
- Darstellung der Entwicklung der THG-Emissionen

#### Modellierung
- Multiple Regression
- Vergleich zum RCP 4.5 Szenario

### 📈 Ergebnisse & Visualisierungen
- Historische Temperaturtrends (1850–2024)
- Monatliche Anomalien im Langzeitvergleich
- Absoluter Temperaturverlauf pro Monat & Jahr
- Prognose der globalen Erwärmung bis 2100
- Wann wird die 1.5°C-Grenze überschritten?
