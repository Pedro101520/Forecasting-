# Forecasting-

# ============================================
# Forecasting com ARIMA e Prophet
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
from prophet import Prophet

# ============================================
# 1. Leitura dos dados
# ============================================
# Espera-se um CSV com colunas: "data","valor"
df = pd.read_csv("serie_historica.csv", parse_dates=["data"])
df = df.set_index("data").asfreq("B")   # frequência dias úteis
serie = df["valor"]

# ============================================
# 2. Divisão treino/teste
# ============================================
train_size = int(len(serie) * 0.8)
train, test = serie[:train_size], serie[train_size:]

# ============================================
# 3. Modelo ARIMA
# ============================================
auto_model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
print("Melhor ARIMA:", auto_model.order)

# Ajuste e previsão
arima_forecast = auto_model.predict(n_periods=len(test))

# Avaliação
rmse_arima = np.sqrt(mean_squared_error(test, arima_forecast))
mae_arima = mean_absolute_error(test, arima_forecast)
mape_arima = np.mean(np.abs((test - arima_forecast) / test)) * 100

print(f"[ARIMA] RMSE={rmse_arima:.2f}, MAE={mae_arima:.2f}, MAPE={mape_arima:.2f}%")

# ============================================
# 4. Modelo Prophet
# ============================================
df_prophet = pd.DataFrame({"ds": train.index, "y": train.values})

model = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=True)
model.fit(df_prophet)

# Horizonte = mesmo tamanho do teste
future = model.make_future_dataframe(periods=len(test), freq="B")
forecast = model.predict(future)

prophet_forecast = forecast["yhat"].iloc[-len(test):].values

# Avaliação
rmse_prophet = np.sqrt(mean_squared_error(test, prophet_forecast))
mae_prophet = mean_absolute_error(test, prophet_forecast)
mape_prophet = np.mean(np.abs((test - prophet_forecast) / test)) * 100

print(f"[Prophet] RMSE={rmse_prophet:.2f}, MAE={mae_prophet:.2f}, MAPE={mape_prophet:.2f}%")

# ============================================
# 5. Comparação visual
# ============================================
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label="Treino")
plt.plot(test.index, test, label="Teste", color="green")
plt.plot(test.index, arima_forecast, label="ARIMA Previsão", color="red")
plt.plot(test.index, prophet_forecast, label="Prophet Previsão", color="orange")
plt.legend()
plt.title("Comparação ARIMA vs Prophet")
plt.show()

# ============================================
# 6. Previsão futura (5 dias úteis após 29/08/2025)
# ============================================

# ---- ARIMA ----
arima_future = auto_model.predict(n_periods=5)
print("\nPrevisão ARIMA para os próximos 5 dias úteis:")
print(arima_future)

# ---- Prophet ----
full_data = pd.DataFrame({"ds": serie.index, "y": serie.values})
model_full = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=True)
model_full.fit(full_data)

future_dates = model_full.make_future_dataframe(periods=5, freq="B")
forecast_future = model_full.predict(future_dates)

print("\nPrevisão Prophet para os próximos 5 dias úteis:")
print(forecast_future[["ds","yhat","yhat_lower","yhat_upper"]].tail(5))
