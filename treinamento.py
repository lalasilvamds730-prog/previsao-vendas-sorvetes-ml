
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dados simulados da sorveteria Gelato Mágico
temperatura = np.array([20, 22, 24, 26, 28, 30, 32, 34, 36]).reshape(-1, 1)
vendas = np.array([100, 120, 145, 170, 200, 230, 260, 290, 320])

# Separação dos dados
X_train, X_test, y_train, y_test = train_test_split(
    temperatura, vendas, test_size=0.2, random_state=42
)

# Treinamento do modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Previsões
previsoes = modelo.predict(X_test)

# Avaliação
mae = mean_absolute_error(y_test, previsoes)
mse = mean_squared_error(y_test, previsoes)
r2 = r2_score(y_test, previsoes)

print("RESULTADOS - GELATO MÁGICO")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# Registro no MLflow
mlflow.set_experiment("Gelato Magico")

with mlflow.start_run():
    mlflow.log_param("algoritmo", "Regressao Linear")
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    mlflow.sklearn.log_model(
        modelo,
        name="modelo_sorvetes"
    )

# Previsão para 30 graus
nova_temperatura = np.array([[30]])
resultado = modelo.predict(nova_temperatura)

print(f"Previsão para 30°C: {resultado[0]:.0f} sorvetes")
