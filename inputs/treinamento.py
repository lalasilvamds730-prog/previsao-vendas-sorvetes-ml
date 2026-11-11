
# Projeto: Gelato Mágico
# Previsão de vendas de sorvetes com Machine Learning

import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Dados históricos simulados
temperatura = np.array([
    20, 22, 24, 26, 28, 30, 32, 34, 36
]).reshape(-1, 1)

vendas = np.array([
    100, 120, 145, 170, 200, 230, 260, 290, 320
])

# 2. Separação dos dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    temperatura,
    vendas,
    test_size=0.2,
    random_state=42
)

# 3. Criação e treinamento do modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 4. Realização das previsões
previsoes = modelo.predict(X_test)

# 5. Avaliação do modelo
mae = mean_absolute_error(y_test, previsoes)
mse = mean_squared_error(y_test, previsoes)
r2 = r2_score(y_test, previsoes)

print("=== RESULTADOS DO MODELO ===")
print(f"Erro médio absoluto (MAE): {mae:.2f}")
print(f"Erro quadrático médio (MSE): {mse:.2f}")
print(f"Coeficiente R²: {r2:.2f}")

# 6. Registro do experimento com MLflow
mlflow.set_experiment("Gelato Magico - Previsao de Vendas")

with mlflow.start_run():
    mlflow.log_param("algoritmo", "Regressao Linear")
    mlflow.log_param("dados_treinamento", len(X_train))
    mlflow.log_param("dados_teste", len(X_test))

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    mlflow.sklearn.log_model(
        modelo,
        name="modelo_previsao_sorvetes"
    )

    print("\nExperimento registrado no MLflow!")

# 7. Previsão para uma nova temperatura
nova_temperatura = np.array([[30]])

venda_prevista = modelo.predict(nova_temperatura)

print(
    f"\nCom temperatura de 30°C, "
    f"a previsão é de {venda_prevista[0]:.0f} sorvetes."
)
