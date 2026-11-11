# 🍦 Gelato Mágico — Previsão de Vendas com Machine Learning

## 📌 Sobre o Projeto

O Gelato Mágico é um projeto de Machine Learning desenvolvido como parte da formação DP-100 da Microsoft na DIO.

O objetivo é desenvolver um modelo preditivo capaz de estimar a quantidade de sorvetes vendidos com base na temperatura ambiente.

A solução busca ajudar uma sorveteria a planejar sua produção, reduzir desperdícios e melhorar seus resultados.

## 🎯 Objetivos

- Treinar um modelo de Machine Learning.
- Aplicar regressão linear para previsão de vendas.
- Avaliar o desempenho do modelo.
- Registrar experimentos utilizando MLflow.
- Explorar possibilidades de implantação em nuvem.

## 🛠️ Tecnologias utilizadas

- Python
- NumPy
- Pandas
- Scikit-learn
- MLflow
- Matplotlib
- GitHub

## 📂 Estrutura do Projeto

```text
previsao-vendas-sorvetes-ml/
│
├── inputs/
│   └── dados_vendas.txt
│
├── treinamento.py
├── requirements.txt
└── README.md
```

## ⚙️ Funcionamento

O projeto utiliza dados simulados de temperatura e vendas para treinar um modelo de regressão linear.

O modelo aprende a relação entre a temperatura e a quantidade de sorvetes vendidos e utiliza esse aprendizado para realizar previsões.

## 📊 Avaliação do Modelo

O desempenho é avaliado utilizando:

- MAE: erro médio absoluto.
- MSE: erro quadrático médio.
- R²: coeficiente de determinação.

### Resultados obtidos

Após executar o treinamento, obtivemos os seguintes resultados:

- **MAE:** 1,43
- **MSE:** 4,08
- **R²:** 0,9994
- **Previsão para 30°C:** 231 sorvetes.

O modelo utilizou Regressão Linear e teve seu experimento registrado no MLflow, incluindo parâmetros, métricas e arquivos do modelo.

**Observação:** foram utilizados dados simulados para fins educacionais.

## 💡 Aprendizados

Durante o desenvolvimento, são explorados conceitos de preparação de dados, treinamento de modelos, regressão, avaliação de resultados e acompanhamento de experimentos com MLflow.

## 🚀 Possibilidades futuras

- Implantar o modelo em ambiente de nuvem.
- Criar uma API para previsões em tempo real.
- Desenvolver um dashboard de acompanhamento das vendas.
- Ampliar a base de dados para melhorar as previsões.

## 👩‍💻 Autora

**Larissa Martins**

Projeto desenvolvido para fins educacionais na formação DP-100 da Microsoft — DIO.

[Meu GitHub](https://github.com/lalasilvamds730-prog)
