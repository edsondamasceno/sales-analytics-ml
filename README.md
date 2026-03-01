# 📊 Sales Analytics ML – Profit Classification

Projeto de Machine Learning para classificação de registros de vendas com o objetivo de identificar transações de **alto lucro**.

O projeto implementa pipeline completo de ciência de dados com:

- Pré-processamento modular
- Treinamento com GridSearch
- Comparação automática de modelos
- Exportação de métricas
- Geração automática de curva ROC

---

## 🎯 Objetivo do Projeto

Criar um modelo de classificação capaz de prever se uma venda será de **alto lucro** (High_Profit) com base em características comerciais e operacionais.

A variável target foi criada utilizando a mediana de `Total Profit`.

---

## 🏗 Estrutura do Projeto

```
sales-analytics-ml/
│
├── data/
│ └── 1000_Sales_Records.csv
│
├── reports/
│ ├── model_comparison_*.csv
│ └── roc_curve_comparison_*.png
│
├── src/
│ ├── features.py
│ ├── preprocessing.py
│ ├── train.py
│
├── main.py
└── README.md
```

---


---

## 🔄 Pipeline Implementado

### 1️⃣ Feature Engineering
- Criação da variável `High_Profit`
- Conversão de `Order_Ship_Days` para variável numérica
- Separação de variáveis numéricas e categóricas

### 2️⃣ Pré-processamento
- StandardScaler para variáveis numéricas
- OneHotEncoder para variáveis categóricas
- Pipeline integrado com modelos

### 3️⃣ Modelos Treinados
- RandomForestClassifier
- XGBoostClassifier

Ambos com otimização via **GridSearchCV (5-fold)**.

---

## 📊 Resultados Obtidos

| Model         | Accuracy | Precision | Recall | F1    | ROC_AUC |
|--------------|----------|-----------|--------|-------|---------|
| RandomForest | 0.7450   | 0.6788    | 0.9300 | 0.7848| 0.7954  |
| XGBoost      | 0.7550   | 0.6711    | 1.0000 | 0.8032| 0.7816  |

---

## 📈 Análise dos Modelos

### 🔹 RandomForest
- Melhor ROC-AUC
- Melhor equilíbrio geral
- Alta capacidade de separação

### 🔹 XGBoost
- Melhor Accuracy
- Recall perfeito (1.0)
- Melhor F1-score
- Modelo mais agressivo na classe positiva

---

## 📂 Outputs Automáticos

A cada execução do `main.py`, o projeto gera automaticamente:

- 📁 CSV comparativo com métricas dos modelos
- 📊 Gráfico ROC comparativo salvo como PNG

Exemplo:

reports/
- model_comparison_20260228_184500.csv
- roc_curve_comparison_20260228_184500.png

---

## 🚀 Como Executar

1. Criar ambiente virtual:

- python -m venv venv
- source venv/bin/activate

---

2. Instalar dependências:

- pip install -r requirements.txt

---

3. Executar projeto:

- python main.py

---

## 🧠 Insights de Negócio

- O modelo XGBoost maximiza a detecção de vendas de alto lucro (Recall = 1.0).
- O RandomForest apresenta melhor capacidade de separação global (ROC-AUC superior).
- A escolha do modelo depende da estratégia:
  - Maximizar detecção → XGBoost
  - Equilíbrio geral → RandomForest

---

## 📌 Próximas Melhorias (Roadmap)

- Ajuste de threshold ótimo
- Curva Precision-Recall
- Feature Importance automática
- Deploy com Streamlit
- Salvamento do modelo final (.pkl)

---

## 👨‍💻 Autor

Edson Damasceno

Projeto desenvolvido para portfólio em Ciência de Dados.
