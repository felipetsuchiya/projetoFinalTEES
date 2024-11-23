# **Projeto de Predição com Classificadores Personalizáveis**

Este projeto é um aplicativo **Flask** interativo que permite ao usuário:

- Fazer upload de datasets estruturados em **CSV**.
- Visualizar **gráficos gerados automaticamente** com base nos dados enviados.
- Escolher entre diferentes **classificadores de Machine Learning** (como KNN, Decision Tree, Random Forest e Logistic Regression).
- Configurar parâmetros específicos de cada classificador.
- Avaliar o desempenho dos modelos com métricas detalhadas e visualizar a **matriz de confusão**.

---

## **Funcionalidades**

### **1. Upload de Dataset**
O usuário pode:
- Fazer upload de um arquivo CSV contendo dados estruturados.
- A aplicação processa automaticamente o dataset, preenche valores ausentes e codifica variáveis categóricas.

### **2. Visualização de Gráficos**
Após o upload, a aplicação:
- Gera **gráficos interativos** como:
  - Histogramas para variáveis numéricas.
  - Gráficos de barras para variáveis categóricas.
- Exibe os gráficos em uma interface amigável.

### **3. Configuração de Classificadores**
O usuário pode:
- Escolher entre os seguintes modelos:
  - **KNN (K-Nearest Neighbors)**.
  - **Decision Tree**.
  - **Random Forest**.
  - **Logistic Regression**.
- Ajustar parâmetros específicos:
  - Número de vizinhos e métrica de distância (KNN).
  - Profundidade máxima e critério de divisão (Decision Tree).
  - Número de estimadores e profundidade máxima (Random Forest).

### **4. Resultados Detalhados**
Após o treinamento e teste:
- A aplicação exibe as métricas de desempenho, incluindo:
  - **Acurácia**.
  - **Precisão**.
  - **Recall**.
  - **F1-Score**.
- Gera uma **matriz de confusão** para avaliação visual das predições.

---

## **Como Usar**

### **1. Pré-requisitos**
Certifique-se de ter instalado:
- Python 3.8 ou superior.
- As bibliotecas Python listadas em `requirements.txt`.

### **2. Instalação**
Clone o repositório e instale as dependências:
```bash
git clone https://github.com/felipetsuchiya/projetoFinalTEES.git
cd projeto-predicao-classificadores
pip install -r requirements.txt
