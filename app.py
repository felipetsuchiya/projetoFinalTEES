import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuração do aplicativo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static/images'

# Variáveis globais
global X_train, X_test, y_train, y_test
graphs = []  # Lista global para armazenar os caminhos dos gráficos gerados


# Rota principal - Upload e geração de gráficos
@app.route('/', methods=['GET', 'POST'])
def index():
    global graphs
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Salvar o arquivo
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Gerar gráficos
            graphs = generate_graphs(filepath)
            process_data(filepath)
            return redirect(url_for('analyze'))
    return render_template('index.html')


# Rota para análise de gráficos
@app.route('/analyze')
def analyze():
    global graphs
    return render_template('graphs.html', graphs=graphs)


# Rota para predição
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global X_train, X_test, y_train, y_test

    if request.method == 'POST':
        model_type = request.form['model']
        if model_type == 'Logistic Regression':
            model = LogisticRegression()
        elif model_type == 'Decision Tree':
            model = DecisionTreeClassifier()
        elif model_type == 'Random Forest':
            model = RandomForestClassifier()
        else:
            return "Modelo não suportado!"

        # Treinar o modelo
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Avaliar o modelo
        metrics = {
            'Acurácia': accuracy_score(y_test, y_pred),
            'Precisão': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted')
        }

        # Gerar matriz de confusão
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Negado (0)", "Aprovado (1)"],
                    yticklabels=["Negado (0)", "Aprovado (1)"])
        plt.title(f'Matriz de Confusão - {model_type}')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        conf_matrix_path = os.path.join(app.config['STATIC_FOLDER'], 'conf_matrix.png')
        plt.savefig(conf_matrix_path)
        plt.close()

        return render_template('results_predict.html', metrics=metrics, model_type=model_type,
                               conf_matrix_path=conf_matrix_path)

    return render_template('predict.html')


# Função para processar os dados do CSV
def process_data(filepath):
    global X_train, X_test, y_train, y_test

    data = pd.read_csv(filepath)

    # Preencher valores ausentes
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)

    # Codificar variáveis categóricas
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

    # Separar features e target
    X = data.drop('loan_status', axis=1)  # Substitua 'loan_status' pelo nome correto, se necessário
    y = data['loan_status']

    # Normalizar os dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Função para gerar gráficos
def generate_graphs(filepath):
    data = pd.read_csv(filepath)
    graph_paths = []

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            # Histograma para colunas numéricas
            plt.figure(figsize=(8, 6))
            sns.histplot(data[column], kde=True, color='blue', label='Distribuição')
            plt.title(f'Histograma - {column}')
            plt.xlabel(column)
            plt.ylabel('Frequência')
            plt.legend(loc='upper right')
            hist_path = os.path.join(app.config['STATIC_FOLDER'], f'hist_{column}.png')
            plt.savefig(hist_path)
            plt.close()
            graph_paths.append(f'images/hist_{column}.png')

        elif pd.api.types.is_categorical_dtype(data[column]) or data[column].dtype == object:
            # Gráfico de barras para colunas categóricas
            plt.figure(figsize=(8, 6))
            data[column].value_counts().plot(kind='bar', color='orange', label='Frequência')
            plt.title(f'Gráfico de Barras - {column}')
            plt.xlabel(column)
            plt.ylabel('Contagem')
            plt.legend(loc='upper right')
            bar_path = os.path.join(app.config['STATIC_FOLDER'], f'bar_{column}.png')
            plt.savefig(bar_path)
            plt.close()
            graph_paths.append(f'images/bar_{column}.png')

    return graph_paths

@app.route('/configure_knn', methods=['GET', 'POST'])
def configure_knn():
    global X_train, X_test, y_train, y_test

    if request.method == 'POST':
        # Obter o modelo escolhido
        model_type = request.form['model']

        # Configurar parâmetros específicos para cada modelo
        if model_type == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            n_neighbors = int(request.form['n_neighbors'])
            metric = request.form['metric']
            model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        elif model_type == 'Decision Tree':
            from sklearn.tree import DecisionTreeClassifier
            max_depth = int(request.form['max_depth'])
            criterion = request.form['criterion']
            model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
        elif model_type == 'Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            n_estimators = int(request.form['n_estimators'])
            max_depth = int(request.form['max_depth_rf'])
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        elif model_type == 'Logistic Regression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
        else:
            return "Modelo não suportado!"

        # Treinar o modelo
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Avaliar o modelo
        metrics = {
            'Acurácia': accuracy_score(y_test, y_pred),
            'Precisão': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted')
        }

        # Gerar matriz de confusão
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Negado (0)", "Aprovado (1)"],
                    yticklabels=["Negado (0)", "Aprovado (1)"])
        plt.title(f'Matriz de Confusão - {model_type}')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        conf_matrix_path = os.path.join(app.config['STATIC_FOLDER'], f'{model_type}_conf_matrix.png')
        plt.savefig(conf_matrix_path)
        plt.close()

        return render_template('results_classifier.html', metrics=metrics, model_type=model_type, conf_matrix_path=conf_matrix_path)

    return render_template('configure_classifier.html')



# Executar o servidor
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['STATIC_FOLDER']):
        os.makedirs(app.config['STATIC_FOLDER'])
    app.run(debug=True)
