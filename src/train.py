import pandas as pd
import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from utils import preprocessar

# Carregar dataset
print("Carregando dataset...")
df = pd.read_csv("../data/itens_treinamento_corredor.csv", sep=",", quoting=3, on_bad_lines="skip", encoding="utf-8")

# Remover valores nulos
print("Removendo valores nulos...")
df.dropna(subset=["produto", "corredor"], inplace=True)
df = df[df["corredor"] != "COMBOS"]

# Pré-processar textos
df["produto"] = df["produto"].astype(str).apply(preprocessar)

# Amostragem para eficiência
amostra = 5_000_000
if len(df) > amostra:
    df = df.sample(amostra, random_state=42)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(df["produto"], df["corredor"], test_size=0.2, random_state=42)

# Vetorização TF-IDF
print("Vetorizando os textos...")
vectorizer = TfidfVectorizer(max_features=10_000)
X_train_tfidf = vectorizer.fit_transform(tqdm(X_train, desc="Vetorizando treino"))
X_test_tfidf = vectorizer.transform(tqdm(X_test, desc="Vetorizando teste"))

# Conversão para formato esparso
X_train_tfidf = csr_matrix(X_train_tfidf).astype(np.float32)
X_test_tfidf = csr_matrix(X_test_tfidf).astype(np.float32)

# Treinamento do modelo
print("Treinando o modelo...")
model = SGDClassifier(loss="hinge", max_iter=1000, n_jobs=-1)
model.fit(X_train_tfidf, y_train)

# Avaliação
print("Acurácia:", accuracy_score(y_test, model.predict(X_test_tfidf)))
print(classification_report(y_test, model.predict(X_test_tfidf)))

# Salvar modelo e vetorizar
joblib.dump(model, "../models/modelo_corredor.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
print("Modelo e vetorizador salvos com sucesso!")
