import joblib
from utils import preprocessar

# Carregar modelo e vetorizar
model = joblib.load("../models/modelo_corredor.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

def prever_corredor(produto):
    """Retorna o corredor previsto para um produto."""
    produto = preprocessar(produto)
    produto_tfidf = vectorizer.transform([produto])
    return model.predict(produto_tfidf)[0]

# Testes
itens = [
    "carne bovina", "shampoo", "arroz integral", "sabão em pó", "leite desnatado",
    "detergente líquido", "banana nanica", "pão de forma integral", "queijo mussarela",
    "iogurte natural", "cerveja lata", "refrigerante cola", "óleo de soja", "farinha de trigo",
    "macarrão espaguete", "creme dental", "chocolate ao leite", "presunto fatiado",
    "desodorante aerosol", "café solúvel", "manteiga com sal", "frango congelado",
    "água mineral", "pipoca de micro-ondas", "batata frita congelada", "molho de tomate",
    "feijão preto", "vinagre de maçã", "azeite de oliva", "cereal matinal"
]

for item in itens:
    corredor = prever_corredor(item).strip().replace('"', '')
    print(f"Produto: {item} -> Corredor previsto: {corredor}")
