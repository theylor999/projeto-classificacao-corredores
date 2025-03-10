import re

def preprocessar(texto):
    """Limpa e padroniza o texto para treinamento/predição."""
    texto = texto.lower()
    texto = re.sub(r"[^a-zA-Z0-9áàãâéêíóôõúç\s]", "", texto)
    return texto.strip()
