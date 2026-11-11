"""Pipeline de PDI/VC em 4 etapas: aquisição -> pré-processamento -> detecção -> características."""
import os

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

TAM_PADRAO = (92, 112)    # (largura, altura) usado na padronização da imagem
TAM_ROI = (96, 96)        # tamanho fixo da face recortada
GRADE = (8, 8)            # grade de células para os histogramas LBP
P_LBP, R_LBP = 8, 2       # vizinhos e raio do LBP (R=2 escolhido no conjunto de desenvolvimento)
PASTA = os.path.dirname(os.path.abspath(__file__))
CASCATA = cv2.CascadeClassifier(os.path.join(PASTA, "modelos", "haarcascade_frontalface_default.xml"))


class EntradaInvalida(Exception):
    """Arquivo ilegível ou sem face detectável."""


def adquirir(caminho):
    """Etapa 1: lê o arquivo, converte para cinza e padroniza o tamanho."""
    img = None
    if os.path.isfile(caminho):
        dados = np.fromfile(caminho, dtype=np.uint8)      # imdecode (em vez de imread) aceita acentos no caminho (Windows)
        img = cv2.imdecode(dados, cv2.IMREAD_GRAYSCALE) if dados.size else None
    if img is None:
        raise EntradaInvalida(f"arquivo ilegível ou inexistente: {caminho}")
    return cv2.resize(img, TAM_PADRAO, interpolation=cv2.INTER_AREA)


def preprocessar(cinza, filtro=True, clahe=True):
    """Etapa 2: filtro Gaussiano (suaviza ruído) + CLAHE (normaliza iluminação/contraste)."""
    saida = cv2.GaussianBlur(cinza, (3, 3), 0.8) if filtro else cinza
    if clahe:
        saida = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(saida)
    return saida


def detectar_face(img):
    """Etapa 3: Haar Cascade. A imagem recebe borda e é ampliada 3x para a cascata (treinada
    com faces maiores) funcionar. Retorna (x, y, l, a) em coordenadas da imagem original."""
    borda, esc = 40, 3
    ampliada = cv2.resize(cv2.copyMakeBorder(img, borda, borda, borda, borda, cv2.BORDER_REPLICATE),
                          None, fx=esc, fy=esc, interpolation=cv2.INTER_CUBIC)
    faces = CASCATA.detectMultiScale(ampliada, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
    if len(faces) == 0:
        raise EntradaInvalida("nenhuma face detectada")
    x, y, l, a = max(faces, key=lambda f: f[2] * f[3])          # fica com a maior face
    x, y, l, a = (int(v / esc) - (borda if i < 2 else 0) for i, v in enumerate((x, y, l, a)))
    x, y = max(x, 0), max(y, 0)
    return x, y, min(l, img.shape[1] - x), min(a, img.shape[0] - y)


def recortar_roi(img, caixa):
    x, y, l, a = caixa
    return cv2.resize(img[y:y + a, x:x + l], TAM_ROI, interpolation=cv2.INTER_AREA)


def extrair_caracteristicas(roi):
    """Etapa 4: LBP uniforme + histogramas por célula (grade 8x8) concatenados e normalizados."""
    lbp = local_binary_pattern(roi, P_LBP, R_LBP, method="uniform")
    gy, gx = GRADE
    h, w = lbp.shape
    hists = []
    for i in range(gy):
        for j in range(gx):
            celula = lbp[i * h // gy:(i + 1) * h // gy, j * w // gx:(j + 1) * w // gx]
            hist, _ = np.histogram(celula, bins=P_LBP + 2, range=(0, P_LBP + 2))
            hists.append(hist / hist.sum())
    return np.concatenate(hists).astype(np.float32)


def processar(caminho, **opcoes):
    """Executa as 4 etapas e devolve também as imagens intermediárias para exibição."""
    original = adquirir(caminho)
    tratada = preprocessar(original, **opcoes)
    try:
        caixa = detectar_face(tratada)
    except EntradaInvalida as erro:
        erro.etapas = {"original": original, "pre": tratada}    # permite exibir as etapas já feitas
        raise
    roi = recortar_roi(tratada, caixa)
    return {"original": original, "pre": tratada, "caixa": caixa, "roi": roi,
            "vetor": extrair_caracteristicas(roi)}
