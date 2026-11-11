"""Enquadramento de fotos reais (selfies, retratos de corpo inteiro) antes do pipeline.

O pipeline padroniza a imagem INTEIRA para 92x112. Numa selfie 720x1280 o rosto fica minúsculo (e a imagem
é achatada), e o Haar Cascade não o encontra. Aqui localizamos a maior face na imagem original e recortamos
uma janela 92:112 em torno dela, com o rosto ocupando ~60% da largura (a mesma escala das imagens ORL).
Se nenhuma face for achada, devolve a imagem como veio (o pipeline então rejeita como entrada inválida)."""
import cv2
import numpy as np

from pipeline import CASCATA

PROPORCAO = 112 / 92        # altura / largura do padrão do pipeline
FRACAO_FACE = 0.60          # largura da face / largura do recorte
LARGURA_BUSCA = 480         # a busca roda numa versão reduzida (mais rápida e estável)


def ler_cinza(caminho):
    dados = np.fromfile(caminho, dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_GRAYSCALE) if dados.size else None


def cinza_de_bytes(dados):
    """Decodifica bytes de uma imagem enviada (JPEG/PNG/JFIF...) para cinza; None se ilegível."""
    arr = np.frombuffer(dados, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE) if arr.size else None


def recortar_retrato(cinza):
    """Retorna (recorte, caixa_da_face) ou (cinza, None) se não houver face."""
    if cinza is None:
        return None, None
    esc = min(1.0, LARGURA_BUSCA / cinza.shape[1])
    pequena = cv2.resize(cinza, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA) if esc < 1 else cinza
    faces = CASCATA.detectMultiScale(pequena, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(int(0.12 * pequena.shape[1]),) * 2)
    if len(faces) == 0:
        return cinza, None
    x, y, l, a = (v / esc for v in max(faces, key=lambda f: f[2] * f[3]))
    larg = l / FRACAO_FACE
    alt = larg * PROPORCAO
    cx, cy = x + l / 2, y + a / 2
    x0, y0 = int(round(cx - larg / 2)), int(round(cy - alt / 2))
    larg, alt = int(round(larg)), int(round(alt))
    # borda replicada para recortes que passam do limite da foto
    m = max(larg, alt)
    ampliada = cv2.copyMakeBorder(cinza, m, m, m, m, cv2.BORDER_REPLICATE)
    recorte = ampliada[y0 + m:y0 + m + alt, x0 + m:x0 + m + larg]
    return recorte, (int(x), int(y), int(l), int(a))
