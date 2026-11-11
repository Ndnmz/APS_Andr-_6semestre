"""Experimento extra: efeito do filtro Gaussiano e do CLAHE (apoia a discussão técnica do relatório).
Resultado puramente descritivo: nada aqui é usado para escolher parâmetros."""
import contextlib
import csv
import io
import os

import numpy as np

import main
from sistema import SistemaBiometrico


def eer_teste(linhas):
    """EER (%) medido nas tentativas de teste: ponto em que FAR = FRR. Não depende do limiar adotado,
    então compara as configurações de pré-processamento em pé de igualdade."""
    def distancias(chave):
        return np.array([l["distancia"] for l in linhas
                         if l["cenario"] == main.CENARIOS[chave] and not np.isnan(l["distancia"])])
    gen, imp = distancias("conhecidos"), distancias("impostores")
    ts = np.unique(np.concatenate([gen, imp]))
    t = ts[int(np.argmin([abs((gen > t).mean() - (imp <= t).mean()) for t in ts]))]
    return float(100 * ((gen > t).mean() + (imp <= t).mean()) / 2)


if __name__ == "__main__":
    configuracoes = [("sem filtro, sem CLAHE", False, False), ("só filtro Gaussiano", True, False),
                     ("só CLAHE", False, True), ("filtro + CLAHE (adotado)", True, True)]
    linhas = []
    for nome, filtro, clahe in configuracoes:
        with contextlib.redirect_stdout(io.StringIO()):
            sistema = SistemaBiometrico(filtro=filtro, clahe=clahe)
            testes = main.executar_testes(sistema)
        far, frr = main.taxas(testes)
        linhas.append({"configuracao": nome, "limiar": round(sistema.limiar, 2),
                       "acuracia_global": main.metricas(testes)[-1]["acuracia"],
                       "FAR": round(far, 1), "FRR": round(frr, 1), "EER_teste": round(eer_teste(testes), 1)})
        print(linhas[-1])
    with open(os.path.join(main.RESULTADOS, "ablacao.csv"), "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=linhas[0].keys())
        escritor.writeheader()
        escritor.writerows(linhas)
