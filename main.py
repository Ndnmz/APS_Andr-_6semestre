"""Protótipo de AUTENTICAÇÃO biométrica facial (1:1) - APS PDI/VC.

Uso:
  python main.py demo                     etapas visuais + mensagens de acesso de 5 tentativas de exemplo
  python main.py avaliar                  3 cenários de teste, métricas (acurácia, FAR, FRR) e gráficos
  python main.py autenticar maria img.jpg o usuário alega ser 'maria' e apresenta a imagem img.jpg
Opção --mostrar abre as figuras na tela (além de salvá-las em resultados/).
"""
import argparse
import csv
import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pipeline import PASTA
from sistema import SistemaBiometrico

RESULTADOS = os.path.join(PASTA, "resultados")
TESTE = os.path.join(PASTA, "base", "teste")
CENARIOS = {"conhecidos": "1. Tentativa correta", "impostores": "2. Identidade alheia/não cadastrado",
            "invalidas": "3. Entrada inválida"}
# Fotos de teste dos cadastrados: base/teste/genuinos/<id>_<n>.jfif|png|jpg (fotos inteiras; o rosto é enquadrado
# automaticamente). As imagens ORL de base/teste/conhecidos e desconhecidos são tentativas de impostor.


def exemplos(sistema):
    """Tentativas de exemplo para a figura das etapas (montadas a partir do que existe em base/teste): fotos genuínas
    de até 2 cadastrados, uma delas alegando a identidade de outro, uma imagem ORL não cadastrada e uma entrada inválida."""
    ids = list(sistema.usuarios)
    por_dono = {}
    for caminho in sorted(glob.glob(os.path.join(TESTE, "genuinos", "*"))):
        por_dono.setdefault(os.path.basename(caminho).split("_")[0], caminho)
    casos = [(c, d) for d, c in por_dono.items() if d in ids][:2]
    if casos and len(ids) > 1:
        casos.append((casos[0][0], ids[(ids.index(casos[0][1]) + 1) % len(ids)]))
    for pasta in ("desconhecidos", "invalidas"):
        arquivos = sorted(glob.glob(os.path.join(TESTE, pasta, "*")))
        if arquivos:
            casos.append((arquivos[0], ids[0]))
    return casos


def salvar(fig, nome, mostrar):
    os.makedirs(RESULTADOS, exist_ok=True)
    fig.savefig(os.path.join(RESULTADOS, nome), dpi=130, bbox_inches="tight")
    if mostrar:
        plt.show()
    plt.close(fig)


def autenticar_arquivo(sistema, caminho, alegado):
    """Enquadra o rosto (se for foto inteira), autentica e apaga o arquivo temporário criado."""
    arquivo = enquadrar(caminho)
    try:
        return sistema.autenticar(arquivo, alegado)
    finally:
        if arquivo != caminho:
            os.remove(arquivo)


def figura_etapas(sistema, casos, nome, mostrar):
    """Uma linha por tentativa (caminho, identidade alegada): original | pré-processada | detecção | ROI."""
    fig, eixos = plt.subplots(len(casos), 4, figsize=(10, 2.7 * len(casos)), squeeze=False)
    for n, (linha, (caminho, alegado)) in enumerate(zip(eixos, casos)):
        res = autenticar_arquivo(sistema, caminho, alegado)
        et = res["etapas"] or {}
        print(f"\n{os.path.basename(caminho)} (alega ser {alegado})\n  {sistema.mensagem(res)}")
        paineis = [et.get("original"), et.get("pre"), et.get("pre"), et.get("roi")]
        titulos = ["1. Original", "2. Pré-processada", "3. Detecção (ROI)", "4. ROI recortada"]
        for k, (ax, img, tit) in enumerate(zip(linha, paineis, titulos)):
            ax.axis("off")
            if img is None:
                ax.text(0.5, 0.5, "não gerada", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if k == 2 and "caixa" in et:
                x, y, l, a = et["caixa"]
                ax.add_patch(plt.Rectangle((x, y), l, a, fill=False, edgecolor="lime", linewidth=2))
            ax.set_title(tit if n == 0 else "", fontsize=9)      # títulos só na primeira linha
        rotulo = {"autenticado": "AUTENTICADO", "rejeitado": "REJEITADO", "invalida": "ENTRADA INVÁLIDA"}[res["status"]]
        linha[0].text(-0.08, 0.5, f"{os.path.basename(caminho)}\nalega: {alegado}\n{rotulo}", rotation=90,
                      fontsize=8, ha="right", va="center", transform=linha[0].transAxes)
    salvar(fig, nome, mostrar)


def montar_tentativas(sistema):
    """Lista (cenário, caminho, identidade alegada, deve_liberar). As alegações são determinísticas:
    cada imagem de usuário cadastrado é usada duas vezes (identidade própria e identidade de outro usuário)."""
    ids = list(sistema.usuarios)
    n = len(ids)
    tentativas = []
    for caminho in sorted(glob.glob(os.path.join(TESTE, "genuinos", "*"))):      # fotos do(s) cadastrado(s)
        dono = os.path.basename(caminho).split("_")[0]
        dono = dono if dono in ids else ids[0]
        tentativas.append(("conhecidos", caminho, dono, True))
        if n > 1:        # a mesma foto alegando a identidade de OUTRO cadastrado (o impostor mais difícil)
            tentativas.append(("impostores", caminho, ids[(ids.index(dono) + 1) % n], False))
    for i, caminho in enumerate(sorted(glob.glob(os.path.join(TESTE, "conhecidos", "*")) +
                                       glob.glob(os.path.join(TESTE, "desconhecidos", "*")))):
        tentativas.append(("impostores", caminho, ids[i % n], False))      # ninguém aqui é cadastrado
    for i, caminho in enumerate(sorted(glob.glob(os.path.join(TESTE, "invalidas", "*")))):
        tentativas.append(("invalidas", caminho, ids[i % n], False))
    return tentativas


def enquadrar(caminho):
    """Fotos inteiras (selfies): recorta o rosto e salva em arquivo temporário PNG. Imagens que já estão no formato
    da base ORL (conhecidos/desconhecidos/invalidas) passam direto."""
    import tempfile
    if os.path.basename(os.path.dirname(caminho)) in ("conhecidos", "desconhecidos", "invalidas", "impostores"):
        return caminho
    from entrada import ler_cinza, recortar_retrato
    recorte, _ = recortar_retrato(ler_cinza(caminho))
    if recorte is None:
        return caminho
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(tmp, recorte)
    return tmp


def executar_testes(sistema):
    """Roda os 3 cenários. Acerto = liberar quem alega a própria identidade e negar todos os demais."""
    linhas = []
    for cen, caminho, alegado, deve_liberar in montar_tentativas(sistema):
        res = autenticar_arquivo(sistema, caminho, alegado)
        liberou = res["status"] == "autenticado"
        linhas.append({"cenario": CENARIOS[cen], "arquivo": os.path.relpath(caminho, TESTE), "alegado": alegado,
                       "esperado": "liberar" if deve_liberar else "negar", "resultado": "liberar" if liberou else "negar",
                       "motivo": res.get("motivo", ""), "distancia": round(res.get("distancia", float("nan")), 4),
                       "acerto": liberou == deve_liberar})
    return linhas


def metricas(linhas):
    tabela = []
    for cenario in list(CENARIOS.values()) + ["Global"]:
        sel = [l for l in linhas if cenario == "Global" or l["cenario"] == cenario]
        acertos = sum(l["acerto"] for l in sel)
        tabela.append({"cenario": cenario, "testes": len(sel), "acertos": acertos,
                       "erros": len(sel) - acertos, "acuracia": round(100 * acertos / len(sel), 1) if sel else None})
    return tabela


def taxas(linhas):
    """FAR (impostores liberados / tentativas impostoras) e FRR (genuínos negados / tentativas genuínas), em %.
    Entradas inválidas ficam de fora: não há uma comparação biométrica a ser aceita ou rejeitada."""
    gen = [l for l in linhas if l["cenario"] == CENARIOS["conhecidos"]]
    imp = [l for l in linhas if l["cenario"] == CENARIOS["impostores"]]
    taxa = lambda grupo: round(100 * sum(not l["acerto"] for l in grupo) / len(grupo), 1) if grupo else float("nan")
    return taxa(imp), taxa(gen)          # FRR = nan enquanto não houver fotos em base/teste/genuinos/


def grafico_distancias(sistema, linhas, mostrar):
    """Esquerda: distâncias das tentativas genuínas x impostoras. Direita: FAR e FRR em função do limiar."""
    dist = {c: np.array([l["distancia"] for l in linhas if l["cenario"] == CENARIOS[c] and not np.isnan(l["distancia"])])
            for c in ("conhecidos", "impostores")}
    if len(dist["conhecidos"]) == 0:        # sem tentativas genuínas não há curva FRR para desenhar
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.hist(dist["conhecidos"], bins=20, alpha=0.6, color="tab:blue", label="genuínas (identidade própria)")
    ax1.hist(dist["impostores"], bins=20, alpha=0.6, color="tab:red", label="impostoras (identidade alheia)")
    ax1.set_xlabel("Distância qui-quadrado à identidade alegada")
    ax1.set_ylabel("Nº de tentativas")
    ts = np.linspace(min(d.min() for d in dist.values()), max(d.max() for d in dist.values()), 200)
    ax2.plot(ts, [100 * (dist["impostores"] <= t).mean() for t in ts], color="tab:red", label="FAR")
    ax2.plot(ts, [100 * (dist["conhecidos"] > t).mean() for t in ts], color="tab:blue", label="FRR")
    ax2.set_xlabel("Limiar de decisão")
    ax2.set_ylabel("Taxa de erro (%)")
    for ax in (ax1, ax2):
        ax.axvline(sistema.limiar, color="k", linestyle="--", label=f"limiar = {sistema.limiar:.3f}")
        ax.legend(fontsize=8)
    salvar(fig, "distancias.png", mostrar)


def avaliar(sistema, mostrar):
    linhas = executar_testes(sistema)
    tabela = metricas(linhas)
    far, frr = taxas(linhas)
    print(f"\nLimiar de decisão (calibrado no cadastro): {sistema.limiar:.4f}")
    print(f"{'Cenário':<38}{'Testes':>7}{'Acertos':>9}{'Erros':>7}{'Acurácia':>10}")
    for t in tabela:
        ac = "n/d" if t["acuracia"] is None else f"{t['acuracia']:.1f}%"
        print(f"{t['cenario']:<38}{t['testes']:>7}{t['acertos']:>9}{t['erros']:>7}{ac:>10}")
    print(f"FAR (falsa aceitação) = {far:.1f}%   FRR (falsa rejeição) = {frr:.1f}%")
    os.makedirs(RESULTADOS, exist_ok=True)
    with open(os.path.join(RESULTADOS, "avaliacao_detalhada.csv"), "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=linhas[0].keys())
        escritor.writeheader()
        escritor.writerows(linhas)
    with open(os.path.join(RESULTADOS, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump({"limiar": sistema.limiar, "FAR": None if np.isnan(far) else round(far, 1),
                   "FRR": None if np.isnan(frr) else round(frr, 1), "tabela": tabela},
                  f, ensure_ascii=False, indent=2)
    grafico_distancias(sistema, linhas, mostrar)
    for l in linhas:
        if not l["acerto"]:
            print(f"  erro: {l['cenario']} | {l['arquivo']} alegando {l['alegado']} -> {l['resultado']} (esperado: {l['esperado']})")
    return tabela


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comando", choices=["demo", "avaliar", "autenticar"])
    ap.add_argument("usuario", nargs="?", help="identidade alegada (comando autenticar)")
    ap.add_argument("imagem", nargs="?", help="imagem apresentada (comando autenticar)")
    ap.add_argument("--mostrar", action="store_true")
    args = ap.parse_args()
    sistema = SistemaBiometrico()
    print(f"{len(sistema.usuarios)} usuários cadastrados, {len(sistema.ids)} imagens de referência.")
    if args.comando == "demo":
        figura_etapas(sistema, exemplos(sistema), "figura_etapas.png", args.mostrar)
    elif args.comando == "avaliar":
        avaliar(sistema, args.mostrar)
    elif args.usuario and args.imagem:
        figura_etapas(sistema, [(args.imagem, args.usuario)], "autenticacao.png", args.mostrar)
    else:
        ap.error("informe a identidade e a imagem: python main.py autenticar s3 caminho/da/imagem.png")


if __name__ == "__main__":
    main()
