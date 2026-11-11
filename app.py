"""Servidor web (Flask) que integra o front-end ao SistemaBiometrico (autenticação facial 1:1).

Uso:  python app.py   ->   http://127.0.0.1:5000
Rotas:
  GET  /                      interface
  GET  /api/info              limiar, nº de usuários e de imagens de referência
  GET  /api/usuarios          lista de usuários (id, nome, nível)
  GET  /api/referencia/<id>   uma imagem de referência do usuário (miniatura)
  GET  /api/metricas          métricas da última avaliação (resultados/metricas.json)
  POST /api/avaliar           executa a avaliação (= python main.py avaliar + demo) e devolve as métricas
  GET  /api/resultados/<arq>  figuras geradas (distancias.png, figura_etapas.png)
  POST /api/autenticar        multipart: usuario=<id>, imagem=<arquivo>
  POST   /api/usuarios        multipart: nome, nivel (1-3), id (opcional), fotos=<3 a 10 arquivos>  -> cadastra
  PATCH  /api/usuarios/<id>   json: {"nivel": 1-3, "nome": "..."}                                  -> edita
  POST   /api/usuarios/<id>/fotos   multipart: fotos=<1 a 10 arquivos>  -> acrescenta referências a quem já existe
  DELETE /api/usuarios/<id>   remove o usuário e suas fotos de referência
Cadastrar/editar/remover recalibra o limiar. O servidor escuta só em 127.0.0.1 e não tem login: é um protótipo local.
"""
import base64
import contextlib
import glob
import io
import json
import os
import re
import shutil
import tempfile
import threading

import matplotlib
matplotlib.use("Agg")                                      # figuras sem janela (o servidor roda em outra thread)
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from entrada import cinza_de_bytes, ler_cinza, recortar_retrato
from pipeline import PASTA, EntradaInvalida, processar
from sistema import SistemaBiometrico, distancia_qui2

app = Flask(__name__, static_folder=os.path.join(PASTA, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024     # até 10 fotos por cadastro

lock = threading.RLock()                                   # cadastro e autenticação não podem rodar ao mesmo tempo
sistema = SistemaBiometrico()                              # cadastra e calibra o limiar uma única vez
print(f"{len(sistema.usuarios)} usuários cadastrados, {len(sistema.ids)} referências, limiar {sistema.limiar:.3f}")


def png_base64(img, caixa=None):
    """Imagem em cinza (ndarray) -> data URI PNG, ampliada para ficar nítida na tela. Desenha a caixa da ROI se houver."""
    if img is None:
        return None
    esc = 3
    img = cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_NEAREST)
    if caixa is not None:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x, y, l, a = (int(v) * esc for v in caixa)
        cv2.rectangle(img, (x, y), (x + l, y + a), (80, 255, 120), 3)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode() if ok else None


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/info")
def info():
    return jsonify({"limiar": sistema.limiar, "usuarios": len(sistema.usuarios), "referencias": int(len(sistema.ids))})


def lista_usuarios():
    return [{"id": u["id"], "nome": u["nome"], "nivel": u["nivel"], "referencias": int((sistema.ids == u["id"]).sum())}
            for u in sistema.usuarios.values()]


@app.get("/api/usuarios")
def usuarios():
    with lock:
        return jsonify(lista_usuarios())


def slug(texto):
    texto = re.sub(r"[^a-z0-9]+", "-", texto.lower().encode("ascii", "ignore").decode() or "usuario").strip("-")
    return texto[:20] or "usuario"


def validar_nivel(valor):
    try:
        nivel = int(valor)
    except (TypeError, ValueError):
        return None
    return nivel if nivel in (1, 2, 3) else None


@app.post("/api/usuarios")
def cadastrar_usuario():
    nome = (request.form.get("nome") or "").strip()[:60]
    nivel = validar_nivel(request.form.get("nivel"))
    uid = (request.form.get("id") or "").strip().lower() or slug(nome)
    fotos = request.files.getlist("fotos")
    if not nome or nivel is None:
        return jsonify({"erro": "informe o nome e um nível de acesso entre 1 e 3"}), 400
    if not re.fullmatch(r"[a-z0-9_-]{1,20}", uid):
        return jsonify({"erro": "id inválido: use 1 a 20 letras minúsculas, números, '-' ou '_'"}), 400
    if not 3 <= len(fotos) <= 10:
        return jsonify({"erro": "envie de 3 a 10 fotos do rosto (de frente, com variações de expressão)"}), 400

    with lock:
        if uid in sistema.usuarios:
            return jsonify({"erro": f"o id '{uid}' já está cadastrado"}), 409
        staging = os.path.join(PASTA, "base", "cadastro", f"_novo_{uid}")
        destino = os.path.join(PASTA, "base", "cadastro", uid)
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging)
        avisos = []
        try:
            vetores = []
            for i, foto in enumerate(fotos, 1):
                recorte, caixa = recortar_retrato(cinza_de_bytes(foto.read()))
                if recorte is None or caixa is None:
                    return jsonify({"erro": f"nenhum rosto detectado na foto {i} ({foto.filename})"}), 422
                caminho = os.path.join(staging, f"{i}.png")
                cv2.imwrite(caminho, cv2.resize(recorte, (184, 224), interpolation=cv2.INTER_AREA))
                try:
                    vetores.append(processar(caminho, **sistema.opcoes)["vetor"])
                except EntradaInvalida as erro:
                    return jsonify({"erro": f"foto {i} ({foto.filename}): {erro}"}), 422
            # rosto parecido com o de outro cadastrado: apenas AVISA (pessoas parecidas existem). Se for a mesma pessoa
            # cadastrada duas vezes, o limiar calibrado fica pior: remova um dos cadastros.
            for outro in sorted(set(sistema.ids)):
                dist = min(float(distancia_qui2(v, sistema.vetores[sistema.ids == outro]).min()) for v in vetores)
                if dist <= sistema.limiar:
                    avisos.append(f"o rosto lembra {sistema.usuarios[outro]['nome']} ({outro}) (distância {dist:.1f}); "
                                  f"se for a mesma pessoa, remova um dos cadastros")
            os.rename(staging, destino)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        sistema.usuarios[uid] = {"id": uid, "nome": nome, "nivel": nivel}
        sistema.salvar_usuarios()
        sistema.recarregar()
        return jsonify({"usuarios": lista_usuarios(), "limiar": sistema.limiar, "avisos": avisos}), 201


@app.post("/api/usuarios/<uid>/fotos")
def adicionar_fotos(uid):
    fotos = request.files.getlist("fotos")
    if not 1 <= len(fotos) <= 10:
        return jsonify({"erro": "envie de 1 a 10 fotos"}), 400
    with lock:
        if uid not in sistema.usuarios:
            return jsonify({"erro": "usuário não encontrado"}), 404
        pasta = os.path.join(PASTA, "base", "cadastro", uid)
        existentes = glob.glob(os.path.join(pasta, "*.png"))
        if len(existentes) + len(fotos) > 20:
            return jsonify({"erro": f"limite de 20 referências por pessoa (já tem {len(existentes)})"}), 400
        staging = os.path.join(PASTA, "base", "cadastro", f"_novo_{uid}")
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging)
        avisos = []
        try:
            for i, foto in enumerate(fotos, 1):
                recorte, caixa = recortar_retrato(cinza_de_bytes(foto.read()))
                if recorte is None or caixa is None:
                    return jsonify({"erro": f"nenhum rosto detectado na foto {i} ({foto.filename})"}), 422
                caminho = os.path.join(staging, f"{i}.png")
                cv2.imwrite(caminho, cv2.resize(recorte, (184, 224), interpolation=cv2.INTER_AREA))
                try:
                    vetor = processar(caminho, **sistema.opcoes)["vetor"]
                except EntradaInvalida as erro:
                    return jsonify({"erro": f"foto {i} ({foto.filename}): {erro}"}), 422
                # foto mais parecida com OUTRA pessoa cadastrada do que com esta: só avisa (pode ser foto trocada)
                outros = [(sistema.distancia(vetor, o), o) for o in sorted(set(sistema.ids)) if o != uid]
                if outros and min(outros)[0] < sistema.distancia(vetor, uid):
                    avisos.append(f"a foto {i} ({foto.filename}) se parece mais com {sistema.usuarios[min(outros)[1]]['nome']} "
                                  f"do que com esta pessoa; confira se não é outra pessoa")
            proximo = max([int(os.path.splitext(os.path.basename(f))[0]) for f in existentes
                           if os.path.splitext(os.path.basename(f))[0].isdigit()], default=0) + 1
            for k, arq in enumerate(sorted(os.listdir(staging), key=lambda n: int(n.split(".")[0]))):
                os.rename(os.path.join(staging, arq), os.path.join(pasta, f"{proximo + k}.png"))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        sistema.recarregar()
        return jsonify({"usuarios": lista_usuarios(), "limiar": sistema.limiar, "avisos": avisos})


@app.patch("/api/usuarios/<uid>")
def editar_usuario(uid):
    dados = request.get_json(silent=True) or {}
    with lock:
        if uid not in sistema.usuarios:
            return jsonify({"erro": "usuário não encontrado"}), 404
        if "nivel" in dados:
            nivel = validar_nivel(dados["nivel"])
            if nivel is None:
                return jsonify({"erro": "nível deve ser 1, 2 ou 3"}), 400
            sistema.usuarios[uid]["nivel"] = nivel
        if str(dados.get("nome", "")).strip():
            sistema.usuarios[uid]["nome"] = str(dados["nome"]).strip()[:60]
        sistema.salvar_usuarios()                           # nível/nome não mudam as imagens: sem recalibrar
        return jsonify({"usuarios": lista_usuarios(), "limiar": sistema.limiar})


@app.delete("/api/usuarios/<uid>")
def remover_usuario(uid):
    with lock:
        if uid not in sistema.usuarios:
            return jsonify({"erro": "usuário não encontrado"}), 404
        if len(sistema.usuarios) == 1:
            return jsonify({"erro": "mantenha ao menos um usuário cadastrado"}), 409
        del sistema.usuarios[uid]
        sistema.salvar_usuarios()
        shutil.rmtree(os.path.join(PASTA, "base", "cadastro", uid), ignore_errors=True)
        sistema.recarregar()
        return jsonify({"usuarios": lista_usuarios(), "limiar": sistema.limiar})


@app.get("/api/referencia/<uid>")
def referencia(uid):
    if uid not in sistema.usuarios:                          # valida o id (evita path traversal)
        return jsonify({"erro": "usuário não encontrado"}), 404
    arquivos = sorted(glob.glob(os.path.join(PASTA, "base", "cadastro", uid, "*.png")))
    if not arquivos:
        return jsonify({"erro": "sem referências"}), 404
    return send_from_directory(os.path.dirname(arquivos[0]), os.path.basename(arquivos[0]))


@app.get("/api/metricas")
def metricas():
    caminho = os.path.join(PASTA, "resultados", "metricas.json")
    if not os.path.isfile(caminho):
        return jsonify({"erro": "ainda não há avaliação: clique em \"Executar avaliação\" (ou rode python main.py avaliar)"}), 404
    with open(caminho, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.post("/api/avaliar")
def avaliar():
    import main as principal                               # reaproveita a avaliação da linha de comando
    with lock:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                principal.avaliar(sistema, False)
                principal.figura_etapas(sistema, principal.exemplos(sistema), "figura_etapas.png", False)
        except Exception as erro:
            return jsonify({"erro": f"falha ao avaliar: {erro}"}), 500
    return metricas()


@app.get("/api/resultados/<nome>")
def resultado(nome):
    if nome not in ("distancias.png", "figura_etapas.png"):
        return jsonify({"erro": "arquivo não encontrado"}), 404
    return send_from_directory(os.path.join(PASTA, "resultados"), nome, max_age=0)


@app.post("/api/autenticar")
def autenticar():
    uid = (request.form.get("usuario") or "").strip()
    arquivo = request.files.get("imagem")
    if not uid or arquivo is None:
        return jsonify({"erro": "informe 'usuario' e 'imagem'"}), 400

    fd, tmp = tempfile.mkstemp(suffix=".png")               # o pipeline lê de um caminho em disco
    os.close(fd)
    try:
        arquivo.save(tmp)
        recorte, _ = recortar_retrato(ler_cinza(tmp))       # enquadra o rosto da selfie (92:112, escala ORL)
        if recorte is not None:
            cv2.imwrite(tmp, recorte)
        with lock:
            res = sistema.autenticar(tmp, uid)
    finally:
        os.remove(tmp)
    if res["status"] == "invalida":                          # não expõe o caminho temporário ao cliente
        res["motivo"] = res["motivo"].replace(tmp, "imagem enviada")

    with lock:
        return montar_resposta(res, uid)


def montar_resposta(res, uid):
    etapas = res.get("etapas") or {}
    caixa = etapas.get("caixa")
    resposta = {
        "status": res["status"],                              # autenticado | rejeitado | invalida
        "mensagem": sistema.mensagem(res),
        "limiar": sistema.limiar,
        "etapas": {
            "original": png_base64(etapas.get("original")),
            "pre": png_base64(etapas.get("pre")),
            "deteccao": png_base64(etapas.get("pre"), caixa) if caixa is not None else None,
            "roi": png_base64(etapas.get("roi")),
        },
    }
    if res["status"] == "invalida":
        resposta["motivo"] = res["motivo"]
    else:
        usuario = sistema.usuarios[uid]
        resposta.update({"distancia": res["distancia"], "usuario": {"id": uid, "nome": usuario["nome"], "nivel": usuario["nivel"]}})
        if res["status"] == "autenticado":
            resposta["acesso"] = [{"nivel": n, "tipo": sistema.info[str(n)]["tipo"], "exemplo": sistema.info[str(n)]["exemplo"]}
                                  for n in range(1, usuario["nivel"] + 1)]
    return jsonify(resposta)


@app.errorhandler(413)
def grande_demais(_):
    return jsonify({"erro": "arquivos grandes demais (limite de 40 MB por envio)"}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
