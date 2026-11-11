"""Cadastro de usuários fictícios, autenticação biométrica 1:1 e camada demonstrativa de controle de acesso."""
import glob
import json
import os

import numpy as np

from pipeline import PASTA, EntradaInvalida, processar

# Multiplica o limiar calibrado. 1.0 = limiar puro; acima de 1 fica MAIS FÁCIL de passar (menos falsas rejeições,
# mais falsas aceitações). Depois de mudar, rode `python main.py avaliar` e confira o FAR.
FOLGA_LIMIAR = 1.03


def carregar_json(nome):
    with open(os.path.join(PASTA, "dados", nome), encoding="utf-8") as arq:
        return json.load(arq)


def distancia_qui2(a, modelos):
    """Distância qui-quadrado entre um histograma e cada linha de 'modelos' (adequada a histogramas)."""
    return ((a - modelos) ** 2 / (a + modelos + 1e-10)).sum(axis=1)


class SistemaBiometrico:
    def __init__(self, **opcoes):
        self.opcoes = opcoes                       # ex.: filtro=False, clahe=False (para experimentos)
        self.info = carregar_json("informacoes_ambientais.json")
        self.vetores_impostores = self.carregar_impostores()      # pool fixo: calcula uma vez só
        self.recarregar()

    def recarregar(self):
        """Relê usuarios.json e base/cadastro/ e recalibra o limiar (chamado após cadastrar/remover/editar)."""
        self.usuarios = {u["id"]: u for u in carregar_json("usuarios.json")["usuarios"]}
        self.cadastrar()
        self.limiar = self.calibrar_limiar()

    def salvar_usuarios(self):
        with open(os.path.join(PASTA, "dados", "usuarios.json"), "w", encoding="utf-8") as arq:
            json.dump({"usuarios": list(self.usuarios.values())}, arq, ensure_ascii=False, indent=2)

    def cadastrar(self):
        """Carrega as imagens de referência de cada usuário e guarda seus vetores de características."""
        ids, vetores = [], []
        for uid in self.usuarios:
            for arq in sorted(glob.glob(os.path.join(PASTA, "base", "cadastro", uid, "*.png"))):
                try:
                    vetores.append(processar(arq, **self.opcoes)["vetor"])
                    ids.append(uid)
                except EntradaInvalida as erro:
                    print(f"[aviso] referência ignorada ({arq}): {erro}")
        self.ids, self.vetores = np.array(ids), np.array(vetores)

    def carregar_impostores(self):
        """Pessoas NÃO cadastradas (base/impostores/*/*.png), usadas só para calibrar o limiar. Com um único
        usuário cadastrado não há outros cadastrados para servir de impostor."""
        vetores = []
        for arq in sorted(glob.glob(os.path.join(PASTA, "base", "impostores", "*", "*.png"))):
            try:
                vetores.append(processar(arq, **self.opcoes)["vetor"])
            except EntradaInvalida:
                pass
        return np.array(vetores)

    def distancia(self, vetor, uid, excluir=None):
        """Autenticação 1:1: menor distância entre o vetor e as referências SÓ do usuário 'uid'.
        'excluir' remove uma referência (usado na calibração, para não comparar a imagem com ela mesma)."""
        mascara = self.ids == uid
        if excluir is not None:
            mascara[excluir] = False
        if not mascara.any():                      # sem referências (ou só havia a excluída)
            return float("inf")
        return float(distancia_qui2(vetor, self.vetores[mascara]).min())

    def calibrar_limiar(self):
        """Escolhe o limiar só com o cadastro (leave-one-out), sem usar imagens de teste: o valor que
        minimiza FRR + FAR. Genuíno: referência x demais referências do MESMO usuário; impostor:
        referência x referências de cada um dos OUTROS usuários + imagens de pessoas não cadastradas."""
        genuinos, impostores = [], []
        for i, (uid, vetor) in enumerate(zip(self.ids, self.vetores)):
            genuinos.append(self.distancia(vetor, uid, excluir=i))
            impostores += [self.distancia(vetor, outro) for outro in sorted(set(self.ids)) if outro != uid]
        for vetor in self.vetores_impostores:                # não cadastrados tentando entrar como cada usuário
            impostores += [self.distancia(vetor, uid) for uid in sorted(set(self.ids))]
        genuinos = np.array([d for d in genuinos if np.isfinite(d)])      # usuário com 1 referência não tem par genuíno
        impostores = np.array([d for d in impostores if np.isfinite(d)])
        if genuinos.size == 0 or impostores.size == 0:
            return 10.0 * FOLGA_LIMIAR             # poucos dados para calibrar: valor padrão
        candidatos = np.unique(np.concatenate([genuinos, impostores]))
        custo = np.array([(genuinos > t).mean() + (impostores <= t).mean() for t in candidatos])
        otimos = np.flatnonzero(custo == custo.min())
        inferior = candidatos[otimos].max()                           # último limiar ótimo
        seguinte = candidatos[otimos.max() + 1] if otimos.max() + 1 < len(candidatos) else inferior
        limiar = float((inferior + seguinte) / 2) if otimos.size == 1 else float(np.median(candidatos[otimos]))  # meio do intervalo ótimo
        return limiar * FOLGA_LIMIAR

    def autenticar(self, caminho, alegado):
        """Autenticação 1:1: o usuário diz quem é ('alegado') e apresenta uma imagem.
        Retorna dicionário com status (autenticado / rejeitado / invalida), distância e etapas."""
        if alegado not in self.usuarios:
            return {"status": "invalida", "motivo": f"identidade '{alegado}' não cadastrada", "etapas": None}
        if not (self.ids == alegado).any():
            return {"status": "invalida", "motivo": f"'{alegado}' não tem imagens de referência válidas", "etapas": None}
        try:
            etapas = processar(caminho, **self.opcoes)
        except EntradaInvalida as erro:
            return {"status": "invalida", "motivo": str(erro), "etapas": getattr(erro, "etapas", None)}
        dist = self.distancia(etapas["vetor"], alegado)
        status = "autenticado" if dist <= self.limiar else "rejeitado"
        return {"status": status, "id": alegado, "distancia": dist, "etapas": etapas}

    def mensagem(self, res):
        """Mensagem de acesso. Regra demonstrativa: o nível N consulta as informações dos níveis 1..N."""
        if res["status"] == "invalida":
            return f"ACESSO NEGADO - entrada inválida ({res['motivo']})."
        usuario = self.usuarios[res["id"]]
        if res["status"] == "rejeitado":
            return (f"ACESSO NEGADO - a imagem não corresponde à identidade alegada: {usuario['nome']} "
                    f"({res['id']}) (distância {res['distancia']:.3f} > limiar {self.limiar:.3f}).")
        nivel = usuario["nivel"]
        liberado = "; ".join(f"nível {n}: {self.info[str(n)]['tipo']}" for n in range(1, nivel + 1))
        return (f"ACESSO LIBERADO - {usuario['nome']} ({res['id']}) autenticado, nível {nivel}. "
                f"Pode consultar -> {liberado}. Exemplo: {self.info[str(nivel)]['exemplo']}")
