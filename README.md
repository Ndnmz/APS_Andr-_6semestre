# APS PDI/VC 2026/2 – Autenticação biométrica facial (1:1)

Protótipo local e acadêmico: o usuário informa quem diz ser e apresenta uma imagem do rosto; o sistema compara **somente**
com as referências daquela identidade e libera (ou nega) o nível de acesso 1, 2 ou 3 às informações ambientais **fictícias**.
Tudo roda no computador local. Não é um sistema de segurança.

## Como executar

```
pip install -r requirements.txt     # use OpenCV 4.x (o opencv-python 5.x não inclui o CascadeClassifier)
python main.py demo                 # resultados/figura_etapas.png (original, pré-processada, detecção, ROI) + mensagens de acesso
python main.py avaliar              # tabela de métricas, FAR/FRR, resultados/distancias.png, CSV e JSON
python main.py autenticar luiza foto.jpg     # "sou a ____ e esta é minha imagem"
python experimento_ablacao.py       # efeito do filtro e do CLAHE (apoia a discussão técnica)
python app.py                       # interface web local (extra): http://127.0.0.1:5000
```
`--mostrar` abre as figuras na tela, além de salvá-las em `resultados/`.

## Exigencias da APS

| Exigência da APS | Onde |
|---|---|
| 1. Aquisição e preparação (leitura, cinza, redimensionamento) | `pipeline.adquirir` |
| 2. Pré-processamento (≥ 2 técnicas): filtro Gaussiano + CLAHE | `pipeline.preprocessar` |
| 3. Detecção da ROI: Haar Cascade, recorte 96x96 da maior face | `pipeline.detectar_face` |
| 4. Características e decisão: LBP uniforme (grade 8x8) + distância qui-quadrado; aceita se `distância <= limiar` | `pipeline.extrair_caracteristicas`, `sistema.autenticar` |
| Cadastro de usuários e nível de acesso (1 a 3) | `dados/usuarios.json`, `base/cadastro/<id>/`, aba **Cadastros** do `app.py` |
| Mensagem com nível e tipo de informação liberada; acesso negado / entrada inválida | `sistema.mensagem`, `dados/informacoes_ambientais.json` |
| Visualização das etapas (≥ 4 imagens de exemplo) | `python main.py demo` |
| 3 cenários de teste, tabela de métricas, FAR e FRR | `python main.py avaliar` |
| Discussão técnica (efeito das técnicas de pré-processamento) | `python experimento_ablacao.py` |
| Extras | `app.py` + `static/index.html` (interface web local e câmera), `entrada.py` (enquadra o rosto de selfies) |

O **limiar** é calibrado só com as imagens de cadastro (leave-one-out, genuíno x impostor), minimizando FRR + FAR; nenhuma
imagem de teste é usada para escolhê-lo. `FOLGA_LIMIAR` (início de `sistema.py`) multiplica esse limiar: valores
acima de 1 deixam a autenticação mais fácil, ao custo de mais falsas aceitações (confira o FAR com `python main.py avaliar`).

## Dados

| Pasta | Conteúdo |
|---|---|
| `base/cadastro/<id>/` | referências dos 4 cadastrados: maria (nível 3), luiza (nível 2), otavio (nível 1), leandro-sampaio (nível 2) |
| `base/teste/genuinos/<id>_<n>.jfif` | fotos dos cadastrados **não** usadas no cadastro (tentativa correta; também alegando a identidade de outro cadastrado) |
| `base/teste/conhecidos`, `desconhecidos` | imagens ORL de pessoas **não** cadastradas (tentativas de impostor) |
| `base/teste/invalidas` | entradas sem rosto (ruído, gradiente etc., sintéticas) e um arquivo corrompido |
| `base/impostores/` | imagens ORL de pessoas não cadastradas, usadas **só** para calibrar o limiar |
| `dados/` | usuários **fictícios** com nível de acesso e textos ambientais **fictícios** |

As fotos dos cadastrados são imagens próprias, usadas **com consentimento**. As imagens ORL vêm da base pública AT&T/ORL
(Samaria e Harter, 1994): cite-a na bibliografia. As fotos dos cadastrados são selfies inteiras; `entrada.py` localiza o rosto
e recorta uma janela 92:112 antes do pipeline (`base/impostores` e as pastas ORL de teste já estão nesse formato).

## Interface web (extra)

`python app.py`: autenticar (arquivo, arrastar ou câmera), ver distância x limiar e as 4 etapas, cadastrar/remover pessoas e
acrescentar fotos de referência a quem já está cadastrado (botão **+ Fotos**), trocar o nível (recalibra o limiar) e ver as métricas. Rosto parecido com o de outro cadastrado gera só um **aviso** (não bloqueia o cadastro); a aba **Métricas** tem o botão *Executar avaliação*, que gera a tabela e as figuras sem terminal. Sem login, só em
127.0.0.1: uso local.

## Limitações

- Base pequena (4 cadastrados, 7 a 9 referências cada), poses frontais, fundo claro e uniforme; pouquíssimas fotos de teste
  por pessoa, então FAR e FRR têm incerteza grande.
- O LBP é sensível a boné, sombra e óculos; fotos de teste mais diferentes das de cadastro aumentam o FRR.
- A imagem é redimensionada para 92x112 sem manter a proporção; o rosto deve estar de frente e em boa luz.
- As imagens ORL de `base/impostores` (calibração) e de `base/teste/conhecidos` são das mesmas pessoas, em fotos diferentes: o FAR
  medido com elas é um pouco otimista. Maria e Leandro ainda não têm fotos em `base/teste/genuinos/`, então o FRR só cobre Luiza e Otávio.
- `FOLGA_LIMIAR` > 1 reduz falsas rejeições, mas o FAR sobe rápido quando o limiar passa da menor distância entre duas pessoas cadastradas.
- As métricas **não estão neste arquivo de propósito**: rode `python main.py avaliar` no seu computador e use os valores gerados.

Samaria, F. S.; Harter, A. C. *Parameterisation of a stochastic model for human face identification.* Proc. 2nd IEEE
Workshop on Applications of Computer Vision, Sarasota, 1994.
