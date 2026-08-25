# Handoff para a sua máquina (T14) — instruções para o Claude Code local

Este projeto foi montado e testado numa sandbox na nuvem, mas essa sandbox
**não tem Docker Hub liberado** (só GitHub Releases), então o teste real
rodou de forma "nativa" (binário do collector + uvicorn direto, sem
container). No seu T14 (i5 11ª gen, 16GB RAM) você tem Docker de verdade —
então dá pra rodar o `docker-compose.yml` como ele foi desenhado.

Cole este arquivo inteiro como prompt pro Claude Code local, ou siga os
passos manualmente. Tudo abaixo assume que você já descompactou o zip e
está com o terminal aberto dentro da pasta do projeto (`otel-sampling-ic/`).

## 0. Contexto rápido (pra você e pro Claude Code local)

- **O que já existe e está pronto:** app Streamlit completo (portfólio CP1
  + laboratório de sampling), 6 pipelines de coletor OTel (head-based e
  tail-based × determinístico/sweet-spot/agressivo), demo-app FastAPI
  instrumentada, `locustfile.py` com parada exata por contagem,
  `docker-compose.yml`, conversor OTLP→CSV, **6 CSVs reais** já gerados em
  `data/real_runs/` (rodados com 20k requisições cada, na sandbox, sem
  Docker), e a apresentação coringa (`coringa/apresentacao_coringa.pptx`)
  já validada e com o bug de bullets corrigido.
- **O que falta e é trabalho seu/do Claude Code local:** (a) dar push pro
  GitHub usando SEU auth local (a sandbox não conseguiu autenticar contra
  `github.com`, só contra a API REST — não vale a pena insistir nisso, é
  limitação do ambiente dela, não do repo); (b) opcionalmente rodar o
  teste completo com os 200k requests via `docker-compose.yml` real, já
  que aqui você tem Docker de verdade; (c) deploy no Streamlit Community
  Cloud; (d) revisão final de conteúdo antes de entregar (nomes, datas,
  cargo — o dashboard tem um aviso pedindo essa revisão).
- **Prazo:** entrega é HOJE, na aula da turma 2ESPH. Priorize nesta ordem:
  push → deploy → revisão de conteúdo → (se sobrar tempo) reteste completo.

## 1. Ambiente

```bash
cd otel-sampling-ic
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Teste local rápido antes de mexer em qualquer coisa:

```bash
streamlit run app.py
```

Abra `http://localhost:8501` e confira as 4 abas do topo (Quem sou eu,
Minhas Qualificações, Skills, Análise de Dados) e, dentro de Análise de
Dados, as 7 sub-abas. Os dados reais de `data/real_runs/` devem carregar
automaticamente (sem precisar fazer upload de nada) — a barra lateral vai
mostrar "teste real (local, reduzido)" em cada um dos 6 slots.

## 2. Push pro GitHub

O repositório git local já está pronto: `git init` feito, 2 commits já
aplicados (`0fd6b31` e `ddaf399`), remote `origin` já apontando para
`https://github.com/CabriniJr/otelabs.git`. Só falta autenticar com o
SEU PAT/gh local e empurrar:

```bash
git remote -v   # confirme que já é https://github.com/CabriniJr/otelabs.git
git log --oneline   # confirme os 2 commits

# se você já tem `gh auth login` feito nessa máquina:
git push -u origin master

# se preferir PAT direto na URL (não fica salvo em lugar nenhum além do
# git credential helper que você já usa):
git push https://SEU_USUARIO:SEU_PAT@github.com/CabriniJr/otelabs.git master
```

Se o repo remoto já tiver algum conteúdo (README criado pelo GitHub, por
exemplo) e o push for rejeitado, rode `git pull --rebase origin master`
primeiro e resolva qualquer conflito trivial (provavelmente só no
README.md).

## 3. (Opcional, se der tempo) Reteste completo com Docker real

Aqui, diferente da sandbox, o `docker-compose.yml` funciona direto —
não precisa da gambiarra do `scripts/run_local_pair.sh` (que só existe
por causa do bloqueio de registry na sandbox).

```bash
docker compose build
docker compose up -d
```

Isso sobe os 6 pares app+coletor (portas documentadas no `EXPERIMENTO.md`
e na aba "Pipeline do experimento" do próprio app). Depois, para cada
combinação, rode o Locust apontando pra porta certa:

```bash
LOCUST_TOTAL_REQUESTS=200000 locust -f locustfile.py \
  --host=http://localhost:8001 \
  --headless -u 200 -r 50 -t 30m
```

(repita para as 6 portas — 8001 a 8006 — trocando o host; ajuste
`-u`/`-r` conforme a CPU aguentar, 200 usuários simultâneos deve ser
tranquilo pro i5 11ª gen). Depois de cada execução, extraia o export do
volume do coletor e rode:

```bash
python scripts/otel_export_to_csv.py --all
```

Isso sobrescreve `data/real_runs/*.csv` com os 200k reais. **Não é
obrigatório** — os CSVs de 20k que já estão no repo já são dados reais e
válidos para a análise estatística (a única diferença é o tamanho da
amostra, que inclusive é interessante mencionar na aba Metodologia: MOE
menor com mais dados). Só vale a pena se sobrar tempo antes da entrega.

Depois de terminar, `docker compose down` para liberar os 16GB de RAM.

## 4. Deploy no Streamlit Community Cloud

1. Confirme que o push do passo 2 já foi feito (o Streamlit Cloud puxa
   direto do GitHub).
2. Acesse https://share.streamlit.io, conecte a conta GitHub
   (CabriniJr), clique em "New app".
3. Repositório: `CabriniJr/otelabs`, branch `master`, main file path:
   `app.py`.
4. Deploy. A primeira build demora ~2-3 min (instala as libs do
   `requirements.txt`).
5. Copie a URL pública gerada e:
   - Cole no slide 8 (Conclusão) da apresentação coringa, no lugar de
     `[preencher após deploy no Streamlit Community Cloud]` — o texto
     está em `coringa/build_deck.js`, procure por essa string, edite, e
     rode `node coringa/build_deck.js` de novo (dentro da raiz do
     projeto, não dentro de `coringa/`) para regerar o `.pptx`.
   - Guarde para colar no formulário/local de entrega do CP1 junto com o
     link do repositório e o zip do código-fonte.

## 5. Revisão final de conteúdo (antes de entregar)

O app tem um aviso na aba "Quem sou eu":
> "Rascunho gerado com o Claude a partir do que já conversamos — revise
> nomes, datas e detalhes antes de entregar."

Isso é sério — releia as abas "Quem sou eu", "Minhas Qualificações" e
"Skills" e confira: cargo atual, nome da empresa anterior (OptDriven),
formação (FIAP — curso certo?), nível de inglês, qualquer detalhe que eu
possa ter generalizado ou errado. É rápido mas importante — é conteúdo
que vai ser avaliado como currículo/perfil profissional seu.

## 6. Checklist final antes de submeter

```bash
python3 -m py_compile app.py   # garante que não quebrou nada
```

- [ ] Push feito, repo público (ou visível pro professor) em
      `github.com/CabriniJr/otelabs`
- [ ] App no ar no Streamlit Community Cloud, testado num navegador
      anônimo (sem estar logado) para confirmar que carrega sem pedir
      nada
- [ ] Conteúdo das abas de portfólio revisado (passo 5)
- [ ] `coringa/apresentacao_coringa.pptx` com o link do dashboard
      preenchido (passo 4) — leve num pendrive ou já aberto, como plano B
      se o link cair na hora da correção
- [ ] Zip do código-fonte pronto para anexar na entrega (pode usar
      `git archive` para gerar limpo, sem `.git/`):

```bash
git archive --format=zip -o otelabs_codigo_fonte.zip HEAD
```

Isso gera um zip só com os arquivos versionados (sem histórico do git,
sem os CSVs de exemplo desnecessários se você quiser cortar mais —
mas por padrão inclui tudo que está no repo, que é o esperado).

## Referência rápida — estrutura do que você recebeu

```
app.py                    # Streamlit — abas do portfólio + laboratório
requirements.txt          # deps do Streamlit
demo-app/                 # FastAPI instrumentada com OTel (alvo do teste)
otel-config/*/*.yaml       # 6 pipelines de coletor (head/tail × 3 configs)
docker-compose.yml         # sobe os 6 pares app+coletor com Docker real
locustfile.py               # gerador de carga, para exata por contagem
scripts/otel_export_to_csv.py   # converte export do coletor pra CSV
scripts/run_local_pair.sh   # só usado na sandbox (sem Docker) — ignore aqui
data/real_runs/*.csv        # 6 CSVs REAIS já gerados (20k reqs/coletor)
data/sample_runs/*.csv      # fallback sintético, só se algum real faltar
coringa/                    # apresentação de backup (pptx) + script gerador
EXPERIMENTO.md              # runbook detalhado das 6 execuções
README.md                   # visão geral do projeto
```

Qualquer dúvida sobre o *porquê* das escolhas (topologia de 3 coletores
separados, fórmula do score composto, por que tail-based retém mais que
a taxa nominal), a aba "Metodologia" dentro do próprio app documenta tudo
— é a mesma explicação que você pode usar se o professor perguntar na
correção.
