# Intervalos de Confiança & Sampling em Coletores OTel

Portfólio para a disciplina de Estatística (FIAP) — análise de intervalos
de confiança aplicada a um experimento real de engenharia de plataforma /
observabilidade: 2 contextos de teste (head-based / tail-based), cada um
com 3 coletores OTel (determinístico, sweet spot, agressivo) rodando em
docker compose, comparando ganho de performance vs. perda de confiança
estatística por sinal (ex.: PIX vs. latência geral do site).

## Como rodar o app Streamlit

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Cada um dos 6 uploads na barra lateral (2 contextos x 3 coletores) é
opcional — o que não for carregado usa o CSV de exemplo correspondente em
`data/sample_runs/` (gerado por `scripts/generate_sample_data.py`), com
aviso na tela. Assim que cada coletor real tiver seu export pronto, é só
subir o arquivo no slot correspondente.

## Como rodar o experimento real (docker compose)

Ver **[EXPERIMENTO.md](EXPERIMENTO.md)** para o passo a passo completo
das 6 execuções (2 contextos x 3 coletores). Resumo da infraestrutura:

```
demo-app/            # app FastAPI instrumentada com OTel (1 span por request,
                      # tag business.domain, status ERROR nos erros simulados)
otel-config/*.yaml    # as 6 pipelines de coletor (receiver -> sampler -> file exporter)
docker-compose.yml    # 6 pares app+coletor, um por contexto x config
locustfile.py         # gera exatamente 200k requisições com o mix 15/20/35/30%
scripts/otel_export_to_csv.py  # converte o export do coletor pro CSV do app
```

A aba **Pipeline do experimento**, dentro do próprio app Streamlit, tem a
mesma tabela de portas/serviços e os comandos, pra referência rápida sem
precisar abrir este arquivo.

## Schema do CSV esperado (cada um dos 6 arquivos)

Uma linha por requisição/span que aquele coletor especificamente
exportou:

| coluna       | tipo                | descrição                                                                 |
|--------------|---------------------|----------------------------------------------------------------------------|
| `trace_id`   | string              | identificador único da requisição/trace                                   |
| `timestamp`  | ISO 8601 ou epoch   | instante da requisição                                                    |
| `domain`     | string              | tag/domínio do sinal (ex.: `pix`, `checkout`, `site_latency`, `api_generic`) |
| `latency_ms` | float               | latência da requisição em milissegundos                                   |
| `is_error`   | bool / 0-1          | se a requisição terminou em erro                                          |

O arquivo **determinístico** de cada contexto é o baseline (100%) usado
como referência para calcular taxa efetiva de retenção e MOE dos outros
dois coletores daquele mesmo contexto — carregue-o sempre que possível.

## Estrutura do projeto

```
app.py                          # app Streamlit (única interface)
utils/stats.py                  # IC (CLT, bootstrap, Wilson), score composto
utils/sampling.py               # políticas de sampling (usadas só pelo gerador de exemplo)
scripts/generate_sample_data.py # gera os 6 CSVs de EXEMPLO (fora do app)
scripts/otel_export_to_csv.py   # converte o export real do coletor pro CSV do app
data/sample_runs/*.csv          # os 6 datasets de exemplo gerados pelo script acima
data/real_runs/*.csv            # (você gera) os 6 CSVs reais, via EXPERIMENTO.md
demo-app/                       # app instrumentada com OTel, alvo do teste de carga
otel-config/*.yaml              # as 6 pipelines de coletor
docker-compose.yml              # sobe os pares app+coletor do experimento real
locustfile.py                   # gerador de carga (200k reqs, parada exata, mix por domínio)
.streamlit/config.toml          # tema (paleta validada para acessibilidade)
```

**O app nunca gera dados sozinho.** Ele só lê os CSVs que cada coletor
real exportou. O único gerador de dados do projeto é
`scripts/generate_sample_data.py`, que roda offline e serve só para testar
a interface antes do experimento real estar pronto.

## O que o app mostra

1. **Visão geral** — baseline determinístico de um contexto: latência
   média com IC por sinal, distribuição de latência, taxa de erro com IC
   de Wilson.
2. **Comparação dos coletores** — determinístico x sweet spot x agressivo,
   lado a lado para os 2 contextos, usando os dados reais exportados por
   cada um (MOE de latência, MOE de erro, redução de exportação).
3. **Trade-off & sweet spot** — curva teórica de MOE/score em função da
   taxa (extrapolada do σ do determinístico), com os 3 pontos reais
   medidos sobrepostos — mostra se o experimento bateu com a teoria.
4. **Head vs. Tail** — compara diretamente os dois contextos nos mesmos
   coletores (sweet spot / agressivo), pra ver onde tail-based preserva
   mais confiança que head-based na mesma taxa nominal.
5. **Sampling dinâmico** — sliders por domínio para explorar,
   hipoteticamente, uma 4ª config com taxa diferente por sinal.
6. **Metodologia** — fórmulas e justificativas estatísticas, para citar
   no relatório do trabalho.

Ver detalhes de cada fórmula na aba **Metodologia** dentro do próprio app.
