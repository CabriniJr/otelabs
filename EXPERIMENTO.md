# Rodando o experimento real (docker compose)

Passo a passo das 6 execuções — 2 contextos (head-based, tail-based) x 3
coletores (determinístico, sweet spot, agressivo). Cada execução é
independente: sobe só o par app+coletor daquela vez, gera a carga, para
tudo, converte o resultado, e segue pra próxima.

## Pré-requisitos

- Docker + Docker Compose instalados.
- Python local com `locust` (`pip install locust`) — o Locust roda no
  seu host, fora do compose, apontando pra porta exposta do app.

## 0. Teste pequeno primeiro (recomendado)

Valide a lógica de geração/parada antes de gastar tempo com 200k
requisições de verdade:

```bash
docker compose up --build otel-collector-head-based-deterministico app-head-based-deterministico
# em outro terminal:
LOCUST_TOTAL_REQUESTS=1000 locust -f locustfile.py --host=http://localhost:8001 \
    --headless -u 50 -r 50 -t 5m
```

Confira em `out/head_based/deterministico/traces.json` que o arquivo foi
criado e tem conteúdo. Se estiver ok, parta pro real (`Ctrl+C` nos dois
terminais, `docker compose down`).

## 1. As 6 execuções

Repita este bloco pra cada uma das 6 combinações (troque só o nome do
serviço e a porta — ver tabela abaixo):

```bash
# 1) sobe o par app+coletor dessa combinação
docker compose up --build -d otel-collector-<contexto>-<config> app-<contexto>-<config>

# 2) gera a carga real (200k requisições, mix 15/20/35/30%)
locust -f locustfile.py --host=http://localhost:<porta> \
    --headless -u 150 -r 30 -t 30m

# 3) espera uns segundos pro coletor esvaziar buffers (batch/decision_wait)
sleep 10

# 4) derruba só esse par (sem apagar o volume de saída)
docker compose stop otel-collector-<contexto>-<config> app-<contexto>-<config>
```

| contexto     | config          | serviço coletor                              | serviço app                          | porta |
|--------------|-----------------|-----------------------------------------------|----------------------------------------|-------|
| head_based   | deterministico  | otel-collector-head-based-deterministico       | app-head-based-deterministico          | 8001  |
| head_based   | sweet_spot      | otel-collector-head-based-sweet-spot           | app-head-based-sweet-spot              | 8002  |
| head_based   | agressivo       | otel-collector-head-based-agressivo            | app-head-based-agressivo               | 8003  |
| tail_based   | deterministico  | otel-collector-tail-based-deterministico       | app-tail-based-deterministico          | 8004  |
| tail_based   | sweet_spot      | otel-collector-tail-based-sweet-spot           | app-tail-based-sweet-spot              | 8005  |
| tail_based   | agressivo       | otel-collector-tail-based-agressivo            | app-tail-based-agressivo               | 8006  |

**Por que subir e derrubar um par por vez:** os 3 coletores de um mesmo
contexto competem pelo mesmo perfil de CPU/rede do seu laptop se
rodarem juntos, o que enviesaria a latência medida. Rodar sequencialmente
mantém as 3 execuções o mais comparáveis possível — e é justamente essa
variância run-a-run (mesmo rodando sequencial) que vira material de
discussão no relatório.

Com `-u 150`, o `demo-app` assíncrono deve sustentar bem mais de 500
req/s (latências médias de 90-380ms), então 200k requisições tendem a
terminar em poucos minutos — o `-t 30m` é só o teto de segurança, o
Locust para sozinho antes disso.

## 2. Convertendo os 6 exports pra CSV

Depois das 6 execuções (os arquivos ficam em `out/<contexto>/<config>/traces.json`):

```bash
python scripts/otel_export_to_csv.py --all
```

Isso gera os 6 CSVs em `data/real_runs/`, já no schema que o app
Streamlit espera. Suba cada um no slot correspondente na barra lateral do
app (ou aponte o app pra ler direto de `data/real_runs/` — ver
README.md).

## 3. Limpando entre tentativas

Se precisar refazer uma execução (ex.: algo deu errado no meio):

```bash
docker compose down -v  # cuidado: apaga containers e (com -v) volumes anônimos
rm -rf out/<contexto>/<config>/*
```

Os volumes de saída são bind mounts em `./out/`, então os arquivos
`traces.json` persistem no seu disco mesmo depois de `docker compose down`
(sem `-v`) — só apague manualmente se quiser recomeçar aquela combinação
do zero.

## 4. Coisas que valem virar nota no relatório

- **Overshoot do Locust**: cada execução deve fechar em ~200.000 +
  `-u` requisições (as que já estavam em voo quando a meta foi
  atingida) — não exatos 200.000. É esperado e documentado em
  `locustfile.py`.
- **Threshold de latência único no tail-based** (`400ms` pra todos os
  domínios, em `otel-config/tail_based/*.yaml`): pix, checkout e
  site_latency têm distribuições de latência bem diferentes — um
  threshold por domínio (via política `and` combinando `latency` +
  `string_attribute` no `tail_sampling`) seria mais correto. Fica como
  próximo passo.
- **`decision_wait: 2s`** no tail_sampling: como cada request vira 1
  trace de 1 span só (sem chamadas encadeadas), a decisão é praticamente
  imediata — não precisa de uma janela grande.
