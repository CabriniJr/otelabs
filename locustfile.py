"""
locustfile.py — gera exatamente TOTAL_REQUESTS requisições (200k por
padrão) contra UM par app+coletor por vez, com o mix de domínios
15/20/35/30% (pix/checkout/site_latency/api_generic) — os mesmos pesos
usados em scripts/generate_sample_data.py.

Uso (headless, apontando pro par "head-based deterministico", porta
8001 — ver docker-compose.yml e EXPERIMENTO.md pras outras portas):

    locust -f locustfile.py --host=http://localhost:8001 \
        --headless -u 100 -r 20 -t 30m

-u 100 -r 20: 100 usuários concorrentes, subindo 20 por segundo.
-t 30m: teto de segurança — o teste para sozinho bem antes disso, assim
que bater TOTAL_REQUESTS (ver o listener abaixo). Ajuste -u pra cima se
quiser terminar mais rápido (mais concorrência = mais throughput, já que
o demo-app é assíncrono).

Validado (testado neste projeto): com o listener abaixo, o teste some
sozinho ~1s depois de bater a meta, em vez de esperar o -t inteiro. É
normal passar um pouco de TOTAL_REQUESTS (a sobra fica ~= nº de usuários
concorrentes, -u) — as requisições já em voo quando a meta é atingida
ainda terminam antes do runner parar de vez.
"""
import itertools
import os
import threading

import gevent
from locust import HttpUser, task, between, events

# LOCUST_TOTAL_REQUESTS permite rodar um teste pequeno primeiro (ex.:
# LOCUST_TOTAL_REQUESTS=1000) pra validar a lógica de parada antes de
# subir pra 200k de verdade.
TOTAL_REQUESTS = int(os.environ.get("LOCUST_TOTAL_REQUESTS", 200_000))

_counter = itertools.count()
_lock = threading.Lock()
_stopped = False
_environment = None


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    global _environment
    _environment = environment


@events.request.add_listener
def contar_e_parar(request_type, name, response_time, response_length,
                    exception, context=None, **kwargs):
    """Roda a cada requisição concluída (sucesso ou falha). Assim que a
    contagem bater TOTAL_REQUESTS, para o teste — garante o MESMO total
    exato em toda execução, o que é essencial pra comparar os 3
    coletores de forma justa ("3 runs iguais")."""
    global _stopped
    with _lock:
        n = next(_counter)
        if n + 1 >= TOTAL_REQUESTS and not _stopped:
            _stopped = True
            if _environment is not None and _environment.runner is not None:
                # spawn_later(0, ...) em vez de chamar quit() direto: o
                # listener roda dentro do greenlet da própria requisição,
                # e runner.quit() tenta encerrar todos os greenlets de
                # usuário (incluindo esse) — chamado direto, ele trava
                # até o -t da CLI expirar. Agendar pra "logo em seguida"
                # evita esse deadlock.
                gevent.spawn_later(0, _environment.runner.quit)


class TrafegoApp(HttpUser):
    # sem pausa entre requisições -> carga contínua, throughput máximo
    # que a concorrência configurada (-u) permitir
    wait_time = between(0, 0)

    @task(15)
    def pix(self):
        self.client.post("/pix/transacao", name="pix")

    @task(20)
    def checkout(self):
        self.client.post("/checkout", name="checkout")

    @task(35)
    def site_latency(self):
        self.client.get("/", name="site_latency")

    @task(30)
    def api_generic(self):
        self.client.get("/api/generic", name="api_generic")
