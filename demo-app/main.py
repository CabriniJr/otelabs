"""
demo-app — aplicação de demonstração instrumentada com OpenTelemetry.

Expõe 4 endpoints, um por domínio/sinal, simulando latência e taxa de
erro realistas (os MESMOS parâmetros usados em
scripts/generate_sample_data.py, de propósito — assim dá pra comparar a
teoria/exemplo com o experimento real). Cada requisição vira 1 span, com
o atributo `business.domain` marcando o sinal e o status do span
marcando erro — é isso que o coletor usa pra decidir o sampling, e é isso
que scripts/otel_export_to_csv.py lê de volta pra gerar o CSV do app
Streamlit.

Rodar localmente (fora do compose), por exemplo:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
    OTEL_SERVICE_NAME=demo-app-local \
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
import asyncio
import os
import random

from fastapi import FastAPI, Response
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "demo-app")

# --- Setup OpenTelemetry (manual, sem auto-instrumentação — mais
# explícito pra fins didáticos) ------------------------------------------
provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
exporter = OTLPSpanExporter(endpoint=f"{OTLP_ENDPOINT.rstrip('/')}/v1/traces")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("demo-app")

# --- Perfis por domínio — IGUAIS aos de scripts/generate_sample_data.py -
import math

DOMAIN_PROFILES = {
    "pix":          dict(mu=math.log(120), sigma=0.35, error_rate=0.005),
    "checkout":     dict(mu=math.log(300), sigma=0.45, error_rate=0.015),
    "site_latency": dict(mu=math.log(180), sigma=0.55, error_rate=0.008),
    "api_generic":  dict(mu=math.log(90),  sigma=0.50, error_rate=0.020),
}

app = FastAPI()


async def handle(domain: str, response: Response):
    prof = DOMAIN_PROFILES[domain]
    latency_ms = random.lognormvariate(prof["mu"], prof["sigma"])
    if random.random() < 0.02:  # pico ocasional (GC, contenção, etc.)
        latency_ms *= random.uniform(3, 8)
    is_error = random.random() < prof["error_rate"]
    if is_error:
        latency_ms *= random.uniform(1.5, 3.0)  # erros tendem a ser mais lentos (timeout)

    with tracer.start_as_current_span(domain) as span:
        span.set_attribute("business.domain", domain)
        await asyncio.sleep(latency_ms / 1000.0)  # não-bloqueante -> alta concorrência
        if is_error:
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("error", True)
            response.status_code = 500
            return {"status": "error", "domain": domain}
    return {"status": "ok", "domain": domain}


@app.get("/")
async def site_latency(response: Response):
    return await handle("site_latency", response)


@app.get("/api/generic")
async def api_generic(response: Response):
    return await handle("api_generic", response)


@app.post("/checkout")
async def checkout(response: Response):
    return await handle("checkout", response)


@app.post("/pix/transacao")
async def pix_transacao(response: Response):
    return await handle("pix", response)


@app.get("/healthz")
async def healthz():
    return {"ok": True}
