#!/usr/bin/env bash
# Roda UM par (coletor nativo + demo-app + locust) sequencialmente, sem
# Docker — usado neste sandbox porque o Docker Hub está bloqueado aqui,
# mas o binário oficial do otelcol-contrib (baixado do GitHub Releases)
# funciona igual. No laptop do usuário (T14), o docker-compose.yml
# normal funciona sem essa gambiarra.
#
# Uso: ./run_local_pair.sh <contexto> <config> <total_requests>
set -euo pipefail

CONTEXT=$1
CONFIG=$2
TOTAL=${3:-25000}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR="$ROOT/out/$CONTEXT/$CONFIG"
mkdir -p "$OUT_DIR"

echo "=== $CONTEXT / $CONFIG (total=$TOTAL) ==="

# 1) coletor
sed "s#/out/traces.json#$OUT_DIR/traces.json#" "$ROOT/otel-config/$CONTEXT/$CONFIG.yaml" > /tmp/otelcol_config.yaml
/tmp/otelcol/otelcol-contrib --config=/tmp/otelcol_config.yaml > "$OUT_DIR/collector.log" 2>&1 &
COLLECTOR_PID=$!
sleep 3

# 2) demo-app
cd "$ROOT/demo-app"
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
  OTEL_SERVICE_NAME="demo-app-$CONTEXT-$CONFIG" \
  uvicorn main:app --host 0.0.0.0 --port 8000 > "$OUT_DIR/app.log" 2>&1 &
APP_PID=$!
sleep 2

# 3) locust
cd "$ROOT"
LOCUST_TOTAL_REQUESTS=$TOTAL locust -f locustfile.py --host=http://localhost:8000 \
  --headless -u 120 -r 40 -t 10m > "$OUT_DIR/locust.log" 2>&1 || true

echo "--- locust summary ---"
tail -15 "$OUT_DIR/locust.log"

# 4) espera o coletor esvaziar buffers e derruba tudo
sleep 5
kill $APP_PID 2>/dev/null || true
sleep 1
kill $COLLECTOR_PID 2>/dev/null || true
sleep 1
wc -l "$OUT_DIR/traces.json" 2>&1 || echo "AVISO: traces.json não foi gerado"
echo "=== fim $CONTEXT / $CONFIG ==="
