"""
scripts/otel_export_to_csv.py

Converte o output do `file` exporter do OTel Collector (JSON lines — um
ExportTraceServiceRequest OTLP/JSON por linha, exatamente o formato que
os 6 configs em otel-config/ usam) no CSV que o app Streamlit espera:
trace_id, timestamp, domain, latency_ms, is_error.

Uso (um arquivo por vez):
    python scripts/otel_export_to_csv.py \
        out/head_based/deterministico/traces.json \
        data/real_runs/head_based_deterministico.csv

Ou para converter os 6 de uma vez (assume a estrutura de pastas criada
pelo docker-compose.yml, out/<contexto>/<config>/traces.json):
    python scripts/otel_export_to_csv.py --all
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone

CONTEXTS = ["head_based", "tail_based"]
RUNS = ["deterministico", "sweet_spot", "agressivo"]


def get_attr(attributes, key):
    for a in attributes or []:
        if a.get("key") == key:
            v = a.get("value", {})
            return v.get("stringValue", v.get("boolValue", v.get("intValue")))
    return None


def convert(in_path: str, out_path: str) -> int:
    rows = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            for rs in doc.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    for span in ss.get("spans", []):
                        trace_id = span.get("traceId")
                        start_ns = int(span.get("startTimeUnixNano", 0))
                        end_ns = int(span.get("endTimeUnixNano", 0))
                        latency_ms = max(0.0, (end_ns - start_ns) / 1e6)
                        domain = get_attr(span.get("attributes"), "business.domain") or "unknown"
                        status = span.get("status", {}) or {}
                        # OTLP StatusCode: 0=UNSET, 1=OK, 2=ERROR
                        is_error = status.get("code") == 2
                        ts = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat()
                        rows.append([trace_id, ts, domain, round(latency_ms, 2), is_error])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace_id", "timestamp", "domain", "latency_ms", "is_error"])
        w.writerows(rows)
    return len(rows)


def main():
    if "--all" in sys.argv:
        for ctx in CONTEXTS:
            for run in RUNS:
                in_path = f"out/{ctx}/{run}/traces.json"
                out_path = f"data/real_runs/{ctx}_{run}.csv"
                if not os.path.exists(in_path):
                    print(f"[pula] {in_path} não existe ainda")
                    continue
                n = convert(in_path, out_path)
                print(f"{n:,} spans -> {out_path}")
        return

    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    n = convert(sys.argv[1], sys.argv[2])
    print(f"{n:,} spans -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
