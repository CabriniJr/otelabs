"""
scripts/generate_sample_data.py

ESTE SCRIPT NÃO FAZ PARTE DO APP STREAMLIT.

Gera um conjunto de dados de EXEMPLO que imita a estrutura do experimento
real: 2 contextos de teste (head-based / tail-based) x 3 coletores
(determinístico / sweet spot / agressivo) = 6 arquivos CSV, cada um
representando o que aquele coletor especificamente exportaria.

Isso serve só para você testar a interface do app ANTES de rodar o
experimento real no seu docker compose. Quando os dados reais estiverem
prontos, suba os 6 CSVs reais no app (mesmo schema) — o app nunca gera ou
re-amostra dados sozinho, só analisa o que for carregado.

Uso:
    python scripts/generate_sample_data.py

Gera em data/sample_runs/:
    head_based_deterministico.csv   (100% do tráfego — ground truth)
    head_based_sweet_spot.csv       (probabilistic_sampler a 15%)
    head_based_agressivo.csv        (probabilistic_sampler a 1%)
    tail_based_deterministico.csv   (100% do tráfego — ground truth)
    tail_based_sweet_spot.csv       (tail_sampling, base_rate 15%)
    tail_based_agressivo.csv        (tail_sampling, base_rate 1%)

Schema de cada CSV (igual ao esperado do teste real — ver README.md):
    trace_id, timestamp, domain, latency_ms, is_error
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import sampling as smp  # noqa: E402

N_TOTAL = 25_000  # demo reduzida; o teste real terá 200_000 por contexto

DOMAIN_PROFILES = {
    "pix":          dict(share=0.15, mu=np.log(120), sigma=0.35, error_rate=0.005),
    "checkout":     dict(share=0.20, mu=np.log(300), sigma=0.45, error_rate=0.015),
    "site_latency": dict(share=0.35, mu=np.log(180), sigma=0.55, error_rate=0.008),
    "api_generic":  dict(share=0.30, mu=np.log(90),  sigma=0.50, error_rate=0.020),
}

SWEET_SPOT_RATE = 0.15
AGGRESSIVE_RATE = 0.01

CONTEXTS = {
    "head_based": dict(seed=42, kind="head"),
    "tail_based": dict(seed=99, kind="tail"),
}


def generate_raw(seed: int) -> pd.DataFrame:
    """Gera o dataset bruto (ground truth, 100% capturado) para um contexto
    de teste — equivalente ao que o coletor determinístico exportaria."""
    rng = np.random.default_rng(seed)
    rows = []
    start = datetime.now(timezone.utc) - timedelta(minutes=30)
    trace_counter = 0
    for domain, prof in DOMAIN_PROFILES.items():
        n = int(round(N_TOTAL * prof["share"]))
        latency = rng.lognormal(mean=prof["mu"], sigma=prof["sigma"], size=n)
        spike_mask = rng.random(n) < 0.02
        latency[spike_mask] *= rng.uniform(3, 8, size=spike_mask.sum())
        is_error = rng.random(n) < prof["error_rate"]
        latency[is_error] *= rng.uniform(1.5, 3.0, size=is_error.sum())
        offsets = np.sort(rng.uniform(0, 30 * 60, size=n))
        for i in range(n):
            trace_counter += 1
            rows.append({
                "trace_id": f"{seed}-trace-{trace_counter:07d}",
                "timestamp": (start + timedelta(seconds=float(offsets[i]))).isoformat(),
                "domain": domain,
                "latency_ms": round(float(latency[i]), 2),
                "is_error": bool(is_error[i]),
            })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def main():
    out_dir = "data/sample_runs"
    os.makedirs(out_dir, exist_ok=True)

    for context_key, cfg in CONTEXTS.items():
        raw = generate_raw(cfg["seed"])

        det_path = f"{out_dir}/{context_key}_deterministico.csv"
        raw.to_csv(det_path, index=False)

        if cfg["kind"] == "head":
            sweet = smp.probabilistic_sample(raw, SWEET_SPOT_RATE, seed=cfg["seed"])
            aggressive = smp.probabilistic_sample(raw, AGGRESSIVE_RATE, seed=cfg["seed"])
        else:
            sweet = smp.tail_based_sample(raw, SWEET_SPOT_RATE, seed=cfg["seed"])
            aggressive = smp.tail_based_sample(raw, AGGRESSIVE_RATE, seed=cfg["seed"])

        sweet.to_csv(f"{out_dir}/{context_key}_sweet_spot.csv", index=False)
        aggressive.to_csv(f"{out_dir}/{context_key}_agressivo.csv", index=False)

        print(f"\n=== {context_key} ===")
        print(f"determinístico: {len(raw):,} linhas -> {det_path}")
        print(f"sweet_spot:     {len(sweet):,} linhas ({len(sweet)/len(raw):.1%} retido)")
        print(f"agressivo:      {len(aggressive):,} linhas ({len(aggressive)/len(raw):.1%} retido)")


if __name__ == "__main__":
    main()
