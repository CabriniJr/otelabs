"""
utils/sampling.py

Implementa, sobre um dataset bruto (ground-truth, 100% capturado no teste
de carga real), as políticas de sampling que um OTel Collector poderia
aplicar. Isso permite comparar diferentes configurações de sampling SEM
precisar reexecutar o teste de carga uma vez por configuração — aplicamos
a política retroativamente sobre os dados reais coletados.

Schema esperado do DataFrame de entrada (df), uma linha por request/span:
    - trace_id / request_id   : identificador único (str)
    - timestamp               : datetime ou epoch (opcional para este módulo)
    - domain                  : tag/domínio do sinal, ex.: "pix", "checkout",
                                 "site_latency", "api_generic"
    - latency_ms              : latência da requisição em milissegundos
    - is_error                : bool — se a requisição terminou em erro

Nenhuma função aqui GERA dados — todas recebem um DataFrame já carregado
a partir do teste de carga real (ver app.py / README.md para o schema
completo do CSV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Taxas padrão de sampling probabilístico (head-based) usadas na comparação
STANDARD_HEAD_RATES = [1.00, 0.50, 0.25, 0.10, 0.05, 0.01]


def probabilistic_sample(df: pd.DataFrame, rate: float, seed: int = 42) -> pd.DataFrame:
    """Sampling probabilístico/head-based: cada requisição tem
    probabilidade `rate` de ser mantida, independente de tudo o mais.
    Isso é o que um `probabilistic_sampler` do OTel Collector faz."""
    if rate >= 1.0:
        return df.copy()
    if rate <= 0.0:
        return df.iloc[0:0].copy()
    return df.sample(frac=rate, random_state=seed)


def tail_based_sample(df: pd.DataFrame, base_rate: float,
                       latency_percentile: float = 95.0,
                       per_domain_threshold: bool = True,
                       seed: int = 42) -> pd.DataFrame:
    """Sampling tail-based: SEMPRE mantém erros e requisições "de cauda"
    (latência acima do percentil `latency_percentile`), e amostra o
    restante ("tráfego normal") na taxa `base_rate`. Reproduz o
    comportamento do processor `tail_sampling` do OTel Collector com uma
    política de erro + latência alta.
    """
    df = df.copy()
    if per_domain_threshold:
        thresholds = df.groupby("domain")["latency_ms"].transform(
            lambda s: np.percentile(s, latency_percentile))
    else:
        thresholds = np.percentile(df["latency_ms"], latency_percentile)

    is_tail = (df["latency_ms"] >= thresholds) | (df["is_error"].astype(bool))
    always_keep = df[is_tail]
    rest = df[~is_tail]
    sampled_rest = probabilistic_sample(rest, base_rate, seed=seed)
    return pd.concat([always_keep, sampled_rest], axis=0).sort_index()


def dynamic_domain_sample(df: pd.DataFrame, rate_by_domain: dict,
                           always_sample_errors: bool = True,
                           seed: int = 42) -> pd.DataFrame:
    """Sampling dinâmico por tag/domínio: cada domínio tem sua própria
    taxa de sampling (ex.: pix=100%, checkout=50%, site_latency=20%,
    api_generic=5%). Opcionalmente, erros são sempre mantidos independente
    do domínio — combina segregação por domínio com uma política de
    tail-based simplificada para erros.

    rate_by_domain: dict {domain: rate}. Domínios não presentes no dict
    usam rate=1.0 (mantidos integralmente) por segurança.
    """
    parts = []
    for domain, group in df.groupby("domain"):
        rate = float(rate_by_domain.get(domain, 1.0))
        if always_sample_errors:
            errors = group[group["is_error"].astype(bool)]
            ok = group[~group["is_error"].astype(bool)]
            sampled_ok = probabilistic_sample(ok, rate, seed=seed)
            parts.append(pd.concat([errors, sampled_ok], axis=0))
        else:
            parts.append(probabilistic_sample(group, rate, seed=seed))
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, axis=0).sort_index()


def export_reduction(baseline_n: int, sampled_n: int) -> float:
    """Fração de redução no volume exportado pelo coletor (proxy direto
    de economia de CPU/rede/armazenamento no backend de observabilidade)."""
    if baseline_n == 0:
        return np.nan
    return 1.0 - (sampled_n / baseline_n)
