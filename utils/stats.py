"""
utils/stats.py

Funções estatísticas usadas no portfólio: intervalos de confiança (via
Teorema Central do Limite e via bootstrap), intervalo de confiança para
proporções (Wilson), e o score composto que combina ganho de throughput
com perda de confiança, ponderado pela importância do sinal.

Todas as funções recebem dados JÁ FILTRADOS por uma política de sampling
(ver utils/sampling.py) — este módulo não sabe nada sobre coleta de dados,
apenas calcula estatística em cima do que recebe.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Intervalos de confiança para médias (ex.: latência média, em ms)
# ---------------------------------------------------------------------------

def clt_mean_ci(data: np.ndarray, confidence: float = 0.95):
    """IC para a média via Teorema Central do Limite (distribuição t de
    Student, apropriada quando o desvio-padrão populacional é desconhecido
    e estimado da amostra).

    Retorna dict com mean, lo, hi, halfwidth, n, std.
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    n = data.size
    if n < 2:
        return {"mean": float(data[0]) if n == 1 else np.nan,
                "lo": np.nan, "hi": np.nan, "halfwidth": np.nan,
                "n": n, "std": np.nan}
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    se = std / np.sqrt(n)
    alpha = 1 - confidence
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=n - 1)
    halfwidth = float(t_crit * se)
    return {"mean": mean, "lo": mean - halfwidth, "hi": mean + halfwidth,
            "halfwidth": halfwidth, "n": n, "std": std}


def bootstrap_mean_ci(data: np.ndarray, confidence: float = 0.95,
                       n_boot: int = 1500, random_state: int = 42):
    """IC para a média via bootstrap percentil (não assume normalidade —
    mais robusto para latências, que costumam ter cauda longa).

    Retorna o mesmo formato de clt_mean_ci.
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    n = data.size
    if n < 2:
        return {"mean": float(data[0]) if n == 1 else np.nan,
                "lo": np.nan, "hi": np.nan, "halfwidth": np.nan,
                "n": n, "std": np.nan}
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = data[idx].mean(axis=1)
    alpha = 1 - confidence
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = float(np.mean(data))
    halfwidth = float((hi - lo) / 2)
    return {"mean": mean, "lo": float(lo), "hi": float(hi),
            "halfwidth": halfwidth, "n": n, "std": float(np.std(data, ddof=1))}


# ---------------------------------------------------------------------------
# Intervalo de confiança para proporções (ex.: taxa de erro, taxa de
# detecção de transações PIX na amostra)
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, confidence: float = 0.95):
    """IC de Wilson para proporções — mais estável que o IC normal quando
    n é pequeno (sampling agressivo) ou a proporção é próxima de 0 ou 1
    (ex.: taxa de erro baixa)."""
    if n == 0:
        return {"p": np.nan, "lo": np.nan, "hi": np.nan, "halfwidth": np.nan, "n": 0}
    p = successes / n
    alpha = 1 - confidence
    z = sp_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2)))) / denom
    lo, hi = max(0.0, center - margin), min(1.0, center + margin)
    return {"p": p, "lo": lo, "hi": hi, "halfwidth": (hi - lo) / 2, "n": n}


# ---------------------------------------------------------------------------
# Métricas derivadas para a análise de trade-off
# ---------------------------------------------------------------------------

def relative_moe(halfwidth: float, reference: float) -> float:
    """Margem de erro relativa (%) — largura do IC dividida pelo valor de
    referência (normalmente a média/taxa no baseline de 100%)."""
    if reference in (0, None) or np.isnan(reference) or halfwidth is None or np.isnan(halfwidth):
        return np.nan
    return 100.0 * halfwidth / abs(reference)


def throughput_gain(sampling_rate: float) -> float:
    """Proxy de ganho de performance: fração de spans/eventos que deixam
    de ser exportados/processados pelo coletor. sampling_rate em [0, 1]."""
    return 1.0 - float(sampling_rate)


def confidence_penalty(moe_pct: float, moe_cap_pct: float = 20.0) -> float:
    """Normaliza a margem de erro relativa (%) para uma penalidade em
    [0, 1]. Acima de moe_cap_pct a penalidade satura em 1 (perda de
    confiança "máxima" para fins do score)."""
    if moe_pct is None or np.isnan(moe_pct):
        return 1.0
    return float(np.clip(moe_pct / moe_cap_pct, 0.0, 1.0))


def composite_score(gain: float, penalty: float, importance: float,
                     alpha: float = 0.5) -> float:
    """Score composto usado para achar o "sweet spot" de sampling.

    score = alpha * ganho_throughput - (1 - alpha) * importância * penalidade_confiança

    - alpha: peso global entre performance (throughput) e confiança
      estatística, escolhido via slider (0 = só importa confiança,
      1 = só importa performance).
    - importance: importância do sinal (0-1). Sinais críticos (ex.: PIX)
      usam importância alta, o que amplifica a penalidade de confiança e
      empurra o sweet spot para taxas de sampling mais altas.
    """
    return float(alpha * gain - (1 - alpha) * importance * penalty)


def find_sweet_spot(rates, scores):
    """Recebe arrays paralelos de taxas de sampling e scores e retorna a
    taxa que maximiza o score (o "sweet spot")."""
    rates = np.asarray(rates, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0 or np.all(np.isnan(scores)):
        return np.nan, np.nan
    i = int(np.nanargmax(scores))
    return float(rates[i]), float(scores[i])
