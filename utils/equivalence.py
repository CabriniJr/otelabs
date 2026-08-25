"""
utils/equivalence.py

Descobre, por TESTE, qual é a menor taxa de sampling que ainda preserva a
informação — em vez de assumir um "sweet spot" escolhido a priori.

A ideia é inverter a pergunta. A aba de trade-off maximiza um score que
depende de escolhas de negócio (α, importância do sinal). Aqui não há
escolha de negócio: fixamos uma **banda de tolerância** ("a estimativa
precisa ficar a no máximo ±5% da verdade") e perguntamos qual a menor taxa
em que a amostra ainda passa nesse critério de forma confiável.

Método — TOST (two one-sided tests) com poder estimado por simulação:

1. O run determinístico (100%) do contexto é tratado como a VERDADE.
2. Para cada taxa candidata, reamostramos esse baseline `n_boot` vezes
   aplicando a **política real daquele contexto**:
   - head-based: cada span tem probabilidade `rate` de ser mantido;
   - tail-based: erros e cauda (≥ p95) são SEMPRE mantidos, e o tráfego
     normal é amostrado a `rate`.
3. Em cada réplica calculamos o IC da média de latência (t de Student) e o
   IC de Wilson da taxa de erro. A réplica "passa" quando o IC inteiro cabe
   dentro da banda de tolerância em torno do valor verdadeiro — este é o
   critério de equivalência (TOST): não basta a estimativa ser próxima, a
   incerteza também tem que caber.
4. A fração de réplicas aprovadas é o **poder** daquela taxa.
5. A taxa recomendada é a MENOR taxa cujo poder atinge o alvo (ex.: 95%).

Nota sobre o nível: usar o IC de `confidence` (ex.: 95%) dentro da banda é
uma versão conservadora do TOST — equivale a testar cada uma das duas
hipóteses unilaterais a α = (1 − confidence)/2. Preferimos a versão
conservadora para não recomendar taxas otimistas demais.

Este módulo só faz estatística sobre arrays; não conhece Streamlit nem o
schema dos CSVs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def _subsample_masks(n, rate, n_boot, rng, always_keep=None):
    """Matriz booleana (n_boot, n): quais spans cada réplica reteve.

    `always_keep` é a máscara de spans que a política tail-based nunca
    descarta (erros e cauda de latência); quando é None, a política é
    puramente probabilística (head-based).
    """
    masks = rng.random((n_boot, n)) < rate
    if always_keep is not None:
        masks |= always_keep[None, :]
    return masks


def _design_ci_batch(values, masks, pis, confidence):
    """IC da média/proporção sob amostragem com probabilidades DESIGUAIS.

    Estimador de razão de Horvitz–Thompson, com peso w = 1/π:

        x̄_w = Σ w·x / Σ w

    e variância pelo método delta, estimada a partir da própria amostra:

        Var(x̄_w) = Σ w²·(1−π)·(x − x̄_w)² / (Σ w)²

    O fator (1−π) é o que dá o resultado central deste módulo: spans que a
    política SEMPRE retém (π = 1) contribuem com variância ZERO. No
    tail-based, todo erro é retido, então a incerteza sobre a taxa de erro
    não vem do numerador (que é conhecido exatamente) e sim do denominador
    — a estimativa do volume total de tráfego. É por isso que o tail-based
    estima taxa de erro com muito mais precisão que uma amostra uniforme do
    mesmo tamanho.

    Com π constante, esta fórmula se reduz à variância clássica de amostra
    aleatória simples com correção de população finita, s²(1−f)/n.

    Retorna (n_efetivo_aproximado, ponto, lo, hi).
    """
    w = np.where(pis > 0, 1.0 / pis, 0.0)
    wm = masks * w[None, :]
    sw = wm.sum(axis=1)
    valid = sw > 0
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = np.where(valid, (wm @ values) / np.where(valid, sw, 1), np.nan)
        dev2 = (values[None, :] - point[:, None]) ** 2
        var = ((wm ** 2) * (1.0 - pis)[None, :] * dev2).sum(axis=1)
        var = np.where(valid, var / np.where(valid, sw, 1) ** 2, np.nan)
        se = np.sqrt(np.maximum(var, 0.0))
        half = z * se
    return masks.sum(axis=1).astype(float), point, point - half, point + half


def _mean_ci_batch(values, masks, confidence, pis=None):
    """IC da média para cada réplica, vetorizado.

    Sem `pis` (retenção uniforme, head-based) usa o IC exato de t de
    Student. Com `pis` desiguais (tail-based reponderado), cai no estimador
    de razão com variância de design.

    Retorna (n_por_replica, media, lo, hi). Réplicas com n < 2 saem NaN.
    """
    if pis is not None:
        return _design_ci_batch(values, masks, pis, confidence)
    n = masks.sum(axis=1)
    valid = n >= 2
    total = masks @ values
    total_sq = masks @ (values ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(valid, total / np.maximum(n, 1), np.nan)
        # variância amostral (ddof=1) a partir das somas acumuladas
        var = np.where(valid,
                       (total_sq - n * mean ** 2) / np.maximum(n - 1, 1),
                       np.nan)
        var = np.maximum(var, 0.0)  # protege contra -0.0 de erro numérico
        se = np.sqrt(var / np.maximum(n, 1))
        t_crit = sp_stats.t.ppf(1 - (1 - confidence) / 2, df=np.maximum(n - 1, 1))
        half = t_crit * se
    return n.astype(float), mean, mean - half, mean + half


def _prop_ci_batch(flags, masks, confidence, pis=None):
    """IC da proporção para cada réplica, vetorizado.

    Sem `pis`, é o IC de Wilson — preferido aqui porque a taxa de erro é
    baixa e o n fica pequeno sob sampling agressivo, situação em que o IC
    normal é ruim. Com `pis` desiguais, usa o estimador de razão.

    Retorna (n_por_replica, p, lo, hi).
    """
    if pis is not None:
        return _design_ci_batch(flags, masks, pis, confidence)
    n = masks.sum(axis=1)
    k = masks @ flags
    valid = n >= 1
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.maximum(n, 1)
        p = np.where(valid, k / nn, np.nan)
        denom = 1 + z ** 2 / nn
        center = (p + z ** 2 / (2 * nn)) / denom
        margin = (z * np.sqrt((p * (1 - p) / nn) + (z ** 2 / (4 * nn ** 2)))) / denom
    return (n.astype(float), p,
            np.maximum(0.0, center - margin), np.minimum(1.0, center + margin))


def baseline_floor(latency, is_error, confidence=0.95):
    """Piso de precisão do próprio baseline 100%.

    Mesmo capturando tudo, a média e a taxa de erro são estimativas com
    incerteza — e a taxa de erro, por ser um evento raro, costuma ter uma
    margem relativa grande. Nenhuma banda de tolerância mais apertada que
    este piso é atingível em taxa nenhuma, nem em 100%. A UI usa isto para
    avisar em vez de simplesmente responder "nenhuma taxa serve".

    Retorna as margens de erro RELATIVAS (em fração, não %) da latência e
    da taxa de erro no baseline completo.
    """
    latency = np.asarray(latency, dtype=float)
    flags = np.asarray(is_error, dtype=float)
    n = latency.size
    if n < 2:
        return {"lat": np.nan, "err": np.nan}
    mean = float(latency.mean())
    se = float(latency.std(ddof=1)) / np.sqrt(n)
    t_crit = sp_stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    lat_floor = (t_crit * se) / mean if mean else np.nan

    k, p = float(flags.sum()), float(flags.mean())
    if p <= 0:
        return {"lat": float(lat_floor), "err": np.nan}
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2)))) / denom
    # margem relativa "de pior lado": o quanto a banda precisa abrir para
    # conter o IC inteiro em torno de p
    err_floor = max(p - (center - margin), (center + margin) - p) / p
    return {"lat": float(lat_floor), "err": float(err_floor), "k": k}


def power_curve(latency, is_error, rates, policy="head", tol_lat=0.05,
                 tol_err=0.20, confidence=0.95, n_boot=200,
                 tail_percentile=95.0, reweight=True, seed=42) -> pd.DataFrame:
    """Curva de poder do teste de equivalência para uma grade de taxas.

    latency  : array de latências do baseline determinístico (a "verdade")
    is_error : array booleano de erros, mesmo comprimento
    rates    : taxas candidatas (base rate da política), em [0, 1]
    policy   : "head" (probabilística) ou "tail" (erro + cauda sempre)
    tol_lat  : banda de tolerância RELATIVA da média de latência (0.05 = ±5%)
    tol_err  : banda de tolerância RELATIVA da taxa de erro
    n_boot   : réplicas de Monte Carlo por taxa
    reweight : no tail-based, corrigir o viés de amostragem ponderando cada
               span por 1/probabilidade de retenção (Horvitz–Thompson)

    Sobre o `reweight`. O tail-based retém 100% dos erros e da cauda: a
    amostra resultante é, DE PROPÓSITO, enviesada — a taxa de erro e a
    latência média medidas nela ficam sistematicamente acima da verdade e
    não convergem para o baseline em taxa nenhuma. Sem correção, o teste de
    equivalência reprova o tail-based sempre, o que diria mais sobre o
    estimador do que sobre a política. Com a correção (o que um pipeline de
    métricas de produção faz ao respeitar a probabilidade de amostragem do
    span), o estimador volta a ser não-enviesado e a comparação entre as
    duas políticas passa a ser justa.

    Retorna um DataFrame com uma linha por taxa: retenção efetiva média,
    poder do teste de latência, do teste de taxa de erro e dos dois juntos,
    além do viés relativo médio de cada estimativa.
    """
    latency = np.asarray(latency, dtype=float)
    flags = np.asarray(is_error, dtype=float)
    n = latency.size
    if n < 2:
        return pd.DataFrame()

    mean_base = float(latency.mean())
    p_base = float(flags.mean())
    lat_lo, lat_hi = mean_base * (1 - tol_lat), mean_base * (1 + tol_lat)
    err_lo, err_hi = p_base * (1 - tol_err), p_base * (1 + tol_err)

    always_keep = None
    if policy == "tail":
        thr = np.percentile(latency, tail_percentile)
        always_keep = (flags > 0) | (latency >= thr)

    rng = np.random.default_rng(seed)
    rows = []
    for rate in rates:
        r = float(rate)
        masks = _subsample_masks(n, r, n_boot, rng, always_keep)
        # π = probabilidade de retenção de cada span. Spans que a política
        # sempre retém têm π = 1; os demais, π = r. Só passamos π adiante
        # quando ele é DESIGUAL — no head-based é constante e as fórmulas
        # exatas (t e Wilson) são melhores.
        pis = None
        if always_keep is not None and reweight and r > 0:
            pis = np.where(always_keep, 1.0, r)

        n_rep, mean_hat, lo, hi = _mean_ci_batch(latency, masks, confidence, pis)
        ok_lat = np.where(np.isnan(lo), False, (lo >= lat_lo) & (hi <= lat_hi))

        # Sem erros no baseline não há taxa de erro para preservar: o teste
        # de proporção não se aplica e não deve contaminar o resultado.
        if p_base > 0:
            _, p_hat, elo, ehi = _prop_ci_batch(flags, masks, confidence, pis)
            ok_err = np.where(np.isnan(elo), False, (elo >= err_lo) & (ehi <= err_hi))
            vies_err = float(np.nanmean(p_hat) / p_base - 1)
        else:
            ok_err = np.full(n_boot, False)
            vies_err = np.nan

        rows.append({
            "rate": r,
            "retencao_efetiva": float(masks.sum(axis=1).mean() / n),
            "n_medio": float(masks.sum(axis=1).mean()),
            "poder_latencia": float(ok_lat.mean()),
            "poder_erro": float(ok_err.mean()) if p_base > 0 else np.nan,
            "poder_conjunto": float((ok_lat & ok_err).mean()) if p_base > 0
                                else float(ok_lat.mean()),
            "vies_latencia": float(np.nanmean(mean_hat) / mean_base - 1),
            "vies_erro": vies_err,
        })
    return pd.DataFrame(rows)


def real_run_ci(base_latency, base_is_error, real_latency, real_is_error,
                 policy="head", confidence=0.95, tail_percentile=95.0):
    """ICs de latência média e taxa de erro para um run REAL de coletor.

    No head-based todos os spans foram retidos com a mesma probabilidade e
    valem os ICs clássicos. No tail-based não: erros e cauda foram retidos
    sempre e o resto a uma taxa-base, então medir contando linhas dá um
    resultado enviesado. Aqui reconstruímos a probabilidade de retenção de
    cada span exportado a partir do baseline — um span é "sempre retido" se
    é erro ou está acima do p95 do baseline; a taxa-base dos demais é
    estimada pela própria retenção observada nessa parcela — e aplicamos o
    mesmo estimador reponderado usado na simulação. Sem isso a validação
    compararia coisas diferentes.

    Retorna {"lat": (ponto, lo, hi), "err": (ponto, lo, hi), "rate_base"}.
    """
    base_latency = np.asarray(base_latency, dtype=float)
    base_flags = np.asarray(base_is_error, dtype=float)
    real_latency = np.asarray(real_latency, dtype=float)
    real_flags = np.asarray(real_is_error, dtype=float)

    pis = None
    rate_base = np.nan
    if policy == "tail" and base_latency.size:
        thr = np.percentile(base_latency, tail_percentile)
        base_always = (base_flags > 0) | (base_latency >= thr)
        real_always = (real_flags > 0) | (real_latency >= thr)
        n_base_rest = int((~base_always).sum())
        n_real_rest = int((~real_always).sum())
        if n_base_rest > 0 and n_real_rest > 0:
            rate_base = min(1.0, n_real_rest / n_base_rest)
            pis = np.where(real_always, 1.0, max(rate_base, 1e-9))

    mask = np.ones((1, real_latency.size), dtype=bool)
    _, lat_p, lat_lo, lat_hi = _mean_ci_batch(real_latency, mask, confidence, pis)
    _, err_p, err_lo, err_hi = _prop_ci_batch(real_flags, mask, confidence, pis)
    return {"lat": (float(lat_p[0]), float(lat_lo[0]), float(lat_hi[0])),
            "err": (float(err_p[0]), float(err_lo[0]), float(err_hi[0])),
            "rate_base": float(rate_base)}


def min_viable_rate(curve: pd.DataFrame, column="poder_conjunto", target=0.95):
    """Menor taxa cujo poder atinge o alvo E se mantém acima dele em todas
    as taxas maiores.

    A exigência de se manter acima evita que o ruído de Monte Carlo em uma
    taxa isolada seja lido como recomendação — o poder é monótono crescente
    na taxa, então uma aprovação seguida de reprovações é ruído.

    Retorna dict com taxa, retenção efetiva e poder; None se nenhuma taxa
    da grade atinge o alvo.
    """
    if curve.empty or column not in curve:
        return None
    c = curve.sort_values("rate").reset_index(drop=True)
    ok = (c[column] >= target).values
    # varre de trás para frente: o candidato só vale se tudo à direita passa
    best = None
    for i in range(len(c) - 1, -1, -1):
        if not ok[i]:
            break
        best = i
    if best is None:
        return None
    row = c.iloc[best]
    return {"rate": float(row["rate"]),
            "retencao_efetiva": float(row["retencao_efetiva"]),
            "n_medio": float(row["n_medio"]),
            "poder": float(row[column])}
