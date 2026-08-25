"""
Verificação de sanidade de utils/equivalence.py.

Não usa pytest de propósito: roda com `python scripts/check_equivalence.py`
sem nenhuma dependência além das que o app já precisa. Confere as
propriedades que a curva de poder tem que ter para a recomendação de taxa
fazer sentido.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import equivalence as eq  # noqa: E402
from utils import stats as st_stats  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FALHA"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_baseline(n=8000, seed=7):
    """Baseline sintético parecido com os dados reais: latência de cauda
    longa (lognormal) e taxa de erro baixa."""
    rng = np.random.default_rng(seed)
    latency = rng.lognormal(mean=4.8, sigma=0.45, size=n)
    is_error = rng.random(n) < 0.012
    return latency, is_error


def main():
    latency, is_error = make_baseline()
    rates = np.geomspace(0.005, 1.0, 24)

    # O piso do baseline dita quais bandas são sequer atingíveis: eventos
    # raros (erros) têm margem relativa grande mesmo capturando 100%.
    floor = eq.baseline_floor(latency, is_error)
    print(f"piso do baseline: latência ±{floor['lat']:.2%}, "
          f"taxa de erro ±{floor['err']:.2%} (k={floor['k']:.0f} erros)\n")
    TOL_LAT, TOL_ERR = 2 * floor["lat"], 2 * floor["err"]

    head = eq.power_curve(latency, is_error, rates, policy="head",
                          tol_lat=TOL_LAT, tol_err=TOL_ERR, n_boot=150)
    tail = eq.power_curve(latency, is_error, rates, policy="tail",
                          tol_lat=TOL_LAT, tol_err=TOL_ERR, n_boot=150)

    # 1. A 100% a amostra É o baseline: com banda acima do piso, o IC cabe
    #    e o poder tem que ser 1.
    check("poder = 1 na taxa 100% (head)",
          head.iloc[-1]["poder_conjunto"] == 1.0,
          f"poder={head.iloc[-1]['poder_conjunto']:.3f}")

    # 1b. Banda mais apertada que o piso é inatingível em QUALQUER taxa —
    #     inclusive 100%. É o caso que a UI precisa avisar.
    impossible = eq.power_curve(latency, is_error, rates, policy="head",
                                tol_lat=TOL_LAT, tol_err=floor["err"] * 0.5,
                                n_boot=80)
    check("banda abaixo do piso ⇒ nenhuma taxa atinge o alvo",
          eq.min_viable_rate(impossible) is None
          and impossible["poder_conjunto"].max() == 0.0,
          f"poder máximo={impossible['poder_conjunto'].max():.3f}")

    # 2. Poder é monótono crescente na taxa (a menos de ruído de Monte Carlo).
    d = np.diff(head["poder_conjunto"].values)
    check("poder cresce com a taxa (head)", d.min() > -0.10,
          f"maior queda={d.min():+.3f}")

    # 3. Taxas muito baixas não podem passar no teste.
    check("poder ~ 0 na taxa mínima (head)",
          head.iloc[0]["poder_conjunto"] < 0.05,
          f"poder={head.iloc[0]['poder_conjunto']:.3f}")

    # 4. No head-based a retenção efetiva é a própria taxa.
    r = head["rate"].values
    check("retenção efetiva ≈ taxa (head)",
          np.allclose(head["retencao_efetiva"].values, r, atol=0.02))

    # 5. No tail-based a retenção é SEMPRE maior que a taxa nominal (erros e
    #    cauda entram de graça) — é o que explica o tail reter mais.
    check("retenção efetiva > taxa nominal (tail)",
          bool((tail["retencao_efetiva"].values >= r - 1e-9).all()),
          f"folga mínima={float((tail['retencao_efetiva'].values - r).min()):+.4f}")

    # 6. O head-based é não-enviesado por construção: a amostra uniforme
    #    estima a verdade sem deslocamento sistemático.
    #
    #    O viés só é mensurável onde o evento raro aparece: a 0,5% sobram
    #    ~40 spans com ~0,4 erros esperados, e o ruído de Monte Carlo sobre
    #    p̂ passa de 10% — medir "viés" ali é medir ruído. Restringimos às
    #    taxas em que se espera ao menos ~25 erros por réplica.
    p_base = float(np.asarray(is_error, float).mean())
    medivel = head[head["n_medio"] * p_base >= 25]
    check("head-based não tem viés relevante (onde é mensurável)",
          abs(medivel["vies_erro"].dropna()).max() < 0.05,
          f"maior viés={abs(medivel['vies_erro'].dropna()).max():+.3%} "
          f"em {len(medivel)} taxas com ≥25 erros esperados")

    # 7. O tail-based SEM correção é enviesado para cima por construção
    #    (retém 100% dos erros e da cauda) — e o viés explode nas taxas
    #    baixas. É o motivo de o reponderamento existir.
    tail_raw = eq.power_curve(latency, is_error, rates, policy="tail",
                              tol_lat=TOL_LAT, tol_err=TOL_ERR,
                              reweight=False, n_boot=150)
    check("tail-based sem correção é enviesado para cima",
          tail_raw["vies_erro"].max() > 0.5 and tail_raw["vies_latencia"].max() > 0.05,
          f"viés máx erro={tail_raw['vies_erro'].max():+.1%}, "
          f"latência={tail_raw['vies_latencia'].max():+.1%}")

    # 8. Com o reponderamento de Horvitz–Thompson o viés some.
    check("reponderamento remove o viés do tail-based",
          abs(tail["vies_erro"].dropna()).max() < 0.10
          and abs(tail["vies_latencia"]).max() < 0.05,
          f"viés máx erro={abs(tail['vies_erro'].dropna()).max():+.1%}, "
          f"latência={abs(tail['vies_latencia']).max():+.1%}")

    # 9. Comparação justa: reponderado e na MESMA retenção efetiva, o
    #    tail-based tem que ir melhor no teste de taxa de erro, porque
    #    guardou todos os erros em vez de uma fração deles.
    target_ret = 0.30
    i_head = int(np.argmin(np.abs(head["retencao_efetiva"].values - target_ret)))
    i_tail = int(np.argmin(np.abs(tail["retencao_efetiva"].values - target_ret)))
    check("com a mesma retenção, tail ≥ head no teste de erro",
          tail.iloc[i_tail]["poder_erro"] >= head.iloc[i_head]["poder_erro"] - 1e-9,
          f"tail={tail.iloc[i_tail]['poder_erro']:.3f} @ "
          f"{tail.iloc[i_tail]['retencao_efetiva']:.1%} vs "
          f"head={head.iloc[i_head]['poder_erro']:.3f} @ "
          f"{head.iloc[i_head]['retencao_efetiva']:.1%}")

    # 7. Banda mais frouxa não pode exigir taxa maior que banda apertada.
    loose = eq.power_curve(latency, is_error, rates, policy="head",
                           tol_lat=3 * TOL_LAT, tol_err=3 * TOL_ERR, n_boot=150)
    mv_tight = eq.min_viable_rate(head)
    mv_loose = eq.min_viable_rate(loose)
    check("tolerância maior ⇒ taxa recomendada menor ou igual",
          mv_loose is not None and mv_tight is not None
          and mv_loose["rate"] <= mv_tight["rate"] + 1e-9,
          f"apertada={mv_tight['rate']:.3%} vs frouxa={mv_loose['rate']:.3%}"
          if mv_tight and mv_loose else "alguma recomendação veio vazia")

    # 8. min_viable_rate tem que devolver uma taxa cujo poder atinge o alvo.
    check("taxa recomendada atinge o alvo",
          mv_tight is not None and mv_tight["poder"] >= 0.95,
          f"poder={mv_tight['poder']:.3f}" if mv_tight else "sem recomendação")

    # 9. Alvo impossível devolve None em vez de mentir.
    check("alvo inatingível devolve None",
          eq.min_viable_rate(head, target=1.01) is None)

    # 10. A média/IC vetorizados batem com a implementação escalar de
    #     referência em utils/stats.py.
    rng = np.random.default_rng(1)
    mask = rng.random(latency.size) < 0.2
    masks = mask[None, :]
    n_rep, mean_v, lo, hi = eq._mean_ci_batch(latency, masks, 0.95)
    ref = st_stats.clt_mean_ci(latency[mask], 0.95)
    check("IC vetorizado == IC escalar de utils/stats.py",
          np.isclose(lo[0], ref["lo"], rtol=1e-9) and np.isclose(hi[0], ref["hi"], rtol=1e-9),
          f"vetorizado=({lo[0]:.6f}, {hi[0]:.6f}) escalar=({ref['lo']:.6f}, {ref['hi']:.6f})")

    # 11. Idem para o IC de Wilson.
    flags = np.asarray(is_error, dtype=float)
    _, p_v, elo, ehi = eq._prop_ci_batch(flags, masks, 0.95)
    ref_w = st_stats.wilson_ci(int(flags[mask].sum()), int(mask.sum()), 0.95)
    check("Wilson vetorizado == Wilson escalar de utils/stats.py",
          np.isclose(elo[0], ref_w["lo"], rtol=1e-9) and np.isclose(ehi[0], ref_w["hi"], rtol=1e-9),
          f"vetorizado=({elo[0]:.6f}, {ehi[0]:.6f}) escalar=({ref_w['lo']:.6f}, {ref_w['hi']:.6f})")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} verificação(ões) falharam: {FAILURES}")
        return 1
    print("Todas as verificações passaram.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
