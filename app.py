"""
Portfólio — Intervalos de Confiança aplicados a Sampling em Coletores OTel
============================================================================

Trabalho de Estatística (FIAP) — análise de um experimento real feito num
docker compose: 2 contextos de teste (head-based / tail-based), cada um
com 3 coletores OTel idênticos exceto pela config de sampling —
determinístico (100%), sweet spot e agressivo (baixa retenção). O app
compara os dados REAIS exportados por cada coletor: ganho de performance
(redução do volume exportado) vs. perda de confiança estatística, por
sinal/domínio, e onde head-based e tail-based divergem.

IMPORTANTE: este app não gera nem re-amostra dados. Cada um dos 6 slots
de upload é o export real de um coletor específico. Por padrão ele já
carrega data/real_runs/*.csv — o teste real (reduzido) rodado localmente
— e só cai para data/sample_runs/*.csv (dados sintéticos de exemplo) se
nem isso existir. Veja README.md e EXPERIMENTO.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats as sp_stats

from utils import stats as st_stats

# ---------------------------------------------------------------------------
# Paleta (dataviz skill — paleta validada, categórica em ordem fixa,
# sequencial azul, divergente azul<->vermelho, cores de status)
# ---------------------------------------------------------------------------
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

DOMAIN_PRIORITY = ["pix", "checkout", "site_latency", "api_generic"]
REQUIRED_COLS = {"trace_id", "timestamp", "domain", "latency_ms", "is_error"}
SOURCE_LABEL = {"upload": "upload manual", "real": "teste real (local, reduzido)",
                    "sample": "exemplo sintético"}

CONTEXTS = [
    {"key": "head_based", "default_label": "Contexto 1 — Head-based (probabilístico)"},
    {"key": "tail_based", "default_label": "Contexto 2 — Tail-based"},
]
RUNS = [
    {"key": "deterministico", "label": "Determinístico (100%)"},
    {"key": "sweet_spot", "label": "Sweet spot"},
    {"key": "agressivo", "label": "Agressivo (baixa retenção)"},
]
RUN_ORDER = [r["key"] for r in RUNS]
RUN_LABEL = {r["key"]: r["label"] for r in RUNS}

st.set_page_config(page_title="IC & Sampling OTel", layout="wide",
                    initial_sidebar_state="expanded")


def base_layout(fig, title, xaxis_title, yaxis_title, legend_title=None):
    fig.update_layout(
        title=dict(text=title, font=dict(color=INK_PRIMARY, size=16)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=INK_SECONDARY, family="system-ui, -apple-system, Segoe UI, sans-serif"),
        legend=dict(title=legend_title, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    fig.update_xaxes(title=xaxis_title, gridcolor=GRID, zerolinecolor=GRID,
                        linecolor="#c3c2b7", tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(title=yaxis_title, gridcolor=GRID, zerolinecolor=GRID,
                        linecolor="#c3c2b7", tickfont=dict(color=INK_MUTED))
    return fig


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Colunas faltando: {sorted(missing)}")
    df["is_error"] = df["is_error"].astype(bool)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    return df.dropna(subset=["latency_ms"])


def domain_colors(domains):
    ordered = [d for d in DOMAIN_PRIORITY if d in domains]
    ordered += [d for d in sorted(domains) if d not in ordered]
    return {d: CATEGORICAL[i % len(CATEGORICAL)] for i, d in enumerate(ordered)}


def default_importance(domain):
    return {"pix": 1.0, "checkout": 0.6, "site_latency": 0.35, "api_generic": 0.2}.get(domain, 0.5)


def mean_ci(values, confidence, method):
    if method.startswith("Bootstrap"):
        return st_stats.bootstrap_mean_ci(values, confidence, n_boot=800)
    return st_stats.clt_mean_ci(values, confidence)


# ---------------------------------------------------------------------------
# Sidebar — carga dos 6 arquivos reais (2 contextos x 3 coletores)
# ---------------------------------------------------------------------------
st.sidebar.header("Dados do experimento")
st.sidebar.caption(
    "Um CSV por coletor: o export REAL daquele run (mesmas colunas: "
    "trace_id, timestamp, domain, latency_ms, is_error). Por padrão o app "
    "já mostra o teste real reduzido (20k reqs/coletor) rodado localmente "
    "— suba um CSV pra substituir por uma rodada sua (ex.: os 200k completos).")

# Prioridade de cada slot: upload manual > data/real_runs (teste real já
# rodado) > data/sample_runs (exemplo sintético, só se nem isso existir).
data = {}
context_labels = {}
sources_used = set()
for ctx in CONTEXTS:
    with st.sidebar.expander(ctx["default_label"], expanded=False):
        label = st.text_input("Rótulo do contexto", ctx["default_label"], key=f"label_{ctx['key']}")
        context_labels[ctx["key"]] = label
        data[ctx["key"]] = {}
        for run in RUNS:
            up = st.file_uploader(run["label"], type=["csv"], key=f"up_{ctx['key']}_{run['key']}")
            real_path = f"data/real_runs/{ctx['key']}_{run['key']}.csv"
            sample_path = f"data/sample_runs/{ctx['key']}_{run['key']}.csv"
            if up is not None:
                try:
                    data[ctx["key"]][run["key"]] = (load_csv(up.getvalue()), "upload")
                except ValueError as e:
                    st.error(f"{run['label']}: {e}")
                    data[ctx["key"]][run["key"]] = (None, None)
            else:
                loaded = False
                for path, src in [(real_path, "real"), (sample_path, "sample")]:
                    if loaded:
                        break
                    try:
                        with open(path, "rb") as f:
                            data[ctx["key"]][run["key"]] = (load_csv(f.read()), src)
                            loaded = True
                    except FileNotFoundError:
                        continue
                if not loaded:
                    data[ctx["key"]][run["key"]] = (None, None)
            sources_used.add(data[ctx["key"]][run["key"]][1])
            st.caption(f"→ {SOURCE_LABEL.get(data[ctx['key']][run['key']][1], 'indisponível')}")

if sources_used == {"sample"}:
    st.sidebar.warning(
        "Mostrando **dados de exemplo sintéticos** — nem upload nem "
        "data/real_runs/ encontrados. Rode o experimento (EXPERIMENTO.md) "
        "ou suba os CSVs manualmente.")
elif "real" in sources_used and "upload" not in sources_used:
    st.sidebar.info(
        "Mostrando o **teste real reduzido** rodado localmente (20k "
        "requisições/coletor). Suba um CSV em qualquer slot pra substituir "
        "por outra rodada (ex.: os 200k completos).")

# domínios: união de todos os arquivos carregados
all_domains = set()
for ctx in CONTEXTS:
    for run in RUNS:
        df, _ = data[ctx["key"]][run["key"]]
        if df is not None:
            all_domains |= set(df["domain"].unique())
domains = sorted(all_domains)
if not domains:
    st.error("Nenhum dado disponível (nem real, nem de exemplo). Verifique os arquivos.")
    st.stop()
colors = domain_colors(domains)

st.sidebar.header("Parâmetros estatísticos")
confidence = st.sidebar.select_slider("Nível de confiança", options=[0.90, 0.95, 0.99],
                                        value=0.95, format_func=lambda v: f"{int(v*100)}%")
ci_method = st.sidebar.radio("Método de IC para latência", ["CLT (t-Student)", "Bootstrap"], index=0)
moe_cap = st.sidebar.slider("MOE% considerada 'confiança perdida' (satura o score)", 5, 50, 20)

st.sidebar.header("Score composto (throughput vs. confiança)")
alpha = st.sidebar.slider("Peso: performance (1.0) vs. confiança (0.0)", 0.0, 1.0, 0.5, 0.05)

st.sidebar.subheader("Importância de cada sinal (0–1)")
importance = {d: st.sidebar.slider(f"Importância — {d}", 0.0, 1.0, default_importance(d), 0.05,
                                     key=f"imp_{d}") for d in domains}

# ---------------------------------------------------------------------------
# Cálculo central: para cada (contexto, run, domínio) -> estatísticas reais,
# usando o run "deterministico" daquele contexto como referência/baseline
# ---------------------------------------------------------------------------
def compute_comparison():
    rows = []
    for ctx in CONTEXTS:
        ck = ctx["key"]
        det_df, _ = data[ck]["deterministico"]
        for run in RUNS:
            df, is_sample = data[ck][run["key"]]
            if df is None:
                continue
            for d in domains:
                sub = df[df["domain"] == d]
                n = len(sub)
                base_sub = det_df[det_df["domain"] == d] if det_df is not None else None
                base_n = len(base_sub) if base_sub is not None else np.nan
                base_mean = base_sub["latency_ms"].mean() if base_sub is not None and len(base_sub) else np.nan
                base_err = base_sub["is_error"].mean() if base_sub is not None and len(base_sub) else np.nan
                if n >= 2:
                    lat = mean_ci(sub["latency_ms"].values, confidence, ci_method)
                else:
                    lat = {"mean": np.nan, "halfwidth": np.nan}
                err = st_stats.wilson_ci(int(sub["is_error"].sum()), n, confidence)
                lat_moe = st_stats.relative_moe(lat["halfwidth"], base_mean if not np.isnan(base_mean) else lat["mean"])
                err_moe = st_stats.relative_moe(err["halfwidth"], base_err if not np.isnan(base_err) else err["p"])
                eff_rate = n / base_n if base_n else np.nan
                rows.append({
                    "contexto": context_labels[ck], "contexto_key": ck,
                    "run": run["key"], "run_label": run["label"],
                    "domain": d, "n": n, "taxa_efetiva": eff_rate,
                    "reducao_exportacao": (1 - eff_rate) if not np.isnan(eff_rate) else np.nan,
                    "latencia_media_ms": lat["mean"], "lat_moe_pct": lat_moe,
                    "taxa_erro": err["p"], "err_moe_pct": err_moe,
                })
    return pd.DataFrame(rows)


cmp_df = compute_comparison()
if cmp_df.empty:
    st.error("Nenhum dado pôde ser processado — confira os CSVs carregados.")
    st.stop()

# ---------------------------------------------------------------------------
# Navegação de topo (estrutura exigida pelo CP1)
# ---------------------------------------------------------------------------
top_home, top_qualif, top_skills, top_present, top_analysis = st.tabs(
    ["Quem sou eu", "Minhas Qualificações", "Skills", "Apresentação", "Análise de Dados"])

with top_home:
    st.title("Luigi")
    st.caption(
        "Rascunho gerado com o Claude a partir do que já conversamos — "
        "revise nomes, datas e detalhes antes de entregar.")
    st.subheader("Platform Engineer Júnior · Observabilidade")
    st.markdown(
        """
Atuo como **Platform Engineer Júnior com foco em Observabilidade** na
**PagBank**, fintech brasileira, onde trabalho com monitoramento,
instrumentação e confiabilidade de sistemas — hoje minha principal frente
é configurar e evoluir testes sintéticos e monitoramento (Datadog,
Splunk/LDAP) e aprofundar instrumentação manual com OpenTelemetry em
Java e Python.

Antes de migrar para observabilidade e plataforma, trabalhei com
**sistemas embarcados** (incluindo NVIDIA Jetson) na OptDriven — essa
base de hardware/baixo nível ajuda no jeito como penso sobre performance
e trade-offs em sistemas distribuídos hoje.

Sou de **São Bernardo do Campo**, Grande São Paulo, e curso atualmente
Data Science e Ciência da Computação Aplicada na **FIAP**, unindo minha
atuação profissional em observabilidade com fundamentos de estatística e
análise de dados — este dashboard é exatamente esse cruzamento: um
laboratório real de sampling em coletores OTel, analisado com os
conceitos estatísticos da disciplina.

Meu momento atual é de **construir as bases**: leitura técnica contínua
(Learning OpenTelemetry, Observability Engineering, Designing
Data-Intensive Applications, o livro de SRE do Google) combinada com um
laboratório prático por semana — este projeto é um desses labs.

🔗 [LinkedIn — Luigi Mendes Cabrini](https://www.linkedin.com/in/luigi-mendes-cabrini-775907349)
        """)

with top_qualif:
    st.title("Minhas Qualificações")
    st.subheader("Formação")
    st.markdown(
        """
- **FIAP** — Data Science e Ciência da Computação Aplicada (em curso).
  Disciplinas cursadas incluem Estatística, Banco de Dados, Java, Redes,
  Programação Dinâmica, Agile e 3D.
        """)
    st.subheader("Experiência profissional")
    st.markdown(
        """
- **PagBank** — Platform Engineer Júnior, Observabilidade *(atual)*
  Configuração de testes sintéticos (Datadog) para monitoramento de
  login Splunk/LDAP, com deployment em localização privada; estudo e
  aplicação de instrumentação manual com OpenTelemetry.
- **OptDriven** — Sistemas embarcados *(anterior)*
  Desenvolvimento em Python para sistemas embarcados, incluindo
  plataformas NVIDIA Jetson.
        """)
    st.subheader("Projetos acadêmicos e pessoais")
    st.markdown(
        """
- **BrasilWatch AI / BrasilFire** (FIAP Global Solution) — previsão de
  incêndios florestais no Brasil usando YOLOv8, LSTM, TimescaleDB/PostGIS
  e dados públicos brasileiros.
- **Este dashboard** — laboratório real de sampling em coletores
  OpenTelemetry (demo-app instrumentada, 6 pipelines de coletor,
  gerador de carga, análise estatística de confiança vs. performance).
- **Atlas** — plataforma pessoal de automação inspirada em conceitos de
  Kubernetes (bot Telegram + dashboard web).
- Contribuição em andamento a um **datasource Grafana para FIWARE**
  (NGSI-v2 + STH-Comet), com foco em portfólio e open source.
        """)
    st.subheader("Idiomas")
    st.markdown(
        """
- **Português** — nativo
- **Inglês** — TOEFL B2 geral (C1 em listening e reading); acompanha
  conteúdo técnico e livros diretamente em inglês
        """)

with top_skills:
    st.title("Skills")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Técnicas")
        st.markdown(
            """
- **Linguagens**: Python, Java
- **Observabilidade**: OpenTelemetry (instrumentação manual, OTel
  Collector, pipelines de sampling), Datadog, conceitos de SRE
- **Infra & dados**: Docker / Docker Compose, conceitos de Kubernetes,
  SQL, Git
- **Análise de dados**: pandas, Streamlit, Plotly, estatística aplicada
  (intervalos de confiança, bootstrap, testes de hipótese)
- **Geração de carga**: Locust
            """)
    with c2:
        st.subheader("Comportamentais")
        st.markdown(
            """
- Autoestudo estruturado (leitura técnica contínua + um laboratório
  prático por semana)
- Comunicação técnica em português e inglês
- Mentoria — dá suporte a uma bolsista de iniciação científica em
  projeto de pesquisa
- Aprendizado por construção: prefere aprender implementando (ex.:
  laboratórios como este) a só ler teoria
            """)
    st.info(
        "As skills e experiências acima vieram do histórico de conversas "
        "com o Claude — revise antes da entrega e ajuste o que não estiver "
        "atualizado.")

with top_present:
    NAVY, DEEPBLUE, TEAL = "#21295C", "#065A82", "#1C7293"
    LIGHT_BLUE, OFFWHITE = "#CADCFC", "#F5F7FA"

    st.markdown(
        f"""
        <style>
        .slide-navy {{background:{NAVY}; color:white; padding:2rem 2.2rem;
            border-radius:0.6rem; margin-bottom:1rem;}}
        .slide-navy h1, .slide-navy h2 {{color:white;}}
        .slide-navy .kicker {{color:#8FA8D6; letter-spacing:2px; font-weight:600;
            font-size:0.85rem; text-transform:uppercase;}}
        .slide-navy .subtitle {{color:{LIGHT_BLUE}; font-size:1.05rem; margin-top:0.4rem;}}
        .quote-box {{background:{DEEPBLUE}; color:white; padding:1.2rem 1.4rem;
            border-radius:0.5rem; font-style:italic; height:100%;}}
        .stat-tile {{background:{OFFWHITE}; border:1px solid #E3E7ED; border-radius:0.5rem;
            padding:1rem; text-align:center;}}
        .stat-tile .value {{font-size:1.6rem; font-weight:700; color:{DEEPBLUE};}}
        .stat-tile .label {{color:#5A6472; font-size:0.85rem; margin-top:0.3rem;}}
        .method-card {{border-radius:0.5rem; padding:1rem; color:white; height:100%;}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Apresentação coringa — plano B caso o dashboard fique indisponível "
        "na correção; o conteúdo é o mesmo, em formato de slides.")

    # Slide 1 — Título
    st.markdown(
        f"""
        <div class="slide-navy">
        <div class="kicker">CP1 · DASHBOARD PROFISSIONAL · FIAP</div>
        <h1>Intervalos de Confiança & Sampling<br>em Coletores OTel</h1>
        <div class="subtitle">Trabalho de Estatística — laboratório real de sampling em
        coletores OpenTelemetry, analisado com os conceitos da disciplina.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Slide 2 — Quem sou eu
    st.subheader("Quem sou eu")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Luigi Mendes Cabrini**")
        st.caption("Platform Engineer Júnior · Observabilidade")
        st.markdown(
            """
- **PagBank** (fintech brasileira) — monitoramento, instrumentação e confiabilidade
- Antes: sistemas embarcados (NVIDIA Jetson) na OptDriven
- Cursa Data Science e Ciência da Computação Aplicada na FIAP
- São Bernardo do Campo, Grande São Paulo
- 🔗 [linkedin.com/in/luigi-mendes-cabrini-775907349](https://www.linkedin.com/in/luigi-mendes-cabrini-775907349)
            """)
    with c2:
        st.markdown(
            '<div class="quote-box">“Este dashboard é exatamente esse cruzamento: '
            'um laboratório real de sampling em coletores OTel, analisado com os '
            'conceitos estatísticos da disciplina.”</div>',
            unsafe_allow_html=True)

    st.divider()

    # Slide 3 — Qualificações & Skills
    st.subheader("Minhas Qualificações & Skills")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Formação & Experiência**")
        st.markdown(
            """
- FIAP — Data Science e Ciência da Computação Aplicada
- PagBank — testes sintéticos Datadog, monitoramento Splunk/LDAP
- OptDriven — Python para sistemas embarcados (Jetson)
- Projeto FIAP Global Solution: BrasilWatch AI / BrasilFire
            """)
    with c2:
        st.markdown("**Skills técnicas**")
        st.markdown(
            """
- Python, Java
- OpenTelemetry (instrumentação manual, OTel Collector, sampling)
- Docker / Docker Compose, conceitos de Kubernetes
- Estatística aplicada (IC, bootstrap), pandas, Plotly, Streamlit
- Locust, SQL, Git
            """)
    st.caption("Inglês: TOEFL B2 geral, C1 em listening/reading · Português nativo")

    st.divider()

    # Slide 4 — O problema
    st.subheader("O problema: sampling em observabilidade")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            """
- Capturar 100% dos spans/traces em produção é caro: CPU do coletor, rede,
  custo de ingestão e armazenamento no backend.
- Reduzir o sampling economiza recursos — mas reduz o tamanho da amostra, o
  que aumenta a margem de erro (MOE) das métricas estimadas (SE = σ/√n).
- Sinais diferentes têm importância diferente: uma transação PIX que falha
  não pode "sumir" por causa do sampling, mas uma página genérica de site
  tolera taxas mais agressivas.
            """)
    with c2:
        s1, s2 = st.columns(2)
        s1.markdown('<div class="stat-tile"><div class="value">IC</div>'
                        '<div class="label">Intervalo de Confiança<br>(CLT, bootstrap, Wilson)</div></div>',
                        unsafe_allow_html=True)
        s2.markdown('<div class="stat-tile"><div class="value">Score</div>'
                        '<div class="label">throughput × confiança<br>× importância do sinal</div></div>',
                        unsafe_allow_html=True)

    st.divider()

    # Slide 5 — Metodologia
    st.subheader("Metodologia: o experimento real")
    st.caption("2 contextos × 3 coletores OTel, rodando de verdade (docker compose / execução local)")
    m1, m2, m3 = st.columns(3)
    cards = [
        (m1, "Determinístico", "100% do tráfego\n(baseline / ground truth)", DEEPBLUE),
        (m2, "Sweet spot", "taxa balanceada\n(head: 15% · tail: erro+cauda sempre + base 15%)", TEAL),
        (m3, "Agressivo", "taxa muito baixa\n(head: 1% · tail: erro+cauda sempre + base 1%)", "#8FA8D6"),
    ]
    for col, title, desc, color in cards:
        col.markdown(
            f'<div class="method-card" style="background:{color};">'
            f'<b>{title}</b><br><span style="font-size:0.85rem;">{desc}</span></div>',
            unsafe_allow_html=True)
    st.markdown(
        """
*Rodado 2×: contexto head-based (probabilistic_sampler) e contexto
tail-based (tail_sampling — erros e cauda p95 sempre mantidos). Demo-app
instrumentada com OpenTelemetry, 4 domínios de sinal (pix, checkout,
site_latency, api_generic), gerador de carga Locust com parada exata na
meta de requisições.*
        """)

    st.divider()

    # Slide 6 — Resultado real
    st.subheader("Resultado real: retenção por contexto")
    st.caption("Teste local já executado (20 mil requisições/coletor) — números reais, não simulados")
    st.table(pd.DataFrame(
        [["Head-based", "20.124 (100%)", "3.016 (15,0%)", "217 (1,1%)"],
         ["Tail-based", "19.968 (100%)", "4.838 (24,2%)", "2.260 (11,3%)"]],
        columns=["Contexto", "Determinístico", "Sweet spot", "Agressivo"],
    ).set_index("Contexto"))
    st.markdown(
        """
*O tail-based retém muito mais que a taxa-base nominal (1% → 11,3% no
agressivo) porque erros e picos de latência são sempre mantidos — é isso
que preserva a confiança nos sinais críticos mesmo sob sampling agressivo.*
        """)

    st.divider()

    # Slide 7 — Trade-off
    st.subheader("Trade-off: performance × confiança")
    st.code("score = α·ganho_throughput − (1−α)·importância·penalidade_confiança", language="text")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            """
- **ganho** = 1 − taxa de sampling (economia no pipeline de observabilidade)
- **penalidade** = MOE% relativa, saturada em um limite configurável
- **importância** = peso do sinal (PIX = 1,0 · latência genérica = 0,2)
- **α** = peso performance × confiança, ajustável no dashboard (aba
  *Sampling dinâmico*)
            """)
    with c2:
        st.markdown(
            f'<div class="method-card" style="background:{NAVY};">'
            'Sinais críticos (PIX) →<br><b>sweet spot mais alto</b><br><br>'
            'Sinais tolerantes (latência genérica) →<br><b>sweet spot mais agressivo</b>'
            '</div>', unsafe_allow_html=True)

    st.divider()

    # Slide 8 — Conclusão
    st.markdown(
        f"""
        <div class="slide-navy">
        <h2>Conclusão</h2>
        <ul>
        <li>Sampling agressivo economiza pipeline, mas custa confiança estatística
        de forma previsível (~1/√n)</li>
        <li>Tail-based preserva confiança em sinais raros/críticos mesmo sob taxas
        baixas — head-based não</li>
        <li>O sweet spot certo depende da importância do sinal, não é um número
        único para o sistema inteiro</li>
        <li>Todo o pipeline (demo-app instrumentada, 6 coletores, gerador de carga,
        conversão OTLP→CSV) está reproduzível — ver EXPERIMENTO.md</li>
        </ul>
        <div class="subtitle">Repositório: github.com/CabriniJr/otelabs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_analysis:
    st.title("Intervalos de Confiança & Sampling em Coletores OTel")
    st.caption(
        "Experimento real (docker compose): 2 contextos x 3 coletores — "
        "determinístico, sweet spot e agressivo. Comparação de ganho de "
        "performance vs. perda de confiança estatística, por sinal.")

    det_total = sum(len(data[c["key"]]["deterministico"][0]) for c in CONTEXTS
                        if data[c["key"]]["deterministico"][0] is not None)
    k1, k2, k3 = st.columns(3)
    k1.metric("Requisições (determinístico, ambos contextos)", f"{det_total:,}".replace(",", "."))
    k2.metric("Domínios/sinais", len(domains))
    k3.metric("Contextos carregados", sum(1 for c in CONTEXTS if data[c["key"]]["deterministico"][0] is not None))

    (tab_overview, tab_compare, tab_tradeoff, tab_headtail, tab_dynamic,
        tab_pipeline, tab_method) = st.tabs(
        ["Visão geral", "Comparação dos coletores", "Trade-off & sweet spot",
         "Head vs. Tail", "Sampling dinâmico", "Pipeline do experimento", "Metodologia"])

    # ---------------------------------------------------------------------------
    # TAB 1 — Visão geral (baseline determinístico, por contexto)
    # ---------------------------------------------------------------------------
    with tab_overview:
        ctx_pick = st.selectbox("Contexto", [c["key"] for c in CONTEXTS],
                                    format_func=lambda k: context_labels[k])
        det_df, src = data[ctx_pick]["deterministico"]
        if det_df is None:
            st.warning("Sem dado determinístico carregado para este contexto.")
        else:
            st.caption(f"Fonte: {SOURCE_LABEL.get(src, src)}")
            st.subheader(f"Baseline determinístico — {context_labels[ctx_pick]}")
            rows = []
            for d in domains:
                sub = det_df[det_df["domain"] == d]
                if len(sub) == 0:
                    continue
                lat = mean_ci(sub["latency_ms"].values, confidence, ci_method)
                err = st_stats.wilson_ci(int(sub["is_error"].sum()), len(sub), confidence)
                rows.append({"domain": d, "n": len(sub), "latencia_media_ms": lat["mean"],
                                "ic_lat_lo": lat["lo"], "ic_lat_hi": lat["hi"],
                                "taxa_erro": err["p"], "ic_erro_lo": err["lo"], "ic_erro_hi": err["hi"]})
            ov = pd.DataFrame(rows)

            fig = go.Figure()
            for _, r in ov.iterrows():
                fig.add_trace(go.Scatter(x=[r["ic_lat_lo"], r["ic_lat_hi"]], y=[r["domain"]] * 2,
                                            mode="lines", line=dict(color=colors[r["domain"]], width=6),
                                            showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=[r["latencia_media_ms"]], y=[r["domain"]], mode="markers",
                                            marker=dict(color=colors[r["domain"]], size=12,
                                            line=dict(color="white", width=1)),
                                            name=r["domain"],
                                            hovertemplate=f"{r['domain']}<br>média: %{{x:.1f}} ms<extra></extra>"))
            fig = base_layout(fig, f"Latência média por sinal — IC {int(confidence*100)}%",
                                "Latência média (ms)", "", "Sinal")
            st.plotly_chart(fig, width='stretch')

            fig2 = go.Figure()
            for d in domains:
                sub = det_df[det_df["domain"] == d]["latency_ms"]
                if len(sub):
                    fig2.add_trace(go.Box(y=sub, name=d, marker_color=colors[d], boxpoints=False))
            fig2 = base_layout(fig2, "Distribuição de latência por sinal", "", "Latência (ms)")
            st.plotly_chart(fig2, width='stretch')

            with st.expander("Ver tabela"):
                show = ov.copy()
                for c in ["latencia_media_ms", "ic_lat_lo", "ic_lat_hi"]:
                    show[c] = show[c].round(1)
                for c in ["taxa_erro", "ic_erro_lo", "ic_erro_hi"]:
                    show[c] = (show[c] * 100).round(2)
                st.dataframe(show, width='stretch')

    # ---------------------------------------------------------------------------
    # TAB 2 — Comparação real dos 3 coletores, lado a lado por contexto
    # ---------------------------------------------------------------------------
    with tab_compare:
        st.subheader("Determinístico vs. sweet spot vs. agressivo — dados reais")
        st.caption(
            "Cada barra usa o dado que o respectivo coletor realmente exportou "
            "naquele run (nada é simulado aqui). A MOE é relativa ao "
            "determinístico do mesmo contexto.")

        metric = st.radio("Métrica", ["MOE de latência (%)", "MOE de taxa de erro (%)", "Redução de exportação (%)"],
                            horizontal=True)
        metric_col = {"MOE de latência (%)": "lat_moe_pct",
                        "MOE de taxa de erro (%)": "err_moe_pct",
                        "Redução de exportação (%)": "reducao_exportacao"}[metric]
        scale = 100 if metric_col != "reducao_exportacao" else 100

        fig3 = make_subplots(rows=1, cols=len(CONTEXTS),
                                subplot_titles=[context_labels[c["key"]] for c in CONTEXTS],
                                shared_yaxes=True)
        for i, ctx in enumerate(CONTEXTS, start=1):
            sub = cmp_df[cmp_df["contexto_key"] == ctx["key"]]
            for d in domains:
                s2 = sub[sub["domain"] == d].set_index("run").reindex(RUN_ORDER)
                fig3.add_trace(go.Bar(x=[RUN_LABEL[r] for r in RUN_ORDER], y=s2[metric_col] * scale,
                                        name=d, marker_color=colors[d],
                                        showlegend=(i == 1)), row=1, col=i)
        fig3.update_layout(barmode="group")
        fig3 = base_layout(fig3, f"{metric} — determinístico x sweet spot x agressivo", "", metric, "Sinal")
        st.plotly_chart(fig3, width='stretch')

        with st.expander("Ver tabela completa"):
            show = cmp_df.copy()
            show["taxa_efetiva"] = (show["taxa_efetiva"] * 100).round(2)
            show["reducao_exportacao"] = (show["reducao_exportacao"] * 100).round(1)
            show["latencia_media_ms"] = show["latencia_media_ms"].round(1)
            show["lat_moe_pct"] = show["lat_moe_pct"].round(2)
            show["taxa_erro"] = (show["taxa_erro"] * 100).round(3)
            show["err_moe_pct"] = show["err_moe_pct"].round(1)
            st.dataframe(show.drop(columns=["contexto_key"]), width='stretch')

    # ---------------------------------------------------------------------------
    # TAB 3 — Trade-off analítico + overlay dos 3 pontos reais medidos
    # ---------------------------------------------------------------------------
    with tab_tradeoff:
        st.subheader("Curva teórica de MOE vs. taxa — com os pontos reais do experimento")
        ctx_pick2 = st.selectbox("Contexto", [c["key"] for c in CONTEXTS],
                                    format_func=lambda k: context_labels[k], key="ctx_tradeoff")
        det_df, _ = data[ctx_pick2]["deterministico"]
        if det_df is None:
            st.warning("Sem dado determinístico carregado para este contexto.")
        else:
            st.caption(
                "A curva usa σ observado no determinístico e a fórmula SE=σ/√n "
                "para extrapolar a MOE em qualquer taxa (sem reamostrar). Os "
                "marcadores ★ são os pontos REAIS medidos nos 3 coletores.")

            rate_grid = np.geomspace(0.005, 1.0, 60)
            score_rows = []
            for d in domains:
                sub = det_df[det_df["domain"] == d]
                if len(sub) < 2:
                    continue
                n_base, mean_base, std_base = len(sub), sub["latency_ms"].mean(), sub["latency_ms"].std(ddof=1)
                p_base = sub["is_error"].mean()
                for r in rate_grid:
                    n_r = max(2, int(round(n_base * r)))
                    se = std_base / np.sqrt(n_r)
                    t_crit = sp_stats.t.ppf(1 - (1 - confidence) / 2, df=n_r - 1)
                    lat_moe = st_stats.relative_moe(t_crit * se, mean_base)
                    err_ci = st_stats.wilson_ci(int(round(p_base * n_r)), n_r, confidence)
                    err_moe = st_stats.relative_moe(err_ci["halfwidth"], p_base)
                    penalty = max(st_stats.confidence_penalty(lat_moe, moe_cap),
                                    st_stats.confidence_penalty(err_moe, moe_cap))
                    gain = st_stats.throughput_gain(r)
                    score = st_stats.composite_score(gain, penalty, importance[d], alpha)
                    score_rows.append({"domain": d, "rate": r, "score": score})
            score_df = pd.DataFrame(score_rows)

            fig4 = go.Figure()
            sweet_spots = []
            for d in domains:
                sub = score_df[score_df["domain"] == d]
                if sub.empty:
                    continue
                fig4.add_trace(go.Scatter(x=sub["rate"], y=sub["score"], mode="lines",
                                            name=d, line=dict(color=colors[d], width=2.5)))
                best_rate, best_score = st_stats.find_sweet_spot(sub["rate"].values, sub["score"].values)
                sweet_spots.append({"domain": d, "taxa_ideal": best_rate, "score": best_score})

                real = cmp_df[(cmp_df["contexto_key"] == ctx_pick2) & (cmp_df["domain"] == d)]
                for _, r in real.iterrows():
                    if np.isnan(r["taxa_efetiva"]):
                        continue
                    fig4.add_trace(go.Scatter(x=[r["taxa_efetiva"]], y=[
                        st_stats.composite_score(
                            st_stats.throughput_gain(r["taxa_efetiva"]),
                            max(st_stats.confidence_penalty(r["lat_moe_pct"], moe_cap),
                                st_stats.confidence_penalty(r["err_moe_pct"], moe_cap)),
                            importance[d], alpha)],
                        mode="markers", marker=dict(color=colors[d], size=13, symbol="star",
                        line=dict(color="white", width=1)), showlegend=False,
                        hovertemplate=f"{d} — {r['run_label']}<br>taxa real: %{{x:.1%}}<extra></extra>"))
            fig4.add_hline(y=0, line=dict(color=INK_MUTED, width=1, dash="dot"))
            fig4.update_xaxes(type="log", tickformat=".0%")
            fig4 = base_layout(fig4, "Score composto vs. taxa (linha = teórico, ★ = medido nos 3 coletores)",
                                "Taxa de sampling (log)", "Score", "Sinal")
            st.plotly_chart(fig4, width='stretch')

            if sweet_spots:
                st.markdown("**Sweet spot teórico por sinal** (dado os pesos atuais):")
                sweet_df = pd.DataFrame(sweet_spots)
                cols = st.columns(len(sweet_df))
                for c, (_, r) in zip(cols, sweet_df.iterrows()):
                    c.metric(r["domain"], f"{r['taxa_ideal']*100:.1f}%", f"score {r['score']:.2f}")

    # ---------------------------------------------------------------------------
    # TAB 4 — Head-based vs. Tail-based, lado a lado
    # ---------------------------------------------------------------------------
    with tab_headtail:
        st.subheader("Head-based vs. tail-based — mesma taxa nominal, confiança diferente")
        st.caption(
            "Compara sweet spot e agressivo dos dois contextos. A vantagem "
            "esperada do tail-based aparece em domínios com poucos erros/cauda "
            "rara: mesmo no agressivo, erros continuam retidos, então a MOE de "
            "taxa de erro deve subir muito menos do que no head-based.")

        focus_runs = st.multiselect("Coletores para comparar", ["sweet_spot", "agressivo"],
                                        default=["sweet_spot", "agressivo"],
                                        format_func=lambda k: RUN_LABEL[k])
        sub = cmp_df[cmp_df["run"].isin(focus_runs)]

        fig5 = make_subplots(rows=1, cols=2, subplot_titles=["MOE de latência (%)", "MOE de taxa de erro (%)"])
        for j, col in enumerate(["lat_moe_pct", "err_moe_pct"], start=1):
            for d in domains:
                s2 = sub[sub["domain"] == d]
                x = [f"{r['contexto']}<br>{RUN_LABEL[r['run']]}" for _, r in s2.iterrows()]
                fig5.add_trace(go.Bar(x=x, y=s2[col], name=d, marker_color=colors[d],
                                        showlegend=(j == 1)), row=1, col=j)
        fig5.update_layout(barmode="group")
        fig5 = base_layout(fig5, "", "", "MOE relativa (%)", "Sinal")
        st.plotly_chart(fig5, width='stretch')

        with st.expander("Ver tabela"):
            show = sub.copy()
            show["reducao_exportacao"] = (show["reducao_exportacao"] * 100).round(1)
            show["lat_moe_pct"] = show["lat_moe_pct"].round(2)
            show["err_moe_pct"] = show["err_moe_pct"].round(1)
            st.dataframe(show[["contexto", "run_label", "domain", "n", "reducao_exportacao",
                                "lat_moe_pct", "err_moe_pct"]], width='stretch')

    # ---------------------------------------------------------------------------
    # TAB 5 — Sampling dinâmico (what-if sobre um determinístico)
    # ---------------------------------------------------------------------------
    with tab_dynamic:
        st.subheader("What-if: sampling dinâmico por tag/domínio")
        st.caption(
            "Explora, sobre um baseline 100% carregado, uma 4ª config "
            "hipotética com taxa diferente por domínio (ex.: pix sempre alto, "
            "tráfego genérico agressivo) — útil para planejar a próxima "
            "rodada do experimento.")

        ctx_pick3 = st.selectbox("Contexto (baseline 100%)", [c["key"] for c in CONTEXTS],
                                    format_func=lambda k: context_labels[k], key="ctx_dyn")
        det_df, _ = data[ctx_pick3]["deterministico"]
        if det_df is None:
            st.warning("Sem dado determinístico carregado para este contexto.")
        else:
            from utils import sampling as smp
            cols = st.columns(len(domains))
            rate_by_domain = {d: c.slider(d, 0.01, 1.0, max(0.05, default_importance(d)), 0.01, key=f"rate_{d}")
                                for d, c in zip(domains, cols)}
            dyn = smp.dynamic_domain_sample(det_df, rate_by_domain)

            rows = []
            for d in domains:
                base_sub = det_df[det_df["domain"] == d]
                dyn_sub = dyn[dyn["domain"] == d]
                if len(base_sub) == 0:
                    continue
                lat = mean_ci(dyn_sub["latency_ms"].values, confidence, ci_method) if len(dyn_sub) >= 2 else {"mean": np.nan, "halfwidth": np.nan}
                lat_moe = st_stats.relative_moe(lat["halfwidth"], base_sub["latency_ms"].mean())
                gain = 1 - len(dyn_sub) / len(base_sub)
                penalty = st_stats.confidence_penalty(lat_moe, moe_cap)
                score = st_stats.composite_score(gain, penalty, importance[d], alpha)
                rows.append({"domain": d, "taxa": rate_by_domain[d], "n": len(dyn_sub),
                                "reducao": gain, "lat_moe_pct": lat_moe, "score": score})
            dyn_df = pd.DataFrame(rows)

            total_reduction = 1 - len(dyn) / len(det_df)
            weighted_score = float(np.average(dyn_df["score"], weights=[importance[d] for d in dyn_df["domain"]])) if len(dyn_df) else np.nan
            k1, k2, k3 = st.columns(3)
            k1.metric("Redução total", f"{total_reduction*100:.1f}%")
            k2.metric("Requisições mantidas", f"{len(dyn):,}".replace(",", "."))
            k3.metric("Score ponderado", f"{weighted_score:.2f}" if not np.isnan(weighted_score) else "—")

            fig6 = go.Figure(go.Bar(x=dyn_df["domain"], y=dyn_df["reducao"] * 100,
                                        marker_color=[colors[d] for d in dyn_df["domain"]]))
            fig6 = base_layout(fig6, "Redução de exportação por domínio (config. dinâmica atual)", "", "Redução (%)")
            st.plotly_chart(fig6, width='stretch')

            with st.expander("Ver tabela"):
                show = dyn_df.copy()
                show["taxa"] = (show["taxa"] * 100).round(1)
                show["reducao"] = (show["reducao"] * 100).round(1)
                show["lat_moe_pct"] = show["lat_moe_pct"].round(2)
                show["score"] = show["score"].round(3)
                st.dataframe(show, width='stretch')

    # ---------------------------------------------------------------------------
    # TAB 6 — Pipeline do experimento (arquitetura + comandos de referência)
    # ---------------------------------------------------------------------------
    with tab_pipeline:
        st.subheader("Arquitetura do experimento real")
        st.caption(
            "6 pares app+coletor (2 contextos x 3 configs), cada um isolado "
            "num par de serviços do docker compose. Passo a passo completo em "
            "EXPERIMENTO.md — aqui vai a referência rápida.")

        st.markdown(
            """
    ```
    Locust (seu host)  ──HTTP──▶  demo-app (FastAPI + OTel SDK)  ──OTLP──▶  OTel Collector  ──file exporter──▶  traces.json
         200k reqs                 1 span por request                       sampler específico            (1 por par app+coletor)
      mix 15/20/35/30%             tag business.domain                      da combinação
                                    status ERROR nos erros                  contexto x config
    ```
            """)

        pipeline_rows = [
            {"contexto": "head_based", "config": "deterministico", "sampler": "probabilistic_sampler 100%",
                "coletor": "otel-collector-head-based-deterministico", "app": "app-head-based-deterministico", "porta": 8001},
            {"contexto": "head_based", "config": "sweet_spot", "sampler": "probabilistic_sampler 15%",
                "coletor": "otel-collector-head-based-sweet-spot", "app": "app-head-based-sweet-spot", "porta": 8002},
            {"contexto": "head_based", "config": "agressivo", "sampler": "probabilistic_sampler 1%",
                "coletor": "otel-collector-head-based-agressivo", "app": "app-head-based-agressivo", "porta": 8003},
            {"contexto": "tail_based", "config": "deterministico", "sampler": "tail_sampling: always_sample",
                "coletor": "otel-collector-tail-based-deterministico", "app": "app-tail-based-deterministico", "porta": 8004},
            {"contexto": "tail_based", "config": "sweet_spot", "sampler": "tail_sampling: erro + cauda p95 + base 15%",
                "coletor": "otel-collector-tail-based-sweet-spot", "app": "app-tail-based-sweet-spot", "porta": 8005},
            {"contexto": "tail_based", "config": "agressivo", "sampler": "tail_sampling: erro + cauda p95 + base 1%",
                "coletor": "otel-collector-tail-based-agressivo", "app": "app-tail-based-agressivo", "porta": 8006},
        ]
        st.dataframe(pd.DataFrame(pipeline_rows), width='stretch', hide_index=True)

        st.markdown("**Comandos de referência** (uma combinação por vez — ver justificativa em EXPERIMENTO.md):")
        st.code(
            "# 1) sobe o par app+coletor\n"
            "docker compose up --build -d otel-collector-<contexto>-<config> app-<contexto>-<config>\n\n"
            "# 2) gera os 200k requests (mix 15/20/35/30%, para sozinho ao bater a meta)\n"
            "locust -f locustfile.py --host=http://localhost:<porta> --headless -u 150 -r 30 -t 30m\n\n"
            "# 3) espera o coletor esvaziar o buffer e derruba só esse par\n"
            "sleep 10 && docker compose stop otel-collector-<contexto>-<config> app-<contexto>-<config>\n\n"
            "# depois das 6 execuções — converte tudo pro CSV que este app lê\n"
            "python scripts/otel_export_to_csv.py --all",
            language="bash")

        with st.expander("O que cada peça faz"):
            st.markdown(
                """
    - **`demo-app/`** — FastAPI com 4 rotas (`pix`, `checkout`, `/`, `/api/generic`),
      cada uma virando 1 span OTel com o atributo `business.domain` e status
      `ERROR` nos erros simulados. Os parâmetros de latência/erro são os
      mesmos de `scripts/generate_sample_data.py`.
    - **`otel-config/*.yaml`** — as 6 pipelines (`receiver otlp` →
      `probabilistic_sampler` ou `tail_sampling` → `file exporter`).
    - **`locustfile.py`** — pesos 15/20/35/30% por domínio e um listener que
      para o teste sozinho assim que bate `TOTAL_REQUESTS` (usa
      `LOCUST_TOTAL_REQUESTS` pra testar pequeno primeiro).
    - **`scripts/otel_export_to_csv.py`** — lê o `traces.json` de cada
      coletor e gera o CSV no schema deste app (`trace_id, timestamp,
      domain, latency_ms, is_error`).
                """)

    # ---------------------------------------------------------------------------
    # TAB 7 — Metodologia
    # ---------------------------------------------------------------------------
    with tab_method:
        st.subheader("Metodologia")
        st.markdown(
            """
    **Desenho do experimento.** Docker compose com 2 contextos de teste
    (head-based e tail-based). Em cada contexto, 3 execuções idênticas do
    mesmo teste de carga são recebidas por 3 coletores OTel configurados de
    forma diferente:
    - **Determinístico** — `probabilistic_sampler` a 100% (ground truth do
      contexto: todo o tráfego é exportado).
    - **Sweet spot** — a taxa (head-based) ou taxa-base (tail-based)
      considerada o melhor equilíbrio entre performance e confiança.
    - **Agressivo** — taxa (ou taxa-base) muito baixa, para mostrar o extremo
      de alta performance / baixa confiança.

    Cada coletor exporta seu próprio CSV com o que **de fato** foi exportado
    — nada é simulado dentro do app. O determinístico de cada contexto serve
    como referência (baseline) para calcular a taxa efetiva de retenção e a
    margem de erro relativa dos outros dois coletores daquele mesmo contexto.

    **Por que 2 contextos.** Comparar head-based com tail-based só faz
    sentido nas mesmas condições de carga — por isso o mesmo desenho de 3
    coletores é repetido duas vezes, uma por estratégia de sampling. Isso
    isola o efeito da estratégia (head vs. tail) do efeito da taxa em si.

    **Intervalos de confiança.**
    - Latência (contínua): IC via CLT (t de Student) ou bootstrap percentil
      (`utils/stats.py`).
    - Taxa de erro (proporção): IC de Wilson — estável mesmo com `n` pequeno
      ou proporções perto de 0, o caso comum de taxas de erro baixas.
    - **MOE relativa (%)** = `halfwidth / valor_de_referência`, onde a
      referência é sempre o valor do determinístico do mesmo contexto.

    **Curva teórica vs. pontos reais (aba Trade-off).** A curva de MOE em
    função da taxa usa o `σ` (desvio-padrão) observado no determinístico e a
    fórmula `SE = σ/√n` para extrapolar analiticamente qualquer taxa entre
    0,5% e 100%, sem precisar reamostrar. Os 3 pontos reais medidos (★) são
    sobrepostos a essa curva — se o experimento estiver bem controlado, eles
    devem cair perto da curva teórica; desvios grandes indicam algo
    interessante para discutir no relatório (viés de amostragem, mudança de
    carga entre execuções, etc.).

    **Score composto e sweet spot.**

    ```
    ganho      = 1 − taxa_efetiva_de_sampling
    penalidade = clip(MOE% / limite_MOE%, 0, 1)
    score      = α·ganho − (1−α)·importância·penalidade
    ```

    `α` e a `importância` por sinal são escolhas de negócio (sliders), não
    estatísticas — o app deixa ambos explícitos. Sinais críticos (ex.: PIX)
    amplificam a penalidade de confiança, empurrando o sweet spot para taxas
    mais altas.

    **Nota sobre "throughput".** O sampling não muda a taxa de requisições
    atendida pela aplicação — ele reduz o **volume exportado pelo pipeline de
    observabilidade** (CPU do coletor, rede, custo de ingestão/armazenamento
    no backend). É esse o "ganho de performance" medido aqui.
            """
        )
