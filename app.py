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
# sequencial azul, divergente azul<->vermelho, cores de status) — ajustada
# para o tema escuro nativo do Streamlit (ver .streamlit/config.toml)
# ---------------------------------------------------------------------------
CATEGORICAL = ["#5b9bf2", "#f0895c", "#3fd19e", "#f0b93f",
                "#f295bd", "#3fd63f", "#8b7ae0", "#f2726f"]
DIVERGING = [[0.0, "#f2726f"], [0.5, "#3a3d47"], [1.0, "#5b9bf2"]]
STATUS = {"good": "#3fd63f", "warning": "#f0b93f", "serious": "#f0895c", "critical": "#f2726f"}
INK_PRIMARY = "#f5f5f3"
INK_SECONDARY = "#c9c8c3"
INK_MUTED = "#8b8a86"
GRID = "#30333c"
SURFACE = "#12151c"

DOMAIN_PRIORITY = ["pix", "checkout", "site_latency", "api_generic"]
REQUIRED_COLS = {"trace_id", "timestamp", "domain", "latency_ms", "is_error"}
SOURCE_LABEL = {"upload": "upload manual", "real": "teste real (local, reduzido)",
                    "sample": "exemplo sintético"}

# Score de prioridade P1–P4 (substitui o slider contínuo de importância):
# cada domínio recebe uma prioridade de negócio discreta, e cada nível
# mapeia para um peso fixo de importância no score composto. P1 = sinal
# crítico (ex.: PIX — precisa de sampling mais determinístico/conservador),
# P4 = sinal tolerante (pode sofrer sampling agressivo sem grande perda).
PRIORITY_LEVELS = {"P1": 1.0, "P2": 0.7, "P3": 0.4, "P4": 0.2}
PRIORITY_DESC = {
    "P1": "crítico — precisa de sampling quase determinístico",
    "P2": "importante — tolera sampling moderado",
    "P3": "secundário — tolera sampling mais agressivo",
    "P4": "tolerante — pode sofrer sampling bem agressivo",
}
DEFAULT_PRIORITY = {"pix": "P1", "checkout": "P2", "site_latency": "P3", "api_generic": "P4"}

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
    """Taxa inicial sugerida para o what-if de sampling dinâmico: o peso da
    prioridade padrão do domínio (P1 = 100%, P4 = 20%)."""
    return PRIORITY_LEVELS[DEFAULT_PRIORITY.get(domain, "P3")]


def mean_ci(values, confidence, method):
    if method.startswith("Bootstrap"):
        return st_stats.bootstrap_mean_ci(values, confidence, n_boot=800)
    return st_stats.clt_mean_ci(values, confidence)


def br(n) -> str:
    """Inteiro no formato brasileiro (ponto como separador de milhar)."""
    return f"{int(n):,}".replace(",", ".")


def extrapolate_at_rate(sub, rate, confidence):
    """Extrapola, a partir do baseline determinístico de UM domínio (`sub`),
    quanto de confiança se perde ao amostrar a `rate`.

    Não reamostra: usa o σ e a taxa de erro observados no run 100% e a
    fórmula SE = σ/√n para calcular analiticamente a margem de erro (MOE)
    relativa em qualquer taxa. É a mesma matemática da curva teórica da aba
    "Trade-off & sweet spot", isolada aqui para ser reusada pelos hovers e
    pelo slider da Visão geral.

    Retorna dict com n_amostrado, lat_moe_pct, err_moe_pct, ganho e a
    penalidade de confiança normalizada.
    """
    n_base = len(sub)
    if n_base < 2:
        return None
    mean_base = float(sub["latency_ms"].mean())
    std_base = float(sub["latency_ms"].std(ddof=1))
    p_base = float(sub["is_error"].mean())
    n_r = max(2, int(round(n_base * rate)))
    se = std_base / np.sqrt(n_r)
    t_crit = sp_stats.t.ppf(1 - (1 - confidence) / 2, df=n_r - 1)
    lat_moe = st_stats.relative_moe(t_crit * se, mean_base)
    err_ci = st_stats.wilson_ci(int(round(p_base * n_r)), n_r, confidence)
    err_moe = st_stats.relative_moe(err_ci["halfwidth"], p_base)
    return {"n_base": n_base, "n": n_r, "mean_base": mean_base, "std_base": std_base,
            "p_base": p_base, "lat_moe_pct": lat_moe, "err_moe_pct": err_moe,
            "halfwidth_ms": float(t_crit * se), "t_crit": float(t_crit),
            "gain": st_stats.throughput_gain(rate)}


def rate_series(df, domain, window_s=3, bucket_s=1):
    """Série temporal de rate() normalizado, no espírito do `rate()` do
    PromQL: conta eventos por bucket de tempo, divide pela duração do
    bucket (eventos/s) e suaviza com uma média móvel de `window_s` buckets.

    O eixo X é o tempo DECORRIDO desde o primeiro evento daquele run, não o
    timestamp absoluto: os 6 coletores foram executados um de cada vez
    (ver EXPERIMENTO.md), então só alinhando pelo início é que as curvas
    ficam de fato sobrepostas e comparáveis.

    Retorna a série em eventos/s do run recebido. Quem chama divide pela
    taxa efetiva de retenção daquele coletor — assim as 3 séries ficam na
    MESMA escala (estimativa da taxa real de requisições) e a única
    diferença visível entre elas é a DISPERSÃO: quanto menos amostra, mais
    ruidosa a estimativa, mesmo com a média correta.
    """
    sub = df[df["domain"] == domain]
    if sub.empty:
        return None
    ts = pd.to_datetime(sub["timestamp"], utc=True, format="mixed")
    elapsed = (ts - ts.min()).dt.total_seconds() // bucket_s * bucket_s
    counts = elapsed.value_counts().sort_index()
    if counts.empty:
        return None
    full_idx = np.arange(counts.index.min(), counts.index.max() + bucket_s, bucket_s)
    counts = counts.reindex(full_idx, fill_value=0)
    rate = counts / bucket_s
    return rate.rolling(window_s, min_periods=1).mean()


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

st.sidebar.subheader("Prioridade de cada sinal (P1–P4)")
st.sidebar.caption(
    "A prioridade de negócio do sinal entra no score composto como peso da "
    "penalidade de confiança: P1 (ex.: PIX) amplifica a perda de confiança e "
    "empurra o sweet spot para taxas mais altas; P4 aceita sampling agressivo.")
priority = {}
for d in domains:
    levels = list(PRIORITY_LEVELS)
    default_level = DEFAULT_PRIORITY.get(d, "P3")
    priority[d] = st.sidebar.selectbox(
        f"Prioridade — {d}", levels, index=levels.index(default_level),
        format_func=lambda p: f"{p} · {PRIORITY_DESC[p]}", key=f"prio_{d}")
importance = {d: PRIORITY_LEVELS[priority[d]] for d in domains}

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
# Navegação de topo — 3 áreas. A primeira agrupa, em sub-abas, as três
# seções de portfólio exigidas pelo CP1 (Quem sou eu / Qualificações /
# Skills); a segunda explica o que foi construído; a terceira é a análise.
# ---------------------------------------------------------------------------
top_cv, top_work, top_analysis = st.tabs(
    ["Currículo", "O que foi feito", "Dados & análise"])

with top_cv:
    top_home, top_qualif, top_skills = st.tabs(
        ["Quem sou eu", "Minhas Qualificações", "Skills"])

with top_home:
    col_photo, col_intro = st.columns([1, 3])
    with col_photo:
        # Avatar do GitHub, versionado localmente (assets/avatar.jpg) para não
        # depender de rede externa no deploy.
        st.image("assets/avatar.jpg", width=200)
        st.markdown(
            "🔗 [github.com/CabriniJr](https://github.com/CabriniJr)  \n"
            "🔗 [LinkedIn](https://www.linkedin.com/in/luigi-mendes-cabrini-775907349)")
    with col_intro:
        st.title("Luigi Mendes Cabrini")
        st.subheader("Platform Engineer Júnior · Observabilidade")
        st.caption("PagBank · FIAP — Engenharia de Software · São Bernardo do Campo, SP")
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
**Engenharia de Software** na **FIAP**, unindo minha atuação profissional
em observabilidade com fundamentos de estatística e análise de dados —
este dashboard é exatamente esse cruzamento: um laboratório real de
sampling em coletores OTel, analisado com os conceitos estatísticos da
disciplina.

Meu momento atual é de **construir as bases**: leitura técnica contínua
(Learning OpenTelemetry, Observability Engineering, Designing
Data-Intensive Applications, o livro de SRE do Google) combinada com um
laboratório prático por semana — este projeto é um desses labs.
        """)

with top_qualif:
    st.title("Minhas Qualificações")
    st.subheader("Formação")
    st.markdown(
        """
- **FIAP** — Engenharia de Software (em curso).
  Disciplinas cursadas incluem Estatística (*Data Science and Statistical
  Computing*), Banco de Dados, Java, Redes, Programação Dinâmica, Agile e 3D.
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
    st.markdown("**Concluídos / em andamento**")
    st.markdown(
        """
- **Este dashboard** ([github.com/CabriniJr/otelabs](https://github.com/CabriniJr/otelabs))
  — laboratório real de sampling em coletores OpenTelemetry: demo-app
  instrumentada, 6 pipelines de coletor, gerador de carga e análise
  estatística de confiança vs. performance.
- **BrasilWatch AI / BrasilFire** (FIAP Global Solution) — previsão de
  incêndios florestais no Brasil usando YOLOv8, LSTM, TimescaleDB/PostGIS
  e dados públicos brasileiros.
- **Atlas** — plataforma pessoal de automação inspirada em conceitos de
  Kubernetes (bot Telegram + dashboard web).
- **Datasource Grafana para FIWARE** (NGSI-v2 + STH-Comet) — contribuição
  open source em andamento: plugin que expõe dados de contexto FIWARE
  diretamente como fonte de dados no Grafana.
        """)
    st.markdown("**Próximo passo**")
    st.markdown(
        """
- **Otelabs** — evoluir este experimento pontual para um conjunto
  permanente de laboratórios de observabilidade: sampling dinâmico por
  tag, comparação head-based × tail-based em escala maior (200k+
  requisições por coletor) e cardinalidade de métricas. Este CP1 é o
  primeiro lab do repositório.
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

with top_work:
    tab_slides, tab_pipeline, tab_method = st.tabs(
        ["Apresentação (slides)", "Pipeline do experimento", "Metodologia"])

with tab_slides:
    # Paleta dos slides, adaptada ao tema escuro do app.
    NAVY, DEEPBLUE, TEAL = "#1b2340", "#0d4b73", "#155f7d"
    LIGHT_BLUE, OFFWHITE = "#a9c6f5", "#171b24"

    st.markdown(
        f"""
        <style>
        .slide-navy {{background:{NAVY}; color:{INK_PRIMARY}; padding:2rem 2.2rem;
            border-radius:0.6rem; margin-bottom:1rem; border:1px solid #2c3654;}}
        .slide-navy h1, .slide-navy h2 {{color:{INK_PRIMARY};}}
        .slide-navy .kicker {{color:{LIGHT_BLUE}; letter-spacing:2px; font-weight:600;
            font-size:0.85rem; text-transform:uppercase;}}
        .slide-navy .subtitle {{color:{LIGHT_BLUE}; font-size:1.05rem; margin-top:0.4rem;}}
        .quote-box {{background:{DEEPBLUE}; color:{INK_PRIMARY}; padding:1.2rem 1.4rem;
            border-radius:0.5rem; font-style:italic; height:100%;}}
        .stat-tile {{background:{OFFWHITE}; border:1px solid {GRID}; border-radius:0.5rem;
            padding:1rem; text-align:center;}}
        .stat-tile .value {{font-size:1.6rem; font-weight:700; color:#5b9bf2;}}
        .stat-tile .label {{color:{INK_SECONDARY}; font-size:0.85rem; margin-top:0.3rem;}}
        .method-card {{border-radius:0.5rem; padding:1rem; color:{INK_PRIMARY}; height:100%;}}
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
- Cursa Engenharia de Software na FIAP
- São Bernardo do Campo, Grande São Paulo
- 🔗 [linkedin.com/in/luigi-mendes-cabrini-775907349](https://www.linkedin.com/in/luigi-mendes-cabrini-775907349)
- 🔗 [github.com/CabriniJr](https://github.com/CabriniJr)
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
- FIAP — Engenharia de Software
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
        (m3, "Agressivo", "taxa muito baixa\n(head: 1% · tail: erro+cauda sempre + base 1%)", "#3a4a6b"),
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

    tab_overview, tab_compare, tab_tradeoff, tab_headtail, tab_dynamic = st.tabs(
        ["Visão geral", "Comparação dos coletores", "Trade-off & sweet spot",
         "Head vs. Tail", "Sampling dinâmico"])

    # ---------------------------------------------------------------------------
    # TAB 1 — Visão geral, em leitura F: a pergunta central e os KPIs na barra
    # superior, os bullets (com hover detalhando a conta) descendo pela haste
    # à esquerda, e os gráficos à direita.
    # ---------------------------------------------------------------------------
    with tab_overview:
        st.markdown(
            f"""
            <style>
            .f-hero {{border-left:4px solid #5b9bf2; padding:0.9rem 1.2rem;
                background:#171b24; border-radius:0 0.5rem 0.5rem 0; margin-bottom:0.8rem;}}
            .f-hero .q {{font-size:1.25rem; font-weight:700; color:{INK_PRIMARY};}}
            .f-hero .sub {{color:{INK_SECONDARY}; font-size:0.92rem; margin-top:0.35rem;}}
            .f-bullets {{overflow:visible;}}
            .f-bullet {{position:relative; border-left:3px solid var(--c);
                padding:0.55rem 0.8rem; margin-bottom:0.55rem; background:#171b24;
                border-radius:0 0.4rem 0.4rem 0; cursor:help;}}
            .f-bullet .head {{font-weight:700; color:{INK_PRIMARY}; font-size:0.95rem;}}
            .f-bullet .badge {{background:var(--c); color:#0d1017; border-radius:0.3rem;
                padding:0.05rem 0.4rem; font-size:0.75rem; font-weight:700; margin-left:0.35rem;}}
            .f-bullet .body {{color:{INK_SECONDARY}; font-size:0.86rem; margin-top:0.25rem;}}
            .f-bullet .hint {{color:{INK_MUTED}; font-size:0.74rem; margin-top:0.2rem;}}
            .f-bullet .tip {{visibility:hidden; opacity:0; position:absolute; z-index:999;
                left:0; top:100%; margin-top:0.3rem; width:min(460px, 92vw);
                background-color:#05070b !important; border:1px solid #3b4252;
                border-radius:0.45rem; padding:0.75rem 0.9rem; color:{INK_SECONDARY};
                font-size:0.8rem; line-height:1.55;
                box-shadow:0 8px 28px rgba(0,0,0,0.85);
                transition:opacity 0.12s ease;}}
            .f-bullet:hover .tip {{visibility:visible; opacity:1;}}
            .f-bullet .tip code {{color:#7fc4ff; background:transparent;}}
            .f-bullet .tip b {{color:{INK_PRIMARY};}}
            </style>
            """, unsafe_allow_html=True)

        # ── Barra superior do F: contexto, a pergunta e o slider de taxa ──
        c_ctx, c_rate = st.columns([1, 2])
        with c_ctx:
            ctx_pick = st.selectbox("Contexto", [c["key"] for c in CONTEXTS],
                                        format_func=lambda k: context_labels[k])
        det_df, src = data[ctx_pick]["deterministico"]

        st.markdown(
            '<div class="f-hero"><div class="q">Quanto precisamos amostrar — e quanta '
            'confiança estamos dispostos a perder — para ganhar desempenho no pipeline '
            'de observabilidade?</div>'
            '<div class="sub">Amostrar menos reduz o volume exportado pelo coletor '
            '(CPU, rede, custo de ingestão), mas encolhe o <i>n</i> e alarga o intervalo '
            'de confiança na proporção de 1/√n. O ponto de equilíbrio não é único: '
            'depende da <b>prioridade do sinal</b> (P1–P4, na barra lateral). '
            'Passe o mouse em cada bullet para ver a conta.</div></div>',
            unsafe_allow_html=True)

        with c_rate:
            rate_pct = st.slider(
                "Taxa de amostragem simulada (%)", 0.5, 100.0, 15.0, 0.5,
                help="Extrapola analiticamente (SE = σ/√n sobre o baseline 100%) "
                     "quanta confiança se perde nesta taxa, sem reamostrar os dados.")
        rate = rate_pct / 100.0

        if det_df is None:
            st.warning("Sem dado determinístico carregado para este contexto.")
        else:
            # ── KPIs da barra superior ──
            ext_by_domain = {}
            for d in domains:
                sub = det_df[det_df["domain"] == d]
                e = extrapolate_at_rate(sub, rate, confidence)
                if e is None:
                    continue
                pen = max(st_stats.confidence_penalty(e["lat_moe_pct"], moe_cap),
                          st_stats.confidence_penalty(e["err_moe_pct"], moe_cap))
                e["penalty"] = pen
                e["score"] = st_stats.composite_score(e["gain"], pen, importance[d], alpha)
                ext_by_domain[d] = e

            if not ext_by_domain:
                st.warning("Dados insuficientes para calcular os indicadores.")
            else:
                worst_lat = max(e["lat_moe_pct"] for e in ext_by_domain.values())
                kept = sum(e["n"] for e in ext_by_domain.values())
                total = sum(e["n_base"] for e in ext_by_domain.values())
                w_score = float(np.average(
                    [e["score"] for e in ext_by_domain.values()],
                    weights=[importance[d] for d in ext_by_domain]))
                p1_ok = all(e["penalty"] < 1 for d, e in ext_by_domain.items()
                            if priority[d] == "P1")

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Redução de exportação", f"{(1 - rate) * 100:.1f}%",
                          f"{br(total - kept)} spans a menos")
                k2.metric("Spans retidos", br(kept), f"de {br(total)}", delta_color="off")
                k3.metric("Pior MOE de latência", f"{worst_lat:.2f}%",
                          f"limite tolerado: {moe_cap}%", delta_color="off")
                k4.metric("Score ponderado por prioridade", f"{w_score:+.3f}",
                          "P1 dentro do limite" if p1_ok else "P1 fora do limite",
                          delta_color="off")

                # ── Haste do F: bullets à esquerda, gráficos à direita ──
                c_bul, c_chart = st.columns([1.05, 1.6])

                with c_bul:
                    st.markdown(f"**Por sinal, a {rate_pct:.1f}% de amostragem**")
                    html = ['<div class="f-bullets">']
                    for d in sorted(ext_by_domain, key=lambda x: priority[x]):
                        e = ext_by_domain[d]
                        p = priority[d]
                        base = extrapolate_at_rate(det_df[det_df["domain"] == d], 1.0, confidence)
                        # A penalidade do score é o PIOR entre os dois sinais de
                        # confiança (latência e taxa de erro): basta um estourar.
                        pen_lat = st_stats.confidence_penalty(e["lat_moe_pct"], moe_cap)
                        pen_err = st_stats.confidence_penalty(e["err_moe_pct"], moe_cap)
                        binding = "latência" if pen_lat >= pen_err else "taxa de erro"
                        verdict = ("dentro do limite" if e["penalty"] < 1
                                   else f"ESTOURA o limite pela {binding}")
                        se = e["std_base"] / np.sqrt(e["n"])
                        html.append(
                            f'<div class="f-bullet" style="--c:{colors[d]}">'
                            f'<div class="head">{d}<span class="badge">{p}</span></div>'
                            f'<div class="body">−{e["gain"] * 100:.0f}% de volume exportado '
                            f'em troca de MOE de latência <b>{base["lat_moe_pct"]:.2f}% → '
                            f'{e["lat_moe_pct"]:.2f}%</b> e de taxa de erro '
                            f'<b>{base["err_moe_pct"]:.1f}% → {e["err_moe_pct"]:.1f}%</b> '
                            f'({verdict}). Score {e["score"]:+.3f}.</div>'
                            f'<div class="hint">▸ passe o mouse para ver a conta</div>'
                            f'<div class="tip">'
                            f'<b>Amostra.</b> n = {br(e["n_base"])} × {rate:.3f} = '
                            f'<b>{br(e["n"])}</b> spans<br>'
                            f'<b>Erro-padrão.</b> SE = σ/√n = {e["std_base"]:.1f} / '
                            f'√{br(e["n"])} = {se:.2f} ms<br>'
                            f'<b>MOE de latência.</b> t<sub>{confidence:.2f}</sub> × SE = '
                            f'{e["t_crit"]:.3f} × {se:.2f} = {e["halfwidth_ms"]:.2f} ms '
                            f'→ relativa: {e["halfwidth_ms"]:.2f} / {e["mean_base"]:.1f} ms = '
                            f'<b>{e["lat_moe_pct"]:.2f}%</b><br>'
                            f'<b>MOE de taxa de erro.</b> IC de Wilson sobre '
                            f'p = {e["p_base"]:.3%} com n = {br(e["n"])} '
                            f'→ relativa: <b>{e["err_moe_pct"]:.1f}%</b><br>'
                            f'<b>Ganho.</b> 1 − {rate:.3f} = <b>{e["gain"]:.3f}</b><br>'
                            f'<b>Penalidade.</b> pior das duas MOEs, saturada em {moe_cap}%: '
                            f'max( min({e["lat_moe_pct"]:.2f}/{moe_cap} ; 1) ; '
                            f'min({e["err_moe_pct"]:.1f}/{moe_cap} ; 1) ) = '
                            f'max({pen_lat:.3f} ; {pen_err:.3f}) = <b>{e["penalty"]:.3f}</b> '
                            f'— quem manda aqui é a <b>{binding}</b><br>'
                            f'<b>Prioridade {p}</b> → importância {importance[d]:.2f} '
                            f'({PRIORITY_DESC[p]})<br>'
                            f'<b>Score.</b> α×ganho − (1−α)×importância×penalidade = '
                            f'{alpha:.2f}×{e["gain"]:.3f} − '
                            f'{1 - alpha:.2f}×{importance[d]:.2f}×{e["penalty"]:.3f} = '
                            f'<b>{e["score"]:+.3f}</b>'
                            f'</div></div>')
                    html.append('</div>')
                    st.markdown("".join(html), unsafe_allow_html=True)

                with c_chart:
                    # Gráfico 1 — rate() normalizado sobreposto: os 3 coletores
                    # na mesma escala, para ver a dispersão crescer.
                    dom_pick = st.selectbox(
                        "Sinal no gráfico de rate()",
                        sorted(ext_by_domain, key=lambda d: priority[d]),
                        format_func=lambda d: f"{d} · {priority[d]}", key="dom_rate")
                    fig_rate = go.Figure()
                    dashes = {"deterministico": "solid", "sweet_spot": "dash", "agressivo": "dot"}
                    widths = {"deterministico": 3, "sweet_spot": 2, "agressivo": 1.6}
                    n_det_dom = len(det_df[det_df["domain"] == dom_pick])
                    for i, run in enumerate(RUNS):
                        rdf, _ = data[ctx_pick][run["key"]]
                        if rdf is None:
                            continue
                        s = rate_series(rdf, dom_pick)
                        if s is None:
                            continue
                        eff = len(rdf[rdf["domain"] == dom_pick]) / n_det_dom if n_det_dom else np.nan
                        if not eff or np.isnan(eff):
                            continue
                        fig_rate.add_trace(go.Scatter(
                            x=s.index, y=s.values / eff, mode="lines",
                            name=f"{run['label']} ({eff:.1%})",
                            line=dict(color=CATEGORICAL[i], width=widths[run["key"]],
                                      dash=dashes[run["key"]]),
                            hovertemplate=(f"{run['label']}<br>rate estimado: "
                                           "%{y:.1f} req/s<extra></extra>")))
                    fig_rate = base_layout(
                        fig_rate,
                        f"rate() normalizado — {dom_pick} (média móvel de 3 s)",
                        "Tempo decorrido do run (s)", "Requisições/s estimadas", "Coletor")
                    st.plotly_chart(fig_rate, width='stretch')
                    st.caption(
                        "As três curvas estimam a MESMA taxa real (cada uma dividida pela "
                        "sua taxa efetiva de retenção). O que muda é a **dispersão**: com "
                        "menos amostra, a média móvel oscila muito mais em torno do mesmo "
                        "valor — é exatamente essa variância extra que o intervalo de "
                        "confiança quantifica.")

                    # Gráfico 2 — IC da latência média no baseline 100%
                    rows = []
                    for d in domains:
                        sub = det_df[det_df["domain"] == d]
                        if len(sub) == 0:
                            continue
                        lat = mean_ci(sub["latency_ms"].values, confidence, ci_method)
                        err = st_stats.wilson_ci(int(sub["is_error"].sum()), len(sub), confidence)
                        rows.append({"domain": d, "n": len(sub), "latencia_media_ms": lat["mean"],
                                        "ic_lat_lo": lat["lo"], "ic_lat_hi": lat["hi"],
                                        "taxa_erro": err["p"], "ic_erro_lo": err["lo"],
                                        "ic_erro_hi": err["hi"]})
                    ov = pd.DataFrame(rows)

                    fig = go.Figure()
                    for _, r in ov.iterrows():
                        fig.add_trace(go.Scatter(x=[r["ic_lat_lo"], r["ic_lat_hi"]],
                                                    y=[r["domain"]] * 2, mode="lines",
                                                    line=dict(color=colors[r["domain"]], width=6),
                                                    showlegend=False, hoverinfo="skip"))
                        fig.add_trace(go.Scatter(x=[r["latencia_media_ms"]], y=[r["domain"]],
                                                    mode="markers",
                                                    marker=dict(color=colors[r["domain"]], size=12,
                                                    line=dict(color=SURFACE, width=1)),
                                                    name=r["domain"],
                                                    hovertemplate=f"{r['domain']}<br>média: %{{x:.1f}} ms<extra></extra>"))
                    fig = base_layout(fig, f"Latência média por sinal — IC {int(confidence*100)}% (baseline 100%)",
                                        "Latência média (ms)", "", "Sinal")
                    st.plotly_chart(fig, width='stretch')

                with st.expander("Ver tabela — baseline determinístico e projeção na taxa escolhida"):
                    show = ov.copy()
                    for c in ["latencia_media_ms", "ic_lat_lo", "ic_lat_hi"]:
                        show[c] = show[c].round(1)
                    for c in ["taxa_erro", "ic_erro_lo", "ic_erro_hi"]:
                        show[c] = (show[c] * 100).round(2)
                    show["prioridade"] = show["domain"].map(priority)
                    show[f"n @ {rate_pct:.1f}%"] = show["domain"].map(
                        lambda d: ext_by_domain[d]["n"] if d in ext_by_domain else np.nan)
                    show[f"MOE lat @ {rate_pct:.1f}% (%)"] = show["domain"].map(
                        lambda d: round(ext_by_domain[d]["lat_moe_pct"], 2) if d in ext_by_domain else np.nan)
                    show["score"] = show["domain"].map(
                        lambda d: round(ext_by_domain[d]["score"], 3) if d in ext_by_domain else np.nan)
                    st.dataframe(show, width='stretch', hide_index=True)

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
            rate_by_domain = {d: c.slider(f"{d} · {priority[d]}", 0.01, 1.0,
                                            max(0.05, importance[d]), 0.01, key=f"rate_{d}")
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

