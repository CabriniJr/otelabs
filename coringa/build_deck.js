const pptxgen = require("pptxgenjs");

const NAVY = "21295C";
const DEEPBLUE = "065A82";
const TEAL = "1C7293";
const WHITE = "FFFFFF";
const INK = "1A1A1A";
const MUTED = "5A6472";
const OFFWHITE = "F5F7FA";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };

function titleSlide(title, subtitle, kicker) {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText(kicker, { x: 0.7, y: 1.7, w: 11.9, h: 0.5, fontFace: "Calibri", fontSize: 14,
    color: "8FA8D6", bold: true, charSpacing: 2 });
  s.addText(title, { x: 0.7, y: 2.2, w: 11.9, h: 1.7, fontFace: "Cambria", fontSize: 40,
    color: WHITE, bold: true });
  s.addText(subtitle, { x: 0.7, y: 3.9, w: 11.0, h: 1.0, fontFace: "Calibri", fontSize: 16,
    color: "CADCFC" });
  return s;
}

function contentSlide(title) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addText(title, { x: 0.6, y: 0.45, w: 12.1, h: 0.8, fontFace: "Cambria", fontSize: 28,
    color: NAVY, bold: true });
  return s;
}

function statTile(s, x, y, w, value, label, color) {
  s.addShape("roundRect", { x, y, w, h: 1.5, fill: { color: OFFWHITE }, line: { color: "E3E7ED", width: 1 },
    rectRadius: 0.08 });
  s.addText(value, { x, y: y + 0.12, w, h: 0.85, align: "center", fontFace: "Calibri", fontSize: 30,
    bold: true, color });
  s.addText(label, { x, y: y + 0.95, w, h: 0.5, align: "center", fontFace: "Calibri", fontSize: 11,
    color: MUTED });
}

// 1) Título
titleSlide(
  "Intervalos de Confiança & Sampling\nem Coletores OTel",
  "Apresentação coringa — dashboard indisponível na correção? Está tudo aqui.",
  "CP1 · DASHBOARD PROFISSIONAL · FIAP"
);

// 2) Quem sou eu
{
  const s = contentSlide("Quem sou eu");
  s.addText("Luigi", { x: 0.6, y: 1.3, w: 6, h: 0.6, fontFace: "Cambria", fontSize: 22, bold: true, color: NAVY });
  s.addText("Platform Engineer Júnior · Observabilidade", { x: 0.6, y: 1.85, w: 6.2, h: 0.4,
    fontFace: "Calibri", fontSize: 14, color: TEAL, bold: true });
  s.addText(
    [
      { text: "PagBank (fintech brasileira) — monitoramento, instrumentação e confiabilidade", options: { breakLine: true, bullet: true } },
      { text: "Antes: sistemas embarcados (NVIDIA Jetson) na OptDriven", options: { breakLine: true, bullet: true } },
      { text: "Cursa Data Science e Ciência da Computação Aplicada na FIAP", options: { breakLine: true, bullet: true } },
      { text: "São Bernardo do Campo, Grande São Paulo", options: { breakLine: false, bullet: true } },
    ],
    { x: 0.6, y: 2.4, w: 6.2, h: 2.2, fontFace: "Calibri", fontSize: 14, color: INK, paraSpaceAfter: 10 }
  );
  s.addShape("roundRect", { x: 7.3, y: 1.3, w: 5.3, h: 4.4, fill: { color: DEEPBLUE }, rectRadius: 0.08 });
  s.addText(
    "“Este dashboard é exatamente esse cruzamento: um laboratório real de sampling em coletores OTel, analisado com os conceitos estatísticos da disciplina.”",
    { x: 7.7, y: 1.7, w: 4.5, h: 3.6, fontFace: "Cambria", fontSize: 15, italic: true, color: WHITE, valign: "middle" }
  );
}

// 3) Qualificações + Skills
{
  const s = contentSlide("Minhas Qualificações & Skills");
  s.addText("Formação & Experiência", { x: 0.6, y: 1.2, w: 5.8, h: 0.4, fontFace: "Calibri", fontSize: 15, bold: true, color: TEAL });
  s.addText(
    [
      { text: "FIAP — Data Science e Ciência da Computação Aplicada", options: { breakLine: true, bullet: true } },
      { text: "PagBank — testes sintéticos Datadog, monitoramento Splunk/LDAP", options: { breakLine: true, bullet: true } },
      { text: "OptDriven — Python para sistemas embarcados (Jetson)", options: { breakLine: true, bullet: true } },
      { text: "Projeto FIAP Global Solution: BrasilWatch AI / BrasilFire", options: { breakLine: false, bullet: true } },
    ],
    { x: 0.6, y: 1.65, w: 5.8, h: 2.6, fontFace: "Calibri", fontSize: 13, color: INK, paraSpaceAfter: 8 }
  );
  s.addText("Skills técnicas", { x: 6.8, y: 1.2, w: 5.8, h: 0.4, fontFace: "Calibri", fontSize: 15, bold: true, color: TEAL });
  s.addText(
    [
      { text: "Python, Java", options: { breakLine: true, bullet: true } },
      { text: "OpenTelemetry (instrumentação manual, OTel Collector, sampling)", options: { breakLine: true, bullet: true } },
      { text: "Docker / Docker Compose, conceitos de Kubernetes", options: { breakLine: true, bullet: true } },
      { text: "Estatística aplicada (IC, bootstrap), pandas, Plotly, Streamlit", options: { breakLine: true, bullet: true } },
      { text: "Locust, SQL, Git", options: { breakLine: false, bullet: true } },
    ],
    { x: 6.8, y: 1.65, w: 5.8, h: 2.9, fontFace: "Calibri", fontSize: 13, color: INK, paraSpaceAfter: 8 }
  );
  s.addText("Inglês: TOEFL B2 geral, C1 em listening/reading  ·  Português nativo",
    { x: 0.6, y: 4.5, w: 12, h: 0.4, fontFace: "Calibri", fontSize: 12, color: MUTED, italic: true });
}

// 4) O problema
{
  const s = contentSlide("O problema: sampling em observabilidade");
  s.addText(
    [
      { text: "Capturar 100% dos spans/traces em produção é caro: CPU do coletor, rede, custo de ingestão e armazenamento no backend.", options: { breakLine: true, bullet: true } },
      { text: "Reduzir o sampling economiza recursos — mas reduz o tamanho da amostra, o que aumenta a margem de erro (MOE) das métricas estimadas (SE = σ/√n).", options: { breakLine: true, bullet: true } },
      { text: "Sinais diferentes têm importância diferente: uma transação PIX que falha não pode “sumir” por causa do sampling, mas uma página genérica de site tolera taxas mais agressivas.", options: { breakLine: false, bullet: true } },
    ],
    { x: 0.6, y: 1.3, w: 7.6, h: 4.2, fontFace: "Calibri", fontSize: 15, color: INK, paraSpaceAfter: 14, lineSpacingMultiple: 1.15 }
  );
  statTile(s, 8.7, 1.3, 3.7, "IC", "Interval de Confiança\n(CLT, bootstrap, Wilson)", DEEPBLUE);
  statTile(s, 8.7, 3.0, 3.7, "Score", "throughput × confiança\n× importância do sinal", TEAL);
}

// 5) Metodologia
{
  const s = contentSlide("Metodologia: o experimento real");
  s.addText("2 contextos × 3 coletores OTel, rodando de verdade (docker compose / execução local)",
    { x: 0.6, y: 1.15, w: 12, h: 0.5, fontFace: "Calibri", fontSize: 14, color: MUTED });
  const cols = [
    { x: 0.6, title: "Determinístico", desc: "100% do tráfego\n(baseline / ground truth)", color: DEEPBLUE },
    { x: 4.7, title: "Sweet spot", desc: "taxa balanceada\n(head: 15% · tail: erro+cauda sempre + base 15%)", color: TEAL },
    { x: 8.8, title: "Agressivo", desc: "taxa muito baixa\n(head: 1% · tail: erro+cauda sempre + base 1%)", color: "8FA8D6" },
  ];
  cols.forEach(c => {
    s.addShape("roundRect", { x: c.x, y: 1.85, w: 3.7, h: 1.7, fill: { color: c.color }, rectRadius: 0.08 });
    s.addText(c.title, { x: c.x + 0.2, y: 2.0, w: 3.3, h: 0.4, fontFace: "Calibri", fontSize: 15, bold: true, color: WHITE });
    s.addText(c.desc, { x: c.x + 0.2, y: 2.4, w: 3.3, h: 1.0, fontFace: "Calibri", fontSize: 11, color: WHITE });
  });
  s.addText(
    "Rodado 2×: contexto head-based (probabilistic_sampler) e contexto tail-based (tail_sampling — erros e cauda p95 sempre mantidos). Demo-app instrumentada com OpenTelemetry, 4 domínios de sinal (pix, checkout, site_latency, api_generic), gerador de carga Locust com parada exata na meta de requisições.",
    { x: 0.6, y: 3.9, w: 12, h: 1.6, fontFace: "Calibri", fontSize: 13, color: INK, italic: true, lineSpacingMultiple: 1.2 });
}

// 6) Resultados — retenção real medida
{
  const s = contentSlide("Resultado real: retenção por contexto");
  s.addText("Teste local já executado (20 mil requisições/coletor) — números reais, não simulados",
    { x: 0.6, y: 1.1, w: 12, h: 0.4, fontFace: "Calibri", fontSize: 13, color: MUTED });

  const rows = [
    ["Contexto", "Determinístico", "Sweet spot", "Agressivo"],
    ["Head-based", "20.124 (100%)", "3.016 (15,0%)", "217 (1,1%)"],
    ["Tail-based", "19.968 (100%)", "4.838 (24,2%)", "2.260 (11,3%)"],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.7, w: 12.1, h: 1.8,
    colW: [3.0, 3.0, 3.0, 3.1],
    fontFace: "Calibri", fontSize: 14, border: { type: "solid", color: "E3E7ED", pt: 1 },
    fill: { color: WHITE },
    autoPage: false,
  });
  s.addText(
    "O tail-based retém muito mais que a taxa-base nominal (1% → 11,3% no agressivo) porque erros e picos de latência são sempre mantidos — é isso que preserva a confiança nos sinais críticos mesmo sob sampling agressivo.",
    { x: 0.6, y: 3.9, w: 12, h: 1.4, fontFace: "Calibri", fontSize: 14, color: INK, italic: true, lineSpacingMultiple: 1.2 });
}

// 7) Trade-off e sweet spot
{
  const s = contentSlide("Trade-off: performance × confiança");
  s.addText(
    "score = α·ganho_throughput − (1−α)·importância·penalidade_confiança",
    { x: 0.6, y: 1.3, w: 12, h: 0.6, fontFace: "Courier New", fontSize: 16, color: NAVY, bold: true });
  s.addText(
    [
      { text: "ganho = 1 − taxa de sampling (economia no pipeline de observabilidade)", options: { breakLine: true, bullet: true } },
      { text: "penalidade = MOE% relativa, saturada em um limite configurável", options: { breakLine: true, bullet: true } },
      { text: "importância = peso do sinal (PIX = 1,0 · latência genérica = 0,2)", options: { breakLine: true, bullet: true } },
      { text: "α = peso performance × confiança, ajustável no dashboard", options: { breakLine: false, bullet: true } },
    ],
    { x: 0.6, y: 2.1, w: 7.6, h: 2.6, fontFace: "Calibri", fontSize: 14, color: INK, paraSpaceAfter: 10 });
  s.addShape("roundRect", { x: 8.7, y: 1.9, w: 3.7, h: 3.1, fill: { color: NAVY }, rectRadius: 0.08 });
  s.addText("Sinais críticos (PIX) →\nsweet spot mais alto",
    { x: 8.9, y: 2.15, w: 3.3, h: 1.2, fontFace: "Calibri", fontSize: 13, color: WHITE, bold: true });
  s.addText("Sinais tolerantes (latência genérica) →\nsweet spot mais agressivo",
    { x: 8.9, y: 3.5, w: 3.3, h: 1.3, fontFace: "Calibri", fontSize: 13, color: "CADCFC" });
}

// 8) Conclusão
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Conclusão", { x: 0.7, y: 0.9, w: 11.9, h: 0.8, fontFace: "Cambria", fontSize: 32, bold: true, color: WHITE });
  s.addText(
    [
      { text: "Sampling agressivo economiza pipeline, mas custa confiança estatística de forma previsível (~1/√n)", options: { breakLine: true, bullet: true } },
      { text: "Tail-based preserva confiança em sinais raros/críticos mesmo sob taxas baixas — head-based não", options: { breakLine: true, bullet: true } },
      { text: "O sweet spot certo depende da importância do sinal, não é um número único para o sistema inteiro", options: { breakLine: true, bullet: true } },
      { text: "Todo o pipeline (demo-app instrumentada, 6 coletores, gerador de carga, conversão OTLP→CSV) está reproduzível — ver EXPERIMENTO.md", options: { breakLine: false, bullet: true } },
    ],
    { x: 0.7, y: 1.9, w: 11.5, h: 3.2, fontFace: "Calibri", fontSize: 16, color: "E8EDF7", paraSpaceAfter: 14 });
  s.addText("Link do dashboard: [preencher após deploy no Streamlit Community Cloud]",
    { x: 0.7, y: 5.3, w: 11.5, h: 0.5, fontFace: "Calibri", fontSize: 13, color: "8FA8D6", italic: true });
}

pres.writeFile({ fileName: "coringa/apresentacao_coringa.pptx" }).then(() => console.log("done"));
