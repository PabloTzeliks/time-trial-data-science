import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
#  CONFIG & TEMA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Time Trial Analytics",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;600&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
      background-color: #0d0d0d;
      color: #f0f0f0;
  }
  [data-testid="stAppViewContainer"] > .main {
      background: radial-gradient(ellipse at 50% 0%, #1a0000 0%, #0d0d0d 65%);
  }
  h1, h2, h3 {
      font-family: 'Orbitron', sans-serif !important;
      letter-spacing: 0.08em;
  }
  h1 { color: #e73735 !important; font-weight: 900; }
  h2, h3 { color: #ffffff !important; font-weight: 700; }
  p, div, span, label {
      font-family: 'Rajdhani', sans-serif !important;
      font-size: 1.05rem;
  }
  [data-testid="stMetric"] {
      background: #111111;
      border: 1px solid #2a2a2a;
      border-left: 4px solid #e73735;
      border-radius: 6px;
      padding: 1.1rem 1.4rem !important;
  }
  [data-testid="stMetricLabel"] {
      font-family: 'Rajdhani', sans-serif !important;
      color: #999999 !important;
      font-size: 0.8rem !important;
      letter-spacing: 0.15em;
      text-transform: uppercase;
  }
  [data-testid="stMetricValue"] {
      font-family: 'Orbitron', sans-serif !important;
      color: #ffffff !important;
      font-size: 1.7rem !important;
  }
  [data-testid="stMetricDelta"] {
      color: #e73735 !important;
      font-family: 'Rajdhani', sans-serif !important;
  }
  hr { border-color: #222222 !important; }
  [data-testid="stDataFrame"] {
      border: 1px solid #222222;
      border-radius: 6px;
      overflow: hidden;
  }
  [data-testid="stSidebar"] { background-color: #0a0a0a; }
  .stButton > button {
      font-family: 'Orbitron', sans-serif !important;
      background: #e73735;
      color: #ffffff;
      border: none;
      border-radius: 4px;
      font-weight: 700;
      letter-spacing: 0.08em;
      padding: 0.5rem 1.4rem;
      transition: background 0.2s;
  }
  .stButton > button:hover { background: #ff5553; color: #ffffff; }
  .ds-insight {
      background: #111111;
      border: 1px solid #2a2a2a;
      border-left: 4px solid #e73735;
      border-radius: 6px;
      padding: 0.9rem 1.2rem;
      margin-bottom: 1rem;
      font-family: 'Rajdhani', sans-serif;
      color: #bbbbbb;
      font-size: 0.95rem;
  }
  .ds-insight strong { color: #ffffff; }
  [data-testid="stSlider"] > div > div > div { background: #e73735 !important; }
  [data-testid="stExpander"] {
      background: #111111 !important;
      border: 1px solid #2a2a2a !important;
      border-radius: 6px !important;
      overflow: hidden !important;
  }
  [data-testid="stExpander"] details { background: #111111 !important; }
  [data-testid="stExpander"] details > summary { background: #111111 !important; }
  [data-testid="stExpander"] details > summary > span:first-child { display: none !important; }
  [data-testid="stExpanderToggleIcon"] svg { fill: #e73735 !important; color: #e73735 !important; }
  [data-testid="stExpander"] details > summary p {
      color: #ffffff !important;
      font-family: 'Rajdhani', sans-serif !important;
      font-size: 1rem !important;
      font-weight: 600 !important;
  }
  [data-testid="stExpanderDetails"] {
      background: #111111 !important;
      border-top: 1px solid #2a2a2a !important;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CARREGAR E PROCESSAR DADOS DA API
#
#  A API retorna lista de pings RFID:
#    { "rfid": "58 59 5A 5B", "timestamp_ms": 562655 }
#
#  O tempo de volta é calculado como a diferença
#  entre timestamps consecutivos do mesmo RFID.
# ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def carregar_dados() -> pd.DataFrame | None:
    try:
        res = requests.get("http://localhost:8080/api/analytics/voltas", timeout=5)
        res.raise_for_status()
        raw = res.json()
    except Exception as e:
        st.error(f"Erro ao conectar na API: {e}")
        return None

    df = pd.DataFrame(raw)

    # Garante colunas esperadas
    if "rfid" not in df.columns or "timestamp_ms" not in df.columns:
        st.error("Formato inesperado da API. Esperado: { rfid, timestamp_ms }")
        return None

    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"])
    df = df.sort_values(["rfid", "timestamp_ms"]).reset_index(drop=True)

    # Calcula tempo de volta = diferença entre pings consecutivos do mesmo RFID
    df["tempo_volta_ms"] = df.groupby("rfid")["timestamp_ms"].diff()

    # Remove o primeiro ping de cada RFID (sem volta anterior) e voltas negativas
    df = df.dropna(subset=["tempo_volta_ms"])
    df = df[df["tempo_volta_ms"] > 0].copy()

    df["tempo_segundos"] = (df["tempo_volta_ms"] / 1000.0).round(3)
    df["timestamp_dt"]   = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df["volta_num"]      = df.groupby("rfid").cumcount() + 1

    return df


def features_por_carro(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("rfid")["tempo_segundos"]
              .agg(tempo_medio="mean", desvio_padrao="std",
                   total_voltas="count", melhor_volta="min")
              .reset_index().fillna(0))


# ─────────────────────────────────────────────
#  PALETA & LAYOUT PLOTLY
# ─────────────────────────────────────────────
CORES = ["#e73735", "#ffffff", "#ff8a89", "#aaaaaa",
         "#cc2220", "#ff5553", "#dddddd", "#888888"]

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Rajdhani", color="#bbbbbb"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#eeeeee")),
    xaxis=dict(gridcolor="#1f1f1f", linecolor="#333333"),
    yaxis=dict(gridcolor="#1f1f1f", linecolor="#333333"),
)


# ─────────────────────────────────────────────
#  CABEÇALHO
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 0.5rem'>
  <span style='font-family:Orbitron,sans-serif; font-size:2.2rem; font-weight:900; color:#e73735; letter-spacing:0.1em'>
    🏁 TIME TRIAL
  </span>
  <span style='font-family:Orbitron,sans-serif; font-size:2.2rem; font-weight:900; color:#ffffff; letter-spacing:0.1em'>
    ANALYTICS
  </span>
  <p style='color:#555555; letter-spacing:0.25em; font-size:0.8rem; margin-top:0.3rem; font-family:Rajdhani,sans-serif'>
    PAINEL DE TELEMETRIA — DATA SCIENCE EDITION
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col_btn, _ = st.columns([1, 8])
with col_btn:
    if st.button("⟳ Atualizar"):
        st.cache_data.clear()
        st.rerun()

df = carregar_dados()

# Para tudo se a API não retornou dados
if df is None or df.empty:
    st.warning("Nenhum dado disponível. Verifique se a API está rodando em http://localhost:8080")
    st.stop()

df_feat = features_por_carro(df)
st.markdown("---")


# ─────────────────────────────────────────────
#  CARDS DE RESUMO
# ─────────────────────────────────────────────
st.markdown("### 📡 Resumo Geral")
melhor = df.loc[df["tempo_segundos"].idxmin()]
c1, c2, c3, c4 = st.columns(4)
c1.metric("🏆 Recorde Absoluto", f"{melhor['tempo_segundos']:.3f}s", f"{melhor['rfid']}")
c2.metric("📊 Média Geral",      f"{df['tempo_segundos'].mean():.3f}s")
c3.metric("🔄 Total de Voltas",  str(len(df)))
c4.metric("🚗 Carros na Pista",  str(df["rfid"].nunique()))
st.markdown("---")


# ─────────────────────────────────────────────
#  1. HISTOGRAMA + KDE
# ─────────────────────────────────────────────
st.markdown("### 📊 1. Curva de Distribuição dos Tempos de Volta")
st.markdown("""<div class="ds-insight">
  <strong>💡 Insight de DS:</strong> Um pico único em sino indica pilotos com perfil similar.
  <strong>Dois picos (bimodal)</strong> revelam dois grupos distintos — ex: adultos vs. crianças ou carros tunados vs. padrão.
  Pontos isolados à esquerda/direita são candidatos a <strong>outliers</strong> (erro de sensor ou atalho).
</div>""", unsafe_allow_html=True)

fig_hist = px.histogram(df, x="tempo_segundos", nbins=40,
                        color_discrete_sequence=["#e73735"], opacity=0.8,
                        labels={"tempo_segundos": "Tempo de Volta (s)"})
if len(df) > 2:
    kde_x = np.linspace(df["tempo_segundos"].min(), df["tempo_segundos"].max(), 300)
    kde_y = gaussian_kde(df["tempo_segundos"])(kde_x)
    bw    = (df["tempo_segundos"].max() - df["tempo_segundos"].min()) / 40
    fig_hist.add_trace(go.Scatter(x=kde_x, y=kde_y * len(df) * bw,
                                  mode="lines", name="Densidade (KDE)",
                                  line=dict(color="#ffffff", width=2.5)))
fig_hist.update_layout(**PL, margin=dict(l=0,r=0,t=20,b=0))
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown("---")


# ─────────────────────────────────────────────
#  2. BOXPLOT + ANOMALIAS
# ─────────────────────────────────────────────
st.markdown("### 📦 2. Consistência e Anomalias por Carro (Boxplot)")
st.markdown("""<div class="ds-insight">
  <strong>💡 Insight de DS:</strong> A caixa mostra 50% das voltas (Q1→Q3); a linha central é a <strong>mediana</strong>.
  Pontos fora dos bigodes são <strong>outliers</strong> — possíveis erros de sensor, carro travado ou atalho.
  São os primeiros candidatos a remoção antes de análises mais profundas.
</div>""", unsafe_allow_html=True)

q1  = df["tempo_segundos"].quantile(0.25)
q3  = df["tempo_segundos"].quantile(0.75)
iqr = q3 - q1
df["is_outlier"] = ((df["tempo_segundos"] < q1 - 1.5*iqr) | (df["tempo_segundos"] > q3 + 1.5*iqr))
n_out = int(df["is_outlier"].sum())

fig_box = px.box(df, x="rfid", y="tempo_segundos", color="rfid",
                 color_discrete_sequence=CORES, points="all",
                 hover_data=["volta_num", "is_outlier"],
                 labels={"rfid": "RFID", "tempo_segundos": "Tempo (s)"})
fig_box.update_layout(**PL, showlegend=False, margin=dict(l=0,r=0,t=20,b=0))
st.plotly_chart(fig_box, use_container_width=True)

oc1, oc2 = st.columns(2)
oc1.metric("⚠️ Outliers Detectados", str(n_out))
oc2.metric("✅ Voltas Válidas", str(len(df) - n_out))

if "show_outliers" not in st.session_state:
    st.session_state.show_outliers = False
if st.button("▼ Ver tabela de outliers" if not st.session_state.show_outliers else "▲ Fechar tabela"):
    st.session_state.show_outliers = not st.session_state.show_outliers
if st.session_state.show_outliers:
    st.dataframe(
        df[df["is_outlier"]][["rfid","tempo_segundos","volta_num","timestamp_dt"]]
          .rename(columns={"rfid":"RFID","tempo_segundos":"Tempo (s)",
                           "volta_num":"Volta","timestamp_dt":"Horário"})
          .sort_values("Tempo (s)"),
        use_container_width=True)
st.markdown("---")


# ─────────────────────────────────────────────
#  3. K-MEANS CLUSTERING
# ─────────────────────────────────────────────
st.markdown("### 🤖 3. Agrupamento de Estilos de Pilotagem (K-Means)")
st.markdown("""<div class="ds-insight">
  <strong>💡 Insight de DS:</strong> Cruzando <strong>Tempo Médio</strong> (velocidade) com
  <strong>Desvio Padrão</strong> (consistência), o K-Means agrupa automaticamente os pilotos em perfis.
  Canto inferior-esquerdo = <em>rápido E consistente</em> (o ideal).
  Canto superior-direito = <em>lento E errático</em>.
</div>""", unsafe_allow_html=True)

n_rfids = df_feat["rfid"].nunique()
max_k   = min(4, max(2, n_rfids - 1))

col_k, _ = st.columns([1, 3])
with col_k:
    n_clusters = st.slider("Número de grupos (k)", 2, max_k, 2)

if n_rfids >= n_clusters:
    X = StandardScaler().fit_transform(df_feat[["tempo_medio","desvio_padrao"]].values)
    df_feat["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X).astype(str)
    df_feat["grupo"]   = df_feat["cluster"].map({"0":"Grupo A","1":"Grupo B","2":"Grupo C","3":"Grupo D"})

    fig_sc = px.scatter(df_feat, x="tempo_medio", y="desvio_padrao",
                        color="grupo", size="total_voltas", text="rfid",
                        color_discrete_sequence=CORES, size_max=40,
                        labels={"tempo_medio":"Tempo Médio (s)","desvio_padrao":"Desvio Padrão (s)",
                                "total_voltas":"Nº Voltas","grupo":"Grupo"})
    fig_sc.update_traces(textposition="top center", textfont=dict(size=10, color="#ffffff"))
    fig_sc.add_hline(y=df_feat["desvio_padrao"].mean(), line_dash="dot", line_color="#333333",
                     annotation_text="média consistência", annotation_font_color="#666666")
    fig_sc.add_vline(x=df_feat["tempo_medio"].mean(), line_dash="dot", line_color="#333333",
                     annotation_text="média velocidade", annotation_font_color="#666666")
    fig_sc.update_layout(**PL, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig_sc, use_container_width=True)
else:
    st.info("Dados insuficientes para clustering. São necessários pelo menos 2 RFIDs com voltas registradas.")
st.markdown("---")


# ─────────────────────────────────────────────
#  4. MATRIZ DE CORRELAÇÃO
# ─────────────────────────────────────────────
st.markdown("### 🌡️ 4. Matriz de Correlação")
st.markdown("""<div class="ds-insight">
  <strong>💡 Insight de DS:</strong> Valores próximos de <strong>+1</strong> = correlação positiva forte.
  Próximos de <strong>-1</strong> = inversa (mais voltas → aprende e fica mais rápido?).
  Próximos de <strong>0</strong> = sem relação aparente.
</div>""", unsafe_allow_html=True)

cols_corr   = ["tempo_medio","desvio_padrao","total_voltas","melhor_volta"]
labels_corr = ["Tempo Médio","Desvio Padrão","Total Voltas","Melhor Volta"]
corr        = df_feat[cols_corr].corr().round(2)

fig_hm = go.Figure(data=go.Heatmap(
    z=corr.values, x=labels_corr, y=labels_corr,
    colorscale=[[0.0,"#0d0d0d"],[0.5,"#555555"],[1.0,"#e73735"]],
    zmin=-1, zmax=1,
    text=corr.values, texttemplate="%{text}",
    textfont=dict(family="Orbitron", size=13, color="#ffffff"),
))
fig_hm.update_layout(**PL, margin=dict(l=0,r=0,t=20,b=0), height=380)
st.plotly_chart(fig_hm, use_container_width=True)
st.markdown("---")


# ─────────────────────────────────────────────
#  EXTRA — Performance & Engajamento
# ─────────────────────────────────────────────
if "show_perf" not in st.session_state:
    st.session_state.show_perf = False
if st.button("▼ Ver Evolução de Performance e Engajamento" if not st.session_state.show_perf else "▲ Fechar gráficos"):
    st.session_state.show_perf = not st.session_state.show_perf
if st.session_state.show_perf:
    cb, cl = st.columns(2, gap="large")
    with cb:
        st.markdown("#### Engajamento por RFID")
        cnt = df.groupby("rfid").size().reset_index(name="total_voltas").sort_values("total_voltas", ascending=False)
        fb  = px.bar(cnt, x="rfid", y="total_voltas", color="rfid",
                     color_discrete_sequence=CORES, text="total_voltas",
                     labels={"rfid":"RFID","total_voltas":"Nº Voltas"})
        fb.update_traces(textposition="outside", marker_line_width=0)
        fb.update_layout(**PL, showlegend=False, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fb, use_container_width=True)
    with cl:
        st.markdown("#### Histórico de Tempos")
        fl = px.line(df, x="volta_num", y="tempo_segundos", color="rfid",
                     markers=True, color_discrete_sequence=CORES,
                     labels={"volta_num":"Volta","tempo_segundos":"Tempo (s)","rfid":"RFID"},
                     hover_data={"timestamp_dt":"|%H:%M:%S"})
        fl.update_traces(line_width=2, marker_size=4)
        fl.update_layout(**PL, margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified")
        st.plotly_chart(fl, use_container_width=True)

st.markdown("---")
st.caption("Time Trial Analytics · dados via RFID · Streamlit + Plotly + scikit-learn")