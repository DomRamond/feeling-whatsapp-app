import re
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import chardet
from pysentimiento import create_analyzer

# Configuração inicial
st.set_page_config(page_title="Analisador de Sentimento WhatsApp", layout="centered")

st.title("?? Analisador de Sentimento de Grupos do WhatsApp")
st.write("Envie um arquivo `.txt` exportado de uma conversa e veja o clima emocional do grupo.")

# Upload do arquivo
uploaded_file = st.file_uploader("?? Escolha o arquivo de conversa (.txt)", type=["txt"])

if uploaded_file:
    # --- Detectar automaticamente a codificação do arquivo ---
    raw_data = uploaded_file.read()

    try:
        # 1?? Detectar codificação com chardet
        detected = chardet.detect(raw_data)
        encoding_detected = detected.get("encoding", None)

        # 2?? Se não detectar, assume Latin-1 (padrão do WhatsApp PT-BR)
        if encoding_detected is None:
            encoding_detected = "latin-1"

        # 3?? Decodificar texto com codificação detectada
        text = raw_data.decode(encoding_detected, errors="ignore").splitlines()

    except Exception:
        # 4?? Se ainda der erro, tenta Latin-1
        text = raw_data.decode("latin-1", errors="ignore").splitlines()

    st.write(f"?? Arquivo detectado como: `{encoding_detected}`")

    # --- Parse do arquivo WhatsApp ---
    pattern = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2})\s*-\s*(.*?):\s*(.*)$')
    rows = []
    for line in text:
        line = line.strip()
        m = pattern.match(line)
        if m:
            date, time, author, msg = m.groups()
            rows.append({'date': date, 'time': time, 'author': author, 'message': msg})
        elif rows and line:
            rows[-1]['message'] += ' ' + line

    df = pd.DataFrame(rows)
    st.success(f"? {len(df)} mensagens carregadas com sucesso!")

    # --- Análise de sentimento ---
    with st.spinner("Analisando sentimentos... (pode levar 1-2 minutos)"):
        analyzer = create_analyzer(task="sentiment", lang="pt")
        preds = df["message"].apply(lambda text: analyzer.predict(text))
        df["sentimento"] = preds.apply(lambda p: p.output)
        df["prob_POS"] = preds.apply(lambda p: p.probas.get("POS", 0))
        df["prob_NEU"] = preds.apply(lambda p: p.probas.get("NEU", 0))
        df["prob_NEG"] = preds.apply(lambda p: p.probas.get("NEG", 0))

    st.write("### ?? Amostra de mensagens analisadas")
    st.dataframe(df.head(20))

    # --- Distribuição geral dos sentimentos ---
    st.write("### ?? Distribuição geral dos sentimentos")
    sent_counts = df["sentimento"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(sent_counts.index, sent_counts.values, color=["green", "gray", "red"])
    ax1.set_xlabel("Sentimento")
    ax1.set_ylabel("Quantidade")
    st.pyplot(fig1)

    # --- Evolução no tempo ---
    df["datetime"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    daily = df.groupby(df["datetime"].dt.date)["sentimento"].value_counts(normalize=True).unstack(fill_value=0)

    st.write("### ? Evolução diária do sentimento")
    fig2, ax2 = plt.subplots()
    for col in daily.columns:
        ax2.plot(daily.index, daily[col], label=col)
    ax2.legend()
    ax2.set_xlabel("Data")
    ax2.set_ylabel("Proporção")
    st.pyplot(fig2)

    # --- Ranking dos participantes ---
    st.write("### ?? Ranking por autor (proporção de mensagens por tipo)")
    author_summary = df.groupby("author")["sentimento"].value_counts(normalize=True).unstack(fill_value=0)
    st.dataframe(author_summary.round(2))

else:
    st.info("Envie o arquivo de conversa (.txt) para começar.")
