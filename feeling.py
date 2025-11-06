import re
import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import chardet
from pysentimiento import create_analyzer

# ==========================
# CONFIGURAÇÕES DO STREAMLIT
# ==========================
st.set_page_config(page_title="Analisador de Sentimento WhatsApp", layout="centered")

st.title("?? Analisador de Sentimento de Grupos do WhatsApp")
st.write("Envie um arquivo `.txt` exportado de uma conversa e veja o clima emocional do grupo.")

# ==========================
# UPLOAD DO ARQUIVO
# ==========================
uploaded_file = st.file_uploader("?? Escolha o arquivo de conversa (.txt)", type=["txt"])

if uploaded_file:
    # ------------------------------------------
    # LEITURA SEGURA DO ARQUIVO (À PROVA DE ERROS)
    # ------------------------------------------
    try:
        uploaded_file.seek(0)
        raw_data = uploaded_file.read()
    except Exception as e:
        st.error(f"? Erro ao ler o arquivo: {e}")
        st.stop()

    # Detectar automaticamente a codificação
    try:
        detected = chardet.detect(raw_data)
        encoding_detected = detected.get("encoding", None)
        if encoding_detected is None:
            encoding_detected = "latin-1"
    except Exception:
        encoding_detected = "latin-1"

    # Garantir decodificação sem falhas
    try:
        text_decoded = raw_data.decode(encoding_detected, errors="ignore")
    except Exception:
        text_decoded = raw_data.decode("latin-1", errors="ignore")

    # Converter para lista de linhas
    text = text_decoded.splitlines()

    st.write(f"?? Arquivo detectado como: `{encoding_detected}`")
    st.caption("Se o texto tiver acentuação incorreta, provavelmente o arquivo foi exportado com codificação Latin-1, o que é comum no WhatsApp PT-BR.")

    # ==========================
    # PARSE DO ARQUIVO WHATSAPP
    # ==========================
    pattern = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4})[\s,]*(\d{1,2}:\d{2})\s*-\s*(.*?):\s*(.*)$')
    rows = []
    for line in text:
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            date, time, author, msg = m.groups()
            rows.append({'date': date, 'time': time, 'author': author, 'message': msg})
        elif rows:
            rows[-1]['message'] += ' ' + line

    if not rows:
        st.error("? Nenhuma mensagem reconhecida no arquivo. Verifique se o formato é válido (Exportação do WhatsApp).")
        st.stop()

    df = pd.DataFrame(rows)
    st.success(f"? {len(df)} mensagens carregadas com sucesso!")

    # ==========================
    # ANÁLISE DE SENTIMENTO
    # ==========================
    with st.spinner("?? Analisando sentimentos... (pode levar 1-2 minutos)"):
        try:
            analyzer = create_analyzer(task="sentiment", lang="pt")
            preds = df["message"].apply(lambda text: analyzer.predict(text))
        except Exception as e:
            st.error(f"? Erro durante a análise de sentimentos: {e}")
            st.stop()

        df["sentimento"] = preds.apply(lambda p: p.output)
        df["prob_POS"] = preds.apply(lambda p: p.probas.get("POS", 0))
        df["prob_NEU"] = preds.apply(lambda p: p.probas.get("NEU", 0))
        df["prob_NEG"] = preds.apply(lambda p: p.probas.get("NEG", 0))

    # ==========================
    # EXIBIÇÃO DE RESULTADOS
    # ==========================
    st.write("### ?? Amostra de mensagens analisadas")
    st.dataframe(df.head(20))

    # --- Distribuição geral dos sentimentos ---
    st.write("### ?? Distribuição geral dos sentimentos")
    sent_counts = df["sentimento"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(sent_counts.index, sent_counts.values, color=["green", "gray", "red"])
    ax1.set_xlabel("Sentimento")
    ax1.set_ylabel("Quantidade")
    ax1.set_title("Distribuição Geral de Sentimentos")
    st.pyplot(fig1)

    # --- Evolução no tempo ---
    df["datetime"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    daily = df.groupby(df["datetime"].dt.date)["sentimento"].value_counts(normalize=True).unstack(fill_value=0)

    st.write("### ? Evolução diária do sentimento")
    if not daily.empty:
        fig2, ax2 = plt.subplots()
        for col in daily.columns:
            ax2.plot(daily.index, daily[col], label=col)
        ax2.legend()
        ax2.set_xlabel("Data")
        ax2.set_ylabel("Proporção")
        ax2.set_title("Evolução Diária dos Sentimentos")
        st.pyplot(fig2)
    else:
        st.info("Não foi possível gerar gráfico temporal (datas ausentes ou inválidas).")

    # --- Ranking dos participantes ---
    st.write("### ?? Ranking por autor (proporção de mensagens por tipo)")
    author_summary = df.groupby("author")["sentimento"].value_counts(normalize=True).unstack(fill_value=0)
    st.dataframe(author_summary.round(2))

else:
    st.info("Envie o arquivo de conversa (.txt) para começar.")
