import re
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

if uploaded_file is not None:
    # Leitura segura do arquivo
    try:
        uploaded_file.seek(0)
        raw_data = uploaded_file.getvalue()
    except Exception as e:
        st.error(f"? Erro ao ler o arquivo: {e}")
        st.stop()

    # Detectar codificação automaticamente
    try:
        detected = chardet.detect(raw_data)
        encoding_detected = detected.get("encoding", "utf-8")
        confidence = detected.get("confidence", 0)
        
        # Se a confiança for baixa, força Latin-1 (comum em WhatsApp PT-BR)
        if confidence < 0.7 or encoding_detected is None:
            encoding_detected = "latin-1"
            
    except Exception:
        encoding_detected = "latin-1"

    # Decodificação com fallback robusto
    text_decoded = None
    encodings_to_try = [encoding_detected, "utf-8", "latin-1", "cp1252", "iso-8859-1"]
    
    for enc in encodings_to_try:
        try:
            text_decoded = raw_data.decode(enc)
            encoding_used = enc
            break
        except (UnicodeDecodeError, AttributeError):
            continue
    
    # Último recurso: ignorar erros
    if text_decoded is None:
        text_decoded = raw_data.decode("latin-1", errors="ignore")
        encoding_used = "latin-1 (com erros ignorados)"

    # Converter em lista de linhas
    text = text_decoded.splitlines()

    st.write(f"?? Codificação utilizada: `{encoding_used}`")
    st.caption("Se houver caracteres estranhos, tente exportar o chat novamente do WhatsApp.")

    # ==========================
    # PARSE DO ARQUIVO WHATSAPP
    # ==========================
    # Padrão mais flexível para diferentes formatos de data/hora
    patterns = [
        re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4})[,\s]+(\d{1,2}:\d{2})\s*[-–]\s*([^:]+?):\s*(.*)$'),
        re.compile(r'^\[?(\d{1,2}/\d{1,2}/\d{2,4})[,\s]+(\d{1,2}:\d{2})\]?\s*([^:]+?):\s*(.*)$'),
        re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2})\s*-\s*([^:]+):\s*(.*)$')
    ]
    
    rows = []
    for line in text:
        line = line.strip()
        if not line:
            continue
            
        matched = False
        for pattern in patterns:
            m = pattern.match(line)
            if m:
                date, time, author, msg = m.groups()
                rows.append({
                    'date': date.strip(),
                    'time': time.strip(),
                    'author': author.strip(),
                    'message': msg.strip()
                })
                matched = True
                break
        
        # Se não deu match e já existem mensagens, anexa à última
        if not matched and rows:
            rows[-1]['message'] += ' ' + line

    if not rows:
        st.error("? Nenhuma mensagem reconhecida no arquivo. Verifique se o formato é válido (Exportação do WhatsApp).")
        st.info("**Dica:** Certifique-se de exportar o chat com a opção 'Sem mídia' diretamente do WhatsApp.")
        st.stop()

    df = pd.DataFrame(rows)
    
    # Remover mensagens do sistema (notificações do WhatsApp)
    system_keywords = [
        "mensagens e chamadas",
        "criptografadas de ponta",
        "adicionou",
        "saiu",
        "removeu",
        "alterou o assunto",
        "alterou a descrição",
        "alterou o ícone"
    ]
    
    for keyword in system_keywords:
        df = df[~df['message'].str.lower().str.contains(keyword, na=False)]
    
    # Remover mensagens vazias ou muito curtas
    df = df[df['message'].str.len() > 2].reset_index(drop=True)
    
    if len(df) == 0:
        st.error("? Nenhuma mensagem válida encontrada após filtragem.")
        st.stop()
    
    st.success(f"? {len(df)} mensagens carregadas com sucesso!")

    # ==========================
    # ANÁLISE DE SENTIMENTO
    # ==========================
    with st.spinner("?? Analisando sentimentos... (pode levar alguns minutos dependendo do tamanho do chat)"):
        try:
            # Criar analisador uma única vez
            analyzer = create_analyzer(task="sentiment", lang="pt")
            
            # Análise com tratamento de erros por mensagem
            sentiments = []
            probs_pos = []
            probs_neu = []
            probs_neg = []
            
            for msg in df["message"]:
                try:
                    pred = analyzer.predict(str(msg)[:500])  # Limita a 500 chars para performance
                    sentiments.append(pred.output)
                    probs_pos.append(pred.probas.get("POS", 0))
                    probs_neu.append(pred.probas.get("NEU", 0))
                    probs_neg.append(pred.probas.get("NEG", 0))
                except Exception:
                    # Se falhar, atribui neutro
                    sentiments.append("NEU")
                    probs_pos.append(0)
                    probs_neu.append(1)
                    probs_neg.append(0)
            
            df["sentimento"] = sentiments
            df["prob_POS"] = probs_pos
            df["prob_NEU"] = probs_neu
            df["prob_NEG"] = probs_neg
            
        except Exception as e:
            st.error(f"? Erro durante a análise de sentimentos: {e}")
            st.info("Certifique-se de que a biblioteca pysentimiento está instalada: `pip install pysentimiento`")
            st.stop()

    # ==========================
    # EXIBIÇÃO DE RESULTADOS
    # ==========================
    st.write("### ?? Amostra de mensagens analisadas")
    st.dataframe(df[['author', 'message', 'sentimento', 'prob_POS', 'prob_NEU', 'prob_NEG']].head(20))

    # --- Distribuição geral dos sentimentos ---
    st.write("### ?? Distribuição geral dos sentimentos")
    sent_counts = df["sentimento"].value_counts()
    
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    colors = {"POS": "green", "NEU": "gray", "NEG": "red"}
    bar_colors = [colors.get(s, "blue") for s in sent_counts.index]
    
    ax1.bar(sent_counts.index, sent_counts.values, color=bar_colors)
    ax1.set_xlabel("Sentimento", fontsize=12)
    ax1.set_ylabel("Quantidade", fontsize=12)
    ax1.set_title("Distribuição Geral de Sentimentos", fontsize=14, fontweight='bold')
    
    # Adicionar valores nas barras
    for i, v in enumerate(sent_counts.values):
        ax1.text(i, v + max(sent_counts.values)*0.01, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    # --- Evolução no tempo ---
    st.write("### ?? Evolução diária do sentimento")
    
    try:
        df["datetime"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df_with_dates = df.dropna(subset=["datetime"])
        
        if len(df_with_dates) > 0:
            daily = df_with_dates.groupby(df_with_dates["datetime"].dt.date)["sentimento"].value_counts(normalize=True).unstack(fill_value=0)
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            
            for col in daily.columns:
                color = colors.get(col, "blue")
                ax2.plot(daily.index, daily[col], label=col, marker='o', color=color, linewidth=2)
            
            ax2.legend(loc='best')
            ax2.set_xlabel("Data", fontsize=12)
            ax2.set_ylabel("Proporção", fontsize=12)
            ax2.set_title("Evolução Diária dos Sentimentos", fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
        else:
            st.info("?? Não foi possível gerar gráfico temporal (datas inválidas no arquivo).")
            
    except Exception as e:
        st.info(f"?? Não foi possível gerar gráfico temporal: {e}")

    # --- Ranking dos participantes ---
    st.write("### ?? Ranking por autor (proporção de mensagens por sentimento)")
    
    author_summary = df.groupby("author")["sentimento"].value_counts(normalize=True).unstack(fill_value=0)
    author_counts = df['author'].value_counts()
    author_summary['Total_Msgs'] = author_counts
    author_summary = author_summary.sort_values('Total_Msgs', ascending=False)
    
    # Formatar percentuais
    percentage_cols = [col for col in author_summary.columns if col != 'Total_Msgs']
    for col in percentage_cols:
        author_summary[col] = (author_summary[col] * 100).round(1)
    
    st.dataframe(author_summary.style.format({col: "{:.1f}%" for col in percentage_cols}))

    # --- Estatísticas adicionais ---
    st.write("### ?? Estatísticas gerais")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de mensagens", len(df))
    with col2:
        st.metric("Participantes", df['author'].nunique())
    with col3:
        sentiment_pct = (sent_counts.get("POS", 0) / len(df) * 100)
        st.metric("% Positivo", f"{sentiment_pct:.1f}%")

else:
    st.info("?? Envie o arquivo de conversa (.txt) para começar a análise!")
    st.markdown("""
    ### Como usar:
    1. Abra o WhatsApp no seu celular
    2. Entre no grupo que deseja analisar
    3. Toque nos três pontos (?) > **Mais** > **Exportar conversa**
    4. Escolha **Sem mídia**
    5. Faça upload do arquivo .txt aqui
    """)