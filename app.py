import os
import json
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px
from PIL import Image
import keras
tf.keras = keras

# ==========================================================
# 1. ORGANIZAÇÃO E ESTADO
# ==========================================================
st.set_page_config(page_title='VAE PneumoniaMNIST', layout='wide')

if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=["Execução", "Classificação", "Erro MSE", "Confiança (%)"])

if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

if "analysis_ran" not in st.session_state:
    st.session_state.analysis_ran = False

# Callback para resetar a interface caso o usuário mude os parâmetros
def reset_analysis():
    st.session_state.analysis_ran = False
    st.toast("Parâmetros alterados. Interface resetada para nova execução.", icon="🔄")

# ==========================================================
# 2. ARQUITETURA DO MODELO E CACHE
# ==========================================================
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(latent_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

def build_decoder(latent_dim: int) -> tf.keras.Model:
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    weights_path = os.path.join(models_dir, 'vae_pneumonia.weights.h5')
    config_path = os.path.join(models_dir, 'config.json')

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        return None, f"Arquivos do modelo ausentes em {models_dir}"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    latent_dim = int(config.get('latent_dim', 16))
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    vae = VAE(encoder, decoder)
    
    dummy = tf.zeros((1, 28, 28, 1))
    _ = vae(dummy, training=False)
    vae.load_weights(weights_path)
    return vae, None

def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != 'L': image = image.convert('L')
    if image.size != (28, 28): image = image.resize((28, 28))
    arr = np.array(image).astype('float32')
    if arr.max() > 1.0: arr = arr / 255.0
    return np.expand_dims(np.expand_dims(arr, axis=-1), axis=0)

@st.cache_data
def compute_reconstruction_error(x: np.ndarray, x_recon: np.ndarray) -> float:
    return float(np.mean((x - x_recon) ** 2))

# ==========================================================
# 3. SIDEBAR - INPUT DE DADOS
# ==========================================================
st.sidebar.header("⚙️ Painel de Controle do VAE")

vae, err = load_model()
if err:
    st.sidebar.error(err)
    st.error("⚠️ O modelo não foi carregado. Certifique-se de que os arquivos `.h5` e `.json` estão na pasta `models/`.")
    st.stop() # Empty State robusto
else:
    st.sidebar.success("✅ Modelo Operacional")

# on_change reseta a interface automaticamente
t_normal = st.sidebar.slider("Limiar Normal (MSE)", 0.000, 0.100, 0.040, 0.001, on_change=reset_analysis)
t_borderline = st.sidebar.slider("Limiar Borderline (MSE)", 0.000, 0.200, 0.080, 0.001, on_change=reset_analysis)
sim_latency = st.sidebar.checkbox("Ativar Pipeline Visual (Latência)", value=True)

if st.sidebar.button("🗑️ Limpar Histórico do Sistema"):
    st.session_state.history_df = pd.DataFrame(columns=["Execução", "Classificação", "Erro MSE", "Confiança (%)"])
    st.session_state.feedback_log = []
    st.rerun()

# ==========================================================
# 4. ÁREA PRINCIPAL E EMPTY STATE
# ==========================================================
st.title("🩻 VAE PneumoniaMNIST")
st.markdown("Sistema de Triagem de Pneumonia com Análise de Erro de Reconstrução.")

uploaded = st.file_uploader("Insira a imagem de Raio-X (PNG/JPG)", type=["png", "jpg", "jpeg"])

if not uploaded:
    st.info("👈 Configure os parâmetros laterais e faça o upload de uma imagem para iniciar.")
    st.stop()

# Separação clara entre Ação e Estado
if st.button("🔍 Iniciar Pipeline de Análise"):
    st.session_state.analysis_ran = True

# ==========================================================
# 5. EXECUÇÃO E DESIGN PARA LATÊNCIA
# ==========================================================
if st.session_state.analysis_ran:
    
    # Uso de st.status combinado com st.progress
    if sim_latency:
        with st.status("Processando imagem através do VAE...", expanded=True) as status:
            st.write("Pré-processando e normalizando canais...")
            time.sleep(0.5)
            st.write("Codificando no espaço latente...")
            time.sleep(0.5)
            st.write("Calculando divergência de reconstrução...")
            prog = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                prog.progress(i + 1)
            status.update(label="Análise Concluída", state="complete", expanded=False)

    # Inferência
    image = Image.open(io.BytesIO(uploaded.read()))
    x = preprocess_image(image)
    recon = vae(x, training=False).numpy()
    mse = compute_reconstruction_error(x, recon)

    if mse < t_normal:
        status, cor = "NORMAL", "green"
    elif mse < t_borderline:
        status, cor = "BORDERLINE", "orange"
    else:
        status, cor = "POSSÍVEL PNEUMONIA", "red"

    conf = max(0, int((1 - mse) * 100)) if mse < 1 else 0

    # Atualização de Histórico Segura
    novo_dado = pd.DataFrame([{
        "Execução": len(st.session_state.history_df) + 1,
        "Classificação": status,
        "Erro MSE": round(mse, 6),
        "Confiança (%)": conf
    }])
    if len(st.session_state.history_df) == 0 or st.session_state.history_df.iloc[-1]["Erro MSE"] != round(mse, 6):
        st.session_state.history_df = pd.concat([st.session_state.history_df, novo_dado], ignore_index=True)

    # Tabs separando contextos, não etapas
    tab_pred, tab_hist, tab_mon = st.tabs(["🎯 Decisão e Revisão", "📊 Histórico Operacional", "📈 Saúde do Modelo"])

    # ==========================================================
    # 6. CONFIDENCE UI E HUMAN-IN-THE-LOOP
    # ==========================================================
    with tab_pred:
        col_orig, col_recon = st.columns(2)
        with col_orig:
            st.caption("Imagem Original")
            st.image(x[0].squeeze(), width=150, clamp=True)
        with col_recon:
            st.caption("Reconstrução VAE")
            st.image(recon[0].squeeze(), width=150, clamp=True)

        st.subheader(f"Diagnóstico Sugerido: :{cor}[{status}]")
        
        # Orientação Comportamental baseada na confiança
        if conf >= 85:
            st.success(f"Alta Confiança ({conf}%). Padrão claro identificado.")
        elif conf >= 60:
            st.warning(f"Confiança Moderada ({conf}%). O VAE encontrou divergências. Revisão recomendada.")
        else:
            st.error(f"Baixa Confiança ({conf}%). Reconstrução falhou. Análise médica urgente necessária.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Erro de Reconstrução (MSE)", f"{mse:.5f}")
        c2.metric("Nível de Confiança", f"{conf}%")
        c3.metric("Thresholds Utilizados", f"{t_normal} / {t_borderline}")

        st.divider()
        st.markdown("#### 👨‍⚕️ Validação do Especialista (Human-in-the-Loop)")
        st.caption("A decisão da IA está alinhada com a avaliação clínica?")
        btn1, btn2 = st.columns(2)
        if btn1.button("✅ Confirmar Diagnóstico da IA"):
            st.session_state.feedback_log.append(True)
            st.toast("Feedback positivo registrado.", icon="📈")
        if btn2.button("❌ Corrigir Diagnóstico da IA"):
            st.session_state.feedback_log.append(False)
            st.toast("Divergência registrada para auditoria.", icon="📉")

    # ==========================================================
    # 7. MONITORAMENTO E HISTÓRICO
    # ==========================================================
    with tab_hist:
        st.markdown("#### Histórico de Classificações")
        st.dataframe(
            st.session_state.history_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confiança (%)": st.column_config.ProgressColumn("Nível de Confiança", min_value=0, max_value=100, format="%d%%"),
                "Erro MSE": st.column_config.NumberColumn(format="%.6f")
            }
        )

    with tab_mon:
        # Métricas agregadas e alerta de degradação
        if st.session_state.feedback_log:
            total_feedbacks = len(st.session_state.feedback_log)
            acertos = sum(st.session_state.feedback_log)
            acuracia = acertos / total_feedbacks
            
            st.markdown("#### Métricas Agregadas")
            m1, m2, m3 = st.columns(3)
            m1.metric("Feedbacks Recebidos", total_feedbacks)
            m2.metric("Diagnósticos Corretos", acertos)
            m3.metric("Acurácia Percebida", f"{acuracia*100:.1f}%")

            # Alerta de degradação explícito
            if acuracia < 0.7 and total_feedbacks >= 3:
                st.error("🚨 **ALERTA DE DEGRADAÇÃO:** A acurácia do modelo caiu abaixo de 70% com base na validação humana. Considere retreinar o VAE ou ajustar os thresholds.")
            else:
                st.success("✅ Modelo operando dentro das margens de segurança estabelecidas.")
        else:
            st.info("O monitoramento de degradação será ativado após as primeiras validações humanas na aba anterior.")

        st.divider()

        if not st.session_state.history_df.empty and len(st.session_state.history_df) > 1:
            # Gráfico mostrando evolução da confiabilidade ao longo das execuções
            fig = px.line(
                st.session_state.history_df, x="Execução", y="Confiança (%)", 
                title="Evolução da Confiança por Interação",
                markers=True,
                color_discrete_sequence=["#1f77b4"]
            )
            # Linha de alerta no gráfico
            fig.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Margem de Risco")
            st.plotly_chart(fig, use_container_width=True)