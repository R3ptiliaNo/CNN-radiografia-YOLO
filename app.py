import streamlit as st
import numpy as np
from PIL import Image
import time
import os
from ultralytics import YOLO

# --- CONFIGURACIÓN DE PÁGINA (Layout Wide para estilo Consola) ---
st.set_page_config(
    page_title="Asistente Diagnóstico IA",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS PERSONALIZADO (Estilo Dark Console + Botones Visibles) ---
st.markdown("""
<style>
    /* Forzar fondo oscuro */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Botones siempre visibles */
    .stButton>button {
        background-color: #1f2937;
        color: #ffffff;
        border: 1px solid #374151;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #374151;
        border-color: #00e5ff;
        color: #00e5ff;
    }
    
    /* Títulos principales */
    .main-header {
        text-align: center;
        font-family: 'Courier New', Courier, monospace;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        margin-bottom: 0px;
    }
    .main-header span {
        color: #00e5ff; /* Cyan brillante */
    }
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 40px;
    }
    
    /* Contenedores tipo tarjeta */
    .card-container {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Cajas de Resultados */
    .resultado-anomalia {
        border: 2px solid #ef4444; /* Rojo */
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background: rgba(239, 68, 68, 0.1);
    }
    .resultado-anomalia h2 { color: #ef4444; margin: 0; font-size: 2rem;}
    
    .resultado-sano {
        border: 2px solid #10b981; /* Verde */
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background: rgba(16, 185, 129, 0.1);
    }
    .resultado-sano h2 { color: #10b981; margin: 0; font-size: 2rem;}
    
    /* Pequeños textos */
    .mini-text { font-size: 0.8rem; color: #94a3b8; }
    .status-dot { color: #10b981; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- ENCABEZADO ---
st.markdown("""
<div class="main-header">
    <h1>🩻 Asistente Diagnóstico: <span>Neumonía</span></h1>
    <div class="sub-header">Arquitectura en Cascada basada en YOLOv11 (Transfer Learning)</div>
</div>
""", unsafe_allow_html=True)

# --- CARGA DE MODELOS ---
@st.cache_resource
def cargar_modelos():
    # En la nube de Streamlit, se asume que los archivos están en la misma ruta
    modelo_a = YOLO("modelo_fase_a.pt") if os.path.exists("modelo_fase_a.pt") else None
    modelo_b = YOLO("modelo_fase_b.pt") if os.path.exists("modelo_fase_b.pt") else None
    return modelo_a, modelo_b

modelo_a, modelo_b = cargar_modelos()

# --- MANEJO DE ESTADO (Para la botonera de Casos de Prueba) ---
if 'imagen_actual' not in st.session_state:
    st.session_state.imagen_actual = None

def cargar_caso_prueba(ruta):
    if os.path.exists(ruta):
        st.session_state.imagen_actual = Image.open(ruta)
    else:
        st.error(f"Falta el archivo: {ruta}. Asegúrate de que esté en la misma carpeta que la app.")

# --- MAQUETADO DE 3 COLUMNAS ---
col1, col2, col3 = st.columns([1, 2.5, 1], gap="large")

# ==========================================
# COLUMNA 1: CONTROLES
# ==========================================
with col1:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### ☁️ Cargar RX")
    st.markdown("<p class='mini-text'>Toca para subir o arrastra (Máx 5MB)</p>", unsafe_allow_html=True)
    archivo_subido = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if archivo_subido:
        st.session_state.imagen_actual = Image.open(archivo_subido)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### CASOS DE PRUEBA")
    # Botonera para inyectar imágenes precargadas (imágenes sueltas en la misma carpeta)
    if st.button("🫁 Cargar Pulmón Sano", use_container_width=True):
        cargar_caso_prueba("sano.jpeg")
    if st.button("🦠 Cargar Infección (Bacteria)", use_container_width=True):
        cargar_caso_prueba("bacteria.jpeg")
    if st.button("🧬 Cargar Infección (Virus)", use_container_width=True):
        cargar_caso_prueba("virus.jpeg")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# VARIABLES GLOBALES DE INFERENCIA
# ==========================================
clase_final = ""
conf_final = 0.0
clase_etiologia = ""
t_pre = 0.0
t_inf = 0.0

# ==========================================
# COLUMNA 2: VISOR PRINCIPAL
# ==========================================
with col2:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span class="mini-text"><span class="status-dot">●</span> SISTEMA ONLINE</span>
    </div>
    """, unsafe_allow_html=True)
    
    visor_contenedor = st.empty()
    
    if st.session_state.imagen_actual is not None:
        if modelo_a is None or modelo_b is None:
            visor_contenedor.error("Modelos no encontrados. Sube los archivos .pt al servidor.")
        else:
            # Botón de acción flotante
            if st.button("🚀 INICIAR DIAGNÓSTICO", type="primary", use_container_width=True):
                # 1. Preparar imagen
                t_inicio = time.time()
                img_array = np.array(st.session_state.imagen_actual.convert('RGB'))
                t_pre = time.time() - t_inicio
                
                # 2. Inferencia Triaje (Fase A)
                with st.spinner("Analizando placa radiográfica..."):
                    t_inicio_inf = time.time()
                    pred_a = modelo_a(img_array, verbose=False)[0]
                    clase_final = pred_a.names[pred_a.probs.top1]
                    conf_final = pred_a.probs.top1conf.item()
                    
                    if clase_final == "PNEUMONIA":
                        # Fase B
                        pred_b = modelo_b(img_array, verbose=False)[0]
                        clase_etiologia = pred_b.names[pred_b.probs.top1]
                    t_inf = time.time() - t_inicio_inf
                
                # Mostramos la imagen limpia sin alteraciones
                visor_contenedor.image(st.session_state.imagen_actual, use_container_width=True)
            else:
                # Si no apretaron el botón, solo mostramos la original
                visor_contenedor.image(st.session_state.imagen_actual, use_container_width=True)
    else:
        # Estado vacío (Pantalla negra)
        visor_contenedor.markdown(
            '<div style="width:100%; height:500px; background-color:#000; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#333;">Esperando imagen...</div>', 
            unsafe_allow_html=True
        )

# ==========================================
# COLUMNA 3: RESULTADOS Y LATENCIA
# ==========================================
with col3:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("<p class='mini-text'>RESULTADO</p>", unsafe_allow_html=True)
    
    if clase_final == "PNEUMONIA":
        st.markdown(f"""
        <div class="resultado-anomalia">
            <h2>⚠️ ANOMALÍA</h2>
            <p style="margin-bottom:0;">Causa: {clase_etiologia}</p>
            <p class="mini-text">Confianza {conf_final*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recomendación Médica
        if clase_etiologia == "BACTERIA":
            st.warning("Recomendación Médica: Evaluar inicio de tratamiento antibiótico.")
        else:
            st.warning("Recomendación Médica: Cuadro compatible con infección viral. Evaluar tratamiento sintomático/antiviral.")

    elif clase_final == "NORMAL":
        st.markdown(f"""
        <div class="resultado-sano">
            <h2>✅ SANO</h2>
            <p style="margin-bottom:0;">Sin Infiltrados</p>
            <p class="mini-text">Confianza {conf_final*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("Fin del diagnóstico. No se detectan infiltrados alveolares.")
        
    else:
        st.markdown("""
        <div style="border: 2px dashed #334155; border-radius: 12px; padding: 20px; text-align: center; color: #64748b;">
            <h2>---</h2>
            <p class="mini-text">A la espera de datos</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- METRICAS DE LATENCIA (EN SEGUNDOS) ---
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### LATENCIA PIPELINE")
    
    st.markdown(f"<div style='display:flex; justify-content:space-between;'><span class='mini-text'>PREPROCESAMIENTO</span><span class='mini-text'>{t_pre:.3f} s</span></div>", unsafe_allow_html=True)
    st.progress(min(t_pre / 0.1, 1.0)) # Barra simbólica (Límite visual 0.1s)
    
    st.markdown(f"<div style='display:flex; justify-content:space-between; margin-top:10px;'><span class='mini-text'>INFERENCIA (GPU/CPU)</span><span class='mini-text'>{t_inf:.3f} s</span></div>", unsafe_allow_html=True)
    st.progress(min(t_inf / 1.0, 1.0)) # Barra simbólica (Límite visual 1s)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- INSTRUCCIONES ---
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### Instrucciones de uso")
    st.markdown("""
    <p class="mini-text">
    1. Suba una radiografía de tórax frontal.<br>
    2. El sistema evaluará primero la presencia de infiltrados (Triaje).<br>
    3. Si detecta anomalías, un segundo modelo determinará la probable etiología (Bacteriana/Viral).<br><br>
    <em>Este es un sistema de soporte al diagnóstico (CAD) y no sustituye el criterio médico profesional.</em>
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
