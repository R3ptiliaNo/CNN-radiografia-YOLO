import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
from ultralytics import YOLO

# --- CONFIGURACIÓN DE PÁGINA (Layout Wide para estilo Consola) ---
st.set_page_config(
    page_title="Consola de Diagnóstico IA",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS PERSONALIZADO (Estilo Dark Console) ---
st.markdown("""
<style>
    /* Forzar fondo oscuro */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
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
    <h1>Consola de <span>Diagnóstico IA</span></h1>
    <div class="sub-header">Visualiza cómo una red neuronal detecta anomalías en radiografías de tórax.</div>
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

# --- FUNCIÓN DE MAPA DE CALOR OPTIMIZADA PARA WEB ---
def generar_mapa_web(img_array, modelo, clase_predicha, prob_base):
    # Reducimos un poco la resolución y agrandamos el paso para que no demore tanto en la web
    img_resized = cv2.resize(img_array, (224, 224))
    mapa_calor = np.zeros((224, 224), dtype=np.float32)
    tamano_parche = 60
    paso = 30 # Saltos grandes para procesar rápido
    
    idx_clase = list(modelo.names.values()).index(clase_predicha)
    
    for y in range(0, 224, paso):
        for x in range(0, 224, paso):
            img_oclusion = img_resized.copy()
            y_fin, x_fin = min(224, y + tamano_parche), min(224, x + tamano_parche)
            img_oclusion[y:y_fin, x:x_fin] = 0 # Oclusión negra
            
            res = modelo(img_oclusion, verbose=False)[0]
            prob_nueva = res.probs.data[idx_clase].item()
            mapa_calor[y:y_fin, x:x_fin] += (prob_base - prob_nueva)
            
    mapa_calor = np.maximum(mapa_calor, 0)
    max_caida = np.max(mapa_calor)
    
    umbral = 0.40 if clase_predicha == 'NORMAL' else 0.05
    if max_caida > umbral:  
        mapa_calor /= max_caida  
    else:
        mapa_calor /= umbral if umbral > 0 else 1
        
    mapa_redimensionado = cv2.resize(mapa_calor, (img_array.shape[1], img_array.shape[0]))
    mapa_color = cv2.applyColorMap(np.uint8(255 * mapa_redimensionado), cv2.COLORMAP_JET)
    mapa_color_rgb = cv2.cvtColor(mapa_color, cv2.COLOR_BGR2RGB)
    
    img_final = cv2.addWeighted(img_array, 0.6, mapa_color_rgb, 0.4, 0)
    return img_final

# --- MANEJO DE ESTADO (Para la botonera de Casos de Prueba) ---
if 'imagen_actual' not in st.session_state:
    st.session_state.imagen_actual = None

def cargar_caso_prueba(ruta):
    if os.path.exists(ruta):
        st.session_state.imagen_actual = Image.open(ruta)
    else:
        st.error(f"Falta el archivo: {ruta}. Asegúrate de crear la carpeta 'casos_prueba'.")

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
    # Botonera para inyectar imágenes precargadas
    if st.button("🫁 Cargar Pulmón Sano", use_container_width=True):
        cargar_caso_prueba("casos_prueba/sano.jpeg")
    if st.button("🦠 Cargar Infección (Bacteria)", use_container_width=True):
        cargar_caso_prueba("casos_prueba/bacteria.jpeg")
    if st.button("🧬 Cargar Infección (Virus)", use_container_width=True):
        cargar_caso_prueba("casos_prueba/virus.jpeg")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# VARIABLES GLOBALES DE INFERENCIA
# ==========================================
img_para_mostrar = None
clase_final = ""
conf_final = 0.0
clase_etiologia = ""
t_pre = t_inf = t_grad = 0

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
                t_pre = int((time.time() - t_inicio) * 1000)
                
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
                    t_inf = int((time.time() - t_inicio_inf) * 1000)
                
                # 3. Generar Mapa de Calor
                with st.spinner("Generando mapa de oclusión (Explicabilidad)..."):
                    t_inicio_grad = time.time()
                    # Seleccionamos qué modelo usar para el mapa
                    modelo_mapa = modelo_a if clase_final == "NORMAL" else modelo_b
                    clase_mapa = clase_final if clase_final == "NORMAL" else clase_etiologia
                    
                    # Para el mapa de neumonía, usamos la confianza de Fase B si es necesario, 
                    # pero mejor usar la confianza de fase A para oclusión
                    prob_base_mapa = conf_final
                    
                    img_mapa = generar_mapa_web(img_array, modelo_a, clase_final, conf_final)
                    t_grad = int((time.time() - t_inicio_grad) * 1000)
                    
                    visor_contenedor.image(img_mapa, use_container_width=True)
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
            <h2>⚠️ ANOMALY</h2>
            <p style="margin-bottom:0;">{clase_etiologia}</p>
            <p class="mini-text">Confianza {conf_final*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    elif clase_final == "NORMAL":
        st.markdown(f"""
        <div class="resultado-sano">
            <h2>✅ NORMAL</h2>
            <p style="margin-bottom:0;">Sin Infiltrados</p>
            <p class="mini-text">Confianza {conf_final*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border: 2px dashed #334155; border-radius: 12px; padding: 20px; text-align: center; color: #64748b;">
            <h2>---</h2>
            <p class="mini-text">A la espera de datos</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- METRICAS DE LATENCIA ---
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### LATENCIA PIPELINE")
    
    st.markdown(f"<div style='display:flex; justify-content:space-between;'><span class='mini-text'>PREPROCESAMIENTO</span><span class='mini-text'>{t_pre} ms</span></div>", unsafe_allow_html=True)
    st.progress(min(t_pre / 100, 1.0)) # Barra simbólica
    
    st.markdown(f"<div style='display:flex; justify-content:space-between; margin-top:10px;'><span class='mini-text'>INFERENCIA (GPU/CPU)</span><span class='mini-text'>{t_inf} ms</span></div>", unsafe_allow_html=True)
    st.progress(min(t_inf / 500, 1.0))
    
    st.markdown(f"<div style='display:flex; justify-content:space-between; margin-top:10px;'><span class='mini-text'>XAI OCLUSIÓN</span><span class='mini-text'>{t_grad} ms</span></div>", unsafe_allow_html=True)
    st.progress(min(t_grad / 5000, 1.0))
    st.markdown('</div>', unsafe_allow_html=True)

    # --- INTERPRETACION ---
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### Interpretación")
    st.markdown("""
    <p class="mini-text">
    Las zonas iluminadas indican los patrones visuales críticos (gradientes de caída) utilizados por el modelo para establecer la predicción actual.
    <br><br>
    <em>Este es un sistema de soporte al diagnóstico (CAD) y no sustituye el criterio médico profesional.</em>
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
