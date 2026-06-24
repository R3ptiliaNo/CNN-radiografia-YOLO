import streamlit as st
import numpy as np
from PIL import Image
import time
import os
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from io import BytesIO

# ============================================
# CLASE YoloClassifier (con soporte para oclusión)
# ============================================
class YoloClassifier:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"🔄 Cargando modelo YOLO desde {self.model_path}...")
        self.model = YOLO(self.model_path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print("✅ Modelo cargado correctamente.")

    def predict(self, image: np.ndarray) -> dict:
        """
        Realiza la predicción y devuelve información detallada.
        """
        results = self.model.predict(image, verbose=False)
        result = results[0]
        probs = result.probs
        top1_index = probs.top1
        confidence = probs.top1conf.item()
        class_name = result.names[top1_index]

        return {
            "class_id": top1_index,
            "class_name": class_name,
            "confidence": confidence,
            "raw_result": result,
            "internal_model": self.model.model  # modelo PyTorch subyacente
        }


# ============================================
# FUNCIÓN PARA MAPA DE OCUSIÓN
# ============================================
def compute_occlusion_map(image_pil, yolo_classifier, target_class_id,
                          patch_size=70, stride=20, mask_value=0):
    """
    Genera un mapa de oclusión rápido basado en el enfoque del código original.
    - Parche grande y stride amplio para acelerar.
    - Umbral dinámico: para clase 'NORMAL' solo pinta caídas > 40%, para 'PNEUMONIA' > 5%.
    - Retorna la imagen superpuesta (PIL).
    """
    # Redimensionar a 224x224 (tamaño de entrada del modelo)
    img_pil = image_pil.convert('RGB').resize((224, 224))
    img_np = np.array(img_pil)

    # Predicción base
    pred_base = yolo_classifier.predict(img_np)
    probs_base = pred_base["raw_result"].probs.data.cpu().numpy()
    conf_base = probs_base[target_class_id]
    class_name = pred_base["class_name"]

    # Inicializar mapa de importancia
    importance_map = np.zeros((224, 224), dtype=np.float32)

    # Deslizar el parche
    for y in range(0, 224, stride):
        for x in range(0, 224, stride):
            # Copia con oclusión (negro = máscara)
            img_occluded = img_np.copy()
            y_end = min(224, y + patch_size)
            x_end = min(224, x + patch_size)
            img_occluded[y:y_end, x:x_end] = mask_value

            # Inferencia
            pred_occluded = yolo_classifier.predict(img_occluded)
            probs_occluded = pred_occluded["raw_result"].probs.data.cpu().numpy()
            conf_occluded = probs_occluded[target_class_id]

            # Caída de confianza
            drop = conf_base - conf_occluded
            importance_map[y:y_end, x:x_end] = max(importance_map[y:y_end, x:x_end].max(), drop)

    # Normalizar
    max_drop = importance_map.max()
    if max_drop == 0:
        max_drop = 1e-6

    # Umbral dinámico según clase
    if class_name == 'NORMAL':
        threshold = 0.40  # solo pinta si cae > 40%
    else:
        threshold = 0.05  # con que caiga > 5% ya lo marca

    # Aplicar umbral: si la caída máxima es menor que el umbral, no pintamos nada
    if max_drop < threshold:
        # Escalamos para que el mapa sea casi invisible (o lo dejamos en cero)
        importance_map = np.zeros_like(importance_map)
    else:
        # Normalizar al rango [0,1] y luego escalar para que el umbral sea el mínimo visible
        importance_map = np.clip((importance_map - threshold) / (max_drop - threshold), 0, 1)

    # Redimensionar mapa al tamaño original (para superposición)
    orig_w, orig_h = image_pil.size
    importance_map_resized = cv2.resize(importance_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Crear mapa de color (jet) y superponer
    heatmap_color = cv2.applyColorMap(np.uint8(255 * importance_map_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Imagen original (en tamaño original)
    img_original = np.array(image_pil.convert('RGB'))
    # Superponer con transparencia (0.6 imagen, 0.4 mapa)
    superimposed = cv2.addWeighted(img_original, 0.6, heatmap_rgb, 0.4, 0)

    # Convertir a PIL
    superimposed_pil = Image.fromarray(superimposed)

    return importance_map, superimposed_pil


# ============================================
# CONFIGURACIÓN DE PÁGINA
# ============================================
st.set_page_config(
    page_title="Asistente Diagnóstico IA",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CSS PERSONALIZADO
# ============================================
st.markdown("""
<style>
    /* ... (igual que antes, no lo repito por brevedad, pero debe ir completo) ... */
</style>
""", unsafe_allow_html=True)

# ============================================
# ENCABEZADO
# ============================================
st.markdown("""
<div class="main-header">
    <h1>🩻 Asistente Diagnóstico: <span>Neumonía</span></h1>
    <div class="sub-header">Arquitectura en Cascada basada en YOLOv11 (Transfer Learning)</div>
</div>
""", unsafe_allow_html=True)

# ============================================
# CARGA DE MODELOS (usando YoloClassifier)
# ============================================
@st.cache_resource
def cargar_modelos():
    modelo_a = None
    modelo_b = None
    if os.path.exists("modelo_fase_a.pt"):
        modelo_a = YoloClassifier("modelo_fase_a.pt")
    if os.path.exists("modelo_fase_b.pt"):
        modelo_b = YoloClassifier("modelo_fase_b.pt")
    return modelo_a, modelo_b

modelo_a, modelo_b = cargar_modelos()

# ============================================
# ESTADO DE LA SESIÓN
# ============================================
if 'imagen_actual' not in st.session_state:
    st.session_state.imagen_actual = None
if 'diagnostico' not in st.session_state:
    st.session_state.diagnostico = {
        "clase_final": "",
        "conf_final": 0.0,
        "clase_etiologia": "",
        "t_pre": 0.0,
        "t_inf": 0.0,
        "imagen_procesada": None  # Para mostrar la imagen original o con heatmap
    }
if 'mostrar_oclusion' not in st.session_state:
    st.session_state.mostrar_oclusion = False

def cargar_caso_prueba(ruta):
    if os.path.exists(ruta):
        st.session_state.imagen_actual = Image.open(ruta)
        st.session_state.mostrar_oclusion = False  # resetear al cambiar imagen

# ============================================
# MAQUETADO DE 3 COLUMNAS
# ============================================
col1, col2, col3 = st.columns([1, 2.5, 1], gap="large")

# ---------- COLUMNA 1: CONTROLES ----------
with col1:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### ☁️ Cargar RX")
    st.markdown("<p class='mini-text'>Toca para subir o arrastra (Máx 5MB)</p>", unsafe_allow_html=True)
    archivo_subido = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if archivo_subido:
        st.session_state.imagen_actual = Image.open(archivo_subido)
        st.session_state.mostrar_oclusion = False
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### CASOS DE PRUEBA")
    if st.button("🫁 Cargar Pulmón Sano", use_container_width=True):
        cargar_caso_prueba("sano.jpeg")
    if st.button("🦠 Cargar Infección (Bacteria)", use_container_width=True):
        cargar_caso_prueba("bacteria.jpeg")
    if st.button("🧬 Cargar Infección (Virus)", use_container_width=True):
        cargar_caso_prueba("virus.jpeg")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- COLUMNA 2: VISOR PRINCIPAL ----------
with col2:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span class="mini-text"><span class="status-dot">●</span> SISTEMA ONLINE</span>
    </div>
    """, unsafe_allow_html=True)
    
    visor_contenedor = st.empty()

    # Decidir qué imagen mostrar
    if st.session_state.imagen_actual is not None:
        if st.session_state.mostrar_oclusion and st.session_state.diagnostico.get("imagen_oclusion") is not None:
            img_a_mostrar = st.session_state.diagnostico["imagen_oclusion"]
        else:
            img_a_mostrar = st.session_state.imagen_actual
        visor_contenedor.image(img_a_mostrar, use_container_width=True)
    else:
        visor_contenedor.markdown(
            '<div style="width:100%; height:500px; background-color:#000; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#333;">Esperando imagen...</div>', 
            unsafe_allow_html=True
        )

# ---------- COLUMNA 3: RESULTADOS Y LATENCIA ----------
with col3:
    # Botón de diagnóstico
    if st.button("🚀 INICIAR DIAGNÓSTICO", type="primary", use_container_width=True):
        if st.session_state.imagen_actual is None:
            st.warning("Primero carga una radiografía.")
        elif modelo_a is None or modelo_b is None:
            st.error("Modelos no encontrados. Sube los archivos .pt al servidor.")
        else:
            # 1. Preprocesamiento
            t_inicio = time.time()
            img_array = np.array(st.session_state.imagen_actual.convert('RGB'))
            t_pre = time.time() - t_inicio

            # 2. Inferencia Fase A (triaje)
            with st.spinner("Analizando placa radiográfica..."):
                t_inicio_inf = time.time()
                pred_a = modelo_a.predict(img_array)
                clase_final = pred_a["class_name"]
                conf_final = pred_a["confidence"]

                clase_etiologia = ""
                if clase_final == "PNEUMONIA":
                    # Fase B
                    pred_b = modelo_b.predict(img_array)
                    clase_etiologia = pred_b["class_name"]
                t_inf = time.time() - t_inicio_inf

            # Guardar en sesión
            st.session_state.diagnostico = {
                "clase_final": clase_final,
                "conf_final": conf_final,
                "clase_etiologia": clase_etiologia,
                "t_pre": t_pre,
                "t_inf": t_inf,
                "imagen_oclusion": None,  # resetear
                "modelo_usado": modelo_b if clase_final == "PNEUMONIA" else modelo_a,
                "target_class_id": pred_a["class_id"] if clase_final != "PNEUMONIA" else pred_b["class_id"]
            }
            st.session_state.mostrar_oclusion = False
            # Forzar actualización
            st.rerun()

    # Mostrar resultados
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("<p class='mini-text'>RESULTADO</p>", unsafe_allow_html=True)
    
    diag = st.session_state.diagnostico
    if diag["clase_final"] == "PNEUMONIA":
        st.markdown(f"""
        <div class="resultado-anomalia">
            <h2>⚠️ ANOMALÍA</h2>
            <p style="margin-bottom:0;">Causa: {diag['clase_etiologia']}</p>
            <p class="mini-text">Confianza {diag['conf_final']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        if diag["clase_etiologia"] == "BACTERIA":
            st.warning("Recomendación Médica: Evaluar inicio de tratamiento antibiótico.")
        else:
            st.warning("Recomendación Médica: Cuadro compatible con infección viral. Evaluar tratamiento sintomático/antiviral.")
    elif diag["clase_final"] == "NORMAL":
        st.markdown(f"""
        <div class="resultado-sano">
            <h2>✅ SANO</h2>
            <p style="margin-bottom:0;">Sin Infiltrados</p>
            <p class="mini-text">Confianza {diag['conf_final']*100:.1f}%</p>
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

    # Botón para mapa de oclusión (solo si hay diagnóstico y clase es PNEUMONIA o NORMAL)
    if diag["clase_final"] in ["PNEUMONIA", "NORMAL"] and st.session_state.imagen_actual is not None:
        if st.button("🧩 Mostrar mapa de oclusión", use_container_width=True):
            with st.spinner("Generando mapa de oclusión... (puede tardar unos segundos)"):
                modelo = diag["modelo_usado"]
                target_id = diag["target_class_id"]
                # Calcular mapa
                _, heatmap_pil = compute_occlusion_map(
                    st.session_state.imagen_actual,
                    modelo,
                    target_id,
                    patch_size=32,
                    stride=16
                )
                st.session_state.diagnostico["imagen_oclusion"] = heatmap_pil
                st.session_state.mostrar_oclusion = True
                st.rerun()

    # ---- Métricas de latencia ----
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### LATENCIA PIPELINE")
    t_pre = diag["t_pre"]
    t_inf = diag["t_inf"]
    st.markdown(f"<div style='display:flex; justify-content:space-between;'><span class='mini-text'>PREPROCESAMIENTO</span><span class='mini-text'>{t_pre:.3f} s</span></div>", unsafe_allow_html=True)
    st.progress(min(t_pre / 0.1, 1.0))
    st.markdown(f"<div style='display:flex; justify-content:space-between; margin-top:10px;'><span class='mini-text'>INFERENCIA (GPU/CPU)</span><span class='mini-text'>{t_inf:.3f} s</span></div>", unsafe_allow_html=True)
    st.progress(min(t_inf / 1.0, 1.0))
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Instrucciones ----
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### Instrucciones de uso")
    st.markdown("""
    <p class="mini-text">
    1. Suba una radiografía de tórax frontal.<br>
    2. El sistema evaluará primero la presencia de infiltrados (Triaje).<br>
    3. Si detecta anomalías, un segundo modelo determinará la probable etiología (Bacteriana/Viral).<br>
    4. Pulse "Mostrar mapa de oclusión" para visualizar las zonas que influyen en la decisión.<br><br>
    <em>Este es un sistema de soporte al diagnóstico (CAD) y no sustituye el criterio médico profesional.</em>
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
