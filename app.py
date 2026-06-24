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
                          patch_size=32, stride=16, mask_value=128):
    """
    Genera un mapa de oclusión para una imagen y una clase objetivo.
    Retorna un heatmap (array 2D) y la imagen superpuesta (PIL).
    """
    # Convertir a numpy y asegurar RGB
    img_np = np.array(image_pil.convert('RGB'))
    h, w, _ = img_np.shape

    # Inicializar mapa de importancia
    importance_map = np.zeros((h, w), dtype=np.float32)

    # Obtener la confianza original para la clase objetivo
    pred_original = yolo_classifier.predict(img_np)
    # Asegurar que probs sea un array numpy
    probs_original = pred_original["raw_result"].probs.data.cpu().numpy()
    conf_original = probs_original[target_class_id]

    # Recorrer la imagen con el parche
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Crear copia de la imagen
            img_copy = img_np.copy()
            # Ocluir el parche con un valor constante (gris medio)
            img_copy[y:y+patch_size, x:x+patch_size] = mask_value

            # Predecir con la imagen ocluida
            pred_occluded = yolo_classifier.predict(img_copy)
            probs_occluded = pred_occluded["raw_result"].probs.data.cpu().numpy()
            conf_occluded = probs_occluded[target_class_id]

            # Caída de confianza (drop)
            drop = conf_original - conf_occluded

            # Asignar el máximo entre el valor actual y drop a toda la región
            # Usamos np.maximum para comparar array con escalar
            importance_map[y:y+patch_size, x:x+patch_size] = np.maximum(
                importance_map[y:y+patch_size, x:x+patch_size],
                drop
            )

    # Normalizar entre 0 y 1
    if importance_map.max() > 0:
        importance_map = importance_map / importance_map.max()

    # Generar visualización con matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    im = ax.imshow(importance_map, cmap='jet', alpha=0.5, interpolation='bilinear')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # Convertir a imagen PIL
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    heatmap_pil = Image.open(buf)
    plt.close(fig)

    return importance_map, heatmap_pil


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
