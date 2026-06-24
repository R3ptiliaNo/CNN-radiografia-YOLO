import streamlit as st
import numpy as np
from PIL import Image
import time
import os
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente Diagnóstico IA",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS PERSONALIZADO (igual que antes) ---
# ... (todo el CSS que ya tenías)

# --- CLASE YOLOCLASSIFIER (adaptada) ---
class YoloClassifier:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"🔄 Cargando modelo YOLO desde {self.model_path}...")
        self.model = YOLO(self.model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print("✅ Modelo cargado correctamente.")

    def predict(self, image: np.ndarray, return_raw=False):
        """
        Retorna dict con predicción. Si return_raw=True, retorna también el resultado completo.
        """
        results = self.model.predict(image, verbose=False)
        result = results[0]
        probs = result.probs
        top1_index = probs.top1
        confidence = probs.top1conf.item()
        class_name = result.names[top1_index]

        out = {
            "class_id": top1_index,
            "class_name": class_name,
            "confidence": confidence,
        }
        if return_raw:
            out["raw_result"] = result
            out["internal_model"] = self.model.model
        return out

# --- FUNCIÓN DE OCULUSIÓN (simple) ---
def occlusion_map(image: np.ndarray, model, patch_size=32, stride=16, target_class=None):
    """
    Genera un mapa de oclusión para la imagen.
    - image: np.array (H,W,3) en RGB
    - model: instancia de YoloClassifier
    - patch_size: tamaño del cuadrado a oscurecer
    - stride: paso entre parches
    - target_class: si es None, usa la clase predicha por el modelo
    Retorna heatmap (H,W) con valores de cambio en confianza.
    """
    # Obtener predicción original
    pred_orig = model.predict(image)
    if target_class is None:
        target_class = pred_orig["class_id"]
    base_conf = pred_orig["confidence"]
    
    h, w, _ = image.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    # Crear copia de la imagen para modificar
    img_copy = image.copy()
    
    # Recorrer la imagen con ventanas
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Guardar el parche original
            patch = img_copy[y:y+patch_size, x:x+patch_size].copy()
            # Oscurecer el parche (poner a 0)
            img_copy[y:y+patch_size, x:x+patch_size] = 0
            # Predecir con parche oscurecido
            pred_occ = model.predict(img_copy)
            # Obtener confianza para la clase objetivo
            conf_occ = pred_occ["confidence"] if pred_occ["class_id"] == target_class else 0.0
            # Cambio en confianza (disminución)
            delta = base_conf - conf_occ
            # Asignar el valor al centro del parche (o a toda la región)
            heatmap[y:y+patch_size, x:x+patch_size] = delta
            # Restaurar el parche
            img_copy[y:y+patch_size, x:x+patch_size] = patch
    
    # Normalizar heatmap entre 0 y 1 para visualización
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    return heatmap

# --- CARGA DE MODELOS USANDO LA CLASE ---
@st.cache_resource
def cargar_modelos():
    modelo_a_path = "modelo_fase_a.pt"
    modelo_b_path = "modelo_fase_b.pt"
    modelo_a = YoloClassifier(modelo_a_path) if os.path.exists(modelo_a_path) else None
    modelo_b = YoloClassifier(modelo_b_path) if os.path.exists(modelo_b_path) else None
    return modelo_a, modelo_b

modelo_a, modelo_b = cargar_modelos()

# --- MANEJO DE ESTADO ---
if 'imagen_actual' not in st.session_state:
    st.session_state.imagen_actual = None
if 'mostrar_oclusion' not in st.session_state:
    st.session_state.mostrar_oclusion = False
if 'ultima_prediccion' not in st.session_state:
    st.session_state.ultima_prediccion = None

def cargar_caso_prueba(ruta):
    if os.path.exists(ruta):
        st.session_state.imagen_actual = Image.open(ruta)
        st.session_state.mostrar_oclusion = False  # resetear
    else:
        st.error(f"Falta el archivo: {ruta}.")

# --- MAQUETADO DE 3 COLUMNAS ---
col1, col2, col3 = st.columns([1, 2.5, 1], gap="large")

# ==========================================
# COLUMNA 1: CONTROLES (igual que antes)
# ==========================================
with col1:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### ☁️ Cargar RX")
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

    # Añadir botón de oclusión
    if st.button("🔍 Mostrar Mapa de Oclusión", use_container_width=True):
        st.session_state.mostrar_oclusion = not st.session_state.mostrar_oclusion

# ==========================================
# VARIABLES GLOBALES (se mantienen)
# ==========================================
clase_final = ""
conf_final = 0.0
clase_etiologia = ""
t_pre = 0.0
t_inf = 0.0

# ==========================================
# COLUMNA 2: VISOR PRINCIPAL (modificado para oclusión)
# ==========================================
with col2:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span class="mini-text"><span class="status-dot">●</span> SISTEMA ONLINE</span>
    </div>
    """, unsafe_allow_html=True)
    
    visor_contenedor = st.empty()
    
    # Si hay imagen, mostrar y procesar
    if st.session_state.imagen_actual is not None:
        if modelo_a is None or modelo_b is None:
            visor_contenedor.error("Modelos no encontrados. Sube los archivos .pt al servidor.")
        else:
            # Botón de acción flotante
            if st.button("🚀 INICIAR DIAGNÓSTICO", type="primary", use_container_width=True):
                # 1. Preparar imagen
                t_inicio = time.time()
                img_pil = st.session_state.imagen_actual.convert('RGB')
                img_array = np.array(img_pil)
                t_pre = time.time() - t_inicio
                
                # 2. Inferencia Triaje (Fase A)
                with st.spinner("Analizando placa radiográfica..."):
                    t_inicio_inf = time.time()
                    pred_a = modelo_a.predict(img_array)
                    clase_final = pred_a["class_name"]
                    conf_final = pred_a["confidence"]
                    
                    if clase_final == "PNEUMONIA":
                        # Fase B
                        pred_b = modelo_b.predict(img_array)
                        clase_etiologia = pred_b["class_name"]
                    else:
                        clase_etiologia = ""
                    t_inf = time.time() - t_inicio_inf
                    
                    # Guardar predicción para oclusión
                    st.session_state.ultima_prediccion = {
                        "clase_final": clase_final,
                        "conf_final": conf_final,
                        "clase_etiologia": clase_etiologia,
                        "img_array": img_array
                    }
                
                # Mostrar imagen (limpiar flags de oclusión)
                st.session_state.mostrar_oclusion = False
                visor_contenedor.image(img_pil, use_container_width=True)
            
            # Si no se ha iniciado diagnóstico, mostrar solo imagen o mapa de oclusión si está activado
            else:
                if st.session_state.mostrar_oclusion and st.session_state.ultima_prediccion is not None:
                    # Generar mapa de oclusión con la última predicción
                    with st.spinner("Generando mapa de oclusión..."):
                        img_array = st.session_state.ultima_prediccion["img_array"]
                        # Usar modelo correspondiente: si la clase final es PNEUMONIA, usar modelo_b (o modelo_a)
                        # Para oclusión, usamos el modelo que hizo la clasificación final
                        # En caso de NORMAL, usamos modelo_a
                        if st.session_state.ultima_prediccion["clase_final"] == "NORMAL":
                            model_occ = modelo_a
                        else:
                            model_occ = modelo_b  # para etiología
                        heatmap = occlusion_map(img_array, model_occ, patch_size=32, stride=16)
                        # Mostrar imagen con heatmap superpuesto
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(img_array)
                        ax.imshow(heatmap, cmap='jet', alpha=0.5, interpolation='bilinear')
                        ax.axis('off')
                        visor_contenedor.pyplot(fig)
                        plt.close(fig)
                else:
                    # Mostrar la imagen original
                    visor_contenedor.image(st.session_state.imagen_actual, use_container_width=True)
    else:
        # Estado vacío
        visor_contenedor.markdown(
            '<div style="width:100%; height:500px; background-color:#000; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#333;">Esperando imagen...</div>', 
            unsafe_allow_html=True
        )

# ==========================================
# COLUMNA 3: RESULTADOS Y LATENCIA (similar, pero usando variables de sesión)
# ==========================================
with col3:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("<p class='mini-text'>RESULTADO</p>", unsafe_allow_html=True)
    
    # Recuperar predicción de sesión si existe
    if st.session_state.ultima_prediccion is not None:
        pred = st.session_state.ultima_prediccion
        clase_final = pred["clase_final"]
        conf_final = pred["conf_final"]
        clase_etiologia = pred["clase_etiologia"]
    else:
        clase_final = ""
        conf_final = 0.0
        clase_etiologia = ""
    
    if clase_final == "PNEUMONIA":
        st.markdown(f"""
        <div class="resultado-anomalia">
            <h2>⚠️ ANOMALÍA</h2>
            <p style="margin-bottom:0;">Causa: {clase_etiologia}</p>
            <p class="mini-text">Confianza {conf_final*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        if clase_etiologia == "BACTERIA":
            st.warning("Recomendación Médica: Evaluar inicio de tratamiento antibiótico.")
        elif clase_etiologia == "VIRUS":
            st.warning("Recomendación Médica: Cuadro compatible con infección viral. Evaluar tratamiento sintomático/antiviral.")
        else:
            st.warning("Recomendación Médica: Consulte con un especialista.")

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
    
    # --- METRICAS DE LATENCIA ---
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### LATENCIA PIPELINE")
    # Aquí t_pre y t_inf se actualizan al hacer diagnóstico; si no, mostrar 0
    st.markdown(f"<div style='display:flex; justify-content:space-between;'><span class='mini-text'>PREPROCESAMIENTO</span><span class='mini-text'>{t_pre:.3f} s</span></div>", unsafe_allow_html=True)
    st.progress(min(t_pre / 0.1, 1.0))
    
    st.markdown(f"<div style='display:flex; justify-content:space-between; margin-top:10px;'><span class='mini-text'>INFERENCIA (GPU/CPU)</span><span class='mini-text'>{t_inf:.3f} s</span></div>", unsafe_allow_html=True)
    st.progress(min(t_inf / 1.0, 1.0))
    st.markdown('</div>', unsafe_allow_html=True)

    # --- INSTRUCCIONES ---
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("#### Instrucciones de uso")
    st.markdown("""
    <p class="mini-text">
    1. Suba una radiografía de tórax frontal.<br>
    2. El sistema evaluará primero la presencia de infiltrados (Triaje).<br>
    3. Si detecta anomalías, un segundo modelo determinará la probable etiología (Bacteriana/Viral).<br>
    4. Presione "Mostrar Mapa de Oclusión" para visualizar las regiones que más influyen en la decisión.<br><br>
    <em>Este es un sistema de soporte al diagnóstico (CAD) y no sustituye el criterio médico profesional.</em>
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
