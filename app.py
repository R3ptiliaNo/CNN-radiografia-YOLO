import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente Diagnóstico IA",
    page_icon="🩻",
    layout="centered"
)

# --- ESTILOS VISUALES ---
st.markdown("""
<style>
    .titulo { font-size: 32px; font-weight: bold; color: #2C3E50; margin-bottom: 5px; }
    .subtitulo { font-size: 18px; color: #7F8C8D; margin-bottom: 20px; }
    .caja-alerta { padding: 15px; border-radius: 8px; margin-top: 10px; font-weight: bold; }
    .sano { background-color: #D4EFDF; color: #1E8449; border: 1px solid #1E8449; }
    .enfermo { background-color: #FADBD8; color: #C0392B; border: 1px solid #C0392B; }
    .info { background-color: #D6EAF8; color: #2874A6; border: 1px solid #2874A6; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="titulo">🩻 Asistente Diagnóstico: Neumonía</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Arquitectura en Cascada basada en YOLOv11 (Transfer Learning)</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
**Instrucciones de uso:**
1. Suba una radiografía de tórax frontal.
2. El sistema evaluará primero la presencia de infiltrados (Triaje).
3. Si detecta anomalías, un segundo modelo determinará la probable etiología (Bacteriana/Viral).
""")
st.markdown("---")

# --- CARGA DE MODELOS (Caché para velocidad) ---
@st.cache_resource
def cargar_modelos():
    try:
        # IMPORTANTE: Asegúrate de que estos archivos estén en la misma carpeta que app.py
        modelo_a = YOLO("modelo_fase_a.pt") 
        modelo_b = YOLO("modelo_fase_b.pt")
        return modelo_a, modelo_b
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None, None

modelo_a, modelo_b = cargar_modelos()

# --- INTERFAZ DE USUARIO ---
if modelo_a and modelo_b:
    archivo_subido = st.file_uploader("Cargue una radiografía de tórax (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if archivo_subido is not None:
        # Mostrar la imagen original
        imagen = Image.open(archivo_subido)
        st.image(imagen, caption="Radiografía cargada", use_container_width=True)
        
        # Preparar imagen para YOLO
        # Convertir a numpy array y asegurar que es RGB
        img_array = np.array(imagen.convert('RGB'))

        if st.button("Analizar Radiografía"):
            with st.spinner("Analizando Fase A (Triaje)..."):
                # Ejecutar Fase A
                pred_a = modelo_a(img_array)[0]
                clase_a = pred_a.names[pred_a.probs.top1]
                conf_a = pred_a.probs.top1conf.item() * 100

            st.markdown("### Resultados del Análisis")
            
            # Lógica en Cascada
            if clase_a == "NORMAL":
                st.markdown(f'<div class="caja-alerta sano">✅ Fase A: Paciente Sano (Confianza: {conf_a:.1f}%)</div>', unsafe_allow_html=True)
                st.info("Fin del diagnóstico. No se detectan infiltrados alveolares.")
                
            elif clase_a == "PNEUMONIA":
                st.markdown(f'<div class="caja-alerta enfermo">⚠️ Fase A: Infección Detectada (Confianza: {conf_a:.1f}%)</div>', unsafe_allow_html=True)
                
                with st.spinner("Analizando Fase B (Etiología)..."):
                    # Ejecutar Fase B solo si hay enfermedad
                    pred_b = modelo_b(img_array)[0]
                    clase_b = pred_b.names[pred_b.probs.top1]
                    conf_b = pred_b.probs.top1conf.item() * 100
                
                st.markdown(f'<div class="caja-alerta info">🔬 Fase B: Causa {clase_b} (Confianza: {conf_b:.1f}%)</div>', unsafe_allow_html=True)
                
                if clase_b == "BACTERIA":
                    st.warning("Recomendación Médica: Evaluar inicio de tratamiento antibiótico.")
                else:
                    st.warning("Recomendación Médica: Cuadro compatible con infección viral. Evaluar tratamiento sintomático/antiviral.")

else:
    st.warning("Esperando modelos. Por favor, asegúrese de que 'modelo_fase_a.pt' y 'modelo_fase_b.pt' estén en el directorio.")