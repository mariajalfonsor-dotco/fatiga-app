"""
app.py
------
Interfaz Streamlit de la aplicación de análisis de fatiga metálica.

Ejecutar con:  streamlit run app.py
"""

import io
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import database
import diagnosis_engine
import habitos_conduccion
import image_analysis
import ocr_reportes
import reportes
import vibration_analysis

st.set_page_config(page_title="Analizador de fatiga metálica", layout="wide",
                   initial_sidebar_state="expanded")


def _es_publico() -> bool:
    """Modo público: no guarda archivos ni análisis en el servidor (protege datos de los usuarios).

    Se activa solo en Streamlit Community Cloud (la app corre en /mount/src) o con la
    variable de entorno MODO_PUBLICO=1.
    """
    if os.environ.get("MODO_PUBLICO", "").lower() in ("1", "true", "si", "sí"):
        return True
    return Path(__file__).resolve().parts[:3] == ("/", "mount", "src")


MODO_PUBLICO = _es_publico()

database.init_db()
reportes.init_tables()

with st.sidebar:
    st.markdown("#### Realizado por")
    st.markdown(
        "- Alfonso Rodríguez, María José\n"
        "- Buitrago Saavedra, Julián Felipe\n"
        "- Herrera Fuentes, Luis Fernando\n"
        "- Tique Rincón, Kenny Alexander"
    )
    st.markdown("**Docente:** Efrén Eduardo Vásquez Díaz")
    st.markdown("**Universidad ECCI**")

if "image_result" not in st.session_state:
    st.session_state.image_result = None
if "vibration_result" not in st.session_state:
    st.session_state.vibration_result = None

st.title("Analizador de fatiga metálica")
st.caption(
    "Proyecto de grado — simulación con datos sintéticos / datasets académicos. "
    "No usar como herramienta de decisión en un caso real sin validación adicional."
)

tab_tel, tab_ana, tab_img, tab_vib, tab_diag, tab_rep, tab_hist, tab_info = st.tabs(
    ["1. Telemetría y hábitos", "2. Análisis de la telemetría", "3. Imagen de fractura",
     "4. Señal de vibración", "5. Diagnóstico", "Reportes e imágenes", "Historial", "Acerca de"]
)

# ---------------------------------------------------------------------------
# Tab 3: Imagen
# ---------------------------------------------------------------------------
with tab_img:
    st.subheader("Cargar y analizar la imagen de la superficie de fractura")
    uploaded_img = st.file_uploader("Imagen (jpg/png)", type=["jpg", "jpeg", "png"], key="img_uploader")

    col1, col2 = st.columns(2)
    if uploaded_img is not None:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Imagen original", use_container_width=True)

        if st.button("Analizar imagen"):
            result = image_analysis.analyze_image(img_bgr)
            st.session_state.image_result = result

    if st.session_state.image_result:
        result = st.session_state.image_result
        with col2:
            st.image(result["preprocessed_image"], caption="Preprocesada (gris + CLAHE)",
                      use_container_width=True, clamp=True)
        st.success(f"Clase predicha: **{result['predicted_class']}** "
                   f"(confianza: {result['confidence']:.1%})")
        st.bar_chart(pd.Series(result["probabilities"], name="probabilidad"))
        with st.expander("Ver características de textura extraídas (GLCM)"):
            st.json(result["features"])

# ---------------------------------------------------------------------------
# Tab 4: Vibración
# ---------------------------------------------------------------------------
with tab_vib:
    st.subheader("Cargar o simular la señal de vibración")
    st.caption("La señal de vibración es complementaria: la telemetría de la flota no la incluye. "
               "Se usa como validación adicional si se dispone de un acelerómetro.")
    source = st.radio("Fuente de la señal", ["Generar señal de demostración", "Cargar archivo CSV"], horizontal=True)

    signal, fs, shaft_freq = None, 12_000.0, 30.0

    if source == "Generar señal de demostración":
        fault_choice = st.selectbox(
            "Tipo de condición a simular",
            vibration_analysis.VIBRATION_CLASSES,
            format_func=lambda c: database.get_vibration_fault(c)["name"],
        )
        shaft_freq = st.slider("Frecuencia de giro del eje (Hz)", 10.0, 60.0, 30.0)
        if st.button("Generar señal"):
            signal = vibration_analysis.generate_demo_signal(fault_choice, fs=fs, shaft_freq=shaft_freq)
            st.session_state["raw_signal"] = (signal, fs, shaft_freq)
    else:
        st.caption("El CSV debe tener una sola columna con los valores de la señal en el tiempo.")
        uploaded_csv = st.file_uploader("Archivo CSV", type=["csv"], key="csv_uploader")
        fs = st.number_input("Frecuencia de muestreo (Hz)", value=12_000.0, step=100.0)
        shaft_freq = st.number_input("Frecuencia de giro del eje (Hz)", value=30.0, step=1.0)
        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv, header=None)
            signal = df.iloc[:, 0].to_numpy(dtype=float)
            st.session_state["raw_signal"] = (signal, fs, shaft_freq)

    if "raw_signal" in st.session_state:
        signal, fs, shaft_freq = st.session_state["raw_signal"]
        st.line_chart(pd.Series(signal[:2000], name="señal (primeras 2000 muestras)"))

        if st.button("Analizar señal"):
            result = vibration_analysis.analyze_vibration(signal, fs, shaft_freq)
            st.session_state.vibration_result = result

    if st.session_state.vibration_result:
        result = st.session_state.vibration_result
        st.success(f"Clase predicha: **{result['predicted_class']}** "
                   f"(confianza: {result['confidence']:.1%})")
        st.bar_chart(pd.Series(result["probabilities"], name="probabilidad"))

        spectrum_df = pd.DataFrame({"frecuencia_hz": result["freqs"], "amplitud": result["spectrum"]})
        spectrum_df = spectrum_df[spectrum_df["frecuencia_hz"] <= shaft_freq * 10]
        st.line_chart(spectrum_df.set_index("frecuencia_hz"))
        with st.expander("Ver características espectrales extraídas"):
            st.json(result["features"])

    st.divider()
    st.subheader("Engranajes de la transmisión: frecuencia de engrane y bandas laterales")
    st.caption(
        "Un diente dañado o una excentricidad modulan la frecuencia de engrane (GMF = dientes × giro) y "
        "generan bandas laterales separadas por la frecuencia de giro. Los umbrales son de referencia y "
        "deben calibrarse con mediciones reales. Referencia normativa: ISO 20816 (reemplaza a ISO 10816), "
        "que no cubre la vibración torsional."
    )
    g1, g2, g3 = st.columns(3)
    with g1:
        dientes = st.number_input("Número de dientes del engranaje", 10, 200, 40, 1)
    with g2:
        giro_g = st.number_input("Frecuencia de giro del eje (Hz)", 5.0, 100.0, 30.0, 1.0, key="gear_shaft")
    with g3:
        fs_g = st.number_input("Frecuencia de muestreo (Hz)", 2000.0, 100000.0, 12000.0, 500.0, key="gear_fs")
    fuente_g = st.radio("Señal a analizar", ["Demostración de engranaje", "Usar la señal cargada arriba"],
                        horizontal=True, key="gear_src")
    cond_g = None
    if fuente_g.startswith("Demostración"):
        cond_g = st.selectbox("Condición a simular", list(vibration_analysis.GEAR_CONDITIONS),
                              format_func=lambda c: vibration_analysis.GEAR_CONDITIONS[c], key="gear_cond")

    if st.button("Analizar engranaje", key="gear_btn"):
        sig_g, fs_use = None, fs_g
        if cond_g is not None:
            sig_g = vibration_analysis.generate_gear_demo_signal(
                cond_g, fs=fs_g, shaft_freq=giro_g, teeth=int(dientes), seed=1)
        elif "raw_signal" in st.session_state:
            sig_g, fs_use, _ = st.session_state["raw_signal"]
        else:
            st.warning("Primero carga o genera una señal en la parte de arriba.")
        if sig_g is not None:
            try:
                st.session_state.gear_result = (
                    vibration_analysis.gear_mesh_analysis(sig_g, fs_use, giro_g, int(dientes)), sig_g, fs_use)
            except ValueError as e:
                st.session_state.gear_result = None
                st.error(str(e))

    if st.session_state.get("gear_result"):
        res_g, sig_g, fs_gu = st.session_state.gear_result
        k1, k2 = st.columns(2)
        k1.metric("Frecuencia de engrane (GMF)", f"{res_g['gmf']:.0f} Hz")
        k2.metric("Bandas laterales / GMF", f"{res_g['ratio']:.2f}" if res_g["ratio"] == res_g["ratio"] else "—")
        {"normal": st.success, "vigilar": st.warning, "alerta": st.error,
         "sin_dato": st.info}[res_g["nivel"]](res_g["estado"])

        tabla_g = pd.DataFrame(res_g["tabla"])
        tabla_g["Amplitud"] = tabla_g["Amplitud"].round(3)
        st.dataframe(tabla_g, use_container_width=True)

        ventana = (res_g["freqs"] >= res_g["gmf"] - 6 * giro_g) & (res_g["freqs"] <= res_g["gmf"] + 6 * giro_g)
        st.markdown("Espectro alrededor de la frecuencia de engrane")
        st.line_chart(pd.DataFrame({"frecuencia_hz": res_g["freqs"][ventana],
                                    "amplitud": res_g["spectrum"][ventana]}).set_index("frecuencia_hz"))
        fmax = float(min(fs_gu / 2, max(2.5 * res_g["gmf"], 1500)))
        st.markdown("Espectrograma (frecuencia en vertical, tiempo en horizontal)")
        st.image(vibration_analysis.spectrogram_image(sig_g, fs_gu, fmax=fmax),
                 caption=f"0 a {fmax:.0f} Hz; los tonos claros indican más energía.", use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5: Diagnóstico combinado
# ---------------------------------------------------------------------------
with tab_diag:
    st.subheader("Diagnóstico combinado")

    img_ok = st.session_state.image_result is not None
    vib_ok = st.session_state.vibration_result is not None

    if not img_ok:
        st.info("Analiza primero una imagen (pestaña 3) para generar el diagnóstico. "
                "La señal de vibración (pestaña 4) es opcional y ayuda a afinarlo.")
    else:
        img_res = st.session_state.image_result
        vib_res = st.session_state.vibration_result

        tel = st.session_state.get("hab_result")
        telemetria_diag = habitos_conduccion.resumen_para_diagnostico(tel[1]) if tel is not None else None
        st.markdown(
            "**Datos que usa el diagnóstico:** imagen (sí) · "
            f"telemetría ({'sí' if tel is not None else 'no, es opcional: pestañas 1 y 2'}) · "
            f"vibración ({'sí' if vib_res else 'no, es opcional: pestaña 4'})"
            + (f" · vehículo analizado: {st.session_state.get('hab_veh_sel')}"
               if tel is not None and st.session_state.get("hab_flota") is not None else ""))
        if tel is not None:
            _, matriz_tel = tel
            hallazgos = matriz_tel[matriz_tel["Prioridad"].isin(["Alta", "Media"])]
            if len(hallazgos):
                st.info("Contexto de telemetría (paso 1): " + "; ".join(
                    f"{r['Hábito / hallazgo']} (prioridad {r['Prioridad'].lower()})"
                    for _, r in hallazgos.iterrows()))

        if img_res["predicted_class"] == "fatiga":
            st.markdown("##### Contexto operativo (para afinar el subtipo de fatiga)")
            st.caption(
                "Los subtipos de fatiga (alto ciclo, bajo ciclo, térmica, por "
                "corrosión, por fricción) casi no se distinguen por textura o "
                "vibración solas — dependen del contexto real de operación."
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                temperatura = st.selectbox("Temperatura de operación", ["normal", "alta", "baja"])
            with c2:
                ambiente = st.selectbox("Ambiente", ["normal", "corrosivo", "abrasivo"])
            with c3:
                carga = st.selectbox(
                    "Tipo de carga", ["alto_ciclo", "bajo_ciclo", "fretting"],
                    format_func=lambda c: {
                        "alto_ciclo": "Cíclica normal (alto ciclo)",
                        "bajo_ciclo": "Cíclica severa (bajo ciclo)",
                        "fretting": "Fricción por micro-movimiento (fretting)",
                    }[c],
                )
        else:
            temperatura, ambiente, carga = "normal", "normal", "alto_ciclo"

        if st.button("Generar diagnóstico", type="primary"):
            diag = diagnosis_engine.diagnose(
                img_res["predicted_class"], img_res["confidence"],
                vib_res["predicted_class"] if vib_res else None,
                vib_res["confidence"] if vib_res else None,
                temperatura=temperatura, ambiente=ambiente, carga=carga,
                telemetria=telemetria_diag,
            )
            st.session_state.diagnosis = diag

        if "diagnosis" in st.session_state:
            diag = st.session_state.diagnosis
            mc1, mc2 = st.columns(2)
            mc1.metric("Confianza de los modelos de IA", f"{diag.combined_confidence:.1%}")
            mc2.metric("Respaldo de la telemetría",
                       "Sí" if diag.telemetry_used else
                       ("Sin reglas aplicables" if diag.telemetry_provided else "No se usó"))
            if diag.telemetry_used:
                st.caption("Hábitos de telemetría considerados: " + "; ".join(diag.telemetry_used))
            st.markdown(f"### Causa raíz probable\n{diag.root_cause}")
            st.markdown(f"**Explicación:** {diag.explanation}")
            st.markdown(f"**Recomendación:** {diag.recommendation}")
            if diag.vibration_class is None and not diag.telemetry_provided:
                st.info("Este diagnóstico usa solo la imagen. Agrega la telemetría (pestañas 1 y 2) o una "
                        "señal de vibración (pestaña 4) y vuelve a generarlo para afinarlo.")
            elif not diag.rule_found:
                st.warning("Esta combinación no tiene una regla específica en la base de "
                           "conocimiento; el resultado es una inferencia general.")

            if MODO_PUBLICO:
                st.caption("En la versión pública no se guardan los análisis.")
            elif st.button("Guardar en el historial"):
                database.save_analysis(
                    diag.image_class, diag.image_confidence,
                    diag.vibration_class, diag.vibration_confidence,
                    diag.root_cause, diag.explanation,
                )
                st.success("Análisis guardado en el historial.")

def _mostrar_ocr(res):
    """Muestra lo que el OCR leyó de la imagen y devuelve las variables numéricas (o None)."""
    st.markdown("##### Lo que se leyó en la imagen")
    if res["total_declarado"]:
        st.metric("Eventos de video que declara el reporte", f"{res['total_declarado']:,}".replace(",", "."))
    ev = res["eventos"]
    if len(ev):
        st.markdown("###### Eventos de video detectados")
        st.dataframe(ev, use_container_width=True)
        g1, g2 = st.columns(2)
        g1.markdown("Por categoría")
        g1.bar_chart(ev["Categoría"].value_counts())
        g2.markdown("Por vehículo")
        g2.bar_chart(ev["Vehículo"].value_counts())
        st.download_button("Descargar eventos (CSV)", ev.to_csv(index=False).encode("utf-8-sig"),
                           file_name="eventos_video.csv", mime="text/csv")
        st.caption("Son eventos de cámara (conducta y seguridad), no mediciones del motor. "
                   "Solo «Aceleración brusca» se relaciona con el esfuerzo del tren motriz.")
    else:
        st.info("No reconocí tarjetas de eventos de video en esta imagen (el texto sí se leyó, míralo abajo).")
    with st.expander("Texto completo leído (OCR)"):
        st.text("\n".join(res["lineas"]) if res["lineas"] else "No se reconoció texto.")
    var = res["variables"]
    if var is not None and len(var):
        st.markdown("###### Variables numéricas reconocidas")
        st.dataframe(var, use_container_width=True)
    return var


# ---------------------------------------------------------------------------
# Tab 1: Telemetría y hábitos de conducción
# ---------------------------------------------------------------------------
with tab_tel:
    st.subheader("Paso 1: telemetría del vehículo y hábitos de conducción")
    st.caption(
        "Sube la telemetría de cualquier vehículo (CSV, PDF o imagen) y detecta hábitos de conducción: lugging "
        "(bajas RPM con carga alta), abuso de embrague a altas RPM y sobrecalentamiento de aceite. Los nombres de "
        "columna, las unidades y el régimen del motor se adaptan a cada vehículo, y si el archivo trae varios "
        "vehículos se analiza cada uno. El análisis con IA y la matriz de mantenimiento están en la pestaña 2."
    )

    fuente_t = st.radio("Fuente de los datos",
                        ["Datos simulados (Geotab Actros)", "Flota simulada (varios vehículos)", "Cargar archivo CSV", "Cargar PDF o imagen"],
                        horizontal=True, key="tel_src")
    df_tel, raw_t = None, None
    if fuente_t.startswith("Datos simulados"):
        st.info("Datos simulados dentro de los rangos del script (no son mediciones reales). "
                "Incluyen unos pocos eventos de lugging y embrague para que el demo muestre resultados.")
        df_tel = habitos_conduccion.cargar_datos_telemetria()
    elif fuente_t.startswith("Flota simulada"):
        st.info("Tres vehículos simulados con regímenes de motor muy distintos (diésel pesado, bus y auto de "
                "gasolina) para mostrar cómo se adaptan los umbrales. No son mediciones reales.")
        df_tel = habitos_conduccion.cargar_datos_flota()
    elif fuente_t == "Cargar archivo CSV":
        st.caption("Los nombres de columna no tienen que ser idénticos: la app los sugiere y tú los puedes corregir. "
                   "Si el archivo trae una columna de vehículo (placa, unidad, activo), se analiza cada uno.")
        up_t = st.file_uploader("CSV de telemetría", type=["csv"], key="tel_csv")
        if up_t is not None:
            raw_t = pd.read_csv(up_t, sep=None, engine="python")
    else:
        st.caption("Sube un PDF o una imagen. Si el PDF tiene tablas con texto, se leen directo; si es una "
                   "captura de pantalla (imagen), se lee con OCR.")
        up_p = st.file_uploader("Reporte PDF o imagen", type=["pdf", "png", "jpg", "jpeg"], key="tel_pdf")
        if up_p is not None:
            data_p = up_p.getvalue()
            es_pdf = up_p.name.lower().endswith(".pdf")
            if es_pdf:
                try:
                    raw_t = habitos_conduccion.extraer_tablas_pdf(data_p)
                except Exception:
                    st.error("No se pudo leer el PDF. Verifica que no esté dañado.")
            if raw_t is None:
                if es_pdf:
                    st.info("Este PDF no tiene tablas con texto: es una imagen. Puedo leerlo con OCR.")
                try:
                    paginas = reportes.pdf_pages_as_png(data_p, max_pages=6, zoom=2.0) if es_pdf else [data_p]
                except Exception:
                    paginas = []
                    st.error("No se pudo abrir el archivo.")
                if paginas:
                    cols_p = st.columns(min(len(paginas), 3))
                    for i, png in enumerate(paginas):
                        cols_p[i % len(cols_p)].image(png, caption=f"Página {i + 1}", use_container_width=True)
                    if st.button("Leer con OCR", type="primary", key="tel_ocr_btn"):
                        with st.spinner("Leyendo la imagen... la primera vez puede tardar un poco."):
                            try:
                                st.session_state.ocr_result = {
                                    "archivo": up_p.name, "res": ocr_reportes.analizar_paginas(paginas)}
                                ev_ocr = st.session_state.ocr_result["res"]["eventos"]
                                st.session_state.eventos_video = ev_ocr if len(ev_ocr) else None
                            except ocr_reportes.OCRNoDisponible as e:
                                st.session_state.ocr_result = None
                                st.error(str(e))
                ocr = st.session_state.get("ocr_result")
                if ocr and ocr["archivo"] == up_p.name:
                    var_ocr = _mostrar_ocr(ocr["res"])
                    if var_ocr is not None and habitos_conduccion.hay_algo_evaluable(var_ocr):
                        raw_t = var_ocr  # variables suficientes: pasan al análisis de telemetría

    if raw_t is not None:
        st.markdown("##### Datos leídos")
        st.dataframe(raw_t.head(8), use_container_width=True)
        sug = habitos_conduccion.sugerir_mapeo(raw_t.columns)
        opciones = ["(no disponible)"] + list(raw_t.columns)
        sin_clave = [c for c in habitos_conduccion.COLUMNAS_OBLIGATORIAS if sug.get(c) is None]
        firma = abs(hash(tuple(raw_t.columns)))
        with st.expander("Asignar columnas (revisa que cada variable apunte a la columna correcta)",
                         expanded=bool(sin_clave) or fuente_t == "Cargar PDF o imagen"):
            mapeo = {}
            cols_ui = st.columns(3)
            for i, canon in enumerate(habitos_conduccion.ORDEN_MAPEO):
                por_defecto = opciones.index(sug[canon]) if sug[canon] in opciones else 0
                elegido = cols_ui[i % 3].selectbox(canon, opciones, index=por_defecto, key=f"map_{firma}_{canon}")
                mapeo[canon] = None if elegido == "(no disponible)" else elegido
            un1, un2 = st.columns(2)
            unidad_temp = un1.selectbox("Unidad de temperatura", ["Detectar automáticamente", "°C", "°F"],
                                        key=f"unit_{firma}")
            torque_nom_in = un2.number_input("Torque nominal del motor (Nm), solo si el torque viene en Nm "
                                             "(0 = estimarlo)", min_value=0.0, value=0.0, step=50.0, key=f"tn_{firma}")
        df_conv = habitos_conduccion.aplicar_mapeo(raw_t, mapeo)
        df_conv, notas_u = habitos_conduccion.normalizar_unidades(
            df_conv, {"Detectar automáticamente": "auto", "°C": "C", "°F": "F"}[unidad_temp], torque_nom_in)
        if notas_u:
            st.info("Unidades ajustadas automáticamente: " + "; ".join(notas_u) + ".")
        if not habitos_conduccion.hay_algo_evaluable(df_conv):
            st.error("Con las columnas asignadas no se puede evaluar ninguna regla. "
                     "Asigna al menos RPM_Motor junto con otras variables.")
        else:
            df_tel = df_conv
            pendientes = habitos_conduccion.descripcion_no_evaluable(df_tel)
            if pendientes:
                st.warning("Con estas columnas no se podrá evaluar: " + "; ".join(pendientes) + ".")

    with st.expander("Umbrales y sensibilidad (ajústalos al vehículo)"):
        d = habitos_conduccion.Umbrales()
        modo_t = st.radio(
            "Umbrales de RPM", ["Automático: se adapta al régimen de cada vehículo (recomendado)",
                                "Manual: RPM fijas"], key="tel_modo")
        auto_t = modo_t.startswith("Automático")
        t1, t2, t3 = st.columns(3)
        with t1:
            lug_carga = st.number_input("Lugging: carga mínima (%)", value=d.lugging_carga_min, step=5.0)
            lug_pedal = st.number_input("Lugging: pedal mínimo (%)", value=d.lugging_pedal_min, step=5.0)
        with t2:
            if auto_t:
                lug_frac = st.number_input("Lugging: RPM por debajo de (% del régimen máximo)",
                                           value=d.lugging_rpm_frac * 100, step=5.0) / 100
                emb_frac = st.number_input("Embrague: RPM por encima de (% del régimen máximo)",
                                           value=d.embrague_rpm_frac * 100, step=5.0) / 100
            else:
                lug_rpm = st.number_input("Lugging: RPM máximas", value=d.lugging_rpm_max, step=50.0)
                emb_rpm = st.number_input("Embrague: RPM altas desde", value=d.embrague_rpm_min, step=50.0)
        with t3:
            temp_crit = st.number_input("Temperatura crítica de aceite (°C)", value=d.temp_aceite_critica, step=1.0)
            contaminacion = st.slider("Proporción esperada de anomalías", 0.01, 0.20, 0.08, 0.01)
        rpm_max_in = 0.0
        if auto_t:
            rpm_max_in = st.number_input("Régimen máximo del motor (RPM). 0 = estimarlo de los datos de cada vehículo",
                                         min_value=0.0, value=0.0, step=100.0)
            st.caption("El régimen máximo se estima con el percentil 99 de las RPM. La temperatura crítica del aceite "
                       "depende del aceite y del fabricante: ajústala si conoces el límite de tu vehículo.")

    if df_tel is not None:
        st.dataframe(df_tel.head(8), use_container_width=True)
        if st.button("Analizar telemetría", type="primary", key="tel_btn"):
            if auto_t:
                umbrales = habitos_conduccion.Umbrales(
                    modo_auto=True, rpm_max_motor=rpm_max_in, lugging_rpm_frac=lug_frac, embrague_rpm_frac=emb_frac,
                    lugging_carga_min=lug_carga, lugging_pedal_min=lug_pedal, temp_aceite_critica=temp_crit)
            else:
                umbrales = habitos_conduccion.Umbrales(
                    modo_auto=False, lugging_rpm_max=lug_rpm, embrague_rpm_min=emb_rpm,
                    lugging_carga_min=lug_carga, lugging_pedal_min=lug_pedal, temp_aceite_critica=temp_crit)
            st.session_state.hab_umbrales = umbrales
            flota_res = habitos_conduccion.run_analysis_flota(df_tel, umbrales, contaminacion)
            if flota_res is not None:
                st.session_state.hab_flota = flota_res
                st.session_state.pop("hab_veh_sel", None)
                st.session_state.hab_result = next(iter(flota_res[0].values()))
            else:
                st.session_state.hab_flota = None
                st.session_state.hab_result = habitos_conduccion.run_analysis(df_tel, umbrales, contaminacion)

    flota_t = st.session_state.get("hab_flota")
    if flota_t is not None:
        st.markdown("##### Resumen de la flota (primero los vehículos con más hallazgos)")
        st.dataframe(flota_t[1], use_container_width=True)
        veh_sel = st.selectbox("Vehículo a revisar en las pestañas 2 y 5", list(flota_t[0]), key="hab_veh_sel")
        st.session_state.hab_result = flota_t[0][veh_sel]

    if st.session_state.get("hab_result") is not None:
        res_t, _ = st.session_state.hab_result
        st.markdown("##### Hábitos detectados")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Muestras analizadas", len(res_t))
        m2.metric("Eventos de lugging", int(res_t["Riesgo_Lugging"].sum()))
        m3.metric("Abuso de embrague", int(res_t["Riesgo_Abuso_Embrague"].sum()))
        m4.metric("Sobrecalentamiento de aceite", int(res_t["Sobrecalentamiento_Aceite"].sum()))

        idx = res_t.set_index("Timestamp") if "Timestamp" in res_t.columns else res_t
        if "RPM_Motor" in idx.columns:
            st.markdown("##### RPM del motor durante el recorrido")
            st.line_chart(idx[["RPM_Motor"]])
        temps = [c for c in ["Temp_Aceite_C", "Temp_Refrigerante_C"] if c in idx.columns]
        if temps:
            st.markdown("##### Temperatura de aceite y refrigerante")
            st.line_chart(idx[temps])
        st.caption("Siguiente: abre la pestaña 2 para ver el análisis con IA y la matriz de mantenimiento.")

# ---------------------------------------------------------------------------
# Tab 2: Análisis de la telemetría (Isolation Forest + matriz de mantenimiento)
# ---------------------------------------------------------------------------
with tab_ana:
    st.subheader("Paso 2: análisis de la telemetría")
    if st.session_state.get("hab_result") is None:
        st.info("Primero carga los datos y presiona «Analizar telemetría» en la pestaña 1.")
    else:
        res_t, matriz = st.session_state.hab_result
        if st.session_state.get("hab_flota") is not None:
            st.markdown(f"**Vehículo:** {st.session_state.get('hab_veh_sel')}")
        ef = habitos_conduccion.umbrales_efectivos(res_t, st.session_state.get("hab_umbrales"))
        partes_u = []
        if ef["rpm_ref"]:
            partes_u.append(f"régimen de referencia {ef['rpm_ref']:.0f} RPM")
        partes_u += [f"lugging: carga > {ef['lugging_carga']:.0f} %, RPM < {ef['lugging_rpm']:.0f}, "
                     f"pedal > {ef['lugging_pedal']:.0f} %",
                     f"embrague: RPM > {ef['embrague_rpm']:.0f}", f"aceite crítico ≥ {ef['temp_critica']:.0f} °C"]
        st.caption("Umbrales aplicados: " + " · ".join(partes_u))
        a1, a2, a3 = st.columns(3)
        a1.metric("Alto riesgo (Isolation Forest)", int((res_t["Riesgo_Tren_Motriz"] == "ALTO RIESGO").sum()))
        a2.metric("Hallazgos de prioridad alta", int((matriz["Prioridad"] == "Alta").sum()))
        a3.metric("Hallazgos de prioridad media", int((matriz["Prioridad"] == "Media").sum()))

        st.markdown("##### Matriz de mantenimiento preventivo")
        st.dataframe(matriz, use_container_width=True)
        st.download_button("Descargar reporte (CSV)", matriz.to_csv(index=False).encode("utf-8-sig"),
                           file_name="reporte_mantenimiento.csv", mime="text/csv")

        idx2 = res_t.set_index("Timestamp") if "Timestamp" in res_t.columns else res_t
        st.markdown("##### Puntaje de anomalía durante el recorrido (más alto = más atípico)")
        st.line_chart(idx2[["Puntaje_Anomalia"]])

        with st.expander("Muestras marcadas como ALTO RIESGO por Isolation Forest"):
            st.dataframe(res_t[res_t["Riesgo_Tren_Motriz"] == "ALTO RIESGO"]
                         .sort_values("Puntaje_Anomalia", ascending=False), use_container_width=True)

    ev_video = st.session_state.get("eventos_video")
    if ev_video is not None and len(ev_video):
        st.markdown("##### Eventos de video leídos de las imágenes (OCR)")
        st.dataframe(ev_video, use_container_width=True)
        st.bar_chart(ev_video["Categoría"].value_counts())

# ---------------------------------------------------------------------------
# Tab: Reportes e imágenes (PDFs + fotos agrupados por caso)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _pdf_pages(data: bytes):
    return reportes.pdf_pages_as_png(data)


with tab_rep:
    st.subheader("Reportes técnicos e imágenes por caso")
    st.caption("Sube reportes en PDF (telemetría, operación) y fotos del daño. Se guardan por caso, "
               "y las imágenes se pueden analizar con el clasificador de fractura.")

    # 1) Caso activo
    caso_sel = None
    if MODO_PUBLICO:
        st.info("En la versión pública los archivos no se guardan en el servidor: se analizan en tu sesión y "
                "se descartan al cerrar la página.")
    else:
        st.markdown("##### 1. Caso")
        casos = reportes.list_cases()
        with st.expander("Crear un caso nuevo", expanded=not casos):
            n1, n2 = st.columns(2)
            with n1:
                nombre_caso = st.text_input("Nombre del caso", placeholder="Falla caja de cambios bus 408")
            with n2:
                vehiculo = st.text_input("Vehículo / ID", placeholder="408")
            notas_caso = st.text_area("Notas", placeholder="Qué pasó, kilometraje, síntomas...")
            if st.button("Crear caso", key="rep_new_case"):
                if not nombre_caso.strip():
                    st.error("Escribe un nombre para el caso.")
                else:
                    reportes.create_case(nombre_caso, vehiculo, notas_caso)
                    st.success("Caso creado.")
                    st.rerun()

        caso_sel = None
        if casos:
            caso_sel = st.selectbox(
                "Caso activo", casos,
                format_func=lambda c: f"#{c['id']} — {c['name']}" + (f" ({c['vehicle']})" if c["vehicle"] else ""),
            )

    # 2) Subida de archivos
    st.markdown("##### 2. Subir archivos")
    categoria = st.selectbox("Tipo de archivo", reportes.CATEGORIES)
    subidos = st.file_uploader("PDF o imágenes (puedes elegir varios)",
                               type=["pdf", "jpg", "jpeg", "png"],
                               accept_multiple_files=True, key="rep_uploader")

    candidatas = {}  # etiqueta -> bytes de imagen, para el análisis
    for f in subidos or []:
        data = f.getvalue()
        if f.name.lower().endswith(".pdf"):
            try:
                paginas = _pdf_pages(data)
            except Exception:
                st.error(f"No se pudo leer {f.name}. Verifica que no esté dañado.")
                continue
            st.markdown(f"**{f.name}** — {len(paginas)} página(s)")
            cols = st.columns(min(len(paginas), 3) or 1)
            for i, png in enumerate(paginas):
                cols[i % len(cols)].image(png, caption=f"Página {i + 1}", use_container_width=True)
                candidatas[f"{f.name} — página {i + 1}"] = png
            texto = reportes.pdf_text(data)
            with st.expander(f"Texto extraído de {f.name}"):
                st.text(texto if texto else "Este PDF no tiene texto seleccionable (es una captura de pantalla).")
        else:
            st.markdown(f"**{f.name}**")
            st.image(data, use_container_width=True)
            candidatas[f.name] = data

    if subidos and not MODO_PUBLICO:
        if st.button("Guardar archivos en el caso", type="primary", key="rep_save"):
            if caso_sel is None:
                st.error("Crea o elige un caso antes de guardar.")
            else:
                for f in subidos:
                    reportes.save_file(caso_sel["id"], f.name, f.getvalue(), categoria)
                st.success(f"{len(subidos)} archivo(s) guardados en el caso #{caso_sel['id']}.")

    # 3) Análisis de imagen
    if candidatas:
        st.markdown("##### 3. Analizar una imagen con el clasificador de fractura")
        st.warning(
            "El clasificador se entrenó con texturas sintéticas de superficies de fractura. "
            "En fotos de piezas completas (engranajes, cajas) o capturas de pantalla, el "
            "resultado no es confiable. Úsalo solo como apoyo."
        )
        elegida = st.selectbox("Imagen a analizar", list(candidatas.keys()))
        st.image(candidatas[elegida], width=320)
        if st.button("Analizar esta imagen", key="rep_analyze"):
            res = image_analysis.analyze_image(reportes.bytes_to_bgr(candidatas[elegida]))
            st.session_state.image_result = res  # alimenta la pestaña de diagnóstico
            st.success(f"Clase predicha: **{res['predicted_class']}** (confianza: {res['confidence']:.1%}). "
                       "Ya quedó disponible en la pestaña 5 (Diagnóstico).")
            st.bar_chart(pd.Series(res["probabilities"], name="probabilidad"))

    # 4) Archivos guardados
    if not MODO_PUBLICO:
        st.markdown("##### 4. Archivos guardados en el caso")
        if caso_sel is None:
            st.caption("Crea un caso para ver sus archivos.")
        else:
            if caso_sel["notes"]:
                st.caption(f"Notas: {caso_sel['notes']}")
            archivos = reportes.list_files(caso_sel["id"])
            if not archivos:
                st.caption("Este caso aún no tiene archivos.")
            else:
                st.dataframe(pd.DataFrame(archivos)[["filename", "category", "kind", "uploaded_at"]],
                             use_container_width=True)
                ver = st.selectbox("Ver archivo guardado", archivos, format_func=lambda a: a["filename"])
                ruta = reportes.Path(ver["path"])
                if ruta.exists():
                    if ver["kind"] == "imagen":
                        st.image(str(ruta), use_container_width=True)
                    else:
                        for i, png in enumerate(_pdf_pages(ruta.read_bytes())):
                            st.image(png, caption=f"Página {i + 1}", use_container_width=True)
                else:
                    st.error("El archivo ya no está en el disco.")

# ---------------------------------------------------------------------------
# Tab 4: Historial
# ---------------------------------------------------------------------------
with tab_hist:
    st.subheader("Historial de análisis guardados")
    history = [] if MODO_PUBLICO else database.get_history()
    if MODO_PUBLICO:
        st.info("En la versión pública no se guardan análisis, para proteger los datos de quienes la usan.")
    elif not history:
        st.caption("Aún no hay análisis guardados.")
    else:
        st.dataframe(pd.DataFrame(history), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5: Acerca de / base de conocimiento
# ---------------------------------------------------------------------------
with tab_info:
    st.subheader("Base de conocimiento")
    st.markdown("**Tipos de fractura**")
    conn = database.get_connection()
    st.dataframe(pd.read_sql("SELECT * FROM fracture_types", conn), use_container_width=True)
    st.markdown("**Tipos de falla por vibración**")
    st.dataframe(pd.read_sql("SELECT * FROM vibration_faults", conn), use_container_width=True)
    st.markdown("**Subtipos de fatiga (según contexto operativo)**")
    st.dataframe(pd.read_sql("SELECT * FROM fatigue_subtypes", conn), use_container_width=True)
    st.markdown("**Reglas de diagnóstico**")
    st.dataframe(pd.read_sql("SELECT * FROM diagnostic_rules", conn), use_container_width=True)
    st.markdown("**Reglas imagen + telemetría** (hipótesis de ingeniería, no validadas con casos reales de campo)")
    st.dataframe(pd.DataFrame([
        {"Fractura": k[0], "Hábito de telemetría": diagnosis_engine.HABITOS[k[1]], "Causa raíz": v[0]}
        for k, v in diagnosis_engine.REGLAS_IMAGEN_TELEMETRIA.items()]), use_container_width=True)
    conn.close()

    st.divider()
    st.markdown(
        "Los modelos de imagen y vibración se entrenan con datos sintéticos "
        "generados a partir de patrones descritos en la literatura de "
        "fractografía y diagnóstico de rodamientos (ver `image_analysis.py` "
        "y `vibration_analysis.py`). Para una validación experimental real, "
        "reemplaza los generadores sintéticos por datos reales etiquetados "
        "(por ejemplo, un subconjunto de un dataset público de fractografía "
        "SEM y el dataset CWRU de vibración de rodamientos) y vuelve a "
        "entrenar los modelos."
    )
