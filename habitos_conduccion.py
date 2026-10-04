"""
habitos_conduccion.py
---------------------
Paso 1 y 2 de la app: análisis de telemetría (Geotab) del tren motriz.

Basado en el script entregado por el equipo, con estas mejoras:
- entrenar_modelo_ia devolvía una tupla por una coma sobrante (`return df,`).
- Se agregó generar_matriz_mantenimiento (faltaba en el script).
- Umbrales editables desde la interfaz.
- Fuentes de datos: simulación, CSV o tablas dentro de un PDF.
- Asignación de columnas: los nombres del CSV/PDF no tienen que ser
  idénticos; se sugieren automáticamente y se pueden corregir a mano.
- Si faltan columnas, solo se omiten las reglas que dependen de ellas.
"""

import re
import unicodedata
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymupdf
from sklearn.ensemble import IsolationForest

COLUMNAS_OBLIGATORIAS = [
    "Carga_Motor_pct", "Torque_Exigido_pct", "Torque_Real_pct", "RPM_Motor",
    "Velocidad_kmh", "Temp_Aceite_C", "Temp_Refrigerante_C",
    "Pedal_Acelerador_pct", "Pedal_Embrague",
]
COLUMNAS_OPCIONALES = ["Vehiculo", "Timestamp", "Pedal_Freno", "Peso_Bruto_kg", "Aceleracion_Longitudinal_g"]

COLUMNAS_IA = [
    "Carga_Motor_pct", "Torque_Exigido_pct", "Torque_Real_pct",
    "RPM_Motor", "Velocidad_kmh", "Temp_Aceite_C",
    "Temp_Refrigerante_C", "Pedal_Acelerador_pct", "Ratio_Speed_RPM",
]

# Columnas que necesita cada regla para poder evaluarse
REGLAS_COLUMNAS = {
    "Riesgo_Lugging": ["Carga_Motor_pct", "RPM_Motor", "Pedal_Acelerador_pct"],
    "Riesgo_Abuso_Embrague": ["Pedal_Embrague", "RPM_Motor"],
    "Sobrecalentamiento_Aceite": ["Temp_Aceite_C"],
}


@dataclass
class Umbrales:
    # Modo automático: las RPM de lugging y de embrague se calculan como fracción del régimen
    # máximo del motor (estimado de los datos de cada vehículo, o indicado a mano). Así sirve
    # para diésel pesado, buses o autos de gasolina. Con modo_auto=False se usan RPM fijas.
    modo_auto: bool = True
    rpm_max_motor: float = 0.0          # 0 = estimar con el percentil 99 de las RPM del vehículo
    lugging_rpm_frac: float = 0.60      # lugging: RPM por debajo de 60 % del régimen máximo
    embrague_rpm_frac: float = 0.75     # embrague: RPM por encima de 75 % del régimen máximo
    lugging_carga_min: float = 85.0     # % de carga del motor (o de torque si no hay carga)
    lugging_pedal_min: float = 80.0     # % de pedal del acelerador
    lugging_rpm_max: float = 1150.0     # solo modo manual
    embrague_rpm_min: float = 1400.0    # solo modo manual
    temp_aceite_critica: float = 118.0  # °C; depende del aceite y del fabricante


def umbrales_efectivos(df: pd.DataFrame, u: Umbrales = None) -> dict:
    """Umbrales realmente aplicados a este vehículo (útil para mostrarlos al usuario)."""
    u = u or Umbrales()
    rpm_ref, lug_rpm, emb_rpm = None, u.lugging_rpm_max, u.embrague_rpm_min
    if u.modo_auto and _tiene(df, "RPM_Motor"):
        rpm_ref = u.rpm_max_motor if u.rpm_max_motor and u.rpm_max_motor > 0 else float(df["RPM_Motor"].quantile(0.99))
        lug_rpm, emb_rpm = u.lugging_rpm_frac * rpm_ref, u.embrague_rpm_frac * rpm_ref
    return {"rpm_ref": rpm_ref, "lugging_rpm": lug_rpm, "embrague_rpm": emb_rpm,
            "lugging_carga": u.lugging_carga_min, "lugging_pedal": u.lugging_pedal_min,
            "temp_critica": u.temp_aceite_critica}


# ==========================================
# 0. LECTURA DE ARCHIVOS Y ASIGNACIÓN DE COLUMNAS
# ==========================================
# Palabras que debe contener el nombre de la columna (todas las de una tupla).
# El orden importa: cada columna del archivo se asigna a una sola variable.
_ALIAS = {
    "RPM_Motor": [("rpm",), ("revoluc",), ("engine", "speed"), ("velocidad", "motor")],
    "Carga_Motor_pct": [("carga", "motor"), ("engine", "load")],
    "Torque_Exigido_pct": [("torque", "exig"), ("torque", "demand"), ("torque", "solicit")],
    "Torque_Real_pct": [("torque", "real"), ("torque", "actual"), ("torque",)],
    "Temp_Aceite_C": [("temp", "aceite"), ("oil", "temp")],
    "Temp_Refrigerante_C": [("temp", "refriger"), ("coolant",)],
    "Pedal_Acelerador_pct": [("acelerador",), ("throttle",), ("accelerator",), ("pedal", "acel")],
    "Pedal_Embrague": [("embrague",), ("clutch",)],
    "Pedal_Freno": [("freno",), ("brake",)],
    "Velocidad_kmh": [("velocidad",), ("vel",), ("speed",), ("km", "h")],
    "Peso_Bruto_kg": [("peso",), ("weight",)],
    "Aceleracion_Longitudinal_g": [("aceleracion",), ("acceleration",)],
    "Timestamp": [("fecha",), ("hora",), ("timestamp",), ("date",)],
    "Vehiculo": [("vehiculo",), ("vehicle",), ("asset",), ("placa",), ("plate",), ("unidad",), ("equipo",)],
}
ORDEN_MAPEO = list(_ALIAS.keys())


def _norm(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def sugerir_mapeo(columnas) -> dict:
    """Propone qué columna del archivo corresponde a cada variable."""
    cols = list(columnas)
    norm = {c: _norm(c) for c in cols}
    usadas, mapeo = set(), {}
    for canon in ORDEN_MAPEO:               # coincidencias exactas primero
        if canon in cols:
            mapeo[canon] = canon
            usadas.add(canon)
    for canon in ORDEN_MAPEO:
        if canon in mapeo:
            continue
        mapeo[canon] = None
        for c in cols:
            if c in usadas:
                continue
            if any(all(tok in norm[c] for tok in toks) for toks in _ALIAS[canon]):
                mapeo[canon] = c
                usadas.add(c)
                break
    return mapeo


def _a_numero(serie: pd.Series) -> pd.Series:
    """Convierte texto como '95 °C', '1,250' o 'Sí' a número."""
    s = serie.astype(str).str.strip()
    num = pd.to_numeric(
        s.str.replace(",", ".", regex=False).str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")
    si, no = {"si", "sí", "on", "true", "yes", "pisado", "activo"}, {"no", "off", "false", "inactivo"}
    palabras = s.str.lower().map(lambda v: 1.0 if v in si else (0.0 if v in no else np.nan))
    return num.fillna(palabras)


def aplicar_mapeo(raw: pd.DataFrame, mapeo: dict) -> pd.DataFrame:
    """Devuelve un DataFrame con los nombres estándar y valores numéricos."""
    out = pd.DataFrame(index=raw.index)
    for canon, col in mapeo.items():
        if not col or col not in raw.columns:
            continue
        if canon == "Timestamp":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ts = pd.to_datetime(raw[col], errors="coerce")
            if ts.notna().any():
                out[canon] = ts
        elif canon == "Vehiculo":
            ids = raw[col].astype(str).str.strip()
            if ids.replace({"": np.nan, "nan": np.nan}).notna().any():
                out[canon] = ids
        else:
            out[canon] = _a_numero(raw[col])
    return out.reset_index(drop=True)


def normalizar_unidades(df: pd.DataFrame, temp_unit: str = "auto", torque_nominal: float = 0.0):
    """Lleva los datos de cualquier vehículo a las unidades que usa el análisis.

    - Temperaturas: °F -> °C (automático si la mediana supera 150).
    - Pedal, carga y torque en fracción 0-1 -> %.
    - Torque en Nm -> % del torque nominal (el indicado o, si no, el percentil 99).
    Devuelve (dataframe, lista de avisos con lo que se convirtió).
    """
    df, notas = df.copy(), []
    for col in ("Temp_Aceite_C", "Temp_Refrigerante_C"):
        if _tiene(df, col) and (temp_unit == "F" or (temp_unit == "auto" and df[col].median() > 150)):
            df[col] = (df[col] - 32) * 5 / 9
            notas.append(f"{col}: convertida de °F a °C")
    for col in ("Pedal_Acelerador_pct", "Carga_Motor_pct", "Torque_Real_pct", "Torque_Exigido_pct"):
        if not _tiene(df, col):
            continue
        mx = df[col].max()
        if col.startswith("Torque") and mx > 120:
            nominal = torque_nominal if torque_nominal and torque_nominal > 0 else float(df[col].quantile(0.99))
            df[col] = df[col] / nominal * 100
            notas.append(f"{col}: parecía venir en Nm; se convirtió a % de {nominal:.0f} Nm")
        elif mx <= 1.0:
            df[col] = df[col] * 100
            notas.append(f"{col}: venía como fracción de 0 a 1; se pasó a %")
    return df, notas


def _limpiar_tabla(df: pd.DataFrame):
    """Valida que una tabla extraída del PDF parezca datos de telemetría."""
    df = df.copy()
    vistos, nombres = {}, []
    for i, c in enumerate(df.columns):      # nombres únicos y sin vacíos
        base = str(c).strip() or f"Col{i}"
        vistos[base] = vistos.get(base, 0) + 1
        nombres.append(base if vistos[base] == 1 else f"{base}_{vistos[base]}")
    df.columns = nombres
    df = df.replace(r"^\s*$", np.nan, regex=True)   # celdas vacías -> NaN
    df = df.dropna(how="all").loc[:, df.notna().any()]
    if df.shape[1] < 3 or df.shape[0] < 3:
        return None
    numericas = sum(_a_numero(df[c]).notna().mean() >= 0.5 for c in df.columns)
    return df.reset_index(drop=True) if numericas >= 2 else None


def extraer_tablas_pdf(data: bytes):
    """Extrae la tabla de datos más grande de un PDF con texto seleccionable.

    Devuelve None si el PDF no tiene tablas legibles (por ejemplo, si es una
    captura de pantalla guardada como PDF: ahí solo hay píxeles, no datos).
    """
    candidatas = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            for estrategia in ("lines", "text"):
                try:
                    tablas = page.find_tables(strategy=estrategia).tables
                except Exception:
                    continue
                hallo = False
                for t in tablas:
                    limpia = _limpiar_tabla(t.to_pandas())
                    if limpia is not None:
                        candidatas.append(limpia)
                        hallo = True
                if hallo:
                    break
    if not candidatas:
        return None
    grupos = {}
    for df in candidatas:
        grupos.setdefault(tuple(df.columns), []).append(df)
    mejor = max(grupos.values(), key=lambda g: sum(len(d) for d in g))
    return pd.concat(mejor, ignore_index=True)


# ==========================================
# Qué se puede evaluar con las columnas disponibles
# ==========================================
def _tiene(df, col) -> bool:
    return col in df.columns and df[col].notna().any()


def _cols_regla(df: pd.DataFrame, regla: str) -> list:
    """Columnas de una regla; sin carga del motor se usa el torque real como indicador de carga."""
    cols = list(REGLAS_COLUMNAS[regla])
    if "Carga_Motor_pct" in cols and not _tiene(df, "Carga_Motor_pct") and _tiene(df, "Torque_Real_pct"):
        cols[cols.index("Carga_Motor_pct")] = "Torque_Real_pct"
    return cols


def reglas_evaluables(df: pd.DataFrame) -> dict:
    return {r: all(_tiene(df, c) for c in _cols_regla(df, r)) for r in REGLAS_COLUMNAS}


def columnas_ia_disponibles(df: pd.DataFrame) -> list:
    cols = [c for c in COLUMNAS_IA if _tiene(df, c)]
    if _tiene(df, "Velocidad_kmh") and _tiene(df, "RPM_Motor") and "Ratio_Speed_RPM" not in cols:
        cols.append("Ratio_Speed_RPM")
    return cols


def hay_algo_evaluable(df: pd.DataFrame) -> bool:
    return any(reglas_evaluables(df).values()) or len(columnas_ia_disponibles(df)) >= 3


def descripcion_no_evaluable(df: pd.DataFrame) -> list:
    """Frases del tipo 'Lugging: faltan Carga_Motor_pct' para avisar al usuario."""
    msgs = []
    nombres = {"Riesgo_Lugging": "Lugging", "Riesgo_Abuso_Embrague": "Abuso de embrague",
               "Sobrecalentamiento_Aceite": "Sobrecalentamiento de aceite"}
    for regla in REGLAS_COLUMNAS:
        faltan = [("Carga_Motor_pct (o Torque_Real_pct)" if c == "Carga_Motor_pct" else c)
                  for c in _cols_regla(df, regla) if not _tiene(df, c)]
        if faltan:
            msgs.append(f"{nombres[regla]}: faltan {', '.join(faltan)}")
    if len(columnas_ia_disponibles(df)) < 3:
        msgs.append("Isolation Forest: se necesitan al menos 3 variables numéricas")
    return msgs


# ==========================================
# 1. CARGA / SIMULACIÓN DE DATOS
# ==========================================
def cargar_datos_telemetria(n_samples: int = 100, inyectar_eventos: bool = True, seed: int = 42) -> pd.DataFrame:
    """Dataset SIMULADO con las variables de Geotab (Mercedes Benz Actros).

    Los valores son aleatorios dentro de los rangos del script original; no
    son mediciones reales. Con inyectar_eventos=True se agregan unos pocos
    eventos de lugging y de abuso de embrague, porque con datos 100 %
    aleatorios casi nunca se cumplen esas condiciones y el demo saldría vacío.
    """
    np.random.seed(seed)

    data = {
        "Timestamp": pd.date_range(start="2026-10-02 05:00", periods=n_samples, freq="1min"),
        "Carga_Motor_pct": np.random.uniform(20, 100, n_samples),
        "Torque_Exigido_pct": np.random.choice([80, 90, 100], n_samples),
        "Torque_Real_pct": np.random.uniform(50, 95, n_samples),
        "RPM_Motor": np.random.uniform(960, 1880, n_samples),
        "Velocidad_kmh": np.random.uniform(14, 60, n_samples),
        "Temp_Aceite_C": np.random.uniform(85, 122, n_samples),
        "Temp_Refrigerante_C": np.random.uniform(90, 102, n_samples),
        "Pedal_Acelerador_pct": np.random.uniform(10, 100, n_samples),
        "Pedal_Embrague": np.random.choice([0, 1], p=[0.95, 0.05], size=n_samples),
        "Pedal_Freno": np.random.choice([0, 1], p=[0.8, 0.2], size=n_samples),
        "Peso_Bruto_kg": np.random.uniform(50680, 52000, n_samples),
        "Aceleracion_Longitudinal_g": np.random.uniform(-0.34, 0.27, n_samples),
    }
    df = pd.DataFrame(data)

    if inyectar_eventos and n_samples >= 20:
        rng = np.random.default_rng(abs(seed - 41))
        idx = rng.choice(n_samples, size=8, replace=False)
        for i in idx[:4]:  # lugging
            df.loc[i, ["Carga_Motor_pct", "RPM_Motor", "Pedal_Acelerador_pct"]] = [
                rng.uniform(90, 100), rng.uniform(960, 1100), rng.uniform(85, 100)]
        for i in idx[4:]:  # abuso de embrague
            df.loc[i, ["Pedal_Embrague", "RPM_Motor"]] = [1, rng.uniform(1500, 1800)]
    return df


def cargar_datos_flota(n_samples: int = 100) -> pd.DataFrame:
    """Flota SIMULADA de 3 vehículos con regímenes de motor muy distintos."""
    perfiles = [("Camión A (diésel pesado)", 1.0, 42), ("Bus B (diésel urbano)", 1.25, 7),
                ("Auto C (gasolina)", 3.0, 13)]
    partes = []
    for nombre, factor, semilla in perfiles:
        d = cargar_datos_telemetria(n_samples, seed=semilla)
        d["RPM_Motor"] = d["RPM_Motor"] * factor
        d.insert(0, "Vehiculo", nombre)
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


# ==========================================
# 2. VARIABLES CALCULADAS (estrés del tren motriz)
# ==========================================
def calcular_caracteristicas_operacionales(df: pd.DataFrame, u: Umbrales = None) -> pd.DataFrame:
    u = u or Umbrales()
    df = df.copy()
    ok = reglas_evaluables(df)
    ef = umbrales_efectivos(df, u)
    carga = df["Carga_Motor_pct"] if _tiene(df, "Carga_Motor_pct") else df.get("Torque_Real_pct", np.nan)

    # A. Lugging: sobreesfuerzo a bajas RPM con alta carga
    df["Riesgo_Lugging"] = np.where(
        ok["Riesgo_Lugging"]
        & (carga > u.lugging_carga_min)
        & (df.get("RPM_Motor", np.nan) < ef["lugging_rpm"])
        & (df.get("Pedal_Acelerador_pct", np.nan) > u.lugging_pedal_min), 1, 0)

    # B. Abuso de embrague (pedal pisado con RPM altas). Sirve con 0/1 o con %.
    df["Riesgo_Abuso_Embrague"] = np.where(
        ok["Riesgo_Abuso_Embrague"]
        & (df.get("Pedal_Embrague", np.nan) > 0)
        & (df.get("RPM_Motor", np.nan) > ef["embrague_rpm"]), 1, 0)

    # C. Estrés térmico crítico
    df["Sobrecalentamiento_Aceite"] = np.where(
        ok["Sobrecalentamiento_Aceite"] & (df.get("Temp_Aceite_C", np.nan) >= u.temp_aceite_critica), 1, 0)

    # D. Relación velocidad / RPM
    if _tiene(df, "Velocidad_kmh") and _tiene(df, "RPM_Motor"):
        df["Ratio_Speed_RPM"] = df["Velocidad_kmh"] / (df["RPM_Motor"] + 1e-5)
    return df


# ==========================================
# 3. MODELO DE IA: DETECCIÓN DE ANOMALÍAS
# ==========================================
def entrenar_modelo_ia(df: pd.DataFrame, contamination: float = 0.08) -> pd.DataFrame:
    df = df.copy()
    feats = columnas_ia_disponibles(df)
    if len(feats) < 3 or len(df) < 10:  # muy pocos datos: no se puede entrenar con sentido
        df["Anomalia_IA"], df["Puntaje_Anomalia"], df["Riesgo_Tren_Motriz"] = 1, 0.0, "NORMAL"
        return df
    X = df[feats].fillna(df[feats].median())

    model = IsolationForest(contamination=contamination, random_state=42)
    df["Anomalia_IA"] = model.fit_predict(X)            # -1 anomalía, 1 normal
    df["Puntaje_Anomalia"] = -model.score_samples(X)    # más alto = más raro
    df["Riesgo_Tren_Motriz"] = np.where(df["Anomalia_IA"] == -1, "ALTO RIESGO", "NORMAL")
    return df


# ==========================================
# 4. MATRIZ DE MANTENIMIENTO PREVENTIVO
# ==========================================
_HALLAZGOS = [
    ("Riesgo_Lugging", "Lugging (bajas RPM con carga alta)",
     "Transmisión (dientes de engranajes), embrague, soportes del motor, cigüeñal",
     "Inspeccionar picaduras y desgaste en engranajes y revisar el embrague; capacitar al conductor para reducir de marcha antes."),
    ("Riesgo_Abuso_Embrague", "Abuso de embrague a altas RPM",
     "Disco y plato de presión, balinera de empuje",
     "Medir espesor del disco y recorrido del pedal; capacitar en el uso del embrague."),
    ("Sobrecalentamiento_Aceite", "Sobrecalentamiento de aceite",
     "Aceite, enfriador de aceite, rodamientos",
     "Revisar nivel y viscosidad, cambiar aceite y filtro, inspeccionar el enfriador."),
    ("Anomalia_IA", "Combinaciones atípicas (Isolation Forest)",
     "A definir según el patrón detectado",
     "Revisar manualmente las muestras marcadas y cruzar con el historial de fallas."),
]


def _prioridad(pct: float) -> str:
    if pct >= 5:
        return "Alta"
    if pct >= 1:
        return "Media"
    if pct > 0:
        return "Baja"
    return "Sin hallazgos"


def generar_matriz_mantenimiento(df: pd.DataFrame, evaluables: dict = None) -> pd.DataFrame:
    n = len(df)
    evaluables = evaluables or {}
    filas = []
    for col, nombre, componente, accion in _HALLAZGOS:
        if not evaluables.get(col, True):
            filas.append({"Hábito / hallazgo": nombre, "Muestras": 0, "% del recorrido": None,
                          "Prioridad": "No evaluable", "Componente en riesgo": componente,
                          "Acción recomendada": "Faltan columnas en los datos para evaluar este punto."})
            continue
        cuenta = int((df[col] == -1).sum()) if col == "Anomalia_IA" else int(df[col].sum())
        pct = 100 * cuenta / n if n else 0.0
        filas.append({
            "Hábito / hallazgo": nombre,
            "Muestras": cuenta,
            "% del recorrido": round(pct, 2),
            "Prioridad": _prioridad(pct),
            "Componente en riesgo": componente,
            "Acción recomendada": accion if cuenta else "Ninguna por ahora.",
        })
    return pd.DataFrame(filas)


_CLAVES_DIAG = {"Riesgo_Lugging": "lugging", "Riesgo_Abuso_Embrague": "embrague",
                "Sobrecalentamiento_Aceite": "temp_aceite", "Anomalia_IA": "anomalias"}


def resumen_para_diagnostico(matriz: pd.DataFrame) -> dict:
    """Convierte la matriz de mantenimiento en {hábito: prioridad} para el motor de diagnóstico."""
    por_nombre = {nombre: _CLAVES_DIAG[col] for col, nombre, _, _ in _HALLAZGOS}
    return {por_nombre[r["Hábito / hallazgo"]]: r["Prioridad"] for _, r in matriz.iterrows()
            if r["Hábito / hallazgo"] in por_nombre}


def run_analysis(df: pd.DataFrame, u: Umbrales = None, contamination: float = 0.08):
    """Pipeline completo. Devuelve (df_con_resultados, matriz)."""
    evaluables = reglas_evaluables(df)
    out = calcular_caracteristicas_operacionales(df, u)
    evaluables["Anomalia_IA"] = len(columnas_ia_disponibles(out)) >= 3 and len(out) >= 10
    out = entrenar_modelo_ia(out, contamination)
    return out, generar_matriz_mantenimiento(out, evaluables)


_RANGO_PRIORIDAD = {"Alta": 3, "Media": 2, "Baja": 1, "Sin hallazgos": 0, "No evaluable": -1}
_COLS_RESUMEN = {"Lugging (bajas RPM con carga alta)": "Lugging (%)",
                 "Abuso de embrague a altas RPM": "Abuso de embrague (%)",
                 "Sobrecalentamiento de aceite": "Aceite crítico (%)",
                 "Combinaciones atípicas (Isolation Forest)": "Alto riesgo IA (%)"}


def run_analysis_flota(df: pd.DataFrame, u: Umbrales = None, contamination: float = 0.08, min_filas: int = 3):
    """Si el archivo trae varios vehículos, analiza cada uno con sus propios umbrales.

    Devuelve ({vehículo: (resultados, matriz)}, resumen de la flota) o None si hay un solo vehículo.
    """
    if "Vehiculo" not in df.columns:
        return None
    ids = [v for v in df["Vehiculo"].dropna().unique() if str(v).strip()]
    if len(ids) < 2:
        return None
    resultados, filas = {}, []
    for v in ids:
        g = df[df["Vehiculo"] == v].drop(columns=["Vehiculo"]).reset_index(drop=True)
        if len(g) < min_filas:
            continue
        res, mat = run_analysis(g, u, contamination)
        resultados[str(v)] = (res, mat)
        fila = {"Vehículo": str(v), "Muestras": len(g)}
        for _, r in mat.iterrows():
            fila[_COLS_RESUMEN[r["Hábito / hallazgo"]]] = r["% del recorrido"]
        fila["Prioridad máxima"] = max(mat["Prioridad"], key=lambda p: _RANGO_PRIORIDAD.get(p, -1))
        filas.append(fila)
    if len(resultados) < 2:
        return None
    resumen = pd.DataFrame(filas)
    resumen["_orden"] = resumen["Prioridad máxima"].map(_RANGO_PRIORIDAD)
    resumen["_suma"] = resumen[list(_COLS_RESUMEN.values())].fillna(0).sum(axis=1)
    resumen = resumen.sort_values(["_orden", "_suma"], ascending=False).drop(columns=["_orden", "_suma"])
    return resultados, resumen.reset_index(drop=True)


if __name__ == "__main__":
    resultado, matriz = run_analysis(cargar_datos_telemetria())
    print(matriz.to_string(index=False))
