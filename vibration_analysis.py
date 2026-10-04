"""
vibration_analysis.py
----------------------
Análisis espectral de la señal de vibración: FFT + extracción de
características + clasificación del tipo de falla (normal, pista interna,
pista externa, elemento rodante), inspirado en el diagnóstico clásico de
fallas de rodamientos (frecuencias BPFI, BPFO, BSF).

IMPORTANTE (léelo para tu documento de tesis):
Igual que en image_analysis.py, el modelo se entrena con señales
SINTÉTICAS generadas a partir de las frecuencias características típicas
de cada tipo de falla (como proporción de la frecuencia de giro). Para una
versión con validez experimental real, reemplaza
`_generate_synthetic_signals()` cargando señales reales (por ejemplo del
dataset público CWRU) y vuelve a entrenar con `train_vibration_model()`.
"""

from pathlib import Path

import cv2
import numpy as np
import joblib
from scipy.fft import rfft, rfftfreq
from scipy.signal import spectrogram
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = Path(__file__).parent / "models" / "vibration_model.joblib"
VIBRATION_CLASSES = ["normal", "pista_interna", "pista_externa", "elemento_rodante"]
FEATURE_NAMES = ["dominant_freq_ratio", "spectral_centroid", "rms", "kurtosis", "crest_factor"]

# Multiplicadores típicos de frecuencia característica de falla respecto a
# la frecuencia de giro del eje (valores de referencia usados en literatura
# de diagnóstico de rodamientos; varían según la geometría real del rodamiento).
FAULT_FREQ_RATIO = {
    "pista_interna": 5.4,
    "pista_externa": 3.6,
    "elemento_rodante": 2.4,
}


# ---------------------------------------------------------------------------
# Extracción de características
# ---------------------------------------------------------------------------

def extract_spectral_features(signal: np.ndarray, fs: float, shaft_freq: float) -> np.ndarray:
    """Extrae características del espectro de una señal de vibración.

    signal: arreglo 1D con la señal en el tiempo.
    fs: frecuencia de muestreo (Hz).
    shaft_freq: frecuencia de giro del eje (Hz), usada para normalizar el
                pico dominante y hacerlo comparable entre condiciones.
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - signal.mean()

    n = len(signal)
    freqs = rfftfreq(n, d=1 / fs)
    spectrum = np.abs(rfft(signal))

    # Ignorar la componente DC y frecuencias por debajo de 0.5x la de giro
    mask = freqs > (0.5 * shaft_freq)
    freqs_m, spectrum_m = freqs[mask], spectrum[mask]

    dominant_freq = freqs_m[np.argmax(spectrum_m)] if len(freqs_m) else 0.0
    dominant_freq_ratio = dominant_freq / shaft_freq if shaft_freq else 0.0

    spectral_centroid = (
        np.sum(freqs_m * spectrum_m) / np.sum(spectrum_m) if np.sum(spectrum_m) > 0 else 0.0
    ) / shaft_freq if shaft_freq else 0.0

    rms = np.sqrt(np.mean(signal ** 2))
    kurt = kurtosis(signal, fisher=True)  # impulsividad: alta en fallas localizadas
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms if rms > 0 else 0.0

    return np.array([dominant_freq_ratio, spectral_centroid, rms, kurt, crest_factor],
                     dtype=np.float32)


# ---------------------------------------------------------------------------
# Datos sintéticos de entrenamiento (ver aviso arriba)
# ---------------------------------------------------------------------------

def _simulate_signal(fault: str, fs: float, duration: float, shaft_freq: float,
                      rng: np.random.Generator) -> np.ndarray:
    t = np.arange(0, duration, 1 / fs)
    # Componente base: giro del eje + un armónico
    signal = 0.5 * np.sin(2 * np.pi * shaft_freq * t)
    signal += 0.15 * np.sin(2 * np.pi * 2 * shaft_freq * t)

    if fault != "normal":
        fault_freq = FAULT_FREQ_RATIO[fault] * shaft_freq
        # Tren de impactos periódicos moduladas (típico de defectos localizados)
        impact_train = np.sin(2 * np.pi * fault_freq * t)
        envelope = (impact_train > 0.95).astype(float)  # picos angostos = impulsivo
        amplitude = {"pista_interna": 1.2, "pista_externa": 0.9, "elemento_rodante": 1.0}[fault]
        signal += amplitude * envelope * np.sin(2 * np.pi * fault_freq * 3 * t)

    signal += rng.normal(0, 0.08, size=t.shape)  # ruido de fondo
    return signal


def _generate_synthetic_dataset(n_per_class: int = 120, seed: int = 7):
    rng = np.random.default_rng(seed)
    fs = 12_000.0  # Hz, igual orden de magnitud que datasets reales tipo CWRU
    duration = 0.5  # segundos por muestra

    X, y = [], []
    for fault in VIBRATION_CLASSES:
        for _ in range(n_per_class):
            shaft_freq = rng.uniform(28, 32)  # ~1740-1920 rpm, rango típico de motores de prueba
            sig = _simulate_signal(fault, fs, duration, shaft_freq, rng)
            feats = extract_spectral_features(sig, fs, shaft_freq)
            X.append(feats)
            y.append(fault)
    return np.vstack(X), np.array(y)


def train_vibration_model():
    X, y = _generate_synthetic_dataset()
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=7)
    clf.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    return clf


def load_vibration_model():
    if not MODEL_PATH.exists():
        return train_vibration_model()
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# API principal del módulo
# ---------------------------------------------------------------------------

def analyze_vibration(signal: np.ndarray, fs: float, shaft_freq: float) -> dict:
    features = extract_spectral_features(signal, fs, shaft_freq)
    clf = load_vibration_model()

    proba = clf.predict_proba(features.reshape(1, -1))[0]
    pred_idx = int(np.argmax(proba))
    pred_class = clf.classes_[pred_idx]
    confidence = float(proba[pred_idx])

    freqs = rfftfreq(len(signal), d=1 / fs)
    spectrum = np.abs(rfft(np.asarray(signal) - np.mean(signal)))

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": dict(zip(clf.classes_, proba.tolist())),
        "features": dict(zip(FEATURE_NAMES, features.tolist())),
        "freqs": freqs,
        "spectrum": spectrum,
    }


def generate_demo_signal(fault: str, fs: float = 12_000.0, duration: float = 0.5,
                          shaft_freq: float = 30.0, seed: int | None = None) -> np.ndarray:
    """Genera una señal de demostración para probar la app sin sensor real."""
    rng = np.random.default_rng(seed)
    return _simulate_signal(fault, fs, duration, shaft_freq, rng)


# ---------------------------------------------------------------------------
# Engranajes de la transmisión: frecuencia de engrane (GMF) y bandas laterales
# ---------------------------------------------------------------------------
# En una caja de cambios, el engrane de un par de ruedas con z dientes produce
# una componente a GMF = z x frecuencia de giro. Un diente dañado o una
# excentricidad modulan esa componente una vez por vuelta, y aparecen bandas
# laterales a GMF +/- k x frecuencia de giro. La razón entre la energía de las
# bandas y la amplitud de GMF es un indicador clásico de daño en engranajes.
GEAR_CONDITIONS = {"sano": "Engranaje sano", "desgaste_leve": "Desgaste leve", "diente_danado": "Diente dañado"}
# Umbrales de REFERENCIA para este prototipo; deben calibrarse con mediciones reales.
GEAR_RATIO_VIGILAR = 0.20
GEAR_RATIO_ALERTA = 0.50


def gear_mesh_analysis(signal: np.ndarray, fs: float, shaft_freq: float, teeth: int,
                       n_sidebands: int = 3) -> dict:
    """Mide GMF, 2xGMF y bandas laterales; devuelve amplitudes e indicador de daño."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    window = np.hanning(n)
    spectrum = np.abs(rfft(x * window)) * 2 / window.sum()  # amplitud aproximada
    freqs = rfftfreq(n, d=1 / fs)
    tol = max(1.5, 1.5 * (freqs[1] - freqs[0]))

    gmf = teeth * shaft_freq
    if gmf + n_sidebands * shaft_freq >= fs / 2:
        raise ValueError("La frecuencia de engrane supera el límite de Nyquist: sube la frecuencia "
                         "de muestreo o reduce dientes / frecuencia de giro.")

    def amp_at(f: float) -> float:
        m = (freqs >= f - tol) & (freqs <= f + tol)
        return float(spectrum[m].max()) if m.any() else 0.0

    a_gmf = amp_at(gmf)
    rows = [{"Componente": "GMF (1× engrane)", "Frecuencia (Hz)": round(gmf, 1), "Amplitud": a_gmf}]
    if 2 * gmf < fs / 2:
        rows.append({"Componente": "2× GMF", "Frecuencia (Hz)": round(2 * gmf, 1), "Amplitud": amp_at(2 * gmf)})
    side_sum = 0.0
    for k in range(1, n_sidebands + 1):
        for sign, label in ((-1, "−"), (1, "+")):
            f = gmf + sign * k * shaft_freq
            a = amp_at(f)
            side_sum += a
            rows.append({"Componente": f"GMF {label} {k}× giro", "Frecuencia (Hz)": round(f, 1), "Amplitud": a})

    ratio = side_sum / a_gmf if a_gmf > 0 else float("nan")
    if not np.isfinite(ratio):
        estado = "No se detecta la frecuencia de engrane: revisa dientes, frecuencia de giro y muestreo."
        nivel = "sin_dato"
    elif ratio >= GEAR_RATIO_ALERTA:
        estado = "Alerta: bandas laterales fuertes; posible diente dañado o excentricidad."
        nivel = "alerta"
    elif ratio >= GEAR_RATIO_VIGILAR:
        estado = "Vigilar: bandas laterales elevadas; posible desgaste incipiente."
        nivel = "vigilar"
    else:
        estado = "Normal: bandas laterales bajas."
        nivel = "normal"

    return {"gmf": gmf, "ratio": ratio, "estado": estado, "nivel": nivel,
            "tabla": rows, "freqs": freqs, "spectrum": spectrum}


def generate_gear_demo_signal(condition: str, fs: float = 12_000.0, duration: float = 1.0,
                              shaft_freq: float = 30.0, teeth: int = 40,
                              seed: int | None = None) -> np.ndarray:
    """Señal SIMULADA de un engranaje: portadora a GMF con modulación por vuelta."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fs)
    gmf = teeth * shaft_freq
    m = {"sano": 0.0, "desgaste_leve": 0.25, "diente_danado": 0.7}[condition]
    signal = 0.4 * np.sin(2 * np.pi * shaft_freq * t)
    signal += (1 + m * np.cos(2 * np.pi * shaft_freq * t)) * np.sin(2 * np.pi * gmf * t)
    signal += 0.3 * np.sin(2 * np.pi * 2 * gmf * t)
    if condition == "diente_danado":  # impacto una vez por vuelta
        phase = (t * shaft_freq) % 1.0
        signal += 0.8 * np.exp(-((phase * 40) ** 2)) * np.sin(2 * np.pi * 3 * gmf * t)
    signal += rng.normal(0, 0.05, size=t.shape)
    return signal


def spectrogram_image(signal: np.ndarray, fs: float, fmax: float = 3000.0, height: int = 360,
                      width: int = 900) -> np.ndarray:
    """Espectrograma (tiempo-frecuencia) como imagen RGB lista para mostrar."""
    x = np.asarray(signal, dtype=np.float64) - np.mean(signal)
    nper = int(min(512, max(64, len(x) // 8)))
    f, t, sxx = spectrogram(x, fs=fs, nperseg=nper, noverlap=int(nper * 0.75))
    keep = f <= fmax
    db = 10 * np.log10(sxx[keep] + 1e-12)
    lo, hi = np.percentile(db, 5), np.percentile(db, 99.5)
    img = np.clip((db - lo) / (hi - lo + 1e-9), 0, 1)
    img = np.flipud(img)  # frecuencias bajas abajo
    img8 = cv2.resize((img * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    demo = generate_demo_signal("pista_interna", seed=1)
    result = analyze_vibration(demo, fs=12_000.0, shaft_freq=30.0)
    print("Clase predicha:", result["predicted_class"])
    print("Confianza:", round(result["confidence"], 3))
    print("Probabilidades:", {k: round(v, 3) for k, v in result["probabilities"].items()})
