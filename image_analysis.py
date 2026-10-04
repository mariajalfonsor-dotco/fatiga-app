"""
image_analysis.py
------------------
Preprocesamiento de la imagen de la fractura + extracción de
características de textura (matriz de co-ocurrencia de niveles de gris,
GLCM) + clasificación del tipo de fractura con un modelo entrenado.

IMPORTANTE (léelo para tu documento de tesis):
El modelo se entrena aquí con datos SINTÉTICOS que imitan, en términos de
textura, las cuatro categorías estándar de fractografía (fatiga, clivaje,
hoyuelos, intergranular). Esto permite tener una app funcional de punta a
punta sin depender de acceso a un dataset real durante el desarrollo.
Para una versión con validez experimental real, reemplaza la función
`_generate_synthetic_dataset()` por la extracción de estas mismas
características sobre imágenes reales etiquetadas (por ejemplo, un
subconjunto de un dataset público de fractografía SEM) y vuelve a
entrenar con `train_image_model()`.
"""

from pathlib import Path

import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = Path(__file__).parent / "models" / "image_model.joblib"
FRACTURE_CLASSES = ["fatiga", "clivaje", "hoyuelos", "intergranular", "sobretorsion", "porosidad"]

FEATURE_NAMES = ["contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"]


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------

def preprocess_image(img_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    """Aplica el pipeline: gris -> reducción de ruido -> CLAHE -> resize.

    Devuelve una imagen en escala de grises, uint8, lista para extraer
    características. No se normaliza a [0,1] aquí porque graycomatrix
    necesita niveles de gris enteros.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # ruido, sin borrar estrías finas
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray


def extract_texture_features(gray: np.ndarray) -> np.ndarray:
    """Extrae descriptores de textura GLCM (a 4 orientaciones, promediados)."""
    # Reducir a 32 niveles de gris para que la GLCM sea manejable
    levels = 32
    quantized = (gray.astype(np.float32) / 256 * levels).astype(np.uint8)
    glcm = graycomatrix(
        quantized, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels, symmetric=True, normed=True,
    )
    feats = [graycoprops(glcm, prop).mean() for prop in
              ["contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"]]
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Datos sintéticos de entrenamiento (ver aviso arriba)
# ---------------------------------------------------------------------------

def _generate_synthetic_dataset(n_per_class: int = 150, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Medias aproximadas de [contrast, homogeneity, energy, correlation, dissimilarity, ASM]
    # calibradas para reflejar la textura típica de cada clase (ver docstring).
    profiles = {
        "fatiga":         np.array([2.5, 0.60, 0.120, 0.88, 1.0, 0.0140]),
        "intergranular":  np.array([4.0, 0.50, 0.090, 0.95, 1.6, 0.0100]),
        "sobretorsion":   np.array([5.0, 0.42, 0.065, 0.75, 2.0, 0.0060]),
        "clivaje":        np.array([7.5, 0.32, 0.050, 0.60, 2.8, 0.0040]),
        "hoyuelos":       np.array([9.5, 0.25, 0.030, 0.40, 3.6, 0.0020]),
        # Cavidades irregulares dispersas -> la más heterogénea de todas
        "porosidad":      np.array([11.0, 0.15, 0.020, 0.25, 4.5, 0.0012]),
    }
    X, y = [], []
    for cls, mean in profiles.items():
        noise = rng.normal(0, np.abs(mean) * 0.10 + 0.01, size=(n_per_class, len(mean)))
        samples = np.clip(mean + noise, 0, None)
        X.append(samples)
        y += [cls] * n_per_class
    return np.vstack(X), np.array(y)


def train_image_model():
    X, y = _generate_synthetic_dataset()
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    clf.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    return clf


def load_image_model():
    if not MODEL_PATH.exists():
        return train_image_model()
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# API principal del módulo
# ---------------------------------------------------------------------------

def analyze_image(img_bgr: np.ndarray) -> dict:
    """Pipeline completo: preprocesa, extrae features, clasifica.

    Devuelve dict con: clase predicha, confianza, y las features (útil
    para mostrar en la interfaz o depurar).
    """
    gray = preprocess_image(img_bgr)
    features = extract_texture_features(gray)
    clf = load_image_model()

    proba = clf.predict_proba(features.reshape(1, -1))[0]
    pred_idx = int(np.argmax(proba))
    pred_class = clf.classes_[pred_idx]
    confidence = float(proba[pred_idx])

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": dict(zip(clf.classes_, proba.tolist())),
        "features": dict(zip(FEATURE_NAMES, features.tolist())),
        "preprocessed_image": gray,
    }


if __name__ == "__main__":
    # Prueba rápida con una imagen sintética (patrón de textura aleatoria)
    fake_img = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
    result = analyze_image(fake_img)
    print("Clase predicha:", result["predicted_class"])
    print("Confianza:", round(result["confidence"], 3))
    print("Probabilidades:", {k: round(v, 3) for k, v in result["probabilities"].items()})
