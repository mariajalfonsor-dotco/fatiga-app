"""
ocr_reportes.py
---------------
Lectura de reportes que son IMÁGENES (capturas de pantalla guardadas como
PDF, o fotos/PNG). Usa OCR para reconocer el texto y luego interpreta:

1. Tarjetas de eventos de video de Geotab (evento, vehículo, conductor, fecha).
2. El total de eventos que declara el reporte ("18205 Eventos en vídeo").
3. Variables numéricas sueltas con nombre reconocible (RPM, temperatura...).

Motores de OCR (el primero que esté disponible):
- RapidOCR:  python -m pip install rapidocr onnxruntime
- Tesseract: requiere instalar el programa Tesseract + pip install pytesseract
"""

import os
import re
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from habitos_conduccion import _ALIAS, ORDEN_MAPEO, _norm


class OCRNoDisponible(Exception):
    pass


MENSAJE_INSTALAR = (
    "Para leer imágenes necesitas un motor de OCR. Cierra la app (Ctrl + C en la ventana negra), "
    "ejecuta  python -m pip install rapidocr onnxruntime  y vuelve a abrir la app. "
    "Si la instalación falla en tu versión de Python, instala el programa Tesseract OCR "
    "y luego  python -m pip install pytesseract."
)

_motor = None


def _crear_motor():
    """Devuelve una función imagen_bgr -> lista de dicts {text, score, x0, y0, x1, y1}."""
    try:
        from rapidocr import RapidOCR
        eng = RapidOCR()

        def run_rapid(img):
            res = eng(img)
            if res is None or res.txts is None:
                return []
            out = []
            for box, txt, sc in zip(res.boxes, res.txts, res.scores):
                xs, ys = [p[0] for p in box], [p[1] for p in box]
                out.append({"text": str(txt).strip(), "score": float(sc),
                            "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)})
            return out
        return run_rapid
    except Exception:
        pass

    try:
        import pytesseract
        for ruta in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                     r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
            if os.path.exists(ruta):
                pytesseract.pytesseract.tesseract_cmd = ruta
                break
        pytesseract.get_tesseract_version()
        lang = "spa+eng" if "spa" in pytesseract.get_languages() else "eng"

        def run_tess(img):
            d = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
            lineas = {}
            for i, t in enumerate(d["text"]):
                if t.strip() and float(d["conf"][i]) >= 0:
                    lineas.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(i)
            out = []
            for idxs in lineas.values():
                out.append({
                    "text": " ".join(d["text"][i] for i in idxs).strip(),
                    "score": float(np.mean([float(d["conf"][i]) for i in idxs])) / 100,
                    "x0": min(d["left"][i] for i in idxs), "y0": min(d["top"][i] for i in idxs),
                    "x1": max(d["left"][i] + d["width"][i] for i in idxs),
                    "y1": max(d["top"][i] + d["height"][i] for i in idxs)})
            return out
        return run_tess
    except Exception:
        raise OCRNoDisponible(MENSAJE_INSTALAR)


def leer_imagen(img_bgr: np.ndarray) -> list:
    global _motor
    if _motor is None:
        _motor = _crear_motor()
    return _motor(img_bgr)


# ---------------------------------------------------------------------------
# Interpretación del texto leído
# ---------------------------------------------------------------------------
_MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
          "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}
_RE_FECHA = re.compile(r"^([A-Za-zñÑ]+)\s+(\d{1,2})\s+at\s+(\d{1,2}):(\d{2})\s*(AM|PM)$", re.I)
_RE_DURACION = re.compile(r"^\d{1,2}:\d{2}$")

_CATEGORIAS = [  # (palabras clave normalizadas, categoría, relevancia)
    (("aceleracion brusca", "harsh accel"), "Aceleración brusca", "Tren motriz"),
    (("frenado brusco", "frenada", "harsh brak"), "Frenado brusco", "Tren motriz"),
    (("rolling stop", "parada rodante"), "Parada rodante", "Conducta / seguridad"),
    (("telefono",), "Uso del teléfono", "Conducta / seguridad"),
    (("cubierta",), "Cámara tapada", "Conducta / seguridad"),
    (("boton",), "Botón de la cámara", "Conducta / seguridad"),
    (("cinturon",), "Cinturón", "Conducta / seguridad"),
    (("velocidad",), "Exceso de velocidad", "Conducta / seguridad"),
    (("pegada", "seguimiento"), "Conducción pegada", "Conducta / seguridad"),
    (("no se esta usan",), "Cámara: uso incorrecto (texto cortado)", "Conducta / seguridad"),
]


def categoria_evento(titulo: str):
    t = _norm(titulo)
    for claves, cat, rel in _CATEGORIAS:
        if any(c in t for c in claves):
            return cat, rel
    return "Otro", "Conducta / seguridad"


def _fecha(m) -> datetime:
    mes = _MESES.get(_norm(m.group(1)))
    if not mes:
        return None
    hora = int(m.group(3)) % 12 + (12 if m.group(5).upper() == "PM" else 0)
    try:  # el reporte no trae el año: se asume el actual
        return datetime(datetime.now().year, mes, int(m.group(2)), hora, int(m.group(4)))
    except ValueError:
        return None


def parsear_eventos(items: list, ancho: int, pagina: int = 1) -> list:
    """Reconstruye tarjetas de eventos de video (rejilla de 2 columnas de Geotab)."""
    eventos = []
    for col in (0, 1):
        columna = sorted((i for i in items if (i["x0"] < ancho * 0.5) == (col == 0)),
                         key=lambda i: i["y0"])
        buf = []
        for it in columna:
            m = _RE_FECHA.match(it["text"])
            if not m:
                buf.append(it)
                continue
            util = [b for b in buf if not _RE_DURACION.match(b["text"])]
            # el conductor de la tarjeta es el más cercano a la fecha (no el filtro del encabezado)
            ic = next((k for k in range(len(util) - 1, -1, -1) if "conductor" in _norm(util[k]["text"])), None)
            if ic is not None and ic > 0:
                titulo = util[ic - 1]["text"]
                vehiculo = next((b["text"] for b in util[ic + 1:]
                                 if re.fullmatch(r"\d{2,4}", b["text"]) and b["score"] >= 0.8), "")
                numeros = re.findall(r"\(\s*(-?\d+(?:\.\d+)?)\s*\)", titulo)
                modelo = next((g for g in re.findall(r"\(([^)]*)\)", titulo)
                               if not re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", g)), "")
                limpio = re.sub(r"\([^)]*\)", "", titulo).strip(" -")
                cat, rel = categoria_evento(limpio)
                fecha = _fecha(m)
                eventos.append({
                    "Página": pagina, "Evento": limpio, "Categoría": cat,
                    "Valor": float(numeros[-1]) if numeros else None, "Modelo": modelo,
                    "Vehículo": vehiculo, "Conductor": util[ic]["text"],
                    "Fecha y hora": fecha.strftime("%d/%m %H:%M") if fecha else it["text"],
                    "Relevancia": rel, "_dt": fecha or datetime.min})
            buf = []
    eventos.sort(key=lambda e: (e["Página"], e["_dt"]), reverse=False)
    for e in eventos:
        e.pop("_dt")
    return eventos


def agrupar_filas(items: list) -> list:
    """Une los textos que están a la misma altura en una sola línea."""
    if not items:
        return []
    alto = np.median([i["y1"] - i["y0"] for i in items]) or 10
    filas, actual, y_ref = [], [], None
    for it in sorted(items, key=lambda i: (i["y0"] + i["y1"]) / 2):
        yc = (it["y0"] + it["y1"]) / 2
        if y_ref is None or abs(yc - y_ref) <= 0.6 * alto:
            actual.append(it)
            y_ref = yc if y_ref is None else y_ref
        else:
            filas.append(actual)
            actual, y_ref = [it], yc
    filas.append(actual)
    return [" ".join(i["text"] for i in sorted(f, key=lambda i: i["x0"])) for f in filas]


def extraer_variables(filas: list) -> dict:
    """Busca 'nombre de variable ... número' en cada línea leída."""
    valores = {}
    for fila in filas:
        n = _norm(fila)
        numeros = re.findall(r"-?\d+(?:[.,]\d+)?", fila)
        if not numeros:
            continue
        for canon in ORDEN_MAPEO:
            if canon == "Timestamp" or canon in valores:
                continue
            if any(all(tok in n for tok in toks) for toks in _ALIAS[canon]):
                valores[canon] = float(numeros[-1].replace(",", "."))
                break
    return valores


def analizar_paginas(paginas_png: list) -> dict:
    """OCR de cada página/imagen (bytes png o jpg) y su interpretación."""
    lineas, eventos, filas_var, total = [], [], [], None
    for n, data in enumerate(paginas_png, start=1):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        items = [i for i in leer_imagen(img) if i["text"]]
        filas = agrupar_filas(items)
        lineas += [f"[p.{n}] {f}" for f in filas]
        eventos += parsear_eventos(items, img.shape[1], n)
        for f in filas:
            m = re.search(r"(\d[\d.,]*)\s+eventos\s+en\s+v[ií]deo", f, re.I)
            if m and total is None:
                total = int(re.sub(r"\D", "", m.group(1)))
        v = extraer_variables(filas)
        if v:
            filas_var.append(v)
    return {
        "lineas": lineas,
        "eventos": pd.DataFrame(eventos),
        "total_declarado": total,
        "variables": pd.DataFrame(filas_var) if filas_var else None,
    }
