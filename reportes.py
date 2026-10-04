"""
reportes.py
-----------
Gestión de casos y archivos técnicos: PDFs de reportes (telemetría,
operación) e imágenes (fotos de daño o fractura).

- Cada caso agrupa los archivos de un vehículo o falla.
- Los archivos se guardan en data/uploads/<id_caso>/ y se registran en la
  misma base SQLite del proyecto (tablas cases y case_files).
- Los PDF se convierten a imágenes (una por página) para poder verlos y
  analizarlos con el clasificador de fractura.
"""

import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pymupdf

import database

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
CATEGORIES = ["Telemetría", "Operación", "Daño / fractura", "Otro"]
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def init_tables():
    conn = database.get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            vehicle TEXT,
            notes TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS case_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            kind TEXT NOT NULL,
            category TEXT,
            path TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases(id)
        );
        """
    )
    conn.commit()
    conn.close()


def create_case(name: str, vehicle: str = "", notes: str = "") -> int:
    conn = database.get_connection()
    cur = conn.execute(
        "INSERT INTO cases (name, vehicle, notes, created_at) VALUES (?, ?, ?, ?)",
        (name.strip(), vehicle.strip(), notes.strip(), datetime.now().strftime("%Y-%m-%d %H:%M")),
    )
    conn.commit()
    case_id = cur.lastrowid
    conn.close()
    return case_id


def list_cases() -> list:
    conn = database.get_connection()
    rows = conn.execute("SELECT * FROM cases ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_files(case_id: int) -> list:
    conn = database.get_connection()
    rows = conn.execute(
        "SELECT * FROM case_files WHERE case_id = ? ORDER BY id DESC", (case_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _safe_name(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", filename)


def save_file(case_id: int, filename: str, data: bytes, category: str) -> str:
    """Guarda el archivo en disco y lo registra en la base de datos."""
    folder = UPLOAD_DIR / str(case_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{datetime.now():%Y%m%d%H%M%S%f}_{_safe_name(filename)}"
    path.write_bytes(data)

    kind = "imagen" if Path(filename).suffix.lower() in IMAGE_EXT else "pdf"
    conn = database.get_connection()
    conn.execute(
        "INSERT INTO case_files (case_id, filename, kind, category, path, uploaded_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (case_id, filename, kind, category, str(path), datetime.now().strftime("%Y-%m-%d %H:%M")),
    )
    conn.commit()
    conn.close()
    return str(path)


# ---------------------------------------------------------------------------
# PDF -> imágenes / texto
# ---------------------------------------------------------------------------

def pdf_pages_as_png(data: bytes, max_pages: int = 10, zoom: float = 1.5) -> list:
    """Devuelve una lista de imágenes PNG (bytes), una por página."""
    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
            pages.append(pix.tobytes("png"))
    return pages


def pdf_text(data: bytes) -> str:
    """Texto seleccionable del PDF (vacío si es solo una captura de pantalla)."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc).strip()


def bytes_to_bgr(data: bytes) -> np.ndarray:
    """Convierte bytes de imagen (png/jpg) al formato BGR de OpenCV."""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
