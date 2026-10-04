"""
database.py
------------
Crea y siembra la base de datos SQLite del proyecto.

La base de datos cumple tres roles:
1) Biblioteca de conocimiento: qué significa cada tipo de fractura, cada
   tipo de firma de vibración y cada subtipo de fatiga, y qué causa raíz
   corresponde a cada combinación (motor de reglas del diagnóstico).
2) Reglas de refinamiento por contexto operativo (temperatura, ambiente,
   tipo de carga) para distinguir subtipos de fatiga que no se pueden
   diferenciar solo por textura o vibración.
3) Historial: registro de cada análisis realizado por la aplicación.

Nota para el proyecto de grado: las descripciones y reglas están basadas
en literatura estándar de fractografía y diagnóstico de fallas por
vibración. Puedes editarlas o ampliarlas citando tus propias fuentes
bibliográficas.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "data" / "fatiga.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(reset: bool = False):
    """Crea las tablas y siembra los datos de referencia si no existen."""
    if reset and DB_PATH.exists():
        DB_PATH.unlink()

    conn = get_connection()
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS fracture_types (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            typical_causes TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS vibration_faults (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            typical_causes TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS fatigue_subtypes (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            typical_causes TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS diagnostic_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fracture_code TEXT NOT NULL,
            vibration_code TEXT NOT NULL,
            root_cause TEXT NOT NULL,
            explanation TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            FOREIGN KEY (fracture_code) REFERENCES fracture_types(code),
            FOREIGN KEY (vibration_code) REFERENCES vibration_faults(code),
            UNIQUE(fracture_code, vibration_code)
        );

        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_class TEXT,
            image_confidence REAL,
            vibration_class TEXT,
            vibration_confidence REAL,
            root_cause TEXT,
            explanation TEXT,
            notes TEXT
        );
        """
    )

    # --- Siembra: tipos de fractura (fractografía) ---
    fracture_types = [
        ("fatiga", "Fractura por fatiga",
         "Superficie con marcas de playa y estrías finas y paralelas, "
         "producto de crecimiento progresivo de grieta bajo carga cíclica.",
         "Carga cíclica repetida, concentradores de esfuerzo (entallas, "
         "agujeros, cambios de sección), corrosión bajo fatiga."),
        ("clivaje", "Fractura frágil por clivaje",
         "Superficie facetada y brillante, con planos de fractura "
         "cristalográficos y poca o nula deformación plástica visible.",
         "Baja temperatura, alta velocidad de carga, fragilización por "
         "hidrógeno, material con baja tenacidad."),
        ("hoyuelos", "Fractura dúctil (hoyuelos)",
         "Superficie rugosa con hoyuelos (dimples) que indican "
         "deformación plástica significativa antes de la separación final.",
         "Sobrecarga estática, exceso de esfuerzo por encima del límite "
         "de fluencia del material."),
        ("intergranular", "Fractura intergranular",
         "La grieta avanza a lo largo de los límites de grano, dejando "
         "una superficie de aspecto granular/poligonal.",
         "Fragilización por hidrógeno, corrosión bajo esfuerzo, "
         "precipitación en límites de grano, sobrecalentamiento."),
        ("sobretorsion", "Fractura por sobretorsión",
         "Superficie en espiral o inclinada aproximadamente 45° respecto "
         "al eje, con marcas de cizallamiento helicoidal características "
         "de un momento torsor excesivo.",
         "Par torsional excesivo, bloqueo súbito del eje, arranque contra "
         "carga trabada, mal dimensionamiento a torsión."),
        ("porosidad", "Fractura asociada a porosidad interna",
         "Superficie con cavidades esféricas o irregulares dispersas "
         "(poros) que actúan como concentradores de esfuerzo internos.",
         "Defecto de fundición o soldadura, gases atrapados durante la "
         "solidificación, falta de fusión en el proceso de manufactura."),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO fracture_types (code, name, description, typical_causes) "
        "VALUES (?, ?, ?, ?)",
        fracture_types,
    )

    # --- Siembra: tipos de falla por vibración (diagnóstico de rodamientos) ---
    vibration_faults = [
        ("normal", "Condición normal",
         "Espectro sin picos dominantes por encima del ruido de fondo; "
         "energía concentrada en la frecuencia de giro fundamental.",
         "Operación dentro de parámetros esperados."),
        ("pista_interna", "Falla en pista interna del rodamiento",
         "Pico dominante en la frecuencia característica de pista interna "
         "(BPFI) y sus armónicos, con bandas laterales a la frecuencia de giro.",
         "Desgaste o picadura en la pista interna, montaje con "
         "desalineación, lubricación deficiente."),
        ("pista_externa", "Falla en pista externa del rodamiento",
         "Pico dominante en la frecuencia característica de pista externa "
         "(BPFO), generalmente más estable en amplitud que la de pista interna.",
         "Desgaste o picadura en la pista externa, carga radial excesiva, "
         "contaminación del rodamiento."),
        ("elemento_rodante", "Falla en elemento rodante (bola/rodillo)",
         "Pico dominante en la frecuencia característica de elemento "
         "rodante (BSF) y sus armónicos.",
         "Picadura o astillado en la bola/rodillo, lubricación deficiente, "
         "contaminación por partículas."),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO vibration_faults (code, name, description, typical_causes) "
        "VALUES (?, ?, ?, ?)",
        vibration_faults,
    )

    # --- Siembra: subtipos de fatiga (refinamiento por contexto operativo) ---
    fatigue_subtypes = [
        ("alto_ciclo", "Fatiga de alto ciclo (HCF)",
         "Carga cíclica de baja amplitud (por debajo del límite elástico) "
         "repetida durante un número muy alto de ciclos antes de la falla.",
         "Vibración mecánica normal sostenida en el tiempo, desbalance o "
         "desalineación leve no corregida durante mucho tiempo de operación."),
        ("bajo_ciclo", "Fatiga de bajo ciclo (LCF)",
         "Carga cíclica de alta amplitud, cercana o superior al límite "
         "elástico del material, que produce falla en relativamente pocos "
         "ciclos.",
         "Arranques y paradas frecuentes con sobrecarga, ciclos de carga "
         "y descarga severos, esfuerzos por encima del diseño nominal."),
        ("termica", "Fatiga térmica",
         "Grietas iniciadas por esfuerzos cíclicos generados por "
         "expansión y contracción repetida debida a variaciones de "
         "temperatura, no por carga mecánica externa.",
         "Ciclos térmicos repetidos (calentamiento/enfriamiento), "
         "gradientes de temperatura mal gestionados, choque térmico."),
        ("corrosion", "Fatiga-corrosión",
         "Acción combinada de carga cíclica y un ambiente corrosivo, que "
         "acelera notablemente la iniciación y propagación de la grieta "
         "respecto a la fatiga en ambiente seco.",
         "Exposición a humedad, químicos o ambientes salinos combinada "
         "con carga cíclica; recubrimiento protector dañado o ausente."),
        ("fretting", "Fatiga por fricción (fretting fatigue)",
         "Daño superficial iniciado por micro-movimientos relativos "
         "repetidos entre dos superficies en contacto bajo carga, que "
         "generan puntos de inicio de grieta.",
         "Ajustes por interferencia con micro-holgura, uniones "
         "atornilladas o prensadas con vibración, falta de lubricación "
         "en superficies en contacto."),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO fatigue_subtypes (code, name, description, typical_causes) "
        "VALUES (?, ?, ?, ?)",
        fatigue_subtypes,
    )

    # --- Siembra: reglas de diagnóstico (combinación imagen + vibración -> causa raíz) ---
    rules = [
        ("fatiga", "pista_interna",
         "Fatiga superficial iniciada en la pista interna del rodamiento",
         "Las estrías de fatiga en la superficie de fractura, sumadas a un "
         "pico dominante en la frecuencia de pista interna (BPFI), indican "
         "que la grieta se originó por carga cíclica repetida sobre un "
         "defecto puntual en la pista interna.",
         "Revisar alineación del eje, verificar ajuste del rodamiento en "
         "el eje y plan de lubricación."),
        ("fatiga", "pista_externa",
         "Fatiga por carga cíclica con defecto en pista externa",
         "La combinación de marcas de playa/estrías con un pico dominante "
         "en la frecuencia de pista externa (BPFO) sugiere un defecto "
         "localizado que actúa como concentrador de esfuerzo bajo carga "
         "repetitiva.",
         "Verificar carga radial y alineación del alojamiento del "
         "rodamiento, revisar sellos e ingreso de contaminantes."),
        ("fatiga", "elemento_rodante",
         "Fatiga de contacto por picadura en elemento rodante",
         "Las estrías de fatiga junto con un pico en la frecuencia de "
         "elemento rodante (BSF) apuntan a fatiga de contacto por "
         "rodadura repetida sobre un defecto en la bola o rodillo.",
         "Inspeccionar lubricante en busca de partículas, revisar carga "
         "y velocidad de operación frente a la especificación del rodamiento."),
        ("fatiga", "normal",
         "Fatiga por carga cíclica normal de operación",
         "Las estrías de fatiga están presentes en la imagen, pero el "
         "espectro de vibración no muestra un defecto de rodamiento "
         "claro; la fatiga probablemente se originó en un concentrador "
         "de esfuerzo geométrico (entalla, cambio de sección) más que en "
         "un defecto de rodamiento.",
         "Revisar geometría de la pieza en busca de concentradores de "
         "esfuerzo y confirmar el historial de cargas cíclicas de operación."),
        ("clivaje", "normal",
         "Fractura frágil por sobrecarga súbita o material poco tenaz",
         "El clivaje (fractura frágil, facetada) sin una firma de "
         "vibración anómala sugiere una falla súbita (impacto, baja "
         "temperatura o material inadecuado) más que un deterioro "
         "progresivo detectable por vibración.",
         "Revisar condiciones de operación a baja temperatura, "
         "verificar la especificación y tenacidad del material usado."),
        ("hoyuelos", "normal",
         "Sobrecarga estática por encima del límite del material",
         "La fractura dúctil con hoyuelos, sin firma de vibración "
         "anómala previa, es típica de una sobrecarga puntual más que "
         "de un deterioro progresivo.",
         "Revisar si hubo un evento de carga excesiva puntual y "
         "confirmar que el material cumple la especificación de diseño."),
        ("intergranular", "pista_interna",
         "Fragilización combinada con fatiga de contacto en pista interna",
         "La fractura intergranular sugiere fragilización del material "
         "(por hidrógeno o corrosión), agravada por fatiga de contacto "
         "originada en un defecto de pista interna detectado en el espectro.",
         "Evaluar exposición a hidrógeno o ambientes corrosivos y "
         "revisar el estado de la pista interna del rodamiento."),
        ("intergranular", "normal",
         "Fragilización del material (hidrógeno, corrosión o sobrecalentamiento)",
         "La fractura intergranular sin firma de vibración anómala "
         "apunta a una fragilización del material por causas químicas o "
         "térmicas más que a un deterioro mecánico progresivo.",
         "Evaluar exposición a hidrógeno, ambientes corrosivos o "
         "temperaturas de servicio elevadas; revisar composición y "
         "tratamiento térmico del material."),
        ("sobretorsion", "normal",
         "Falla por sobrecarga torsional súbita",
         "La superficie en espiral típica de sobretorsión, sin una firma "
         "de vibración anómala previa, indica un evento súbito de par "
         "torsional excesivo más que un deterioro progresivo.",
         "Revisar si hubo un bloqueo del eje, arranque contra carga "
         "trabada, o si el par de diseño fue superado en operación."),
        ("sobretorsion", "pista_interna",
         "Sobretorsión posiblemente agravada por desalineación del eje",
         "La fractura por sobretorsión combinada con un defecto de pista "
         "interna sugiere que una desalineación del eje pudo generar "
         "tanto el desgaste del rodamiento como picos de par torsional "
         "anómalos.",
         "Revisar alineación del eje y del acople, y verificar el "
         "historial de picos de par de la máquina."),
        ("porosidad", "normal",
         "Falla iniciada en un defecto de fabricación (porosidad)",
         "Los poros internos actúan como concentradores de esfuerzo desde "
         "la fabricación de la pieza; sin una firma de vibración anómala, "
         "la falla probablemente se originó directamente en ese defecto "
         "bajo la carga normal de operación.",
         "Revisar el proceso de fundición o soldadura de la pieza y "
         "considerar inspección por ultrasonido o radiografía en piezas "
         "similares."),
        ("porosidad", "pista_interna",
         "Fatiga acelerada por poro coincidente con defecto de pista interna",
         "Un poro interno cercano a la zona de carga del rodamiento actúa "
         "como concentrador de esfuerzo adicional, acelerando la fatiga "
         "de contacto detectada en la pista interna.",
         "Revisar calidad de fabricación de la pieza y el estado de la "
         "pista interna del rodamiento; considerar inspección no "
         "destructiva antes del montaje."),
        ("clivaje", "pista_interna",
         "Fractura frágil precipitada por vibración anómala de pista interna",
         "Aunque el clivaje suele asociarse a eventos súbitos, un defecto "
         "de pista interna genera impactos repetidos que pueden precipitar "
         "una fractura frágil en un material de baja tenacidad.",
         "Revisar tenacidad del material especificado y el estado de la "
         "pista interna del rodamiento."),
        ("clivaje", "elemento_rodante",
         "Fractura frágil precipitada por impacto de elemento rodante dañado",
         "Un elemento rodante picado genera impactos de alta frecuencia que, "
         "sobre un material de baja tenacidad, pueden actuar como el evento "
         "súbito que origina el clivaje.",
         "Revisar tenacidad del material y reemplazar el elemento rodante "
         "dañado."),
        ("hoyuelos", "pista_interna",
         "Sobrecarga súbita provocada por agarrotamiento del rodamiento",
         "Un defecto severo en la pista interna puede provocar un "
         "agarrotamiento súbito del eje, generando la sobrecarga estática "
         "que produce la fractura dúctil con hoyuelos.",
         "Inspeccionar el rodamiento por agarrotamiento y revisar el "
         "sistema de lubricación."),
        ("intergranular", "elemento_rodante",
         "Fragilización combinada con fatiga de contacto en elemento rodante",
         "La fractura intergranular sugiere fragilización del material, "
         "agravada por la fatiga de contacto generada por un defecto en "
         "el elemento rodante detectado en el espectro.",
         "Evaluar exposición a hidrógeno o ambientes corrosivos y revisar "
         "el estado del elemento rodante."),
        ("sobretorsion", "elemento_rodante",
         "Sobretorsión posiblemente ligada a agarrotamiento de un elemento rodante",
         "Un elemento rodante dañado puede generar un agarrotamiento "
         "puntual que se traduce en un pico de par torsional suficiente "
         "para producir la fractura por sobretorsión.",
         "Revisar el estado del elemento rodante y el historial de picos "
         "de par de la máquina."),
        ("porosidad", "elemento_rodante",
         "Fatiga acelerada por poro coincidente con defecto en elemento rodante",
         "Un poro interno cercano a la zona de carga actúa como "
         "concentrador de esfuerzo adicional, acelerando la fatiga de "
         "contacto generada por el defecto del elemento rodante.",
         "Revisar calidad de fabricación de la pieza y el estado del "
         "elemento rodante; considerar inspección no destructiva."),
        ("clivaje", "pista_externa",
         "Fractura frágil precipitada por impactos de un defecto en pista externa",
         "Un defecto en la pista externa (BPFO) genera impactos repetidos "
         "y estables que, sobre un material de baja tenacidad, pueden "
         "actuar como el evento que origina el clivaje.",
         "Revisar tenacidad del material, carga radial y estado de la "
         "pista externa y su alojamiento."),
        ("hoyuelos", "pista_externa",
         "Sobrecarga estática con daño previo en pista externa",
         "La fractura dúctil con hoyuelos indica sobrecarga; el defecto en "
         "pista externa sugiere una carga radial elevada o contaminación "
         "que pudo contribuir al evento de sobrecarga.",
         "Revisar la carga radial aplicada, el alojamiento del rodamiento "
         "y confirmar si hubo un evento de sobrecarga puntual."),
        ("hoyuelos", "elemento_rodante",
         "Sobrecarga súbita asociada a daño en elemento rodante",
         "Un elemento rodante dañado puede provocar un agarrotamiento "
         "puntual que genera la sobrecarga estática responsable de la "
         "fractura dúctil con hoyuelos.",
         "Inspeccionar el elemento rodante, la lubricación y el historial "
         "de sobrecargas de la máquina."),
        ("intergranular", "pista_externa",
         "Fragilización combinada con fatiga de contacto en pista externa",
         "La fractura intergranular sugiere fragilización del material, "
         "agravada por la fatiga de contacto de un defecto en la pista "
         "externa detectado en el espectro.",
         "Evaluar exposición a hidrógeno o ambientes corrosivos y revisar "
         "el estado de la pista externa."),
        ("sobretorsion", "pista_externa",
         "Sobretorsión posiblemente ligada a carga radial excesiva o desalineación",
         "Un defecto en pista externa suele asociarse a carga radial "
         "excesiva o desalineación del alojamiento, condiciones que pueden "
         "coincidir con picos de par torsional anómalos.",
         "Revisar alineación del alojamiento y del acople, y el historial "
         "de picos de par de la máquina."),
        ("porosidad", "pista_externa",
         "Fatiga acelerada por poro coincidente con defecto en pista externa",
         "Un poro interno cercano a la zona de carga actúa como "
         "concentrador de esfuerzo adicional, acelerando la fatiga de "
         "contacto de la pista externa.",
         "Revisar la calidad de fabricación de la pieza y el estado de la "
         "pista externa; considerar inspección no destructiva."),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO diagnostic_rules "
        "(fracture_code, vibration_code, root_cause, explanation, recommendation) "
        "VALUES (?, ?, ?, ?, ?)",
        rules,
    )

    conn.commit()
    conn.close()


def get_fracture_type(code: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM fracture_types WHERE code = ?", (code,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_vibration_fault(code: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM vibration_faults WHERE code = ?", (code,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_fatigue_subtype(code: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM fatigue_subtypes WHERE code = ?", (code,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_rule(fracture_code: str, vibration_code: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM diagnostic_rules WHERE fracture_code = ? AND vibration_code = ?",
        (fracture_code, vibration_code),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_analysis(image_class, image_conf, vibration_class, vibration_conf,
                   root_cause, explanation, notes=""):
    conn = get_connection()
    conn.execute(
        "INSERT INTO analysis_history "
        "(timestamp, image_class, image_confidence, vibration_class, "
        "vibration_confidence, root_cause, explanation, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), image_class, image_conf,
         vibration_class, vibration_conf, root_cause, explanation, notes),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db(reset=True)
    print(f"Base de datos creada en: {DB_PATH}")
