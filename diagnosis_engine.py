"""
diagnosis_engine.py
--------------------
Motor de diagnóstico híbrido (enfoque B del diseño): la IA se encarga de
la percepción (clasificar imagen y señal por separado); las REGLAS,
almacenadas en la base de datos, se encargan de traducir esa combinación
en una causa raíz explicable.

Además del cruce imagen + vibración, cuando la fractura es de tipo
"fatiga" se incorpora un CONTEXTO OPERATIVO (temperatura, ambiente, tipo
de carga) para distinguir el subtipo de fatiga (alto ciclo, bajo ciclo,
térmica, por corrosión, por fricción/fretting). Esto es deliberado: esos
subtipos casi no se distinguen por textura o vibración por sí solas, así
que pedírselo al modelo sería forzar una respuesta poco confiable. Es más
honesto (y más acertado) resolverlo con la información de contexto que ya
conoce quien reporta la falla.
"""

from dataclasses import dataclass, field

import database

# Orden de prioridad para resolver el subtipo de fatiga cuando hay más de
# un factor de contexto presente a la vez (el más "dominante" primero,
# según cuánto acelera/explica la falla en la literatura de fatiga).
_SUBTYPE_PRIORITY = [
    ("ambiente", "corrosivo", "corrosion"),
    ("carga", "fretting", "fretting"),
    ("temperatura", "alta", "termica"),
    ("carga", "bajo_ciclo", "bajo_ciclo"),
]
_DEFAULT_SUBTYPE = "alto_ciclo"  # el más común si no hay ningún factor especial


def resolve_fatigue_subtype(temperatura: str = "normal", ambiente: str = "normal",
                             carga: str = "alto_ciclo") -> str:
    """Decide el subtipo de fatiga a partir del contexto operativo.

    temperatura: 'normal' | 'alta' | 'baja'
    ambiente:    'normal' | 'corrosivo' | 'abrasivo'
    carga:       'alto_ciclo' | 'bajo_ciclo' | 'fretting'
    """
    context = {"temperatura": temperatura, "ambiente": ambiente, "carga": carga}
    for field_name, value, subtype in _SUBTYPE_PRIORITY:
        if context[field_name] == value:
            return subtype
    return _DEFAULT_SUBTYPE


# ---------------------------------------------------------------------------
# Cruce IMAGEN + TELEMETRÍA
# ---------------------------------------------------------------------------
# Cada regla une un tipo de fractura con un hábito de operación detectado en la
# telemetría y propone una causa raíz. Son HIPÓTESIS de ingeniería basadas en
# mecanismos de falla conocidos (fatiga, desgaste tribológico, sobrecarga); no
# están validadas con casos reales de campo.
HABITOS = {
    "lugging": "Lugging (bajas RPM con carga alta)",
    "embrague": "Abuso de embrague a altas RPM",
    "temp_aceite": "Sobrecalentamiento de aceite",
    "anomalias": "Combinaciones atípicas (Isolation Forest)",
}
_ORDEN_PRIORIDAD = {"Alta": 2, "Media": 1}

_FATIGA_LUGGING = (
    "Fatiga por sobrecarga torsional cíclica asociada a lugging",
    "Las estrías de fatiga indican crecimiento progresivo de grieta bajo carga cíclica. La telemetría "
    "muestra operación frecuente a bajas RPM con carga y pedal altos (lugging), que somete engranajes y "
    "ejes a torques elevados y puede generar vibración torsional, un mecanismo que acelera la iniciación "
    "de grietas.",
    "Inspeccionar picaduras y grietas en dientes y ejes; capacitar a los conductores para reducir de "
    "marcha antes y evitar el lugging.")
_FATIGA_EMBRAGUE = (
    "Fatiga por cargas transitorias asociadas al abuso de embrague",
    "Las estrías de fatiga indican carga cíclica repetida. La telemetría muestra el pedal de embrague "
    "pisado con RPM altas: genera fricción, calor y cargas de choque en arranques y cambios que se "
    "repiten y contribuyen a la fatiga de los componentes de la transmisión.",
    "Medir el desgaste del disco y del plato de presión, revisar ejes y estrías, y capacitar en el uso "
    "del embrague.")
_FATIGA_ACEITE = (
    "Fatiga superficial acelerada por lubricación deficiente (aceite sobrecalentado)",
    "Las estrías de fatiga indican crecimiento progresivo de grieta. La telemetría muestra temperatura de "
    "aceite por encima del umbral crítico: el lubricante pierde viscosidad y película protectora, lo que "
    "favorece el desgaste y la fatiga superficial (picaduras) en flancos de engranajes y rodamientos.",
    "Revisar nivel, viscosidad y estado del aceite, cambiar aceite y filtro, inspeccionar el enfriador y "
    "buscar partículas metálicas en el lubricante.")
_FATIGA_ANOMALIA = (
    "Fatiga en un régimen de operación atípico",
    "Las estrías de fatiga indican carga cíclica. El modelo de IA marcó combinaciones atípicas de "
    "parámetros de operación, que pueden corresponder a ciclos de carga más severos que los normales.",
    "Revisar las muestras marcadas en la pestaña 2 y cruzarlas con el historial de mantenimiento.")


def _sobrecarga(modo: str, causa: str):
    return (f"{modo} asociada a {causa}",
            f"El tipo de fractura es compatible con un exceso de esfuerzo sobre el material. La telemetría "
            f"muestra {causa}, una condición que genera torques o cargas por encima de lo normal en la transmisión.",
            "Revisar el estado de engranajes, ejes y embrague; verificar los límites de torque del fabricante "
            "y corregir el hábito de operación identificado.")


# (tipo de fractura, hábito) -> (causa raíz, explicación, recomendación)
REGLAS_IMAGEN_TELEMETRIA = {
    ("fatiga", "lugging"): _FATIGA_LUGGING,
    ("fatiga", "embrague"): _FATIGA_EMBRAGUE,
    ("fatiga", "temp_aceite"): _FATIGA_ACEITE,
    ("fatiga", "anomalias"): _FATIGA_ANOMALIA,
    ("clivaje", "lugging"): _sobrecarga("Fractura frágil por sobrecarga a baja velocidad", "lugging (bajas RPM con carga alta)"),
    ("clivaje", "embrague"): _sobrecarga("Fractura frágil por cargas de impacto", "abuso de embrague a altas RPM"),
    ("clivaje", "temp_aceite"): (
        "Fractura frágil asociada a choques térmicos y sobrecalentamiento",
        "La superficie facetada es típica de fractura frágil. La telemetría muestra temperaturas de aceite "
        "críticas; los choques térmicos y los gradientes de temperatura reducen la tenacidad efectiva y "
        "favorecen la fractura sin deformación plástica.",
        "Revisar el sistema de enfriamiento y los ciclos térmicos de la transmisión, y verificar la tenacidad "
        "del material con el fabricante."),
    ("hoyuelos", "lugging"): _sobrecarga("Sobrecarga dúctil por torque excesivo", "lugging (bajas RPM con carga alta)"),
    ("hoyuelos", "embrague"): _sobrecarga("Sobrecarga dúctil por arranques bruscos", "abuso de embrague a altas RPM"),
    ("hoyuelos", "temp_aceite"): (
        "Sobrecarga dúctil con material debilitado por temperatura",
        "Los hoyuelos indican deformación plástica antes de la rotura. La telemetría muestra temperatura de "
        "aceite crítica, que reduce la resistencia del material y de la lubricación, de modo que cargas "
        "normales pueden superar el límite de fluencia.",
        "Revisar el enfriamiento y la lubricación, y confirmar los límites de torque a la temperatura de operación."),
    ("sobretorsion", "lugging"): _sobrecarga("Sobretorsión por par excesivo", "lugging (bajas RPM con carga alta)"),
    ("sobretorsion", "embrague"): _sobrecarga("Sobretorsión por par transitorio", "abuso de embrague a altas RPM"),
    ("intergranular", "temp_aceite"): (
        "Fragilización por sobrecalentamiento",
        "La fractura sigue los límites de grano. La telemetría muestra temperaturas críticas sostenidas, y el "
        "sobrecalentamiento es una causa típica de fragilización en límites de grano.",
        "Confirmar el tratamiento térmico del material con el fabricante, revisar el enfriamiento y hacer "
        "análisis metalográfico."),
}
# La porosidad es un defecto del material: la operación pudo agravarlo, no originarlo
for _h, _txt in (("lugging", "lugging"), ("embrague", "abuso de embrague"),
                 ("temp_aceite", "sobrecalentamiento de aceite"), ("anomalias", "condiciones atípicas de operación")):
    REGLAS_IMAGEN_TELEMETRIA[("porosidad", _h)] = (
        "Defecto interno del material (porosidad) agravado por la operación",
        f"La porosidad es un defecto de fabricación o fundición. La telemetría muestra {_txt}, que pudo "
        f"acelerar la falla pero no es su origen.",
        "Solicitar análisis metalográfico y trazabilidad del lote de la pieza; corregir también el hábito de operación.")


def _habitos_activos(telemetria: dict) -> list:
    """Hábitos con prioridad alta o media, de mayor a menor prioridad."""
    orden = list(HABITOS)
    activos = [h for h, pr in (telemetria or {}).items() if pr in _ORDEN_PRIORIDAD and h in HABITOS]
    return sorted(activos, key=lambda h: (-_ORDEN_PRIORIDAD[telemetria[h]], orden.index(h)))


@dataclass
class DiagnosisResult:
    image_class: str
    image_confidence: float
    vibration_class: str
    vibration_confidence: float
    root_cause: str
    explanation: str
    recommendation: str
    combined_confidence: float
    rule_found: bool
    fatigue_subtype: dict = field(default=None)
    telemetry_used: list = field(default_factory=list)   # hábitos que sustentan la causa
    telemetry_provided: bool = False


def diagnose(image_class: str, image_confidence: float,
             vibration_class: str = None, vibration_confidence: float = None,
             temperatura: str = "normal", ambiente: str = "normal",
             carga: str = "alto_ciclo", telemetria: dict = None) -> DiagnosisResult:
    """Combina imagen, telemetría (opcional) y vibración (opcional) en una causa raíz.

    telemetria: {'lugging': 'Alta', 'embrague': 'Media', 'temp_aceite': 'Sin hallazgos', ...}
    """
    telemetry_provided = bool(telemetria)
    activos = _habitos_activos(telemetria)
    fracture_info = database.get_fracture_type(image_class)
    fname = fracture_info["name"] if fracture_info else image_class

    # Regla de telemetría: el hábito activo más prioritario que tenga regla con esta fractura
    tel_rule, tel_habito = None, None
    for h in activos:
        if (image_class, h) in REGLAS_IMAGEN_TELEMETRIA:
            tel_rule, tel_habito = REGLAS_IMAGEN_TELEMETRIA[(image_class, h)], h
            break
    otros = [HABITOS[h] for h in activos if h != tel_habito]

    rule = database.get_rule(image_class, vibration_class) if vibration_class else None

    if rule:  # imagen + vibración (y telemetría como refuerzo)
        root_cause, explanation, recommendation = rule["root_cause"], rule["explanation"], rule["recommendation"]
        rule_found = True
        if tel_rule:
            explanation += f" Además, la telemetría respalda este cuadro: {tel_rule[1]}"
            recommendation += f" Por la telemetría: {tel_rule[2]}"
    elif tel_rule:  # imagen + telemetría
        root_cause, explanation, recommendation = tel_rule
        rule_found = True
        if vibration_class:
            explanation += (f" La señal de vibración clasificada como '{vibration_class}' no tiene una regla "
                            f"específica con esta fractura.")
    elif vibration_class:  # vibración sin regla
        vibration_info = database.get_vibration_fault(vibration_class)
        root_cause = (f"Causa no catalogada específicamente: posible combinación de '{fname}' con "
                      f"'{vibration_info['name'] if vibration_info else vibration_class}'")
        explanation = ("No existe una regla específica en la base de conocimiento para esta combinación. "
                       f"{fracture_info['description'] if fracture_info else ''} "
                       f"{vibration_info['description'] if vibration_info else ''}").strip()
        recommendation = "Se recomienda revisión manual por un especialista."
        rule_found = False
    else:  # solo imagen (con o sin telemetría sin reglas aplicables)
        root_cause = f"Causa no determinada: fractura de tipo '{fname}'"
        explanation = (fracture_info["description"] + " " + fracture_info["typical_causes"]) if fracture_info else ""
        if telemetry_provided and not activos:
            explanation += (" La telemetría analizada no muestra hábitos de prioridad alta o media que "
                            "expliquen este tipo de fractura.")
            recommendation = "Confirmar con inspección de un especialista; considerar medir vibraciones."
        elif telemetry_provided:
            explanation += (" La telemetría muestra hábitos de riesgo, pero no hay una regla específica "
                            "entre ellos y este tipo de fractura.")
            recommendation = "Confirmar con inspección de un especialista; considerar medir vibraciones."
        else:
            recommendation = ("Complementar con la telemetría del vehículo (pestañas 1 y 2) o una señal de "
                              "vibración, y confirmar con inspección de un especialista.")
        rule_found = False

    if tel_rule and otros:
        explanation += " También se detectó en la telemetría: " + "; ".join(otros) + "."

    fatigue_subtype_info = None
    if image_class == "fatiga":
        subtype_code = resolve_fatigue_subtype(temperatura, ambiente, carga)
        fatigue_subtype_info = database.get_fatigue_subtype(subtype_code)
        if fatigue_subtype_info:
            root_cause = f"{root_cause} — subtipo: {fatigue_subtype_info['name']}"
            explanation = (f"{explanation} Según el contexto operativo indicado, esto corresponde "
                           f"específicamente a {fatigue_subtype_info['name'].lower()}: "
                           f"{fatigue_subtype_info['description']}")
            recommendation = (f"{recommendation} Adicionalmente, dado el subtipo identificado: "
                              f"{fatigue_subtype_info['typical_causes']}")

    if vibration_confidence is None:
        combined_confidence = round(image_confidence, 3)
    else:
        combined_confidence = round((image_confidence + vibration_confidence) / 2, 3)

    return DiagnosisResult(
        image_class=image_class,
        image_confidence=round(image_confidence, 3),
        vibration_class=vibration_class,
        vibration_confidence=None if vibration_confidence is None else round(vibration_confidence, 3),
        root_cause=root_cause,
        explanation=explanation,
        recommendation=recommendation,
        combined_confidence=combined_confidence,
        rule_found=rule_found,
        fatigue_subtype=fatigue_subtype_info,
        telemetry_used=[HABITOS[tel_habito]] + otros if tel_rule else [],
        telemetry_provided=telemetry_provided,
    )


if __name__ == "__main__":
    database.init_db()
    result = diagnose("fatiga", 0.93, "pista_interna", 0.88,
                       temperatura="normal", ambiente="corrosivo", carga="alto_ciclo")
    print("Causa raíz:", result.root_cause)
    print("Explicación:", result.explanation)
    print("Recomendación:", result.recommendation)
    print("Confianza combinada:", result.combined_confidence)
