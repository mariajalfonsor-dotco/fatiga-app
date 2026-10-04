# Analizador de fatiga metálica (proyecto de grado)

Aplicación de escritorio (Streamlit) que analiza una imagen de una superficie
de fractura y una señal de vibración, y propone una causa raíz probable de
la falla combinando ambos resultados con una base de conocimiento en SQLite.

## Instalación

```bash
python -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

Esto abre la aplicación en el navegador (http://localhost:8501). La primera
vez que se ejecuta, se crean automáticamente:
- la base de datos SQLite (`data/fatiga.db`) con las tablas y el
  conocimiento de referencia (tipos de fractura, tipos de falla por
  vibración, reglas de diagnóstico),
- los modelos de clasificación (`models/*.joblib`), entrenados con datos
  sintéticos (ver más abajo).

## Estructura del proyecto

| Archivo                 | Responsabilidad                                                   |
|--------------------------|--------------------------------------------------------------------|
| `app.py`                 | Interfaz Streamlit (las 5 pestañas de la app)                     |
| `database.py`            | Esquema SQLite, siembra de conocimiento, historial                |
| `image_analysis.py`      | Preprocesamiento OpenCV + textura GLCM + clasificador de fractura |
| `vibration_analysis.py`  | FFT + características espectrales + clasificador de falla        |
| `diagnosis_engine.py`    | Combina ambas predicciones y consulta las reglas de diagnóstico  |

## Enfoque de diseño: IA híbrida (percepción + reglas)

Se eligió deliberadamente un enfoque híbrido en vez de un modelo extremo a
extremo de caja negra:

1. Dos modelos de Machine Learning (Random Forest) hacen la parte de
   **percepción**: clasifican el tipo de fractura (6 clases: fatiga,
   clivaje, hoyuelos, intergranular, sobretorsión, porosidad) a partir de
   textura de la imagen, y el tipo de falla (4 clases: normal, pista
   interna, pista externa, elemento rodante) a partir de características
   espectrales de la vibración.
2. Un **motor de reglas**, almacenado en la base de datos, traduce la
   combinación de ambas clasificaciones en una causa raíz explicada paso a
   paso.
3. Cuando la fractura es de tipo "fatiga", un **tercer nivel de
   contexto operativo** (temperatura, ambiente, tipo de carga —
   preguntado directamente al usuario en la interfaz) refina cuál de los
   5 subtipos de fatiga aplica: alto ciclo, bajo ciclo, térmica, por
   corrosión o por fricción (fretting). Esto es deliberado: esos subtipos
   casi no se distinguen por textura o vibración solas, así que pedírselo
   al modelo sería forzar una respuesta poco confiable — es más honesto
   resolverlo con la información de contexto que ya conoce quien reporta
   la falla.

Esto permite justificar cada conclusión ante un jurado ("el sistema
concluyó X porque detectó Y en la imagen, Z en la vibración, y W en el
contexto operativo") y requiere muchos menos datos etiquetados que
entrenar un único modelo extremo a extremo.

## Sobre los datos usados (léelo para tu documento de tesis)

Por defecto, **los dos modelos se entrenan con datos sintéticos**:

- `image_analysis.py` genera vectores de características de textura (GLCM)
  con distribuciones calibradas para imitar las seis categorías de fractura
  del sistema: fatiga, clivaje (frágil), hoyuelos (dúctil), intergranular,
  sobretorsión y porosidad.
- `vibration_analysis.py` genera señales simuladas con las frecuencias
  características típicas de fallas de rodamiento (pista interna, pista
  externa, elemento rodante) como múltiplos de la frecuencia de giro del
  eje, más ruido de fondo.

Esto permite tener una aplicación funcional de punta a punta sin depender
de acceso a un laboratorio o a un dataset real durante el desarrollo — es
apropiado para el alcance de una simulación de proyecto de grado, siempre
que lo declares así explícitamente en el documento.

**Para subir el nivel de rigor experimental**, puedes reemplazar los datos
sintéticos por datos reales sin cambiar la arquitectura:

- Imágenes: existen datasets académicos publicados de fractografía SEM ya
  clasificados en las mismas 4 categorías (cleavage, dimple, fatigue,
  intergranular) citados en varios papers recientes de clasificación de
  fracturas con CNN. Extrae las mismas características GLCM sobre esas
  imágenes reales y vuelve a entrenar con `train_image_model()`.
- Vibración: el dataset público **CWRU** (Case Western Reserve University
  Bearing Data Center) es el estándar académico para diagnóstico de fallas
  de rodamiento por vibración, con señales reales de falla en pista
  interna, pista externa y elemento rodante. Extrae las mismas
  características espectrales sobre esas señales reales y vuelve a
  entrenar con `train_vibration_model()`.

En ambos casos, la interfaz, la base de datos y el motor de diagnóstico no
necesitan cambios — solo el origen de los datos de entrenamiento.

## Limitaciones conocidas (para la sección de alcance/limitaciones de la tesis)

- Es una simulación de software: no ha sido validada con casos reales de
  campo ni con un laboratorio de ensayos de fatiga.
- Las reglas de diagnóstico cubren un conjunto acotado de combinaciones;
  combinaciones no cubiertas producen una respuesta general con menor
  confianza en vez de una causa específica inventada.
- Los multiplicadores de frecuencia de falla (BPFI/BPFO/BSF) usados para
  generar las señales sintéticas son valores de referencia típicos, no los
  de un rodamiento específico; en un caso real deben calcularse a partir
  de la geometría exacta del rodamiento usado.
