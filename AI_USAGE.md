# AI_USAGE — Uso de Inteligencia Artificial en el desafío

> Documento vivo. Se actualiza por etapa. Redactado con apoyo de Claude (Anthropic)
> y revisado/editado por mí; los ajustes finales y el criterio son propios.

## Enfoque

Usé IA como **mentor técnico y par de programación**, no como generador automático de
la solución. El principio que guió el uso: la IA acelera y explica, pero **cada decisión
la reviso, cada salida la ejecuto y verifico, y documento explícitamente dónde la IA se
equivocó o propuso algo subóptimo**. No incorporo nada que no pueda explicar y defender.

## Herramienta

- **Claude (Anthropic)**, vía interfaz de chat, en un proyecto con contexto de mi perfil.
- **Usos:** exploración guiada de la data, generación de código base (scripts y notebook),
  explicación de conceptos (validación temporal, madurez del target, calibración,
  economía de la decisión de aprobación) y revisión crítica de mis propias decisiones.

## Registro por etapa

| Etapa | Aporte de la IA | Mi criterio / decisión |
|---|---|---|
| Estrategia | Lectura de la rúbrica y plan de trabajo | Prioricé criterio, validación honesta y negocio por sobre performance |
| EDA | Exploró la data y propuso chequeos | Dirigí las hipótesis (error de unidad, madurez del target) y las validé con datos |
| Limpieza (`data_prep.py`) | Generó la función `clean()` determinística | Definí las reglas y umbrales; exigí simetría train/test para evitar leakage |
| Estructura del repo | Propuso una estructura inicial | Redirigí a arquitectura medallion (raw/silver/gold) por cercanía a producción |

## Casos donde la IA se equivocó o fue subóptima (y cómo lo resolví)

1. **Bug en código generado — import mal ubicado.** La anotación de tipo `Path` en una
   función fallaba porque el `import` estaba solo dentro de `__main__`. Lo detecté al
   **ejecutar** el módulo (`NameError`), no al leerlo. Corrección: subí el import al tope
   del módulo.

2. **Estado inconsistente al reeditar.** Al modificar `data_prep.py` por partes, el bloque
   `__main__` quedó en una versión previa. Lo detecté al ejecutar y revisar el archivo, y
   lo reescribí. Lección: verificar el archivo final, no confiar en que cada edición quedó
   como se esperaba.

3. **Formato del EDA.** La IA propuso el EDA como script `.py`. Decidí que debía ser un
   **notebook**, porque el análisis exploratorio se beneficia de la narrativa intercalada
   con tablas y gráficos, y el desafío lo acepta. La IA lo convirtió.

4. **Estructura de datos.** La IA propuso una limpieza en memoria sin materializar salidas.
   Pedí una **estructura medallion** (raw → silver → gold) con generación de archivos por
   etapa, más cercana a un entorno productivo. Se implementó cuidando de **no** meter
   transformaciones con estado (imputación, codificación, escalado) en las capas de datos,
   para evitar leakage: esas viven en el pipeline del modelo, ajustadas solo con train.

5. **Rutas relativas al directorio de trabajo.** Los scripts asumían ejecución desde la
   raíz del repo; al correr desde `src/` fallaban (`FileNotFoundError`). Corrección: anclar
   las rutas al archivo (`Path(__file__)`) para que corran desde cualquier directorio.

6. **Detección de tipos frágil.** El código comparaba `dtype == "object"` para hallar las
   categóricas; al leer el parquet las strings no volvían como `object` y quedaban 0
   categóricas (el imputador de mediana explotó). Corrección: usar `is_numeric_dtype`.

## Casos donde la IA cuestionó mi criterio (diálogo, no aceptación pasiva)

- **Justificación del ingreso corregido.** Mi primera justificación para corregir los
  ingresos anómalos fue débil ("más datos, más información"). La IA la cuestionó y me llevó
  a **validar la hipótesis** de error de unidad con evidencia: magnitud del factor,
  reconciliación de la distribución completa al multiplicar ×1000, y tasa de default normal
  del grupo. Mantuve la decisión, pero ahora con una justificación defendible.

## Detección de leakage (criterio propio sobre performance sospechosa)

- Un baseline logístico dio AUC ~0.98, irrealmente alto para riesgo crediticio. En vez de
  reportarlo, investigué: la variable `num_contactos_ult_trimestre` (correlación 0.735 con
  el target) explicaba sola ~0.14 de AUC (0.98 → 0.83 al quitarla). El diccionario no
  aclara si esos contactos son previos o posteriores a la solicitud → riesgo de leakage.
  **Decisión:** reportar la performance honesta sin esa variable y documentar el riesgo,
  en vez de exhibir un AUC inflado. La validación out-of-time NO detecta este tipo de leak
  (no es temporal): se detecta con razonamiento de dominio.

## Verificación (cómo controlo la calidad de lo generado)

- Ejecuto todo el código generado y **reproduzco los números** antes de commitear.
- Leo cada módulo; no incorporo código que no pueda explicar en una entrevista.
- Reproducibilidad: la solución se regenera desde `data/raw/` + código.

## Extensiones planificadas (con apoyo de IA, pendientes)

- **Perfilado automático del EDA** con `ydata-profiling` (HTML), como complemento —no
  reemplazo— del análisis manual.
- **Panel HTML comparativo** de la serie de modelos propuestos (métricas out-of-time).
- **Panel de monitoreo MLOps**: drift de features (PSI), drift de score y desempeño en el
  tiempo; motivado por el drift real detectado en los datos (default de 7% a 13%).
