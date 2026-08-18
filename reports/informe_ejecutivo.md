# Informe ejecutivo — Modelo de probabilidad de default

**Para:** Gerencia de Riesgo · **Autor:** Luis Altamirano · **Fecha:** Agosto 2026

## Resumen

Construimos un modelo que estima, para cada solicitud, la probabilidad de que el cliente
caiga en mora (90+ días dentro de 12 meses). Aplicado como **política de aprobación**, el
modelo **casi cuadruplica la ganancia del portafolio** (de 198 a 734 millones de CLP en la
prueba histórica) y **reduce a la mitad la mora entre los créditos aprobados** (de 12,5% a
6,1%), aprobando el 76% de las solicitudes. A continuación, el detalle.

## 1. Problemas encontrados en la data

La data venía del warehouse "tal como está" y presentaba varios problemas que corregimos:

- **Ingresos mal registrados (~12% de los casos):** aparecían valores imposibles (por
  ejemplo, 280 pesos de ingreso mensual). Verificamos que era un error de unidad —estaban
  en miles, no en pesos— y lo corregimos con evidencia, no por suposición.
- **Datos faltantes (16–18% en ingreso y antigüedad laboral):** los completamos y, además,
  marcamos *que faltaban*, porque el hecho de faltar suele ser informativo (por ejemplo, un
  trabajador informal).
- **Valores imposibles** (edades sobre 100 años, antigüedades de 78 años): corregidos.
- **Una variable "demasiado buena":** una variable (número de contactos del cliente)
  predecía casi perfecto. Al investigarla, concluimos que probablemente reflejaba
  información *posterior* a la decisión de crédito (contactos de gestión de un cliente que
  ya venía cayendo). Incluirla habría inflado los resultados en el papel y **fallado en
  producción**, donde esa información no existe al momento de decidir. **La excluimos.**
- **La mora viene subiendo:** de 7% a 13% a lo largo del período de datos. Esto confirma el
  problema que motivó el proyecto y condiciona todas las decisiones siguientes.

## 2. Enfoque

- **Modelo:** una regresión logística. La elegimos por sobre alternativas más complejas
  (que probamos y **igualaron su desempeño, sin superarlo**) porque es **interpretable**:
  permite explicar por qué se rechaza a un cliente, algo clave para el regulador y para el
  propio negocio.
- **Validación honesta:** medimos el modelo sobre un período **futuro que no vio durante el
  entrenamiento**, imitando lo que ocurrirá en producción. No lo evaluamos sobre los mismos
  datos con que aprendió (eso siempre da un resultado optimista y engañoso).
- **Probabilidades confiables:** ajustamos el modelo para que cuando diga "15% de riesgo",
  de verdad caiga cerca del 15% de esos casos. Esto es esencial porque la decisión de
  aprobación usa la probabilidad como insumo económico.

## 3. Performance esperada y su justificación

El modelo **distingue bien a quién paga de quién cae:** puesto un cliente que cae en mora y
uno que paga, le asigna mayor riesgo al que cae en el ~83% de los casos — un nivel **bueno y
realista** para crédito de consumo.

Es importante subrayar que este es un número **honesto**. Al excluir la variable que
filtraba información, renunciamos a un resultado aparente mucho más alto (pero falso) a
cambio de un modelo que **sí funcionará en producción**. Como el desempeño se midió sobre
datos futuros no vistos, esperamos que se sostenga en la práctica.

## 4. Política de aprobación recomendada y ganancia estimada

**Regla:** aprobar una solicitud cuando su **valor esperado sea positivo** — es decir,
cuando la ganancia esperada (si paga) supere la pérdida esperada (si cae), usando la
probabilidad de default y la economía del producto (interés, plazo y pérdida por default).

Esto define un **umbral de riesgo que depende del plazo** (un crédito más largo genera más
interés, por lo que tolera más riesgo):

| Plazo | Aprobar si el riesgo de default es menor a |
|---|---|
| 6 meses | 5,2% |
| 12 meses | 9,8% |
| 24 meses | 17,9% |
| 48 meses | 30,4% |

**Impacto estimado** (prueba sobre datos históricos con resultados reales, 8.300 solicitudes):

| | Aprobar todo | Política del modelo |
|---|---|---|
| Solicitudes aprobadas | 100% | 76% |
| Mora entre aprobados | 12,5% | **6,1%** |
| Ganancia total | 198 MM CLP | **734 MM CLP** |
| Ganancia por solicitud | 23.900 CLP | **88.400 CLP** |

La política **casi cuadruplica la ganancia** (+536 MM CLP) rechazando el ~24% de solicitudes
más riesgosas, sin dejar de aprobar a la gran mayoría de los buenos clientes.

## 5. Limitaciones y próximos pasos

- **La mora sube en el tiempo.** El modelo puede subestimar el riesgo del período más
  reciente. Lo mitigamos calibrando con los datos más nuevos disponibles, pero recomendamos
  **recalibración periódica** y un **margen de conservadurismo** al fijar umbrales.
- **La ganancia es una estimación histórica.** El resultado real dependerá de que las
  condiciones del mercado se mantengan.
- **Confirmar la variable excluida.** Si el negocio verifica que "número de contactos" está
  disponible *antes* de decidir el crédito, podría reincorporarse.
- **Monitoreo en producción.** Recomendamos un tablero que vigile cambios en el perfil de
  los solicitantes y en el desempeño del modelo en el tiempo (data drift), para saber cuándo
  reentrenar.
- **Mejoras futuras:** más variables de negocio y un ajuste fino del modelo.

---

*Detalle técnico (para referencia): discriminación AUC ≈ 0,83 / KS ≈ 0,51 sobre validación
out-of-time; modelo logístico calibrado (isotónica); variable excluida por leakage;
validación temporal de tres vías (train/validación/holdout).*