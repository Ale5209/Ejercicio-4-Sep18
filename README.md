# 📈 Mezcla óptima de productos · Frontera Eficiente de Markowitz

App en **Streamlit** que, a partir de la base de datos de ventas históricas, calcula la mezcla de productos que
alcanza un **margen esperado** con el **menor riesgo posible** y la traduce en **qué producir y cuántas unidades**.

## Estructura

```
├── app.py                 # Interfaz Streamlit
├── markowitz.py           # Núcleo de cálculo (sin Streamlit; reutilizable y probado)
├── requirements.txt
├── data/Base_Datos_Gestion_Productos_Markowitz.xlsx
└── tests/test_markowitz.py
```

## Ejecutar en local

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Pruebas (opcional): `pip install pytest && pytest -q`

## Subir a GitHub y desplegar en Streamlit Community Cloud

```bash
git init
git add .
git commit -m "Optimizador de mezcla de productos (Markowitz)"
git branch -M main
git remote add origin https://github.com/<usuario>/<repositorio>.git
git push -u origin main
```

1. Entra a <https://share.streamlit.io> e inicia sesión con GitHub.
2. **New app** → elige el repositorio y la rama `main`.
3. **Main file path:** `app.py` → **Deploy**.

> Si el repositorio es público, los datos de `data/` también lo serán. Para datos sensibles usa un repositorio
> privado, o no subas el Excel y carga el archivo desde la barra lateral de la app.

## Inputs

| Input | Dónde | Qué hace |
|---|---|---|
| Número de productos a optimizar | Sección 1 | Cuántos productos entran al modelo |
| Productos | Sección 1 | Selección manual, o automática por mejor ratio margen/riesgo |
| Costos unitarios y precios de venta | Sección 2 | Tabla editable (por omisión, último dato histórico en MXN) |
| Margen esperado | Sección 3 | Margen bruto (sobre precio) que debe alcanzar la mezcla |
| Intervalo de confianza | Barra lateral | Nivel de confianza del VaR |
| Plazo para VaR | Barra lateral | Horizonte del VaR, en días |
| Periodicidad de los precios | Barra lateral | Mensual / Trimestral / Anual: agregación de ventas para calcular rendimientos |
| Plazo a calcular | Barra lateral | 7 días, 3 meses, 6 meses, YTD, 1 año, 5 años: ventana histórica usada para estimar el riesgo |
| Ventas objetivo por periodo | Sección 3 | Volumen total a repartir; convierte participaciones en unidades |

## Outputs

* **Nombre de los productos a producir** y **cantidad de unidades** de cada uno (tabla + gráfica + descarga en Excel/CSV).
* Margen esperado, riesgo, VaR y margen en riesgo (MXN).
* Frontera eficiente con la cartera elegida, la de varianza mínima y la de máximo ratio margen/riesgo.
* Composición de la mezcla a lo largo de la frontera, correlaciones y covarianzas.

## Metodología (resumen)

* Cada producto es un activo; la cartera es la **mezcla de ventas** (participaciones ≥ 0 que suman 100 %).
* **Rendimiento = margen:** `(precio − costo) / precio`. Margen de la mezcla = promedio ponderado.
* **Riesgo:** desviación estándar de la variación periodo contra periodo de las ventas; covarianzas estimadas con el
  historial según periodicidad y plazo (con periodicidad mensual y todo el historial coinciden con la hoja
  `Covarianzas` del Excel).
* **Optimización:** minimiza la varianza sujeta al margen esperado (SLSQP, scipy). Si el margen pedido queda por debajo
  del de la cartera de varianza mínima, se usa esta última (zona ineficiente de la curva).
* **VaR paramétrico normal:** `−(μ·h + z·σ·√h)` con `h = días del VaR / días del periodo`; el margen en riesgo es el VaR
  aplicado al margen esperado en MXN.
* **Unidades:** `participación × ventas objetivo / precio`, redondeadas a enteros.

## Notas sobre los datos

* Fecha de corte = última fecha del archivo; **YTD** y los plazos se cuentan hacia atrás desde ella.
* Con datos mensuales, un plazo de **7 días** contiene a lo sumo 1 observación, y con periodicidad **anual** solo
  hay 2 rendimientos: la app amplía automáticamente la ventana a un mínimo de 3 observaciones (si existen), aplica
  encogimiento de covarianza cuando hay menos de 2 observaciones por producto y muestra un aviso.
* El historial incluido cubre 36 meses (ene-2024 a dic-2026): **5 años** equivale a todo el historial disponible.
* Se usan las columnas `*_MXN` de `Ventas_Historicas`. En la hoja `Productos` el Producto C figura en USD; si sus
  precios históricos realmente están en dólares, conviértelos a MXN antes de cargar el archivo.

## Formato del Excel que acepta la app

Hoja `Productos` (columna `Producto`) y hoja `Ventas_Historicas` con las columnas `Fecha`, `Producto`,
`Precio_Unitario_MXN`, `Costo_Variable_Unitario_MXN` y `Ventas_MXN`, un registro por producto y mes.

---
*Herramienta de apoyo a la decisión; los supuestos (normalidad, estabilidad de correlaciones) pueden no cumplirse.*
