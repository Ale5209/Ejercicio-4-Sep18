"""
app.py — Optimizador de mezcla de productos (Frontera Eficiente de Markowitz)

Ejecutar localmente:   streamlit run app.py
"""
from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import markowitz as mk

RUTA_DATOS = Path(__file__).parent / "data" / "Base_Datos_Gestion_Productos_Markowitz.xlsx"

st.set_page_config(
    page_title="Mezcla óptima de productos · Frontera eficiente",
    page_icon="📈",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Leyendo la base de datos…")
def cargar(contenido: bytes | None):
    fuente = BytesIO(contenido) if contenido else RUTA_DATOS
    return mk.cargar_datos(fuente)


def mostrar_avisos(avisos):
    for nivel, texto in avisos:
        (st.warning if nivel == "warning" else st.info)(texto)


def mxn(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%"


def tabla_plan(plan: pd.DataFrame, con_total: bool = True) -> pd.DataFrame:
    """Formatea el plan de producción para mostrarlo."""
    p = plan.copy()
    filas = pd.DataFrame(
        {
            "Producto": p["Producto"],
            "Unidades a producir": p["Unidades a producir"].map(lambda v: f"{v:,.0f}"),
            "Participación en ventas": p["Participación en ventas"].map(pct),
            "Precio (MXN)": p["Precio de venta (MXN)"].map(lambda v: f"{v:,.2f}"),
            "Costo unit. (MXN)": p["Costo unitario (MXN)"].map(lambda v: f"{v:,.2f}"),
            "Ventas (MXN)": p["Ventas (MXN)"].map(mxn),
            "Margen (MXN)": p["Margen (MXN)"].map(mxn),
            "Margen (%)": p["Margen (%)"].map(pct),
        }
    )
    if con_total and len(p) > 1:
        ventas, margen = p["Ventas (MXN)"].sum(), p["Margen (MXN)"].sum()
        filas.loc[len(filas)] = [
            "TOTAL",
            f"{p['Unidades a producir'].sum():,.0f}",
            pct(p["Participación en ventas"].sum()),
            "",
            "",
            mxn(ventas),
            mxn(margen),
            pct(margen / ventas if ventas > 0 else 0),
        ]
    return filas


def figura_frontera(res: mk.Resultado) -> go.Figure:
    fig = go.Figure()
    al = res.aleatorias
    fig.add_trace(
        go.Scatter(
            x=al["Riesgo"] * 100,
            y=al["Margen"] * 100,
            mode="markers",
            name="Carteras posibles",
            marker=dict(size=4, color="rgba(140,140,140,0.35)"),
            hoverinfo="skip",
        )
    )
    fr = res.frontera
    fig.add_trace(
        go.Scatter(
            x=fr["Riesgo"] * 100,
            y=fr["Margen"] * 100,
            mode="lines",
            name="Frontera eficiente",
            line=dict(color="#1f4e9c", width=3),
            hovertemplate="Riesgo %{x:.2f}%<br>Margen %{y:.2f}%<extra></extra>",
        )
    )
    sigma = np.sqrt(np.diag(res.cov.values))
    fig.add_trace(
        go.Scatter(
            x=sigma * 100,
            y=res.margenes.values * 100,
            mode="markers+text",
            text=res.productos,
            textposition="top center",
            name="Productos individuales",
            marker=dict(size=10, color="#444444"),
            hovertemplate="%{text}<br>Riesgo %{x:.2f}%<br>Margen %{y:.2f}%<extra></extra>",
        )
    )
    estilos = {
        "Varianza mínima": ("square", "#2a9d8f"),
        "Máx. margen/riesgo": ("diamond", "#e9a100"),
        "Margen esperado (elegida)": ("star", "#d62828"),
    }
    for nombre, (simbolo, color) in estilos.items():
        w = res.carteras[nombre]
        mg, rs = mk.estadistica_cartera(w.values, res.margenes.values, res.cov.values)
        fig.add_trace(
            go.Scatter(
                x=[rs * 100],
                y=[mg * 100],
                mode="markers",
                name=nombre,
                marker=dict(size=16, symbol=simbolo, color=color, line=dict(width=1, color="white")),
                hovertemplate=f"{nombre}<br>Riesgo %{{x:.2f}}%<br>Margen %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        xaxis_title="Riesgo: desviación estándar de las ventas (% por periodo)",
        yaxis_title="Margen esperado (%)",
        legend=dict(orientation="h", yanchor="top", y=-0.18),
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def figura_composicion(res: mk.Resultado) -> go.Figure:
    fig = go.Figure()
    fr = res.frontera
    for p in res.productos:
        fig.add_trace(
            go.Scatter(
                x=fr["Margen"] * 100,
                y=fr[p] * 100,
                mode="lines",
                stackgroup="mezcla",
                name=p,
                hovertemplate=f"{p}<br>Margen %{{x:.2f}}%<br>Participación %{{y:.1f}}%<extra></extra>",
            )
        )
    fig.add_vline(
        x=res.margen_objetivo_efectivo * 100,
        line_dash="dash",
        annotation_text="Margen elegido",
    )
    fig.update_layout(
        xaxis_title="Margen esperado de la cartera (%)",
        yaxis_title="Participación en ventas (%)",
        legend=dict(orientation="h", yanchor="top", y=-0.18),
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def figura_unidades(plan: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=plan["Producto"],
            y=plan["Unidades a producir"],
            text=plan["Unidades a producir"].map(lambda v: f"{v:,.0f}"),
            textposition="outside",
            marker_color="#1f4e9c",
        )
    )
    fig.update_layout(
        yaxis_title="Unidades a producir",
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def figura_correlacion(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Encabezado y barra lateral
# --------------------------------------------------------------------------- #
st.title("📈 Mezcla óptima de productos")
st.caption(
    "Optimización bajo la Frontera Eficiente de Markowitz: encuentra qué productos producir y en qué "
    "cantidad para alcanzar el margen esperado con el menor riesgo posible."
)

with st.sidebar:
    st.header("Datos")
    archivo = st.file_uploader(
        "Base de datos (opcional)",
        type=["xlsx"],
        help="Si no subes nada se usa data/Base_Datos_Gestion_Productos_Markowitz.xlsx. "
        "El archivo debe tener las hojas 'Productos' y 'Ventas_Historicas'.",
    )

try:
    productos_df, ventas = cargar(archivo.getvalue() if archivo is not None else None)
except Exception as e:  # noqa: BLE001 - se muestra al usuario
    st.error(f"No se pudo leer la base de datos: {e}")
    st.stop()

todos = [p for p in productos_df["Producto"].tolist() if p in set(ventas["Producto"])]
if len(todos) < 2:
    st.error("La base de datos necesita al menos 2 productos con ventas históricas.")
    st.stop()

with st.sidebar:
    st.caption(f"Historial: {ventas['Fecha'].min():%b-%Y} a {ventas['Fecha'].max():%b-%Y} · {len(todos)} productos")
    st.header("Parámetros de riesgo")
    periodicidad = st.selectbox(
        "Periodicidad de los precios",
        list(mk.PERIODICIDADES),
        help="Frecuencia con la que se agregan las ventas para calcular rendimientos, varianzas y covarianzas.",
    )
    plazo = st.selectbox(
        "Plazo a calcular",
        mk.PLAZOS,
        index=mk.PLAZOS.index("5 años"),
        help="Ventana histórica (hasta la última fecha de la base) usada para estimar riesgo y correlaciones. "
        "Si el plazo tiene muy pocas observaciones se amplía automáticamente.",
    )
    confianza = (
        st.slider("Intervalo de confianza (VaR)", 90.0, 99.5, 95.0, 0.5, help="Nivel de confianza en %.") / 100
    )
    plazo_var = st.number_input(
        "Plazo para VaR (días)", min_value=1, max_value=3650, value=30, step=1,
        help="Horizonte del VaR; se escala con la raíz del tiempo.",
    )
    incluir_media = st.checkbox(
        "Incluir el rendimiento medio en el VaR", value=True,
        help="Si se desmarca, el VaR es más conservador (media cero).",
    )

# Rendimientos históricos en la periodicidad y plazo elegidos
niveles = mk.niveles_ventas(ventas, periodicidad)[todos]
try:
    rets_todos = mk.retornos(niveles)
    rets_v, avisos_ventana = mk.ventana_retornos(rets_todos, plazo, periodicidad)
except ValueError as e:
    st.error(str(e))
    st.stop()

base = mk.precios_costos_recientes(ventas, todos)
margen_base = mk.margen_unitario(
    base.set_index("Producto")["Precio de venta (MXN)"], base.set_index("Producto")["Costo unitario (MXN)"]
)

# --------------------------------------------------------------------------- #
# 1. Selección de productos
# --------------------------------------------------------------------------- #
st.subheader("1 · Productos a optimizar")
c1, c2 = st.columns([1, 3])
n_prod = int(
    c1.number_input(
        "Número de productos",
        min_value=2,
        max_value=len(todos),
        value=len(todos),
        step=1,
        help="Cuántos productos entran a la optimización.",
    )
)
modo = c2.radio(
    "Selección", ["Manual", "Automática (mejor ratio margen/riesgo)"], horizontal=True
)
if modo == "Manual":
    sel = c2.multiselect(
        "Productos",
        todos,
        default=todos[:n_prod],
        max_selections=n_prod,
        key=f"sel_{n_prod}",
    )
    if len(sel) != n_prod:
        st.warning(f"Selecciona exactamente {n_prod} productos (llevas {len(sel)}).")
        st.stop()
else:
    ratio = (margen_base / rets_v.std()).sort_values(ascending=False)
    top = set(ratio.index[:n_prod])
    sel = [p for p in todos if p in top]
    c2.write("Seleccionados: " + ", ".join(f"**{p}**" for p in sel))

mostrar_avisos(avisos_ventana)

# --------------------------------------------------------------------------- #
# 2. Costos y precios
# --------------------------------------------------------------------------- #
st.subheader("2 · Costos unitarios y precios de venta")
st.caption("Valores iniciales: último dato histórico en MXN. Puedes editarlos.")
base_sel = base[base["Producto"].isin(sel)].reset_index(drop=True)
editado = st.data_editor(
    base_sel,
    key="editor_" + "|".join(sel),
    hide_index=True,
    disabled=["Producto"],
    column_config={
        "Costo unitario (MXN)": st.column_config.NumberColumn(min_value=0.0, format="%.2f", required=True),
        "Precio de venta (MXN)": st.column_config.NumberColumn(min_value=0.01, format="%.2f", required=True),
    },
)
editado = editado.dropna()
if set(editado["Producto"]) != set(sel):
    st.error("Completa el costo y el precio de todos los productos.")
    st.stop()
precios = pd.Series(editado["Precio de venta (MXN)"].values, index=editado["Producto"])
costos = pd.Series(editado["Costo unitario (MXN)"].values, index=editado["Producto"])
margenes = mk.margen_unitario(precios, costos)
if (margenes <= 0).any():
    st.warning(
        "Hay productos con margen ≤ 0 (precio ≤ costo): "
        + ", ".join(margenes.index[margenes <= 0])
        + ". Difícilmente serán elegidos por el optimizador."
    )
st.dataframe(
    pd.DataFrame({"Margen unitario (sobre precio)": margenes.map(pct)}).T,
)

# --------------------------------------------------------------------------- #
# 3. Objetivo
# --------------------------------------------------------------------------- #
st.subheader("3 · Objetivo de margen y volumen")

with st.expander("Restricciones avanzadas (opcional)"):
    a1, a2, a3 = st.columns(3)
    peso_max = a1.slider(
        "Participación máxima por producto (% de ventas)",
        min_value=math.ceil(100 / n_prod), max_value=100, value=100, step=1,
    ) / 100
    peso_min = a2.slider(
        "Participación mínima por producto (%)",
        min_value=0, max_value=int(100 // n_prod), value=0, step=1,
        help="Con un mínimo > 0 todos los productos seleccionados se producen.",
    ) / 100
    encogimiento = a3.slider(
        "Encogimiento de covarianza", 0.0, 0.9, 0.0, 0.05,
        help="Suaviza las covarianzas hacia la diagonal; útil con pocas observaciones. "
        "Se aplica automáticamente (50 %) si hay menos de 2 observaciones por producto.",
    )

try:
    lo, hi = mk.rango_margen(margenes.values, peso_min, peso_max)
except ValueError as e:
    st.error(str(e))
    st.stop()

o1, o2 = st.columns(2)
lo_pct, hi_pct = math.ceil(lo * 1000) / 10, math.floor(hi * 1000) / 10
if lo_pct < hi_pct:
    margen_obj = (
        o1.slider(
            "Margen esperado (%)",
            min_value=lo_pct,
            max_value=hi_pct,
            value=round((lo_pct + hi_pct) / 2, 1),
            step=0.1,
            format="%.1f",
            help="Margen bruto (sobre precio) que debe alcanzar la mezcla. El rango depende de los "
            "márgenes de los productos elegidos.",
        )
        / 100
    )
else:
    margen_obj = lo
    o1.info(f"Los productos elegidos solo permiten un margen de {pct(lo)}.")

ventas_default = mk.ventas_promedio(niveles, rets_v.index, sel)
ventas_obj = float(
    o2.number_input(
        f"Ventas objetivo por periodo {periodicidad.lower()} (MXN)",
        min_value=1000.0,
        value=float(max(round(ventas_default, -3), 1000.0)),
        step=10000.0,
        format="%.0f",
        key=f"ventas_{periodicidad}_{plazo}_{'|'.join(sel)}",
        help="Ventas totales que se repartirán entre los productos según la mezcla óptima. "
        "Sirve para convertir participaciones en unidades. Por omisión: promedio histórico.",
    )
)

# --------------------------------------------------------------------------- #
# Cálculo
# --------------------------------------------------------------------------- #
try:
    res = mk.optimizar_mezcla(
        rets_v,
        sel,
        precios,
        costos,
        margen_objetivo=margen_obj,
        confianza=confianza,
        plazo_var_dias=float(plazo_var),
        periodicidad=periodicidad,
        ventas_objetivo=ventas_obj,
        peso_min=peso_min,
        peso_max=peso_max,
        encogimiento=encogimiento,
        incluir_media=incluir_media,
    )
except (ValueError, RuntimeError) as e:
    st.error(f"No se pudo optimizar: {e}")
    st.stop()

mostrar_avisos(res.avisos)
met = res.metricas

st.divider()
st.subheader("Resultado")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Margen esperado", pct(met["margen_esperado"]))
k2.metric(f"Riesgo (σ por periodo {periodicidad.lower()})", pct(met["riesgo_periodo"]))
k3.metric(f"VaR {confianza:.1%} a {int(plazo_var)} días", pct(met["var_pct"]), help="Pérdida máxima esperada de ventas con esa confianza.")
k4.metric("Margen en riesgo (VaR, MXN)", mxn(met["var_margen_mxn"]))

tab_res, tab_fr, tab_comp, tab_datos = st.tabs(
    ["Qué producir", "Frontera eficiente", "Composición de la frontera", "Datos y supuestos"]
)

with tab_res:
    a_producir = res.plan[res.plan["Unidades a producir"] > 0].reset_index(drop=True)
    excluidos = [p for p in res.productos if p not in set(a_producir["Producto"])]
    st.markdown("#### Productos a producir y cantidades")
    st.dataframe(tabla_plan(a_producir), hide_index=True)
    st.plotly_chart(figura_unidades(a_producir))
    if excluidos:
        st.caption("No se producen (participación óptima 0 %): " + ", ".join(excluidos))
    st.caption(
        f"Con {mxn(ventas_obj)} de ventas objetivo por periodo, el plan genera {mxn(met['ventas_plan_mxn'])} "
        f"en ventas y {mxn(met['margen_plan_mxn'])} de margen ({pct(met['margen_plan_pct'])}); las unidades se "
        "redondean a enteros."
    )
    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Descargar plan (Excel)",
        data=mk.resultado_a_excel(res),
        file_name="plan_produccion_markowitz.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    d2.download_button(
        "⬇️ Descargar plan (CSV)",
        data=a_producir.to_csv(index=False).encode("utf-8-sig"),
        file_name="plan_produccion_markowitz.csv",
        mime="text/csv",
    )

with tab_fr:
    st.plotly_chart(figura_frontera(res))
    st.caption(
        "Cualquier cartera bajo la curva azul es ineficiente: se puede obtener más margen con el mismo riesgo "
        "o el mismo margen con menos riesgo. La estrella roja es la mezcla recomendada para tu margen esperado."
    )
    resumen = res.resumen.copy()
    for c in resumen.columns[1:]:
        if c == "Ratio margen/riesgo":
            resumen[c] = resumen[c].map(lambda v: f"{v:.2f}")
        else:
            resumen[c] = resumen[c].map(pct)
    st.dataframe(resumen, hide_index=True)

with tab_comp:
    st.plotly_chart(figura_composicion(res))
    st.caption("Cómo cambia la mezcla de ventas conforme se exige más margen (y por tanto más riesgo).")

with tab_datos:
    st.markdown(
        f"**Observaciones usadas:** {met['observaciones']} periodos {periodicidad.lower()}s "
        f"({met['desde']:%b-%Y} a {met['hasta']:%b-%Y}) · **Plazo:** {plazo}"
    )
    est = pd.DataFrame(
        {
            "Margen unitario": res.margenes.map(pct),
            "Rendimiento medio de ventas": res.mu_ret.map(pct),
            "Volatilidad (σ)": pd.Series(np.sqrt(np.diag(res.cov.values)), index=res.productos).map(pct),
        }
    )
    st.dataframe(est)
    cc1, cc2 = st.columns(2)
    cc1.markdown("**Correlaciones**")
    cc1.plotly_chart(figura_correlacion(res.corr))
    cc2.markdown("**Matriz de covarianzas**")
    cc2.dataframe(res.cov.round(6))
    with st.expander("Rendimientos históricos utilizados"):
        st.dataframe(res.retornos.round(4))
    with st.expander("Metodología y supuestos"):
        st.markdown(
            """
- **Cartera:** cada producto es un activo; la ponderación es su participación en las ventas (suma 100 %, sin posiciones cortas).
- **Rendimiento = margen:** `margen_i = (precio_i − costo_i) / precio_i`; el margen de la mezcla es el promedio ponderado.
- **Riesgo:** desviación estándar de la variación periodo contra periodo de las ventas, con la matriz de covarianzas del
  historial (periodicidad y plazo elegidos).
- **Optimización:** se minimiza la varianza sujeta a alcanzar el margen esperado (SLSQP). Si el margen pedido está por
  debajo de la cartera de varianza mínima se usa esta última, porque esa zona de la curva es ineficiente.
- **VaR paramétrico (normal):** `VaR = −(μ·h + z·σ·√h)`, con `h = plazo en días / días del periodo`. El margen en riesgo
  es el VaR aplicado al margen esperado en MXN.
- **Cantidades:** `unidades_i = participación_i × ventas objetivo / precio_i`, redondeadas a enteros.
- **Limitaciones:** la normalidad y la estabilidad de correlaciones son supuestos; con pocas observaciones la
  estimación del riesgo es poco confiable. Herramienta de apoyo a la decisión, no una recomendación financiera.
"""
        )
