"""
markowitz.py
============
Núcleo de cálculo del optimizador de mezcla de productos basado en la
Frontera Eficiente de Markowitz.

No depende de Streamlit, de modo que puede probarse y reutilizarse por separado.

Modelo (resumen)
----------------
* Cada producto es un "activo" y la cartera es la mezcla de ventas
  (w_i = participación del producto i en las ventas; suma 100 %, sin cortos).
* Rendimiento de la cartera  -> margen esperado:  m_p = sum(w_i * m_i),
  con  m_i = (precio_i - costo_i) / precio_i   (valores editables por el usuario).
* Riesgo de la cartera       -> desviación estándar de la variación de ventas:
  s_p = sqrt(w' * Cov * w), con Cov estimada con los rendimientos históricos de
  ventas en la periodicidad y el plazo elegidos.
* Optimización: minimizar la varianza sujeta a alcanzar el margen esperado.
* VaR paramétrico (normal) sobre la variación de ventas de la mezcla, convertido
  a MXN de margen en riesgo.
* Cantidades: unidades_i = w_i * ventas_objetivo / precio_i.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.stats import norm

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
PERIODICIDADES = {
    # nombre: (alias de frecuencia pandas, meses por periodo, días por periodo)
    "Mensual": ("MS", 1, 30.4375),
    "Trimestral": ("QS", 3, 91.3125),
    "Anual": ("YS", 12, 365.25),
}
PLAZOS = ["7 días", "3 meses", "6 meses", "YTD", "1 año", "5 años"]

# Longitud (en meses) de cada plazo; None = caso especial (7 días / YTD)
_MESES_PLAZO = {"3 meses": 3, "6 meses": 6, "1 año": 12, "5 años": 60}

MIN_RETORNOS = 3        # observaciones mínimas deseadas para estimar riesgo
MIN_RETORNOS_ABS = 2    # por debajo de esto no se puede calcular nada
OBS_CONFIABLES = 12     # por debajo de esto se advierte de baja confiabilidad
UMBRAL_PESO = 1e-4      # participaciones menores se consideran cero
ENCOGIMIENTO_MIN_POCOS_DATOS = 0.5  # se aplica si hay menos de 2 observaciones por producto

COLUMNAS_VENTAS = [
    "Fecha",
    "Producto",
    "Precio_Unitario_MXN",
    "Costo_Variable_Unitario_MXN",
    "Ventas_MXN",
]

Aviso = tuple[str, str]  # (nivel: "info" | "warning", texto)


# --------------------------------------------------------------------------- #
# Carga y preparación de datos
# --------------------------------------------------------------------------- #
def cargar_datos(fuente) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee el Excel (ruta u objeto tipo archivo) y devuelve (productos, ventas)."""
    with pd.ExcelFile(fuente) as xls:
        for hoja in ("Productos", "Ventas_Historicas"):
            if hoja not in xls.sheet_names:
                raise ValueError(f"Falta la hoja '{hoja}' en el archivo de Excel.")
        productos = xls.parse("Productos")
        ventas = xls.parse("Ventas_Historicas")

    if "Producto" not in productos.columns:
        raise ValueError("La hoja 'Productos' debe tener la columna 'Producto'.")
    faltantes = [c for c in COLUMNAS_VENTAS if c not in ventas.columns]
    if faltantes:
        raise ValueError(
            "La hoja 'Ventas_Historicas' no tiene las columnas: " + ", ".join(faltantes)
        )

    ventas = ventas.copy()
    ventas["Fecha"] = pd.to_datetime(ventas["Fecha"]).dt.to_period("M").dt.to_timestamp()
    ventas = ventas[ventas["Producto"].isin(productos["Producto"])]
    return productos, ventas


def precios_costos_recientes(ventas: pd.DataFrame, nombres: list[str]) -> pd.DataFrame:
    """Último precio y costo unitario (MXN) observados de cada producto."""
    ult = ventas.sort_values("Fecha").groupby("Producto").tail(1).set_index("Producto")
    return pd.DataFrame(
        {
            "Producto": nombres,
            "Costo unitario (MXN)": ult.loc[nombres, "Costo_Variable_Unitario_MXN"].round(2).values,
            "Precio de venta (MXN)": ult.loc[nombres, "Precio_Unitario_MXN"].round(2).values,
        }
    )


def niveles_ventas(ventas: pd.DataFrame, periodicidad: str) -> pd.DataFrame:
    """Ventas (MXN) por producto agregadas a la periodicidad elegida.

    Las ventas son un flujo, por lo que se suman dentro de cada periodo. Los
    periodos incompletos (p. ej. un trimestre con solo 2 meses) quedan como NaN.
    """
    freq, meses, _ = PERIODICIDADES[periodicidad]
    pv = (
        ventas.pivot_table(index="Fecha", columns="Producto", values="Ventas_MXN", aggfunc="sum")
        .sort_index()
        .asfreq("MS")
    )
    if meses == 1:
        return pv
    return pv.resample(freq).sum(min_count=meses)


def retornos(niveles: pd.DataFrame) -> pd.DataFrame:
    """Variación porcentual periodo contra periodo (sin filas incompletas)."""
    r = niveles / niveles.shift(1) - 1
    return r.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def ventana_retornos(
    rets: pd.DataFrame, plazo: str, periodicidad: str
) -> tuple[pd.DataFrame, list[Aviso]]:
    """Recorta los rendimientos al 'plazo a calcular' respecto a la última fecha.

    Si el plazo abarca menos de MIN_RETORNOS observaciones (p. ej. 7 días con
    datos mensuales) se amplía automáticamente a las últimas MIN_RETORNOS.
    """
    if len(rets) < MIN_RETORNOS_ABS:
        raise ValueError(
            f"Con periodicidad {periodicidad.lower()} solo hay {len(rets)} rendimiento(s) "
            "en el historial; se necesitan al menos 2. Usa una periodicidad más corta."
        )
    if plazo not in PLAZOS:
        raise ValueError(f"Plazo no reconocido: {plazo}")

    ref = rets.index.max()
    _, meses_periodo, _ = PERIODICIDADES[periodicidad]
    avisos: list[Aviso] = []

    if plazo == "7 días":
        sel = rets[rets.index > ref - pd.Timedelta(days=7)]
    elif plazo == "YTD":
        sel = rets[rets.index >= pd.Timestamp(year=ref.year, month=1, day=1)]
    else:
        sel = rets[rets.index > ref - pd.DateOffset(months=_MESES_PLAZO[plazo])]

    n_plazo = len(sel)
    if n_plazo < MIN_RETORNOS:
        sel = rets.iloc[-min(len(rets), MIN_RETORNOS):]
        avisos.append(
            (
                "info",
                f"El plazo «{plazo}» contiene solo {n_plazo} observación(es) con periodicidad "
                f"{periodicidad.lower()}; se amplió automáticamente a los últimos {len(sel)} periodos "
                "para poder estimar el riesgo.",
            )
        )
    elif plazo in _MESES_PLAZO:
        esperados = _MESES_PLAZO[plazo] / meses_periodo
        if n_plazo == len(rets) and n_plazo < esperados:
            avisos.append(
                (
                    "info",
                    f"El historial disponible ({n_plazo} periodos de rendimiento, desde "
                    f"{rets.index.min():%b-%Y}) es más corto que el plazo «{plazo}»; "
                    "se usó todo el historial.",
                )
            )

    if len(sel) < OBS_CONFIABLES:
        avisos.append(
            (
                "warning",
                f"Solo hay {len(sel)} observaciones para estimar varianzas y covarianzas; "
                "el riesgo estimado es poco confiable. Considera una periodicidad más corta, "
                "un plazo más largo o subir el encogimiento de covarianza.",
            )
        )
    return sel, avisos


def ventas_promedio(niveles: pd.DataFrame, indice: pd.Index, productos: list[str]) -> float:
    """Ventas totales promedio por periodo de los productos, en las fechas dadas."""
    return float(niveles.loc[indice, productos].sum(axis=1).mean())


# --------------------------------------------------------------------------- #
# Estadísticos
# --------------------------------------------------------------------------- #
def estadisticos(rets: pd.DataFrame, encogimiento: float = 0.0) -> tuple[pd.Series, pd.DataFrame]:
    """Media y matriz de covarianzas (opcionalmente encogida hacia la diagonal)."""
    mu = rets.mean()
    cov = rets.cov()
    d = float(np.clip(encogimiento, 0.0, 1.0))
    if d > 0:
        diag = pd.DataFrame(np.diag(np.diag(cov)), index=cov.index, columns=cov.columns)
        cov = (1 - d) * cov + d * diag
    # Pequeña regularización para evitar matrices singulares con pocas observaciones
    cov = cov + 1e-10 * np.eye(len(cov))
    return mu, cov


def margen_unitario(precios: pd.Series, costos: pd.Series) -> pd.Series:
    """Margen sobre precio: (precio - costo) / precio."""
    return (precios - costos) / precios


# --------------------------------------------------------------------------- #
# Optimización
# --------------------------------------------------------------------------- #
def _limpiar(w: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    w = np.clip(w, wmin, wmax)
    if wmin <= 0:
        w = np.where(w < UMBRAL_PESO, 0.0, w)
    return w / w.sum()


def _minimizar_varianza(cov, margenes, objetivo, wmin, wmax) -> np.ndarray:
    """min w'Cov w  s.a.  sum(w)=1, [margenes'w = objetivo], wmin<=w<=wmax."""
    n = len(margenes)
    escala = float(np.trace(cov)) / n or 1.0
    S = cov / escala

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    if objetivo is not None:
        cons.append(
            {"type": "eq", "fun": lambda w: margenes @ w - objetivo, "jac": lambda w: margenes}
        )
    bounds = [(wmin, wmax)] * n

    inicios = [np.full(n, 1.0 / n)] + [np.eye(n)[i] * 0.6 + 0.4 / n for i in range(n)]
    for x0 in inicios:
        res = minimize(
            lambda w: w @ S @ w,
            x0,
            jac=lambda w: 2 * S @ w,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={"ftol": 1e-14, "maxiter": 500},
        )
        w = res.x
        ok = res.success and abs(w.sum() - 1) < 1e-6
        if objetivo is not None:
            ok = ok and abs(margenes @ w - objetivo) < 1e-6
        if ok:
            return _limpiar(w, wmin, wmax)
    raise RuntimeError("El optimizador no encontró una solución factible.")


def rango_margen(margenes: np.ndarray, wmin: float, wmax: float) -> tuple[float, float]:
    """Margen mínimo y máximo alcanzable respetando las cotas de participación."""
    n = len(margenes)
    if wmin * n > 1 + 1e-9 or wmax * n < 1 - 1e-9:
        raise ValueError(
            "Las cotas de participación son inviables: con "
            f"{n} productos la suma de participaciones no puede ser 100 %."
        )
    kw = dict(A_eq=[np.ones(n)], b_eq=[1.0], bounds=[(wmin, wmax)] * n, method="highs")
    bajo = linprog(margenes, **kw)
    alto = linprog(-margenes, **kw)
    if bajo.status != 0 or alto.status != 0:
        raise ValueError("No se pudo determinar el rango de margen alcanzable.")
    return float(bajo.fun), float(-alto.fun)


def cartera_varianza_minima(cov, margenes, wmin=0.0, wmax=1.0) -> np.ndarray:
    return _minimizar_varianza(cov, margenes, None, wmin, wmax)


def cartera_para_margen(cov, margenes, objetivo, wmin=0.0, wmax=1.0) -> np.ndarray:
    return _minimizar_varianza(cov, margenes, objetivo, wmin, wmax)


def cartera_max_ratio(cov, margenes, wmin, wmax, w0: np.ndarray) -> np.ndarray:
    """Cartera con mayor ratio margen/riesgo (análoga al portafolio tangente con rf=0)."""
    n = len(margenes)

    def neg_ratio(w):
        return -(margenes @ w) / np.sqrt(max(w @ cov @ w, 1e-16))

    res = minimize(
        neg_ratio,
        w0,
        bounds=[(wmin, wmax)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    cand = _limpiar(res.x, wmin, wmax) if res.success else w0
    return cand if neg_ratio(cand) <= neg_ratio(w0) + 1e-12 else w0


def estadistica_cartera(w, margenes, cov) -> tuple[float, float]:
    """(margen esperado, riesgo por periodo) de una cartera."""
    return float(margenes @ w), float(np.sqrt(max(w @ cov @ w, 0.0)))


def frontera(cov, margenes, productos, wmin=0.0, wmax=1.0, puntos=60) -> pd.DataFrame:
    """Puntos de la frontera eficiente (tramo superior, desde la varianza mínima)."""
    w_mv = cartera_varianza_minima(cov, margenes, wmin, wmax)
    m_mv = float(margenes @ w_mv)
    _, m_max = rango_margen(margenes, wmin, wmax)
    objetivos = np.linspace(m_mv, m_max - 1e-9 * abs(m_max), puntos)

    filas = []
    for t in objetivos:
        try:
            w = cartera_para_margen(cov, margenes, t, wmin, wmax)
        except RuntimeError:
            continue
        m, s = estadistica_cartera(w, margenes, cov)
        filas.append({"Margen": m, "Riesgo": s, **dict(zip(productos, w))})
    return pd.DataFrame(filas)


def carteras_aleatorias(margenes, cov, wmin=0.0, wmax=1.0, n=3000, semilla=42) -> pd.DataFrame:
    """Nube de carteras posibles (Dirichlet) para dibujar el conjunto factible."""
    rng = np.random.default_rng(semilla)
    k = len(margenes)
    base = np.vstack(
        [rng.dirichlet(np.ones(k), n // 2), rng.dirichlet(np.full(k, 0.3), n - n // 2)]
    )
    W = wmin + (1 - k * wmin) * base
    W = W[(W <= wmax + 1e-12).all(axis=1)]
    m = W @ margenes
    s = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    return pd.DataFrame({"Margen": m, "Riesgo": s})


# --------------------------------------------------------------------------- #
# VaR y plan de producción
# --------------------------------------------------------------------------- #
def var_parametrico(w, mu_ret, cov, confianza, horizonte, incluir_media=True) -> float:
    """VaR paramétrico (normal) de la variación de ventas, como fracción (>= 0).

    `horizonte` está en periodos; la media escala con h y la desviación con raiz(h).
    """
    mu_p = float(mu_ret @ w) * horizonte if incluir_media else 0.0
    sigma_p = float(np.sqrt(max(w @ cov @ w, 0.0) * horizonte))
    z = norm.ppf(1 - confianza)  # negativo
    return max(-(mu_p + z * sigma_p), 0.0)


def plan_produccion(pesos: pd.Series, precios: pd.Series, costos: pd.Series, ventas_objetivo: float):
    """Convierte participaciones en unidades enteras a producir."""
    unidades = np.rint(pesos * ventas_objetivo / precios).astype(int)
    ventas = unidades * precios
    costo_total = unidades * costos
    margen_mxn = ventas - costo_total
    plan = pd.DataFrame(
        {
            "Producto": pesos.index,
            "Participación en ventas": pesos.values,
            "Precio de venta (MXN)": precios.values,
            "Costo unitario (MXN)": costos.values,
            "Unidades a producir": unidades.values,
            "Ventas (MXN)": ventas.values,
            "Costo total (MXN)": costo_total.values,
            "Margen (MXN)": margen_mxn.values,
        }
    )
    ventas_seguras = plan["Ventas (MXN)"].replace(0, np.nan)
    plan["Margen (%)"] = (plan["Margen (MXN)"] / ventas_seguras).fillna(0.0)
    return plan


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
@dataclass
class Resultado:
    productos: list[str]
    margenes: pd.Series
    mu_ret: pd.Series
    cov: pd.DataFrame
    corr: pd.DataFrame
    retornos: pd.DataFrame
    frontera: pd.DataFrame
    aleatorias: pd.DataFrame
    carteras: dict[str, pd.Series]
    resumen: pd.DataFrame
    plan: pd.DataFrame
    metricas: dict
    parametros: dict
    rango_margen: tuple[float, float]
    margen_objetivo_efectivo: float
    avisos: list[Aviso] = field(default_factory=list)


def optimizar_mezcla(
    rets: pd.DataFrame,
    productos: list[str],
    precios: pd.Series,
    costos: pd.Series,
    margen_objetivo: float,
    confianza: float,
    plazo_var_dias: float,
    periodicidad: str,
    ventas_objetivo: float,
    peso_min: float = 0.0,
    peso_max: float = 1.0,
    encogimiento: float = 0.0,
    incluir_media: bool = True,
) -> Resultado:
    """Calcula la mezcla óptima, la frontera eficiente, el VaR y el plan de producción."""
    productos = list(productos)
    k = len(productos)
    if k < 2:
        raise ValueError("Selecciona al menos 2 productos para optimizar.")
    precios = precios.loc[productos].astype(float)
    costos = costos.loc[productos].astype(float)
    if (precios <= 0).any():
        raise ValueError("Todos los precios de venta deben ser mayores que cero.")
    if (costos < 0).any():
        raise ValueError("Los costos unitarios no pueden ser negativos.")

    margenes = margen_unitario(precios, costos)
    r = rets[productos]
    avisos: list[Aviso] = []
    encogimiento_ef = float(encogimiento)
    if len(r) < 2 * k and encogimiento_ef < ENCOGIMIENTO_MIN_POCOS_DATOS:
        encogimiento_ef = ENCOGIMIENTO_MIN_POCOS_DATOS
        avisos.append(
            (
                "warning",
                f"Con solo {len(r)} observaciones para {k} productos la matriz de covarianzas es "
                "inestable (puede producir carteras de riesgo cero que son un artefacto estadístico); "
                f"se aplicó un encogimiento de covarianza de {encogimiento_ef:.0%}.",
            )
        )
    mu_ret, cov = estadisticos(r, encogimiento_ef)
    corr = r.corr()
    m, C = margenes.values, cov.values

    lo, hi = rango_margen(m, peso_min, peso_max)
    objetivo = float(np.clip(margen_objetivo, lo, hi))
    if abs(objetivo - margen_objetivo) > 1e-9:
        avisos.append(
            (
                "warning",
                f"El margen esperado solicitado ({margen_objetivo:.2%}) no es alcanzable con los "
                f"productos y restricciones elegidos (rango {lo:.2%} – {hi:.2%}); se ajustó a {objetivo:.2%}.",
            )
        )

    w_mv = cartera_varianza_minima(C, m, peso_min, peso_max)
    m_mv = float(m @ w_mv)
    fr = frontera(C, m, productos, peso_min, peso_max)
    if objetivo < m_mv - 1e-9:
        w_obj = w_mv
        avisos.append(
            (
                "info",
                f"El margen esperado ({objetivo:.2%}) está por debajo del de la cartera de varianza mínima "
                f"({m_mv:.2%}). Esa zona es ineficiente (más riesgo y menos margen), por lo que se usa la "
                "cartera de varianza mínima.",
            )
        )
    else:
        w_obj = cartera_para_margen(C, m, objetivo, peso_min, peso_max)

    idx0 = int(np.argmax((fr["Margen"] / fr["Riesgo"].replace(0, np.nan)).fillna(0).values))
    w0 = fr[productos].iloc[idx0].values
    w_ratio = cartera_max_ratio(C, m, peso_min, peso_max, w0)

    carteras = {
        "Margen esperado (elegida)": pd.Series(w_obj, index=productos),
        "Varianza mínima": pd.Series(w_mv, index=productos),
        "Máx. margen/riesgo": pd.Series(w_ratio, index=productos),
    }

    h = plazo_var_dias / PERIODICIDADES[periodicidad][2]
    filas = []
    for nombre, w in carteras.items():
        mg, rs = estadistica_cartera(w.values, m, C)
        var = var_parametrico(w.values, mu_ret.values, C, confianza, h, incluir_media)
        filas.append(
            {
                "Cartera": nombre,
                "Margen esperado": mg,
                "Riesgo (σ por periodo)": rs,
                "Ratio margen/riesgo": mg / rs if rs > 0 else np.nan,
                f"VaR {confianza:.1%} (% ventas)": var,
                **{p: w[p] for p in productos},
            }
        )
    resumen = pd.DataFrame(filas)

    w_sel = carteras["Margen esperado (elegida)"]
    plan = plan_produccion(w_sel, precios, costos, ventas_objetivo)
    mg, rs = estadistica_cartera(w_sel.values, m, C)
    var = var_parametrico(w_sel.values, mu_ret.values, C, confianza, h, incluir_media)
    if var == 0.0 and incluir_media:
        avisos.append(
            (
                "info",
                "El VaR resulta 0 porque el rendimiento medio histórico de ventas supera al riesgo en ese "
                "horizonte. Desmarca «Incluir el rendimiento medio en el VaR» para una medida más conservadora.",
            )
        )
    ventas_plan = float(plan["Ventas (MXN)"].sum())
    margen_plan = float(plan["Margen (MXN)"].sum())
    metricas = {
        "margen_esperado": mg,
        "riesgo_periodo": rs,
        "riesgo_horizonte": rs * np.sqrt(h),
        "horizonte_periodos": h,
        "var_pct": var,
        "var_ventas_mxn": var * ventas_plan,
        "var_margen_mxn": var * margen_plan,
        "ventas_plan_mxn": ventas_plan,
        "margen_plan_mxn": margen_plan,
        "margen_plan_pct": margen_plan / ventas_plan if ventas_plan > 0 else 0.0,
        "observaciones": len(r),
        "desde": r.index.min(),
        "hasta": r.index.max(),
    }
    parametros = {
        "Productos": ", ".join(productos),
        "Periodicidad": periodicidad,
        "Observaciones usadas": len(r),
        "Periodo de datos": f"{r.index.min():%Y-%m} a {r.index.max():%Y-%m}",
        "Margen esperado solicitado": f"{margen_objetivo:.2%}",
        "Margen esperado usado": f"{objetivo:.2%}",
        "Nivel de confianza VaR": f"{confianza:.1%}",
        "Plazo VaR (días)": plazo_var_dias,
        "Ventas objetivo por periodo (MXN)": ventas_objetivo,
        "Participación mínima": f"{peso_min:.0%}",
        "Participación máxima": f"{peso_max:.0%}",
        "Encogimiento de covarianza": encogimiento_ef,
        "VaR incluye rendimiento medio": "Sí" if incluir_media else "No",
    }

    return Resultado(
        productos=productos,
        margenes=margenes,
        mu_ret=mu_ret,
        cov=cov,
        corr=corr,
        retornos=r,
        frontera=fr,
        aleatorias=carteras_aleatorias(m, C, peso_min, peso_max),
        carteras=carteras,
        resumen=resumen,
        plan=plan,
        metricas=metricas,
        parametros=parametros,
        rango_margen=(lo, hi),
        margen_objetivo_efectivo=objetivo,
        avisos=avisos,
    )


def resultado_a_excel(res: Resultado) -> bytes:
    """Exporta plan, carteras, frontera y parámetros a un libro de Excel."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        res.plan.to_excel(xw, sheet_name="Plan de producción", index=False)
        res.resumen.to_excel(xw, sheet_name="Carteras", index=False)
        res.frontera.to_excel(xw, sheet_name="Frontera eficiente", index=False)
        pd.DataFrame(
            {"Parámetro": list(res.parametros), "Valor": [str(v) for v in res.parametros.values()]}
        ).to_excel(xw, sheet_name="Parámetros", index=False)
    return buf.getvalue()
