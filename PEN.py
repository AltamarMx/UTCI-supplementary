
```{python}

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------
# 1) Parámetro único: región a graficar
# --------------------------------------------
reg = "PEN"  # <- Cámbialo por la región que necesites

# --------------------------------------------
# 2) Lee datos y muestrea
# --------------------------------------------
f = "data/demanda_utci.parquet"
utci = pd.read_parquet(f)
utci = utci.sample(n=30000, random_state=42)

# --------------------------------------------
# 3) Definición de categorías y colores
# --------------------------------------------
stress_categories = [
    ("MC", -13, 0),
    ("SC", 0, 9),
    ("NT", 9, 26),
    ("MH", 26, 32),
    ("SH", 32, 38),
    ("VSH", 38, 46)
]

seasons = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11]
}

season_colors = {
    'Winter': 'rgb(31, 119, 180)',
    'Spring': 'rgb(44, 160, 44)',
    'Summer': 'rgb(214, 39, 40)',
    'Autumn': 'rgb(255, 127, 14)'
}

# --------------------------------------------
# 4) Verifica que existan las columnas para esta región
# --------------------------------------------
col_utci = f"{reg}_utci"
col_dem  = f"{reg}_demanda"
if col_utci not in utci.columns or col_dem not in utci.columns:
    raise ValueError(f"El DataFrame no contiene '{col_utci}' y/o '{col_dem}'.")

# --------------------------------------------
# 5) Calcula rango de demanda (ymin, ymax)
# --------------------------------------------
serie_dem = utci[col_dem].dropna()
if not serie_dem.empty:
    ymin, ymax = serie_dem.min(), serie_dem.max()
else:
    ymin, ymax = 0, 1

# --------------------------------------------
# 6) Crea subplots 2×2:
#    - fila 1, col 1: barra marginal de UTCI ponderada por demanda
#    - fila 1, col 2: en blanco
#    - fila 2, col 1: scatter UTCI vs demanda con líneas de promedio y rectángulos
#    - fila 2, col 2: histograma de densidad de DEMANDA por temporada (eje Y compartido)
# --------------------------------------------
fig = make_subplots(
    rows=2, cols=2,
    row_heights=[0.2, 0.8],
    column_widths=[0.7, 0.3],
    horizontal_spacing=0.08,
    vertical_spacing=0.10,
    specs=[
        [{"type": "xy"}, {"type": "domain"}],
        [{"type": "xy"}, {"type": "xy"}]
    ],
    shared_yaxes=True  # asegura que la segunda columna comparta eje Y con la primera
)

# --------------------------------------------
# 7) FILA 1, COL 1 (subplot x1/y1): Barra marginal
# --------------------------------------------
mask_full = utci[col_utci].notna() & utci[col_dem].notna()
valores_utci = utci.loc[mask_full, col_utci].values
valores_dem  = utci.loc[mask_full, col_dem].values

bins = 30
counts, bin_edges = np.histogram(
    valores_utci,
    bins=bins,
    weights=valores_dem,
    density=True
)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_width   = bin_edges[1] - bin_edges[0]

fig.add_trace(
    go.Bar(
        x=bin_centers,
        y=counts,
        width=bin_width,
        marker=dict(color='gray', opacity=0.32),
        name="Marginal UTCI×Demanda"
    ),
    row=1, col=1
)
fig.update_xaxes(visible=False, row=1, col=1)
fig.update_yaxes(visible=False, row=1, col=1)

# --------------------------------------------
# 8) FILA 2, COL 1 (subplot x3/y3): Scatter + líneas + rectángulos
# --------------------------------------------
for i, (abbr, xmin, xmax) in enumerate(stress_categories):
    color_fill = 'lightgrey' if (i % 2 == 0) else 'white'
    fig.add_shape(
        type='rect',
        x0=xmin, x1=xmax,
        y0=ymin, y1=ymax,
        xref='x3', yref='y3',
        fillcolor=color_fill,
        opacity=0.20,
        line_width=0,
        layer='below'
    )
    mid = xmin + 0.5 * (xmax - xmin)
    fig.add_annotation(
        x=mid,
        y=ymax * 1.02,
        xref='x3', yref='y3',
        text=abbr,
        showarrow=False,
        font=dict(size=9),
        xanchor='center',
        yanchor='bottom'
    )

for season_name, months in seasons.items():
    df_season = utci[utci.index.month.isin(months)].copy()

    # Scatter “raw”
    fig.add_trace(
        go.Scatter(
            x=df_season[col_utci],
            y=df_season[col_dem],
            mode='markers',
            marker=dict(size=4, opacity=0.20, color=season_colors[season_name]),
            name=f"{season_name} (raw)",
            legendgroup=season_name,
            showlegend=True
        ),
        row=2, col=1
    )

    # Promedio horario
    grp = df_season.groupby(df_season.index.hour)
    x_mean = grp[col_utci].mean()
    y_mean = grp[col_dem].mean()

    fig.add_trace(
        go.Scatter(
            x=x_mean,
            y=y_mean,
            mode='lines',
            line=dict(color=season_colors[season_name], width=2),
            name=f"{season_name} (avg)",
            legendgroup=season_name,
            showlegend=True
        ),
        row=2, col=1
    )

    # Marcadores de promedio (excepto horas 12 y 23)
    rest_hours = [h for h in x_mean.index if h not in {12, 23}]
    fig.add_trace(
        go.Scatter(
            x=x_mean.loc[rest_hours],
            y=y_mean.loc[rest_hours],
            mode='markers',
            marker=dict(size=4, color=season_colors[season_name], opacity=0.75),
            showlegend=False
        ),
        row=2, col=1
    )

    # Puntos para horas 12 y 23
    for hour, symbol in {12: 'circle', 23: 'square'}.items():
        if hour in x_mean.index:
            fig.add_trace(
                go.Scatter(
                    x=[x_mean.loc[hour]],
                    y=[y_mean.loc[hour]],
                    mode='markers',
                    marker=dict(symbol=symbol, size=8, color=season_colors[season_name]),
                    showlegend=False
                ),
                row=2, col=1
            )

fig.update_xaxes(
    title_text="UTCI",
    row=2, col=1,
    showgrid=True, gridwidth=0.3, gridcolor='lightgray'
)
fig.update_yaxes(
    title_text="Demanda [MWh]",
    row=2, col=1,
    showgrid=True, gridwidth=0.3, gridcolor='lightgray',
    tickformat=',d'
)
fig.add_annotation(
    x=0, y=1.08,
    xref='x3 domain', yref='y3 domain',
    text=reg,
    showarrow=False,
    font=dict(size=16, color='black'),
    xanchor='left',
    yanchor='bottom'
)

# --------------------------------------------
# 9) FILA 2, COL 2 (subplot x4/y4): Histograma de densidad de DEMANDA por temporada
#    (ahora comparte el mismo eje Y con el subplot de la izquierda)
# --------------------------------------------
# Calculamos la densidad máxima entre todas las temporadas para fijar el rango X
max_density = 0
for season_name, months in seasons.items():
    df_s = utci[utci.index.month.isin(months)].copy()
    if df_s[col_dem].dropna().empty:
        continue
    counts_s, _ = np.histogram(
        df_s[col_dem].dropna(),
        bins=30,
        density=True
    )
    max_density = max(max_density, counts_s.max())

# Dibujamos cada curva de densidad
for season_name, months in seasons.items():
    df_s = utci[utci.index.month.isin(months)].copy()
    if df_s[col_dem].dropna().empty:
        continue

    counts_s, bin_edges_s = np.histogram(
        df_s[col_dem].dropna(),
        bins=30,
        density=True
    )
    bin_centers_s = 0.5 * (bin_edges_s[:-1] + bin_edges_s[1:])

    fig.add_trace(
        go.Scatter(
            x=counts_s,
            y=bin_centers_s,
            mode='lines',
            line=dict(color=season_colors[season_name], width=1.5),
            fill='tozerox',
            opacity=0.45,
            name=f"{season_name} densidad"
        ),
        row=2, col=2
    )

# Ajustes de ejes en la columna derecha
fig.update_xaxes(
    range=[0, max_density * 1.1],  # un 10% extra para que no quede justo al borde
    visible=False,
    row=2, col=2
)
fig.update_yaxes(
    matches='y3', 
    title_text="Demanda [MWh]", 
    row=2, col=2
)

# --------------------------------------------
# 10) Layout final y mostrar
# --------------------------------------------
fig.update_layout(
    height=700,
    width=1100,
    margin=dict(l=50, r=20, t=40, b=40),
    showlegend=True
)

fig.show();


``` 