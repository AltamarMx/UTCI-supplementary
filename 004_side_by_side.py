# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 1) Cargar DataFrames desde Parquet
ciudades = pd.read_parquet("data/demanda_utci_ciudades.parquet")
regiones = pd.read_parquet("data/demanda_utci_regions.parquet")

# 2) Definir categorías de estrés UTCI
stress_categories = [
    ("MC", -13,   0),    # Moderated cold
    ("SC",   0,   9),    # Slight cold
    ("NT",   9,  26),    # No thermal
    ("MH",  26,  32),    # Moderated heat
    ("SH",  32,  38),    # Strong heat
    ("VSH", 38,  46),    # Very strong heat
    # ("EH", 46, np.inf), # (opcional: Extreme heat)
]

# 3) Definir estaciones
seasons = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11]
}

# 4) Lista de regiones que existen en ambos DataFrames
regions = ['BCS', 'PEN', 'BC', 'NO', 'N', 'CEN', 'NE', 'ORI', 'OCC']

# 5) Crear figura con 9 filas y 2 columnas
fig, axes = plt.subplots(
    nrows=9, ncols=2,
    figsize=(14, 36),       # ajusta el alto porque son 9 filas
    sharex=False, sharey=False,
    constrained_layout=False
)

# 6) Función auxiliar que dibuja sobre un eje dado
def plot_utci_demanda(ax, df, reg_abbr):
    """
    Dibuja en 'ax' el scatter y medias para la región 'reg_abbr' sobre el DataFrame 'df'.
    También agrega histograma marginal derecho (demanda) y superior (UTCI ponderado).
    """
    # ————— a) Fondo de bandas de categorías UTCI —————
    for i, (abbr, xmin, xmax) in enumerate(stress_categories):
        ax.axvspan(
            xmin, xmax,
            color='lightgrey' if (i % 2 == 0) else 'white',
            alpha=0.2, linewidth=0, zorder=0
        )
        # Etiqueta de la categoría en la parte superior del recuadro
        x0, x1 = ax.get_xlim()
        xmin_plot = xmin if np.isfinite(xmin) else x0
        xmax_plot = xmax if np.isfinite(xmax) else x1
        mid = xmin_plot + (xmax_plot - xmin_plot) / 2
        ax.text(
            mid, 0.92, abbr,
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom',
            fontsize=8, clip_on=False, zorder=5
        )

    # ————— b) Scatter + curvas de promedio horario + marcadores especiales —————
    season_colors = {}
    for season_name, meses in seasons.items():
        df_season = df[df.index.month.isin(meses)]

        # 1) Scatter de puntos brutos (hourly raw)
        ax.scatter(
            df_season[f'{reg_abbr}_UTCI'],
            df_season[f'{reg_abbr}_DEMANDA'],
            alpha=0.03, marker='.', s=1, zorder=1
        )

        # 2) Cálculo de promedios por hora
        grp = df_season.groupby(df_season.index.hour)
        x_hour = grp[f'{reg_abbr}_UTCI'].mean()      # UTCI promedio por hora
        y_hour = grp[f'{reg_abbr}_DEMANDA'].mean()   # Demanda promedio por hora

        # 3) Dibujar curva de promedio horario
        line, = ax.plot(
            x_hour, y_hour, '-', label=season_name, zorder=2
        )
        season_colors[season_name] = line.get_color()

        # 4) Marcadores intermedios (todos excepto horas especiales)
        special = {12: 'o', 23: 's'}  # marcadores para hora 12 y 23
        rest_hours = [h for h in x_hour.index if h not in special]
        ax.scatter(
            x_hour.loc[rest_hours],
            y_hour.loc[rest_hours],
            marker='.', color=line.get_color(),
            alpha=0.7, zorder=3
        )

        # 5) Marcadores “especiales” (sin etiqueta de leyenda adicional)
        for h, marker_shape in special.items():
            if h in x_hour.index:
                ax.scatter(
                    x_hour.loc[h],
                    y_hour.loc[h],
                    marker=marker_shape,
                    color=line.get_color(),
                    s=40,
                    label=None,
                    zorder=4
                )

    # ————— c) Ajustes generales del subplot —————
    ax.set_title(reg_abbr, loc='left', fontsize=12, fontweight='bold', pad=4)
    ax.set_ylabel("Demanda [$MWh$]")
    ax.grid(alpha=0.3, zorder=5)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    # Etiqueta X sólo si es fila inferior (se pone más adelante)
    # ax.set_xlabel("UTCI")

    # ————— d) Histograma marginal derecho: distribución de demanda —————
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="15%", pad=0)
    for season_name in seasons:
        df_s = df[df.index.month.isin(seasons[season_name])]
        cax.hist(
            df_s[f'{reg_abbr}_DEMANDA'],
            orientation='horizontal',
            bins=30, density=True,
            color=season_colors[season_name],
            alpha=0.4
        )
    cax.axis('off')

    # ————— e) Histograma marginal superior: UTCI ponderado por demanda —————
    top_ax = divider.append_axes("top", size="15%", pad=0.1)
    s_utci = df[f'{reg_abbr}_UTCI']
    s_dem = df[f'{reg_abbr}_DEMANDA']
    mask = s_utci.notna() & s_dem.notna()
    top_ax.hist(
        s_utci[mask],
        bins=30,
        weights=s_dem[mask],
        density=True,
        color='gray',
        alpha=0.3
    )
    # Asegurar que el histograma superior comparta el mismo rango X que el eje principal
    top_ax.set_xlim(ax.get_xlim())
    top_ax.axis('off')
    # Elevar el zorder del eje principal para que quede encima de top_ax
    ax.set_zorder(top_ax.get_zorder() + 1)


# 7) Bucle para cada región: dibujar en columna 0 (ciudades) y columna 1 (regiones)
for i, reg_abbr in enumerate(regions):
    ax_city = axes[i, 0]
    ax_reg  = axes[i, 1]

    # a) Dibujar comparación para “ciudades”
    plot_utci_demanda(ax_city, ciudades, reg_abbr)

    # b) Dibujar comparación para “regiones”
    plot_utci_demanda(ax_reg, regiones, reg_abbr)

    # c) Etiquetar eje X solamente en la fila inferior (i == 8)
    if i == len(regions) - 1:
        ax_city.set_xlabel("UTCI")
        ax_reg.set_xlabel("UTCI")

# 8) Ajustar márgenes y colapsar leyenda global
plt.subplots_adjust(
    left=0.05, right=0.93,
    top=0.95, bottom=0.05,
    hspace=0.45, wspace=0.35
)

# 9) Crear una única leyenda global al fondo (con los cuatro nombres de estación)
handles, labels = axes[0, 0].get_legend_handles_labels()
unique = dict(zip(labels, handles))
fig.legend(
    unique.values(), unique.keys(),
    loc='lower center',
    ncol=len(seasons),
    frameon=False,
    bbox_to_anchor=(0.5, 0.00)
)

plt.savefig('side_by_side.png')
plt.show()

# %%
