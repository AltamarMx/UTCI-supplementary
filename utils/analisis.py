import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

def compare_utci(region,
                 region_file='data/demanda_utci_regiones.parquet',
                 city_file  ='data/demanda_utci_ciudades.parquet'):
    """
    Compara para la región dada:
      - IZQ: datos de region_file
      - DER: datos de city_file
    Ambos deben tener columnas:
        f"{region}_UTCI" y f"{region}_DEMANDA"
    """
    # 1) Configuración común
    stress_cats = [
        ("MC", -13,  0),
        ("SC",   0,  9),
        ("NT",   9, 26),
        ("MH",  26, 32),
        ("SH",  32, 38),
        ("VSH",38, 46),
    ]
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11],
    }

    # 2) Leo ficheros y aseguro DateTimeIndex
    df_reg  = pd.read_parquet(region_file)
    df_city = pd.read_parquet(city_file)
    for df in (df_reg, df_city):
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    # 3) Extraigo las series de UTCI y DEMANDA, quitando NaNs
    s_utci_reg = df_reg[f"{region}_UTCI"].dropna()
    s_dem_reg  = df_reg[f"{region}_DEMANDA"].dropna()
    s_utci_city= df_city[f"{region}_UTCI"].dropna()
    s_dem_city = df_city[f"{region}_DEMANDA"].dropna()

    # 4) Preparo figura con dos ejes
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14, 6), sharex=True, sharey=True
    )
    panels = [
        (axL, s_utci_reg,  s_dem_reg,  f"Region"),
        (axR, s_utci_city, s_dem_city, f"City")
    ]

    for ax, s_utci, s_dem, title in panels:
        # 4a) sombreado de categorías
        for i, (abbr, x0, x1) in enumerate(stress_cats):
            ax.axvspan(x0, x1,
                       color='lightgrey' if i%2==0 else 'white',
                       alpha=0.2, zorder=0)
            mid = x0 + (x1-x0)/2
            ax.text(mid, 0.92, abbr,
                    transform=ax.get_xaxis_transform(),
                    ha='left', va='bottom',
                    fontsize=8, clip_on=False, zorder=5)

        # 4b) scatter + medias por estación
        colors = {}
        df = pd.DataFrame({'UTCI': s_utci, 'DEMANDA': s_dem})
        for season, meses in seasons.items():
            sub = df[df.index.month.isin(meses)]
            # puntos crudos
            ax.scatter(sub['UTCI'], sub['DEMANDA'],
                       s=1, alpha=0.05, marker='.', zorder=1)
            # medias horarias
            grp = sub.groupby(sub.index.hour)
            x = grp['UTCI'].mean()
            y = grp['DEMANDA'].mean()
            line, = ax.plot(x, y, '-', label=season, zorder=2)
            colors[season] = line.get_color()
            # marcadores
            special = {12:'o', 23:'s'}
            rest = [h for h in x.index if h not in special]
            ax.scatter(x.loc[rest], y.loc[rest],
                       marker='.', color=line.get_color(),
                       alpha=0.7, zorder=3)
            for h, m in special.items():
                if h in x.index:
                    ax.scatter(x.loc[h], y.loc[h],
                               marker=m, s=40,
                               color=line.get_color(),
                               zorder=4)

        # 4c) ejes, formato y título
        ax.set_title(title, loc='center', fontsize=12, pad=6)
        ax.set_xlabel("UTCI")
        ax.set_ylabel("Demand [MWh]")
        ax.grid(alpha=0.3, zorder=5)
        ax.yaxis.set_major_formatter(
            mticker.StrMethodFormatter("{x:,.0f}")
        )

        # 4d) histogramas marginales
        div = make_axes_locatable(ax)
        # derecha: demanda
        ax_hist_y = div.append_axes("right", size="15%", pad=0)
        for season, meses in seasons.items():
            sel = df[df.index.month.isin(meses)]['DEMANDA']
            ax_hist_y.hist(sel, orientation='horizontal',
                           bins=30, density=True,
                           color=colors[season], alpha=0.4)
        ax_hist_y.axis('off')

        # arriba: UTCI ponderado por demanda
        ax_hist_x = div.append_axes("top", size="15%", pad=0.1)
        mask = df['UTCI'].notna() & df['DEMANDA'].notna()
        ax_hist_x.hist(
            df.loc[mask, 'UTCI'],
            bins=30,
            weights=df.loc[mask, 'DEMANDA'],
            density=True,
            color='gray', alpha=0.3
        )
        ax_hist_x.set_xlim(ax.get_xlim())
        ax_hist_x.axis('off')
        ax.set_zorder(ax_hist_x.get_zorder() + 1)

    # 5) leyenda global
    handles, labels = axL.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(), by_label.keys(),
        loc='lower center', ncol=len(seasons),
        frameon=False, bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()

def analyze(region, data_file, title):
    """
    Analiza y grafica la relación entre UTCI y demanda para una sola fuente de datos.

    Parámetros:
      - region: nombre de la región (ej. 'BC', 'CEN', etc.)
      - data_file: ruta al archivo parquet con datos (de regiones o de ciudades)
      - title: texto para el título del gráfico (ej. 'Region' o 'City')
    """
    # 1) Configuración común
    stress_cats = [
        ("MC", -13,  0),
        ("SC",   0,  9),
        ("NT",   9, 26),
        ("MH",  26, 32),
        ("SH",  32, 38),
        ("VSH", 38, 46),
    ]
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11],
    }

    # 2) Leer archivo y asegurar DateTimeIndex
    df = pd.read_parquet(data_file)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 3) Extraer series y DataFrame base
    s_utci = df[f"{region}_UTCI"].dropna()
    s_dem  = df[f"{region}_DEMANDA"].dropna()
    df_panel = pd.DataFrame({'UTCI': s_utci, 'DEMANDA': s_dem})

    # 4) Crear figura
    fig, ax = plt.subplots(figsize=(5, 5))

    # 4a) sombreado de categorías y etiquetas
    for i, (abbr, x0, x1) in enumerate(stress_cats):
        ax.axvspan(x0, x1,
                   color='lightgrey' if i%2==0 else 'white',
                   alpha=0.2, zorder=0)
        mid = x0 + (x1 - x0) / 2
        ax.text(mid, 0.92, abbr,
                transform=ax.get_xaxis_transform(),
                ha='left', va='bottom', fontsize=8,
                clip_on=False, zorder=5)

    # 4b) scatter y medias por estación
    colors = {}
    for season, meses in seasons.items():
        sub = df_panel[df_panel.index.month.isin(meses)]
        # puntos crudos
        ax.scatter(sub['UTCI'], sub['DEMANDA'],
                   s=1, alpha=0.05, marker='.', zorder=1)
        # medias horarias
        grp = sub.groupby(sub.index.hour)
        x = grp['UTCI'].mean()
        y = grp['DEMANDA'].mean()
        line, = ax.plot(x, y, '-', label=season, zorder=2)
        colors[season] = line.get_color()
        # marcadores especiales
        special = {12: 'o', 23: 's'}
        rest = [h for h in x.index if h not in special]
        ax.scatter(x.loc[rest], y.loc[rest],
                   marker='.', color=line.get_color(),
                   alpha=0.7, zorder=3)
        for h, m in special.items():
            if h in x.index:
                ax.scatter(x.loc[h], y.loc[h],
                           marker=m, s=40,
                           color=line.get_color(), zorder=4)

    # 4c) ejes, formato y título
    ax.set_title(title, loc='center', fontsize=12, pad=6)
    ax.set_xlabel("UTCI")
    ax.set_ylabel("Demand [MWh]")
    ax.grid(alpha=0.3, zorder=5)
    ax.yaxis.set_major_formatter(
        mticker.StrMethodFormatter("{x:,.0f}")
    )

    # 4d) histogramas marginales
    div = make_axes_locatable(ax)
    # demanda a la derecha
    ax_hist_y = div.append_axes("right", size="15%", pad=0)
    for season, meses in seasons.items():
        sel = df_panel[df_panel.index.month.isin(meses)]['DEMANDA']
        ax_hist_y.hist(sel, orientation='horizontal',
                       bins=30, density=True,
                       color=colors[season], alpha=0.4)
    ax_hist_y.axis('off')

    # UTCI ponderado arriba
    ax_hist_x = div.append_axes("top", size="15%", pad=0.1)
    mask = df_panel['UTCI'].notna() & df_panel['DEMANDA'].notna()
    ax_hist_x.hist(
        df_panel.loc[mask, 'UTCI'],
        bins=30,
        weights=df_panel.loc[mask, 'DEMANDA'],
        density=True,
        color='gray', alpha=0.3
    )
    ax_hist_x.set_xlim(ax.get_xlim())
    ax_hist_x.axis('off')
    ax.set_zorder(ax_hist_x.get_zorder() + 1)

    plt.tight_layout()
    plt.show()



# Definiciones globales
seasons = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11]
}
stress_categories = [
    ('No Thermal Stress', 9, 26),
    ('Moderate Heat',     26, 32),
    ('Strong Heat',       32, 38),
    ('Very Strong Heat',  38, 46)
]
# color asociado a las categorías
color_map = {
    'No Thermal Stress': '#F7F7F7',
    'Moderate Heat':     '#EF9B7A',
    'Strong Heat':       '#67001F',
    'Very Strong Heat':  'firebrick'
}

def get_stress_category(val):
    for abbr, low, high in stress_categories:
        if low <= val < high:
            return abbr
    return 'Unknown'


def plot_demand_increase(data_file: str,
                         region: str,
                         kind: str = 'Region',
                         figsize: tuple = (5, 5)) -> plt.Figure:
    """
    Grafica el incremento de demanda (%) respecto a Winter para una región o ciudad,
    anotando el valor de UTCI (°C) y coloreando según la categoría de estrés térmico.
    """
    # 1) Leer datos
    df = pd.read_parquet(data_file)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 2) Calcular estadísticas por temporada
    records = []
    utci_col = f"{region}_UTCI"
    dem_col  = f"{region}_DEMANDA"
    for season, months in seasons.items():
        sub = df[df.index.month.isin(months)]
        if utci_col not in sub.columns or dem_col not in sub.columns:
            continue
        # media horaria de UTCI y demanda
        grp = sub.groupby(sub.index.hour)
        utci_mean = grp[utci_col].mean()
        dem_mean  = grp[dem_col].mean()
        if utci_mean.empty:
            continue
        # hora de max UTCI y sus valores
        h_max    = utci_mean.idxmax()
        utci_max = round(utci_mean.loc[h_max], 1)
        dem_at   = round(dem_mean.loc[h_max], 1)
        # categoría de estrés
        cat = get_stress_category(utci_max)
        records.append({
            'Season': season,
            'Avg_Demand': dem_at,
            'UTCI': utci_max,
            'Stress': cat
        })
    df_stats = pd.DataFrame(records).set_index('Season')

    # 3) Calcular cambio relativo vs Winter
    if 'Winter' not in df_stats.index:
        raise ValueError(f"Falta datos de Winter para {region}")
    base_dem = df_stats.at['Winter', 'Avg_Demand']
    df_stats['Δ_Demand'] = ((df_stats['Avg_Demand'] - base_dem) / base_dem * 100).round(1)
    df_stats = df_stats.reindex(['Winter','Spring','Summer','Autumn'])

    # 4) Graficar
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df_stats.index, df_stats['Δ_Demand'], linestyle='--', color='gray', alpha=0.7)
    ax.axhline(0, linestyle=':', color='black', linewidth=0.8)

    # puntos con color y anotaciones
    for season in df_stats.index:
        y = df_stats.at[season, 'Δ_Demand']
        ut = df_stats.at[season, 'UTCI']
        cat = df_stats.at[season, 'Stress']
        ax.scatter(season, y,
                   color=color_map.get(cat, 'gray'),
                   s=100, edgecolor='k', zorder=3)
        ax.text(season, y + 2,
                f"{ut:.1f}°C",
                ha='center', va='bottom', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='none', pad=1))

    ax.set_title(f"{kind}", fontsize=14)
    ax.set_ylabel('Δ Demand (%)')
    ax.set_xlabel('Season')
    ax.set_ylim(0,100)
    ax.grid(alpha=0.3)

    # leyenda de estrés
    handles = [Line2D([0],[0], marker='o', color='w', label=label,
                      markerfacecolor=color, markersize=10,
                      markeredgecolor='k')
               for label, color in color_map.items()]
    fig.legend(handles=handles,
               loc='lower center', ncol=len(color_map),
               frameon=False, bbox_to_anchor=(0.5, -0.15),
            #    title='UTCI Stress Category'
               )

    plt.tight_layout(rect=[0,0.05,1,1])
    plt.show()
