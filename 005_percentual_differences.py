# %% 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import Markdown
import pandas as pd
# %% 

f = 'data/demanda_utci_regions.parquet'
f2 = 'data/demanda_utci_ciudades.parquet'

utci = pd.read_parquet(f)
utci2 = pd.read_parquet(f2)

regions = ['BCS','PEN','BC','NO','N','CEN','NE','ORI','OCC']

# %% 

# --- 2) Define seasons ---
seasons = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11]
}
# %% 
stress_categories = [
    ("MC", -13,   0),   # Moderated cold
    ("SC",   0,    9),  # Slight cold
    ("No Thermal Stress",  9,   26),   # No thermal
    ("Moderate Heat",     26,   32),   # Moderate heat
    ("Strong Heat",       32,   38),   # Strong heat
    ("Very Strong Heat",  38,   46)    # Very strong heat
]
# %% 

def get_stress_abbr(val):
    for abbr, low, high in stress_categories:
        if low <= val < high:
            return abbr
    return None
# %% 

# 1) Calcula por región y temporada (en inglés)
results = []
for reg in regions:
    for season_name, months in seasons.items():
        df_season  = utci[utci.index.month.isin(months)]
        df_season2  = utci2[utci2.index.month.isin(months)]
        
        grp_utci   = df_season.groupby(df_season.index.hour)[f'{reg}_UTCI']
        grp_utci2   = df_season2.groupby(df_season2.index.hour)[f'{reg}_UTCI']

        grp_demand = df_season.groupby(df_season.index.hour)[f'{reg}_DEMANDA']

        if grp_utci.mean().empty and grp_utci2.mean().empty:
            continue

        h_max      = grp_utci.mean().idxmax()
        h_max2      = grp_utci2.mean().idxmax()


        utci_max   = grp_utci.mean().max()
        utci_max2   = grp_utci2.mean().max()


        dem_at_max = grp_demand.mean().loc[h_max]
        dem_at_max2 = grp_demand.mean().loc[h_max2]
        results.append({
            'Region':         reg,
            'Season_ES':      season_name,   # ya está en inglés: "Winter", "Spring", etc.

            'Max_UTCI_Region':       utci_max,
            'Max_UTCI_City':       utci_max2,

            'Average_Demand_Region': dem_at_max,       
            'Average_Demand_City': dem_at_max2,  
        })
# %% 

# 2) DataFrame y redondeo
df_max = pd.DataFrame(results).round(1)

# 3) Renombrar “Season_ES” a “Season” (sin traducir)
df_max.rename(columns={'Season_ES': 'Season'}, inplace=True)
# %% 

# 4) Asignar categoría de estrés térmico
df_max['Stress_Category_Region'] = df_max['Max_UTCI_Region'].apply(get_stress_abbr)
df_max['Stress_Category_City'] = df_max['Max_UTCI_City'].apply(get_stress_abbr)

# 5) Reordenar columnas
df_max = df_max[
    ['Region', 'Season','Average_Demand_Region', 'Max_UTCI_Region', 'Stress_Category_Region','Average_Demand_City', 'Max_UTCI_City', 'Stress_Category_City']
]

# %% 

# 1) (Optional) Rename columns if you want user‐friendly headers in English:
df_to_print = df_max.rename(columns={
    'Region':          'Region',
    'Season':          'Season',
    'Max_UTCI_Region': 'Max Max_UTCI_Region',
    'Max_UTCI_City': 'Max Max_UTCI City',
    'Stress_Category_Region': 'Thermal Stress Category Region',
    'Stress_Category_City': 'Thermal Stress Category City'
})
df_max

# %% 


# 1) Creamos dos diccionarios que guarden el valor de Winter para cada región
winter_utci_reg  = (
    df_max[df_max['Season']=='Winter']
      .set_index('Region')['Max_UTCI_Region']
      .to_dict()
)
winter_utci_city = (
    df_max[df_max['Season']=='Winter']
      .set_index('Region')['Max_UTCI_City']
      .to_dict()
)


winter_demand_reg = (
    df_max[df_max['Season']=='Winter']
      .set_index('Region')['Average_Demand_Region']
      .to_dict()
)

winter_demand_city = (
    df_max[df_max['Season']=='Winter']
      .set_index('Region')['Average_Demand_City']
      .to_dict()
)
winter_demand_city
# %% 

# 2) Definimos funciones que calculen el cambio relativo:
def rel_change_utci(row):
    base_reg = winter_utci_reg[row['Region']]
    base_city = winter_utci_city[row['Region']]

    return (row['Max_UTCI_Region'] - base_reg,row['Max_UTCI_City'] - base_city )

def rel_change_demand(row):
    base_region = winter_demand_reg[row['Region']]
    base_ciudad = winter_demand_city[row['Region']]
    return (((row['Average_Demand_Region'] - base_region) / base_region *100), ((row['Average_Demand_City'] - base_ciudad) / base_ciudad *100))


# 3) Añadimos columnas nuevas al DataFrame
df_max[['RelChange_Max_UTCI_Region', 'RelChange_Max_UTCI_City']] = (df_max.apply(rel_change_utci, axis=1, result_type='expand').round(1))
df_max[['RelChange_Demand_Region', 'RelChange_Demand_City']]= (df_max.apply(rel_change_demand, axis=1, result_type='expand').round(1))
df_max
# %% 

# 4) Para claridad, reordenamos columnas y mostramos resultado
cols = [
    'Region', 'Season', 'Max_UTCI_Region', 'Max_UTCI_City','RelChange_Max_UTCI_Region', 'RelChange_Max_UTCI_City','RelChange_Demand_Region',
    'RelChange_Demand_City','Stress_Category_Region', 'Stress_Category_City',
]
df_with_rel = df_max[cols]
# %%
df_with_rel
# %%
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
color_map = {
    'No Thermal Stress': '#F7F7F7',
    'Moderate Heat':     '#EF9B7A',
    'Strong Heat':       '#67001F',
    'Very Strong Heat':  'firebrick'
}
regions = df_with_rel['Region'].unique()

fig, axes = plt.subplots(
    nrows = len(regions),
    ncols = 2,
    figsize=(12, 6*len(regions))
)
fig.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.15)

for i, reg in enumerate(regions):
    df_reg = df_with_rel[df_with_rel['Region'] == reg]
    
    ax = axes[i, 0]
    utci_vals   = [df_reg.loc[df_reg['Season']==s, 'Max_UTCI_Region'].iat[0] for s in seasons]
    pct_demand  = [df_reg.loc[df_reg['Season']==s, 'RelChange_Demand_Region'].iat[0] for s in seasons]
    cats_utci   = [df_reg.loc[df_reg['Season']==s, 'Stress_Category_Region'].iat[0] for s in seasons]
    
    ax.plot(seasons, pct_demand, linestyle='--', color='gray', alpha=0.7)
    ax.axhline(0, color='black', linestyle=':', linewidth=0.8)
    for j, s in enumerate(seasons):
        ax.scatter(
            s, pct_demand[j],
            color=color_map[cats_utci[j]],
            s=130, edgecolor='k', zorder=3
        )
        ax.text(
            s, pct_demand[j] + 2,
            f"{utci_vals[j]:.1f}°C",
            ha='center', va='bottom', fontsize=14,
            bbox=dict(facecolor='white', edgecolor='none', pad=1)
        )
    ax.set_ylim(-2, 105)
    ax.set_xlim(-0.3, 3.3)
    ax.set_title(reg, fontsize=16, fontweight='bold')
    if 0 == 0:  # siempre para la columna 0
        ax.set_ylabel('Electricity demand differences (%)', fontsize=14)
    if i == len(regions)-1:
        ax.set_xticklabels(seasons, fontsize=15)
    else:
        ax.set_xticklabels([])

    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=15)
    
    ax = axes[i, 1]
    utci_vals   = [df_reg.loc[df_reg['Season']==s, 'Max_UTCI_City'].iat[0] for s in seasons]
    pct_demand  = [df_reg.loc[df_reg['Season']==s, 'RelChange_Demand_City'].iat[0] for s in seasons]
    cats_utci   = [df_reg.loc[df_reg['Season']==s, 'Stress_Category_City'].iat[0] for s in seasons]
    
    ax.plot(seasons, pct_demand, linestyle='--', color='gray', alpha=0.7)
    ax.axhline(0, color='black', linestyle=':', linewidth=0.8)
    for j, s in enumerate(seasons):
        ax.scatter(
            s, pct_demand[j],
            color=color_map[cats_utci[j]],
            s=130, edgecolor='k', zorder=3
        )
        ax.text(
            s, pct_demand[j] + 2,
            f"{utci_vals[j]:.1f}°C",
            ha='center', va='bottom', fontsize=14,
            bbox=dict(facecolor='white', edgecolor='none', pad=1)
        )
    ax.set_ylim(-2, 105)
    ax.set_xlim(-0.3, 3.3)
    ax.set_title(reg, fontsize=16, fontweight='bold')
    ax.set_yticklabels([])
    if i == len(regions)-1:
        ax.set_xticklabels(seasons, fontsize=15)
    else:
        ax.set_xticklabels([])

    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=15)

#leyenda abajo
handles = [
    Line2D([0], [0], marker='o', color='w', label=cat,
           markerfacecolor=color_map[cat], markersize=10,
           markeredgecolor='k')
    for cat in color_map
]
fig.legend(
    handles=handles,
    loc='lower center',
    ncol=4,
    frameon=False,
    fontsize=16,
    bbox_to_anchor=(0.5, 0.03),
    title='UTCI Category',
    title_fontsize=17,
)

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.show()
# %%
