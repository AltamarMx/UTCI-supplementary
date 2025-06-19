# %% 
import pandas as pd
import plotly.express as px  
# %%
f = './data/001_raw/central.csv'
cen = pd.read_csv(f,index_col=0,parse_dates=True)
cen.info()
# %%
f = './data/001_raw/pacifico.csv'
pacifico = pd.read_csv(f,index_col=0,parse_dates=True)
pacifico
# %%
ciudades = pd.concat([cen,pacifico],axis=1)
ciudades
# %%
f = './data/demanda_utci_regions.parquet'
regions = pd.read_parquet(f)
columnas = regions.columns.to_list()
columnas = [columna.upper() for columna in columnas]
regions.columns = columnas
cen_rename = {'CENTRAL_DEMANDA':'CEN_DEMANDA',
              'CENTRAL_UTCI':'CEN_UTCI'}
regions.rename(columns=cen_rename,inplace=True)
regions.to_parquet(f)
# %%
ciudades = pd.concat([cen,pacifico],axis=1)
# ciudades = ciudades.reset_index()
px.line(
    data_frame=ciudades.reset_index(),
    x = 'time',
    y = ['PEN_UTCI','ORI_UTCI','BCS_UTCI']
)

# %%
regions.columns
# %%
ciudades.columns
# %%
demanda_cols = [columna for columna in regions.columns if 'DEMANDA' in columna]
demanda_cols

# %%
ciudades_utci_demanda = pd.concat([regions[demanda_cols],ciudades],axis=1)

# %%
ciudades_utci_demanda.to_parquet('data/demanda_utci_ciudades.parquet')
# %%
