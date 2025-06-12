# %%  
from utils.analisis import compare_utci, analyze

# %%
compare_utci(
    region='BCS',
    region_file='./data/demanda_utci_regions.parquet',
    city_file='./data/demanda_utci_ciudades.parquet'
)

# %%
analyze('BCS', './data/demanda_utci_ciudades.parquet', 'City')


# %%
