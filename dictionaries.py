# python dictionaries to be used in python/functions.py

# define the base directory where the data is stored
base_dir = "/home/users/benhutch/skill-maps-processed-data"

base_dir_historical = "/home/users/benhutch/skill-maps-processed-data/historical"

# define the directory where the plots will be saved
plots_dir = base_dir + "/plots"

gif_plots_dir = base_dir + "/plots/gif"

# list of the test model
test_model = ["CMCC-CM2-SR5"]
test_model_bcc = ["BCC-CSM2-MR"]
test_model2 = ["EC-Earth3"]
test_model_norcpm = ["NorCPM1"]
test_model_hadgem = ["HadGEM3-GC31-MM"]
test_model_cesm = ["CESM1-1-CAM5-CMIP5"]


# List of the full models
models = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "MPI-ESM1-2-LR",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

# List of the test model
test_model = ["MIROC6"]

# define the paths for the observations
obs_psl = "/home/users/benhutch/ERA5_psl/long-ERA5-full.nc"

# For the north atlantic region
obs_psl_na = "/home/users/benhutch/ERA5_psl/long-ERA5-full-north-atlantic.nc"

# Global
obs_psl_glob = "/home/users/benhutch/ERA5_psl/long-ERA5-full-global.nc"

# the variable has to be extracted from these
obs_tas = "/home/users/benhutch/ERA5/adaptor.mars.internal-1687448519.6842003-11056-8-3ea80a0a-4964-4995-bc42-7510a92e907b.nc"
obs_sfcWind = "/home/users/benhutch/ERA5/adaptor.mars.internal-1687448519.6842003-11056-8-3ea80a0a-4964-4995-bc42-7510a92e907b.nc"
#

paper_1_variables = ["tas", "sfcWind", "rsds", "psl"]

obs_rsds = "not/yet/implemented"

obs = "/home/users/benhutch/ERA5/adaptor.mars.internal-1691509121.3261805-29348-4-3a487c76-fc7b-421f-b5be-7436e2eb78d7.nc"

gridspec_global = "/home/users/benhutch/gridspec/gridspec-global.txt"

gridspec_north_atlantic = "/home/users/benhutch/gridspec/gridspec-north-atlantic.txt"

obs_regrid = "/home/users/benhutch/ERA5/ERA5_full_global.nc"

# Define the labels for the plots - wind
sfc_wind_label = "10-metre wind speed"
sfc_wind_units = "m s\u207b\u00b9"

# Define the labels for the plots - temperature
tas_label = "2-metre temperature"
tas_units = "K"

psl_label = "Sea level pressure"
psl_units = "hPa"

rsds_label = "Surface solar radiation downwards"
rsds_units = "W m\u207b\u00b2"

canari_base_dir = "/gws/nopw/j04/canari/users/benhutch"
canari_plots_dir = "/gws/nopw/j04/canari/users/benhutch/plots"
scratch_base_dir = "/work/scratch-nopw2/benhutch"
home_dir = "/home/users/benhutch"

tas_models_noEC_historical = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "MPI-ESM1-2-LR",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "NorCPM1",
]

rsds_models_historical = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "MPI-ESM1-2-LR",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "NorCPM1",
]

sfcWind_models_historical = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "MPI-ESM1-2-LR",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
]

psl_models_historical = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "MPI-ESM1-2-LR",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "NorCPM1",
]

historical_models_ua = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "MPI-ESM1-2-LR",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "NorCPM1",
]

# Define a dictionary to map the season strings to their corresponding months
season_month_map = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "MAY": [3, 4, 5],
    "JJA": [6, 7, 8],
    "ULG": [6, 7, 8],
    "JJAS": [6, 7, 8, 9],
    "SON": [9, 10, 11],
    "SOND": [9, 10, 11, 12],
    "NDJF": [11, 12, 1, 2],
    "DJFM": [12, 1, 2, 3],
    "djfm": [12, 1, 2, 3],
    "ONDJFM": [10, 11, 12, 1, 2, 3],
    "ondjfm": [10, 11, 12, 1, 2, 3],
    "JFMA": [1, 2, 3, 4],
    "AYULGS": [4, 5, 6, 7, 8, 9],
    "AMJJAS": [4, 5, 6, 7, 8, 9],
}

pr_models = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "MPI-ESM1-2-LR",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "NorCPM1",
]

# Observations for pr
# Convert to .nc first
obs_pr = "/gws/nopw/j04/canari/users/benhutch/ERA5/adaptor.mars.internal-1708982310.3666325-24382-12-6f0e95b3-7128-4ad4-8bc6-416040163640.nc"

# Define a dictionary to set up the matching historical models
# for the different variables
historical_models_map = {
    "psl": psl_models_historical,
    "tas": tas_models_noEC_historical,
    "sfcWind": sfcWind_models_historical,
    "rsds": rsds_models_historical,
    "ua": historical_models_ua,
    "pr": pr_models,
}


save_dir = "/gws/nopw/j04/canari/users/benhutch/NAO-matching"

obs_ws_var_names = ["ua", "va", "var131", "var132", "pr", "var228", "var246", "var247", "u100", "v100"]

# Define a dictionary to map the variable names to their corresponding names in the obs dataset
var_name_map = {
    "psl": "msl",
    "msl": "msl",
    "tas": "t2m",
    "t2m": "t2m",
    "sfcWind": "si10",
    "si10": "si10",
    "rsds": "ssrd",
    "ssrd": "ssrd",
    "tos": "sst",
    "sst": "sst",
    "ua": "var131",
    "va": "var132",
    "var131": "var131",
    "var132": "var132",
    "pr": "var228",
    "var228": "var228",
    "u100": "var246",
    "v100": "var247",
    "var246": "var246",
    "var247": "var247",
}

season_map = {"1": "DJFM", "2": "MAM", "3": "JJA", "4": "SON"}

# Define the dimensions for the grids
# for processing the observations
north_atlantic_grid = {"lon1": 280, "lon2": 37.5, "lat1": 77.5, "lat2": 20}

# define the NA grid for obs
north_atlantic_grid_obs = {"lon1": 100, "lon2": 217.5, "lat1": 77.5, "lat2": 20}

# Define the dimensions for the gridbox for the azores
azores_grid = {"lon1": 152, "lon2": 160, "lat1": 36, "lat2": 40}

# Define the dimensions for the gridbox for the azores
azores_grid_corrected = {"lon1": -28, "lon2": -20, "lat1": 36, "lat2": 40}

# Define the dimensions for the gridbox for iceland
iceland_grid = {"lon1": 155, "lon2": 164, "lat1": 63, "lat2": 70}

# Define the dimensions for the gridbox for the azores
iceland_grid_corrected = {"lon1": -25, "lon2": -16, "lat1": 63, "lat2": 70}

# Set up a northern europe grid box
northern_europe_grid = {"lon1": -10, "lon2": 25, "lat1": 55, "lat2": 70}

# SPNA grid box
# 49 - 57 * N
# 50 - 25 * W
spna_grid_strommen = {"lon1": -50, "lon2": -25, "lat1": 49, "lat2": 57}

# Create a dictionary containing both azores grid corrected
# and iceland grid corrected
nao_skill_grid = {"south": azores_grid_corrected, "north": iceland_grid_corrected}
# Define the dimensions for the summertime NAO (SNAO) southern pole
# As defined in Wang and Ting 2022
# This is the pointwise definition of the SNAO
# Which is well correlated with the EOF definition from Folland et al. 2009
snao_south_grid = {
    "lon1": -25,  # degrees west
    "lon2": 5,  # degrees east
    "lat1": 45,
    "lat2": 55,
}

# Define the dimensions for the summertime NAO (SNAO) northern pole
# As defined in Wang and Ting 2022
snao_north_grid = {
    "lon1": -52,  # degrees west
    "lon2": -22,  # degrees west
    "lat1": 60,
    "lat2": 70,
}

# Define the summer NAO skill grid
snao_skill_grid = {"south": snao_south_grid, "north": snao_north_grid}

# Define the dimensions for the gridbox for the North Sea Region
north_sea_grid = {"lon1": -2, "lon2": 8, "lat1": 50, "lat2": 65}

# Define the dimensions for the gridbox for Central Europe
central_europe_grid = {"lon1": -10, "lon2": 25, "lat1": 36, "lat2": 65}

# define the uk grid from Clarke et al. 2017
uk_grid = {"lon1": -10, "lon2": 3, "lat1": 50, "lat2": 60}

north_sea_kay = {
    "lon1": 1,  # degrees east
    "lon2": 7,
    "lat1": 53,  # degrees north
    "lat2": 59,
}

# define the Norway grid
norway_grid = {"lon1": 4, "lon2": 16, "lat1": 58, "lat2": 71}

# define a grid to plot for the north atlantic
north_atlantic_grid_plot = {"lon1": -70, "lon2": 40, "lat1": 20, "lat2": 90}

# Wider pacific grid for the plot
pacific_grid_plot = {"lon1": -180, "lon2": 40, "lat1": 10, "lat2": 90}

# Test pacific grid for the plot
pacific_grid_plot_test = {"lon1": 180, "lon2": 360, "lat1": 10, "lat2": 90}

# Set up a grid centred over the Rocky Mountains
rocky_mountain_grid = {"lon1": -120, "lon2": -90, "lat1": 30, "lat2": 50}

# Define a grid to plot for China
china_grid_plot = {"lon1": 75, "lon2": 130, "lat1": 15, "lat2": 55}

# Define a smaller gridbox within the bounds of china_grid_plot
eastern_china_gridbox = {"lon1": 108, "lon2": 122, "lat1": 22, "lat2": 36}

# Define the dimensions for another central europe grid for RSDS correlations
central_europe_grid_rsds = {"lon1": 165, "lon2": 220, "lat1": 30, "lat2": 70}

# Define a gribox for the SPNA region as in Smith et al. 2019
spna_grid_smith = {"lon1": -60, "lon2": -10, "lat1": 50, "lat2": 65}


# Define the dimensions for the gridbox for the N-S UK index
# From thornton et al. 2019
uk_n_box = {"lon1": 153, "lon2": 201, "lat1": 57, "lat2": 70}

# Define this but corrected
uk_n_box_corrected = {"lon1": -27, "lon2": 21, "lat1": 57, "lat2": 70}

# And for the southern box
uk_s_box = {"lon1": 153, "lon2": 201, "lat1": 38, "lat2": 51}

# Define this but corrected
uk_s_box_corrected = {"lon1": -27, "lon2": 21, "lat1": 38, "lat2": 51}

# Define the scandi gridbox to be used for precip
# From Landgren et al., 2014
scandi_box = {
    "lon1": 2,  # degrees east
    "lon2": 23,
    "lat1": 56,  # degrees north
    "lat2": 71,
}

# Define the mediterranean box for ssrd
# from (Calì Quaglia et al., 2022)
med_box = {
    "lon1": -11,  # degrees east
    "lon2": 37,
    "lat1": 31,  # degrees north
    "lat2": 52,
}

# Focus this box on spain, france and italy
# loosely based on (Calì Quaglia et al., 2022)
med_box_focus = {
    "lon1": -11,  # degrees east
    "lon2": 2,
    "lat1": 35,  # degrees north
    "lat2": 45,
}

sfcWind_models_numbers = [1, 2, 3, 5, 8, 9, 10, 11]
no_sfcWind_models = 8
sfcWind_models = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "HadGEM3-GC31-MM",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
]
sfcWind_models_noMIROC = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "HadGEM3-GC31-MM",
    "FGOALS-f3-L",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
]

rsds_models_numbers = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
no_rsds_models = 11
rsds_models = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

rsds_models_noCMCC = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

common_models_noCMCC = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

tas_models_numbers = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
no_tas_models = 11
tas_models = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

psl_models_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
no_psl_models = 12
psl_models = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "MPI-ESM1-2-LR",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

# psl models no CanESM5
psl_models_noCan = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "MPI-ESM1-2-LR",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

# Psl models no CanESM5
psl_models_noCan_noEC_noMPI_LR = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

# common models for different variables (not sfcWind)
common_models = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

# Common models no IPSL, no CANESM5
common_models_noIPSL_noCan = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "FGOALS-f3-L",
    "MIROC6",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

common_models_noCMCC = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "HadGEM3-GC31-MM",
    "EC-Earth3",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

common_models_noHadGEM = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "EC-Earth3",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

common_models_noEC = [
    "BCC-CSM2-MR",
    "MPI-ESM1-2-HR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "HadGEM3-GC31-MM",
    "FGOALS-f3-L",
    "MIROC6",
    "IPSL-CM6A-LR",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]

# Set up the tos models
tos_models = [
    "CanESM5",
    "HadGEM3-GC31-MM",
    "FGOALS-f3-L",
    "MIROC6",
    "CESM1-1-CAM5-CMIP5",
    "NorCPM1",
]
no_tos_models = 6
tos_models_numbers = [3, 5, 8, 9, 11, 12]

# Define a seasons list for the subplotting function
seasons_list_obs = ["DJFM", "MAM", "JJA", "SON"]

# Define a seasons list for the subplotting function for tos
# missing MAM
seasons_list_obs_tos = ["DJFM", "JJA", "SON"]

# Define a seasons list for the observations
seasons_list_model = ["DJFM", "MAY", "ULG", "SON"]

# Define a seasons list for the observations
seasons_list_model_tos = ["DJFM", "ULG", "SON"]

# Set up a list of variables for plotting
variables = ["psl", "tas", "sfcWind", "rsds"]

# Set up a list of variables for plotting
variables_850u = ["psl", "tas", "ua", "rsds"]

# Variables no tos
variables_no_tos = ["psl", "sfcWind", "tas"]

# Variables no psl
variables_no_psl = ["tos", "sfcWind", "tas"]

# Set up a list of variables for plotting
obs_var_names = ["psl", "tos", "sfcWind", "tas"]

# Obs var names for 850U
obs_var_names_850u = ["psl", "tos", "ua", "tas"]

obs_var_names_no_tos = ["psl", "sfcWind", "tas"]

obs_var_names_no_psl = ["tos", "sfcWind", "tas"]


models_list = [psl_models, tos_models, sfcWind_models, tas_models]

# Common models list
# for updated variables
# which is psl, tas, sfcwind and rsds
updated_models_list = [
    common_models,
    common_models,
    sfcWind_models,
    common_models_noCMCC,
]

# Substituting 850U in the place of sfcWind
updated_models_list_850u = [
    common_models,
    common_models,
    common_models,
    common_models_noCMCC,
]

obs_ua_va = "/gws/nopw/j04/canari/users/benhutch/ERA5/adaptor.mars.internal-1694423850.2771118-29739-1-db661393-5c44-4603-87a8-2d7abee184d8.nc"

obs_ua_va_925_1000 = "/gws/nopw/j04/canari/users/benhutch/ERA5/adaptor.mars.internal-1695119417.6957593-19943-15-3e812c68-d8c5-42b1-bac9-d086b50ae406.nc"

# for the new set of variables subplots
variables_list_updated = ["psl", "tas", "wind", "rsds"]
models_list_updated = [
    common_models,
    common_models,
    common_models,
    common_models_noCMCC,
]

# Updates obs var names
obs_var_names_updated = ["psl", "tas", "wind", "rsds"]

variables_list_ws_compare = ["psl", "rsds", "wind", "sfcWind"]
models_list_ws_compare = [
    common_models,
    common_models_noCMCC,
    common_models,
    sfcWind_models,
]
