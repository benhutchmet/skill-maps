# python dictionaries to be used in python/functions.py

# define the base directory where the data is stored
base_dir = "/home/users/benhutch/skill-maps-processed-data"

# define the directory where the plots will be saved
plots_dir = base_dir + "/plots"

gif_plots_dir = base_dir + "/plots/gif"

# list of the test model
test_model = [ "CMCC-CM2-SR5" ]
test_model_bcc = [ "BCC-CSM2-MR" ]
test_model2 = [ "EC-Earth3" ]
test_model_norcpm = [ "NorCPM1" ]
test_model_hadgem = [ "HadGEM3-GC31-MM" ]
test_model_cesm = [ "CESM1-1-CAM5-CMIP5" ]

# List of the full models
models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5", "NorCPM1" ]

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

obs_rsds="not/yet/implemented"

obs = "/home/users/benhutch/ERA5/adaptor.mars.internal-1691509121.3261805-29348-4-3a487c76-fc7b-421f-b5be-7436e2eb78d7.nc"

gridspec_global = "/home/users/benhutch/gridspec/gridspec-global.txt"

gridspec_north_atlantic = "/home/users/benhutch/gridspec/gridspec-north-atlantic.txt"

obs_regrid = "/home/users/benhutch/ERA5/ERA5_full_global.nc"

# Define the labels for the plots - wind
sfc_wind_label="10-metre wind speed"
sfc_wind_units = 'm s\u207b\u00b9'

# Define the labels for the plots - temperature
tas_label="2-metre temperature"
tas_units="K"

psl_label="Sea level pressure"
psl_units="hPa"

rsds_label="Surface solar radiation downwards"
rsds_units="W m\u207b\u00b2"

# Define the dimensions for the grids
# for processing the observations
north_atlantic_grid = {
    'lon1': 280,
    'lon2': 37.5,
    'lat1': 77.5,
    'lat2': 20
}

# define the NA grid for obs
north_atlantic_grid_obs = {
    'lon1': 100,
    'lon2': 217.5,
    'lat1': 77.5,
    'lat2': 20
}

# Define the dimensions for the gridbox for the azores
azores_grid = {
    'lon1': 152,
    'lon2': 160,
    'lat1': 36,
    'lat2': 40
}

# Define the dimensions for the gridbox for iceland
iceland_grid = {
    'lon1': 155,
    'lon2': 164,
    'lat1': 63,
    'lat2': 70
}

# Define the dimensions for the gridbox for the N-S UK index
# From thornton et al. 2019
uk_n_box = {
    'lon1': 153,
    'lon2': 201,
    'lat1': 57,
    'lat2': 70
}

# And for the southern box
uk_s_box = {
    'lon1': 153,
    'lon2': 201,
    'lat1': 38,
    'lat2': 51
}

