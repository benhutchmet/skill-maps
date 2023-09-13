# a script to test predictability by rescaling of the NAO and model member selection (Doug Smith's method).

import iris
import iris.coord_categorisation as coord_cat
import functions
import functions_verification
import numpy as np
import scipy
import iris.plot as iplt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm  # for Brewer colourmap
import pdb
import iris.quickplot as qplt

#command to run prior to running script
#module load scitools/production-os44-1

datadir = '/data/users/hadhz/Multi_model_predictability/'
output_dir = '/net/home/h01/hadhz/Python/Multi_model_predictability/'
season = 'DJF'
variable = 'MSLP'   
region = False  # False (giving a global field) or 'Europe'
filename = 'Multi_model_rescaled'
expt = 'Multi_model_rescaled'
leadtime = 'One' # One, Two, Three  leadtime in months
start_month, start_month_number = functions.start_month_from_leadtime(leadtime, season)
start_year = 1993  # Can't change as hard coded in functions below.
year_list= start_year + np.arange(24) # makes array from 1993 to 2016 (start dates)
end_year = year_list[-1]

print('------------------Read and process obs ------------------')
obs, anom_obs, running_seasonal_mean_obs= functions.read_obs(variable, season, data_type='mid_month')
print(anom_obs)


print('----------------- Read multi model cube ---------------')
#read in cube of anomaly fields (each model relative to its own climatology)
infile = datadir+'MULTI-MODEL/MultiModel_'+variable+'_ANOM_Glob_start_month_'+start_month+'_season_'+season+'_'+str(start_year)+'_'+str(end_year)+'.nc'
print(infile)
anom_MM = iris.load_cube(infile)
print(anom_MM)

print('----------------- Calculate the NAO ---------------')
obs_NAO, mod_NAO = functions.calc_NAO_and_plot(anom_MM, anom_obs, variable, season, leadtime, filename, expt, output_dir, datadir, start_year, end_year, multimodel = True, rm_timemean = True, plot_graphics = False)
print('obs_NAO =', obs_NAO)
print('mod_NAO =', mod_NAO)

print('----------------- Rescale the NAO ---------------')
rescaled_NAO, orig_NAO = functions.rescale_NAO(mod_NAO, obs_NAO, season, leadtime, filename, output_dir, datadir, start_year, end_year, multimodel = True) 
print('obs_NAO:', obs_NAO.data)
print('The original ensemble mean NAO forecast', orig_NAO.data)
print('The rescaled NAO ', rescaled_NAO.data)

print('Minimum of NAO predictions is ', np.min(mod_NAO.data))
print('Maximum of NAO predictions is ', np.max(mod_NAO.data))

print('----------------- Calculate nearest ensemble members ---------------')
cL = iris.cube.CubeList()
for i, cube_i in enumerate(mod_NAO.slices_over('realization')):

    NAO_diff_i = cube_i - rescaled_NAO  
    NAO_diff_i = iris.analysis.maths.abs(NAO_diff_i)
    realization_coord = iris.coords.AuxCoord(i, standard_name='realization')
    NAO_diff_i.add_aux_coord(realization_coord)                
    cL.append(NAO_diff_i)
NAO_diff = cL.merge_cube() 
print(NAO_diff)
print(NAO_diff.data) 

no_members = 40
cL = iris.cube.CubeList()
for i, cube_i in enumerate(NAO_diff.slices_over('time')): 
    # work out X nearest members per year  (no_members)    
    sorted_members = np.argsort(cube_i.data)
    new_cube = anom_MM[sorted_members[0:no_members],i]
    new_cube.coord('realization').points = np.arange(no_members)
    iris.util.promote_aux_coord_to_dim_coord(new_cube, 'realization')
    cL.append(new_cube)
     
selected_members = cL.merge_cube()   # cube of selected ensemble members 
print(selected_members)

#calculate ensemble mean
ens_mean = selected_members.collapsed(['realization'], iris.analysis.MEAN)

#calculate temporal correlation and plot
functions.calculate_corr_and_plot(ens_mean, anom_obs, variable, season, leadtime, output_dir, datadir, expt, start_year, end_year, multimodel = True)

print('----------------- Calculate the NAO on rescaled members ---------------')
obs_NAO, mod_NAO = functions.calc_NAO_and_plot(selected_members, anom_obs, variable, season, leadtime, filename, expt, output_dir, datadir, start_year, end_year, multimodel = True, rm_timemean = True, plot_graphics = True)
print('obs_NAO =', obs_NAO)
print('mod_NAO =', mod_NAO)



print('----------------- Calculate temperature/precip skill using selected members ---------------')
for variable in ['MSLP', 'Temp', 'Precip']:
#for variable in ['MSLP']:
    obs, anom_obs, running_seasonal_mean_obs= functions.read_obs(variable, season, data_type='mid_month')
    print(anom_obs)

    #read in cube of anomaly fields (each model relative to its own climatology)
    infile = datadir+'MULTI-MODEL/MultiModel_'+variable+'_ANOM_Glob_start_month_'+start_month+'_season_'+season+'_'+str(start_year)+'_'+str(end_year)+'.nc'
    print(infile)
    anom_MM_var = iris.load_cube(infile)
    anom_MM_var_ens_mean = anom_MM_var.collapsed(['realization'], iris.analysis.MEAN) 
    print(anom_MM_var)

    #calculate original RMSE       
    error = anom_MM_var_ens_mean.copy()
    error.data = anom_obs.data - anom_MM_var_ens_mean.data
    rmse = error.collapsed(['time'], iris.analysis.RMS)   
        
    #limit members to those selected by NAO
    cL = iris.cube.CubeList()
    for i, cube_i in enumerate(NAO_diff.slices_over('time')): 
        # work out X nearest members per year  (no_members)    
        sorted_members = np.argsort(cube_i.data)
        new_cube = anom_MM_var[sorted_members[0:no_members],i]
        new_cube.coord('realization').points = np.arange(no_members)
        iris.util.promote_aux_coord_to_dim_coord(new_cube, 'realization')
        cL.append(new_cube)
         
    selected_members_var = cL.merge_cube()   # cube of selected ensemble members 
    print(selected_members_var)
    
   # detrend_temp = 'TRUE'
   # if variable == 'Temp' and detrend_temp == 'TRUE':
   #     cL = iris.cube.CubeList()
   #     for i, cube_i in enumerate(selected_members_var.slices_over('realization')):   
   #         print('realization', i)
   #         detrended = functions.linear_trend_removal_2d(cube_i)                
   #         cL.append(detrended)
   #     selected_members_var = cL.merge_cube()
     
    #calculate ensemble mean
    ens_mean_var = selected_members_var.collapsed(['realization'], iris.analysis.MEAN)

    #calculate temporal correlation and plot
    functions.calculate_corr_and_plot(ens_mean_var, anom_obs, variable, season, leadtime, output_dir, datadir, expt, start_year, end_year, multimodel = True)
 
    #calculate rescaled RMSE
    error_rescaled = ens_mean_var.copy()
    error_rescaled.data = anom_obs.data - ens_mean_var.data
    rmse_rescaled = error_rescaled.collapsed(['time'], iris.analysis.RMS)   

    #plot variable field for a selection of years for original and rescaled to see if makes a difference.
        
    for winter_year in [1998, 2006, 2010, 2012, 2015]:
    #for winter_year in [1998]:            
        plot_name = output_dir + str(winter_year) + '_' + variable + '_'+str(no_members)+'_members.png'
        functions.individual_year_rescaled_plot(winter_year, variable, anom_obs, anom_MM_var_ens_mean, ens_mean_var, error, error_rescaled, plot_name)
               
    #plot the difference in the correlation or RMSE field between original and rescaled    
    for measure in ['Correlation', 'RMSE']:
        
        plot_name = output_dir + season + '_' + variable +'_' + measure + '_skill_rescaled_difference_' + str(no_members) + '_members.png'    
        
        if measure == 'Correlation':
            name, plot_str, expt_str = functions.create_filenames(leadtime, expt)
            orig = iris.load_cube(datadir + season + '_'+ variable + '_skill_' + name + '_month_leadtime_Multi_model.nc')
            rescaled = iris.load_cube(datadir + season + '_'+ variable + '_skill_' + name + '_month_leadtime_Multi_model_rescaled.nc')
            orig_sig = iris.load_cube(datadir + season + '_'+ variable + '_significance_' + name + '_month_leadtime_Multi_model.nc')
            rescaled_sig = iris.load_cube(datadir + season + '_'+ variable + '_significance_' + name + '_month_leadtime_Multi_model_rescaled.nc')
            
            functions.plot_skill_difference(variable, orig, rescaled, measure, plot_name, orig_sig = orig_sig, rescaled_sig = rescaled_sig)
        
        if measure == 'RMSE':
            functions.plot_skill_difference(variable, rmse, rmse_rescaled, measure, plot_name)
    
#Test code is working
    
#test = ens_mean - ens_mean_var
#print(test)
#print(np.max(test.data))
    
    
    
    
