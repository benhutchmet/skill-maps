import numpy as np
import random
from scipy import signal, stats
from scipy.stats import pearsonr


def fcsts_assess(obs, fcst1, fcst2, do_detrend = False):

    """

    Assess and compare two forecasts, using block bootstrap for uncertainties.

    Doug Smith : Nov 2018

                Oct 2019 added msss and 10 member skill

    Inputs:

        obs = timeseries of observations

        fcst1[member,time] = fcst1 ensemble

        fcst2[member,time] = fcst2 ensemble

        do_detrend = True for detrended timeseries

    Outputs:

        corr1: correlation between fcst1 ensemble mean and obs

        corr1_min, corr1_max, corr1_p: 5% to 95% uncertainties and p value

        corr2: correlation between fcst2 ensemble mean and obs

        corr2_min, corr2_max, corr2_p: 5% to 95% uncertainties and p value

        corr10: correlation between fcst1 ensemble mean and obs for 10 ensemble members

        corr10_min, corr10_max, corr10_p: 5% to 95% uncertainties and p value

        msss1: mean squared skill score between fcst1 ensemble mean and obs

        msss1_min, msss1_max, msss1_p: 5% to 95% uncertainties and p value

        rpc1: ratio of predictable components for fcst1

        rpc1_min, rpc1_max, rpc1_p: 5% to 95% uncertainties and p value

        rpc2: ratio of predictable components for fcst2

        rpc2_min, rpc2_max, rpc2_p: 5% to 95% uncertainties and p value

        corr_diff: corr1-corr2

        corr_diff_min, corr_diff_max, corr_diff_p: 5% to 95% uncertainties and p value

        partialr: partial correlation between obs and fcst1 ensemble mean...

            ...after removing the influence of fcst2 ensemble mean

        partialr_min, partialr_max, partialr_p: 5% to 95% uncertainties and p value

        partialr_bias: bias in partial correlation

        obs_resid: residual after regressing out fcst2 ensemble mean

        fcst1_em_resid: residual after regressing out fcst2 ensemble mean

    """

    # Set up output dictionary

    mdi = -9999.0

    fcsts_stats = {

        'corr1':mdi, 'corr1_min':mdi, 'corr1_max':mdi, 'corr1_p':mdi,

        'corr2':mdi, 'corr2_min':mdi, 'corr2_max':mdi, 'corr2_p':mdi,

        'corr10':mdi, 'corr10_min':mdi, 'corr10_max':mdi, 'corr10_p':mdi,

        'msss1':mdi, 'msss1_min':mdi, 'msss1_max':mdi, 'msss1_p':mdi,

        'corr12':mdi, 'corr12_min':mdi, 'corr12_max':mdi, 'corr12_p':mdi,

        'rpc1':mdi, 'rpc1_min':mdi, 'rpc1_max':mdi, 'rpc1_p':mdi,

        'rpc2':mdi, 'rpc2_min':mdi, 'rpc2_max':mdi, 'rpc2_p':mdi,

        'corr_diff':mdi, 'corr_diff_min':mdi, 'corr_diff_max':mdi, 'corr_diff_p':mdi,

        'partialr':mdi, 'partialr_min':mdi, 'partialr_max':mdi, 'partialr_p':mdi, 'partialr_bias':mdi,

        'obs_resid':[], 'fcst1_em_resid':[]

        }

    n_times = len(obs)

    nens1 = fcst1.shape[0] ; nens2 = fcst2.shape[0] ; nens2_2 = int(nens2/2+1)

    #nens = min(nens1,nens2)

    # detrend

    if do_detrend:

        obs = signal.detrend(obs)

        fcst1 = signal.detrend(fcst1)

        fcst2 = signal.detrend(fcst2)

    # bootstrap for uncertianties

    nboot = 1000

    r_partial_boot = np.zeros(nboot) ; r_partial_bias_boot = np.zeros(nboot)

    r1o_boot = np.zeros(nboot) ; r2o_boot = np.zeros(nboot) ; r12_boot = np.zeros(nboot)

    rdiff_boot = np.zeros(nboot) ; rpc1_boot = np.zeros(nboot) ; rpc2_boot = np.zeros(nboot)

    r_ens_10_boot = np.zeros(nboot) ; msss1_boot = np.zeros(nboot)

    block = 5 # for block bootstrap

    nblocks = int(n_times/block)

    if(nblocks*block < n_times):

        nblocks = nblocks+1

    index_time = range(n_times-block+1)

    index_ens1 = range(fcst1.shape[0])

    index_ens2 = range(fcst2.shape[0])

    for iboot in np.arange(nboot):

        # select ensemble members and starting indices for blocks

        if(iboot == 0): # raw data

            ind_ens1_this = index_ens1

            ind_ens2_this = index_ens2

            ind_time_this = range(0,n_times,block)

        else: # random samples

            # create an array containing random indices
            ind_ens1_this = np.array([random.choice(index_ens1) for _ in index_ens1])

            ind_ens2_this = np.array([random.choice(index_ens2) for _ in index_ens2])

            ind_time_this = np.array([random.choice(index_time) for _ in range(nblocks)])

        obs_boot = np.zeros(n_times)

        fcst1_boot = np.zeros(shape=(nens1,n_times)) ; fcst2_boot = np.zeros(shape=(nens2,n_times))

        fcst10_boot = np.zeros(shape=(10,n_times))

        # loop over blocks

        itime = 0

        for ithis in ind_time_this:

            # loop over start dates within block

            ind_block=np.arange(ithis,ithis+block)

            ind_block[(ind_block>n_times-1)] = ind_block[(ind_block>n_times-1)]-n_times

            ind_block = ind_block[:min(block,n_times-itime)]

            for iblck in ind_block:

                obs_boot[itime] = obs[iblck]

                fcst1_boot[:,itime] = fcst1[ind_ens1_this,iblck]

                fcst2_boot[:,itime] = fcst2[ind_ens2_this,iblck]

                fcst10_boot[:,itime] = fcst1[ind_ens1_this[0:10],iblck]

                itime = itime+1

 

        # stats

        o = obs_boot

        f1 = np.mean(fcst1_boot,axis=0) ; f2 = np.mean(fcst2_boot,axis=0)

        f10 = np.mean(fcst10_boot,axis=0)

        corr = pearsonr(f1,f2) ; r12 = corr[0]

        corr = pearsonr(f1,o) ; r1o = corr[0]

        corr = pearsonr(f2,o) ; r2o = corr[0]

        corr = pearsonr(f10,o) ; r_ens_10_boot[iboot] = corr[0]

        msss1_boot[iboot] = msss(o, f1)

        r1o_boot[iboot] = r1o ; r2o_boot[iboot] = r2o ; r12_boot[iboot] = r12

        rdiff_boot[iboot] = r1o-r2o

        var_noise_f1 = np.var(fcst1_boot-f1,ddof=n_times)/nens1

        var_noise_f2 = np.var(fcst2_boot-f2,ddof=n_times)/nens2

        sig_f1 = np.std(f1) ; sig_f2 = np.std(f2)

        rpc1_boot[iboot] = r1o/(sig_f1/np.std(fcst1_boot))

        rpc2_boot[iboot] = r2o/(sig_f2/np.std(fcst2_boot))

        # biased partial correlation

        denom_sq = (1.0-r2o**2)*(1.0-r12**2)

        r_partial_boot[iboot] = (r1o-r12*r2o)/np.sqrt(denom_sq)

        # compute bias by removing independent estimates of f2

        f2_1 = np.mean(fcst2_boot[:nens2_2,:],axis=0) # first half

        f2_2 = np.mean(fcst2_boot[nens2_2:-1,:],axis=0) # second half

        corr = pearsonr(f1,f2_1) ; r12_1 = corr[0]

        corr = pearsonr(f2_1,o) ; r2o_1 = corr[0]

        corr = pearsonr(f2_2,o) ; r2o_2 = corr[0]

        sigo = np.std(o) ; sig_f2_1 = np.std(f2_1)

        res_f1 = f1 - r12_1*f2_1*sig_f1/sig_f2_1

        res_o_1 = o - r2o_1*f2_1*sigo/sig_f2_1

        res_o_2 = o - r2o_2*f2_2*sigo/np.std(f2_2)

        corr = pearsonr(res_f1,res_o_1) ; rp_biased = corr[0]

        corr = pearsonr(res_f1,res_o_2) ; rp = corr[0]

        r_partial_bias_boot[iboot] = rp_biased-rp

 

    # stats

    fcsts_stats['corr1'] = r1o_boot[0]

    fcsts_stats['corr1_min'] = np.percentile(r1o_boot,5)

    fcsts_stats['corr1_max'] = np.percentile(r1o_boot,95)

    count_vals = sum(i < 0.0 for i in r1o_boot)

    fcsts_stats['corr1_p'] = float(count_vals)/nboot

    fcsts_stats['corr2'] = r2o_boot[0]

    fcsts_stats['corr2_min'] = np.percentile(r2o_boot,5)

    fcsts_stats['corr2_max'] = np.percentile(r2o_boot,95)

    count_vals = sum(i < 0.0 for i in r2o_boot)

    fcsts_stats['corr2_p'] = float(count_vals)/nboot

    fcsts_stats['corr10'] = np.percentile(r_ens_10_boot,50)

    fcsts_stats['corr10_min'] = np.percentile(r_ens_10_boot,5)

    fcsts_stats['corr10_max'] = np.percentile(r_ens_10_boot,95)

    count_vals = sum(i < 0.0 for i in r_ens_10_boot)

    fcsts_stats['corr10_p'] = float(count_vals)/nboot

    fcsts_stats['msss1'] = msss1_boot[0]

    fcsts_stats['msss1_min'] = np.percentile(msss1_boot,5)

    fcsts_stats['msss1_max'] = np.percentile(msss1_boot,95)

    count_vals = sum(i < 0.0 for i in msss1_boot)

    fcsts_stats['msss1_p'] = float(count_vals)/nboot

    fcsts_stats['corr12'] = r12_boot[0]

    fcsts_stats['corr12_min'] = np.percentile(r12_boot,5)

    fcsts_stats['corr12_max'] = np.percentile(r12_boot,95)

    count_vals = sum(i < 0.0 for i in r12_boot)

    fcsts_stats['corr12_p'] = float(count_vals)/nboot

    fcsts_stats['corr_diff'] = rdiff_boot[0]

    fcsts_stats['corr_diff_min'] = np.percentile(rdiff_boot,5)

    fcsts_stats['corr_diff_max'] = np.percentile(rdiff_boot,95)

    count_vals = sum(i < 0.0 for i in rdiff_boot)

    fcsts_stats['corr_diff_p'] = float(count_vals)/nboot

    fcsts_stats['rpc1'] = rpc1_boot[0]

    fcsts_stats['rpc1_min'] = np.percentile(rpc1_boot,5)

    fcsts_stats['rpc1_max'] = np.percentile(rpc1_boot,95)

    count_vals = sum(i < 1.0 for i in rpc1_boot)

    fcsts_stats['rpc1_p'] = float(count_vals)/nboot

    fcsts_stats['rpc2'] = rpc2_boot[0]

    fcsts_stats['rpc2_min'] = np.percentile(rpc2_boot,5)

    fcsts_stats['rpc2_max'] = np.percentile(rpc2_boot,95)

    count_vals = sum(i < 1.0 for i in rpc2_boot)

    fcsts_stats['rpc2_p'] = float(count_vals)/nboot

    # adjusted partial correlation

    adjust_bias = np.percentile(r_partial_bias_boot,50)

    r_partial_boot = r_partial_boot-adjust_bias

    fcsts_stats['partialr_bias'] = adjust_bias

    fcsts_stats['partialr'] = r_partial_boot[0]

    fcsts_stats['partialr_min'] = np.percentile(r_partial_boot,5)

    fcsts_stats['partialr_max'] = np.percentile(r_partial_boot,95)

    count_vals = sum(i < 0.0 for i in r_partial_boot)

    fcsts_stats['partialr_p'] = float(count_vals)/nboot

    # residuals

    f1 = np.mean(fcst1,axis=0) ; f2 = np.mean(fcst2,axis=0)

    sig1 = np.std(f1) ; sig2 = np.std(f2) ; sigo = np.std(obs)

    fcsts_stats['obs_resid'] = obs - r2o_boot[0]*f2*sigo/sig2

    fcsts_stats['fcst1_em_resid'] = f1 - r12_boot[0]*f2*sig1/sig2

    return fcsts_stats