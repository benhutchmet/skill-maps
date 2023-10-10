#!/usr/bin/env python

"""
plot_init_benefit.py
--------------------

A script which plots a series of subplots.
Each row shows the total skill of the initialized ensemble mean and 
the benefit of the initialization relative to the uninitialized ensemble mean.

Usage:

    $ python plot_init_benefit.py <variable> <region> <season> <forecast_range>
                                <method> <no_bootstraps>
    Args:                            
        <variable>      :   variable to plot
        <region>        :   region to plot
        <season>        :   season to plot
        <forecast_range>:   forecast range to plot
        <method>        :   method to plot
        <no_bootstraps> :   number of bootstraps to plot

    Returns:
        None
                                
"""

#