#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main script, writed by Ivan Mysin for article "How computate Hodgkin-Huxley equations faster"

this script generate figure 1 and 2 

parameters of all neurons are got from  "Minimal Hodgkin–Huxley type models for different classes of cortical and thalamic neurons" (Pospischil at al, 2008) 
"""
import numpy as np
import HH_classes as HH
import matplotlib.pyplot as plt
import timeit

# function FS neuron object with set parameters
def get_fs_neuron():
    
    neurons_params = {
        "V0" : -60, # mV, initial value of potential
        "C"  : 1,   # mkF, capacity of memrane
    }
    
    Vt = -62        # mV, threshold of spike generation
    
    external_current = {
        "Iext" : 0.7, # nA, value of current for patch-clump electrode imitation
    }
    
    leak_current = {
        "Erev" : -70.0, # mV, reversal potentiol of leak current
        "gmax" : 0.015, # mS, conductance of leak current
    }
    
    sodium_current_params = {
        "Erev" : 50.0,   # mV, reversal potentiol of sodium current
        "gmax" : 50.0,   # mS, maximal conductance of sodium current
        "Vt"   : Vt,
    }
    
    potassium_current_params = {
        "Erev" : -90.0, # mV, reversal potentiol of potassium current
        "gmax" : 10.0,  # mS, maximal conductance of potassium current
        "Vt"   : Vt,
    }

    neuron = HH.Neuron(neurons_params) # declare neuron object
    # add currents
    neuron.addCurrent( HH.Current(external_current) )
    neuron.addCurrent( HH.LeakCurrent(leak_current) )
    neuron.addCurrent( HH.SodiumCurrent(sodium_current_params) )
    neuron.addCurrent( HH.PotassiumCurrent(potassium_current_params) )
    
    return neuron

# function RS neuron object with set parameters
def get_rs_neuron():
    
    neurons_params = {
        "V0" : -60, # mV, initial value of potential
        "C"  : 1,   # mkF, capacity of memrane
    }
    
    Vt = -56.2     # mV, threshold of spike generation
    
    external_current = {
        "Iext" : 0.9,    # nA, value of current for patch-clump electrode imitation   
    }
    
    leak_current = {
        "Erev" : -70.3, # mV, reversal potentiol of leak current
        "gmax" : 0.02,  # mS, conductance of leak current
    }
    
    sodium_current_params = {
        "Erev" : 50.0,  # mV, reversal potentiol of sodium current
        "gmax" : 56.0,  # mS, maximal conductance of sodium current
        "Vt"   : Vt,
    }
    
    potassium_current_params = {
        "Erev" : -90.0, # mV, reversal potentiol of potassium current
        "gmax" : 6.0,   # mS, maximal conductance of potassium current
        "Vt"   : Vt,
    }
    
    slow_potassium_current_params = {
        "Erev" : -90.0,   # mV, reversal potentiol of potassium current
        "gmax" : 0.07,    # mS, maximal conductance of slow potassium current
        "tau_max" : 608,  # ms, maximal tau_inf of p
    }

    neuron = HH.Neuron(neurons_params) # declare neuron object
    # add currents
    neuron.addCurrent( HH.Current(external_current) )
    neuron.addCurrent( HH.LeakCurrent(leak_current) )
    neuron.addCurrent( HH.SodiumCurrent(sodium_current_params) )
    neuron.addCurrent( HH.PotassiumCurrent(potassium_current_params) )
    neuron.addCurrent( HH.SlowPotassiumCurrent(slow_potassium_current_params) )
    
    return neuron

# function LTS neuron object with set parameters
def get_LTS_neuron():
    
    neurons_params = {
        "V0" : -60,  # mV, initial value of potential
        "C"  : 1,    # mkF, capacity of memrane
    }

    Vt = -50.0       # mV, threshold of spike generation
    
    external_current = {
        "Iext" : 0.3,  # nA, value of current for patch-clump electrode imitation         
    }
    
    leak_current = {
        "Erev" : -50.0, # mV, reversal potentiol of leak current
        "gmax" : 0.019, # mS, conductance of leak current
    }
    
    sodium_current_params = {
        "Erev" : 50.0, # mV, reversal potentiol of sodium current
        "gmax" : 50.0, # mS, maximal conductance of sodium current
        "Vt"   : Vt,
    }
    
    potassium_current_params = {
        "Erev" : -90.0, # mV, reversal potentiol of potassium current
        "gmax" : 4.0,   # mS, maximal conductance of potassium current
        "Vt"   : Vt,   
    }
    
    slow_potassium_current_params = {
        "Erev" : -90.0,    # mV, reversal potentiol of potassium current
        "gmax" : 0.028,    # mS, maximal conductance of slow potassium current  
        "tau_max" : 4000,  # ms, maximal tau_inf of p
    }
    
    t_type_calcium_current_params = {
        "Erev" : 120.0,  # mV, reversal potentiol of calcium current
        "gmax" : 0.4,    # mS, maximal conductance of calcium current of T type
        "Vx"   : -7,     # mV, parameter for calculation  
    }
    
    neuron = HH.Neuron(neurons_params) # declare neuron object
    # add currents
    neuron.addCurrent( HH.Current(external_current) )
    neuron.addCurrent( HH.LeakCurrent(leak_current) )
    neuron.addCurrent( HH.SodiumCurrent(sodium_current_params) )
    neuron.addCurrent( HH.PotassiumCurrent(potassium_current_params) )
    neuron.addCurrent( HH.SlowPotassiumCurrent(slow_potassium_current_params) )
    neuron.addCurrent( HH.CalciumCurrentTType(t_type_calcium_current_params) )
    
    return neuron

def get_bursting_neuron():
    
    neurons_params = {
        "V0" : -60, # mV, initial value of potential
        "C"  : 1,   # mkF, capacity of memrane
    }
    
    Vt = -58        # mV, threshold of spike generation
    
    external_current = {
        "Iext" : 0.5, # nA, value of current for patch-clump electrode imitation  
    }
        
    leak_current = {
        "Erev" : -75.0, # mV, reversal potentiol of leak current
        "gmax" : 0.01,  # mS, conductance of leak current
    }
    
    sodium_current_params = {
        "Erev" : 50.0,   # mV, reversal potentiol of sodium current
        "gmax" : 50.0,   # mS, maximal conductance of sodium current
        "Vt"   : Vt,
    }
    
    potassium_current_params = {
        "Erev" : -90.0, # mV, reversal potentiol of potassium current
        "gmax" : 4.2,   # mS, maximal conductance of potassium current
        "Vt"   : Vt,
    }
    
    slow_potassium_current_params = {
        "Erev" : -90.0,   # mV, reversal potentiol of potassium current
        "gmax" : 0.042,   # mS, maximal conductance of slow potassium current   
        "tau_max" : 1000, # ms, maximal tau_inf of p
    }
    
    l_type_calcium_current_params = {
        "Erev" : 120.0,  # mV, reversal potentiol of calcium current
        "gmax" : 0.12,   # mS, maximal conductance of calcium current of L type
    }
    
    neuron = HH.Neuron(neurons_params) # declare neuron object
    # add currents
    neuron.addCurrent( HH.Current(external_current) )
    neuron.addCurrent( HH.LeakCurrent(leak_current) )
    neuron.addCurrent( HH.SodiumCurrent(sodium_current_params) )
    neuron.addCurrent( HH.PotassiumCurrent(potassium_current_params) )
    neuron.addCurrent( HH.SlowPotassiumCurrent(slow_potassium_current_params) )
    neuron.addCurrent( HH.CalciumCurrentLType(l_type_calcium_current_params) )
    
    return neuron

# set parameters of figures
plt.rc('axes', linewidth=2)
plt.rc('axes', linewidth=2)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('lines', linewidth=3) 
plt.rc('lines', markersize=4)
plt.rc('lines', color="black")

# decalre object for figure 1 and exes
fig1, axs = plt.subplots(nrows=8, ncols=2, tight_layout=True, figsize=(15, 15))
# declate time array for ploting
time4plots = np.linspace(0, 1000, 100002)

# declare lists for saving spent time on the simulations
t_slow_mean = []
t_fast_mean = []

t_slow_std = []
t_fast_std = []

# make 10 simulations of FS neuron with accurate comutation of x_inf and T functions
t_slow_fs = np.array([], dtype=float)
for idx in range(10):
    neuron = get_fs_neuron()
    tmp = timeit.timeit("neuron.run(0.01, 1000)", setup="", globals=globals(), number=1)
    t_slow_fs = np.append(t_slow_fs, tmp)
# plot results
axs[0, 0].plot(time4plots, neuron.Vhist)
axs[0, 0].set_title("FS (during simulation)", fontsize=18, loc='right')
axs[0, 1].plot(time4plots, neuron.Vhist)
# save mean and std of spent time
t_slow_mean.append(t_slow_fs.mean())
t_slow_std.append(t_slow_fs.std())

########################
# make 10 simulations of FS neuron with precomputed x_inf and T functions
t_fast_fs = np.array([], dtype=float)
for idx in range(10):
    neuron = get_fs_neuron()
    tmp = timeit.timeit("neuron.runfast(0.01, 1000)", setup="", globals=globals(), number=1)
    t_fast_fs = np.append(t_fast_fs, tmp)

# plot results
axs[1, 0].plot(time4plots, neuron.Vhist)
axs[1, 0].set_title("FS (precomputed)", fontsize=18, loc='right')
axs[1, 1].plot(time4plots, neuron.Vhist)
# save mean and std of spent time
t_fast_mean.append(t_fast_fs.mean())
t_fast_std.append(t_fast_fs.std())
#########################
# make 10 simulations of RS neuron with accurate comutation of x_inf and T functions
t_slow_rs = np.array([], dtype=float)
for idx in range(10):
    neuron = get_rs_neuron()
    tmp = timeit.timeit("neuron.run(0.01, 1000)", setup="", globals=globals(), number=1)
    t_slow_rs = np.append(t_slow_rs, tmp)
# plot results
axs[2, 0].plot(time4plots, neuron.Vhist)
axs[2, 0].set_title("RS (during simulation)", fontsize=18, loc='right')
axs[2, 1].plot(time4plots, neuron.Vhist)
# save mean and std of spent time
t_slow_mean.append(t_slow_rs.mean())
t_slow_std.append(t_slow_rs.std())
########################
# make 10 simulations of RS neuron with precomputed x_inf and T functions
t_fast_rs = np.array([], dtype=float)
for idx in range(10):
    neuron = get_rs_neuron()
    tmp = timeit.timeit("neuron.runfast(0.01, 1000)", setup="", globals=globals(), number=1)
    t_fast_rs = np.append(t_fast_rs, tmp)
# plot results
axs[3, 0].plot(time4plots, neuron.Vhist)
axs[3, 0].set_title("RS (precomputed)", fontsize=18, loc='right')
axs[3, 1].plot(time4plots, neuron.Vhist)
# save mean and std of spent time
t_fast_mean.append(t_fast_rs.mean())
t_fast_std.append(t_fast_rs.std())
######################
# make 10 simulations of LTS neuron with accurate comutation of x_inf and T functions
t_slow_lts = np.array([], dtype=float)
for idx in range(10):
    neuron = get_LTS_neuron()
    tmp = timeit.timeit("neuron.run(0.01, 1000)", setup="", globals=globals(), number=1)
    t_slow_lts = np.append(t_slow_lts, tmp)
# plot results
axs[4, 0].plot(time4plots, neuron.Vhist)
axs[4, 0].set_title("LTS (during simulation)", fontsize=18, loc='right')
axs[4, 1].plot(time4plots, neuron.Vhist)
# save mean and std of spent time
t_slow_mean.append(t_slow_lts.mean())
t_slow_std.append(t_slow_lts.std())
######################
# make 10 simulations of LTS neuron with precomputed x_inf and T functions
t_fast_lts = np.array([], dtype=float)
for idx in range(10):
    neuron = get_LTS_neuron()
    tmp = timeit.timeit("neuron.runfast(0.01, 1000)", setup="", globals=globals(), number=1)
    t_fast_lts = np.append(t_fast_lts, tmp)
# plot results
axs[5, 0].plot(time4plots, neuron.Vhist)
axs[5, 0].set_title("LTS (precomputed)", fontsize=18, loc='right')
axs[5, 1].plot(time4plots, neuron.Vhist)
# save mean and std of spent time
t_fast_mean.append(t_fast_lts.mean())
t_fast_std.append(t_fast_lts.std())
######################
# make 10 simulations of IB neuron with accurate comutation of x_inf and T functions
t_slow_ib = np.array([], dtype=float)
for idx in range(10):
    neuron = get_bursting_neuron()
    tmp = timeit.timeit("neuron.run(0.01, 1000)", setup="", globals=globals(), number=1)
    t_slow_ib = np.append(t_slow_ib, tmp)
# plot results
axs[6, 0].plot(time4plots, neuron.Vhist)
axs[6, 0].set_title("IB(during simulation)", fontsize=18, loc='right')
axs[6, 1].plot(time4plots, neuron.Vhist)
# save mean and std of spent time
t_slow_mean.append(t_slow_ib.mean())
t_slow_std.append(t_slow_ib.std())

######################
# make 10 simulations of IB neuron with precomputed x_inf and T functions
t_fast_ib = np.array([], dtype=float)
for idx in range(10):
    neuron = get_bursting_neuron()
    tmp = timeit.timeit("neuron.runfast(0.01, 1000)", setup="", globals=globals(), number=1)
    t_fast_ib = np.append(t_fast_ib, tmp)
# plot results
axs[7, 0].plot(time4plots, neuron.Vhist)
axs[7, 0].set_title("IB(precomputed)", fontsize=18, loc='right')
axs[7, 1].plot(time4plots, neuron.Vhist)
# save mean and std of spent time
t_fast_mean.append(t_fast_ib.mean())
t_fast_std.append(t_fast_ib.std())

# set parameters for figure 1
for ax in axs[:, 0]:
    ax.set_xlim(0, 1000)
    ax.set_ylim(-95, 65)
    ax.set_ylabel("V, mV")
ax.set_xlabel("time, ms")

for ax in axs[:, 1]:
    ax.set_xlim(0, 100)
    ax.set_ylim(-95, 65)
    ax.set_ylabel("V, mV")
ax.set_xlabel("time, ms")

fig1.savefig("figure_1.png", dpi=300)


# declare object for figure 2
fig2, ax = plt.subplots()
width = 0.3
ind = np.arange(len(t_slow_mean))
rects1 = ax.bar(ind, t_slow_mean, width, color='b', yerr=t_slow_std) # plot spent time for accurate computation of x_inf and T
rects2 = ax.bar(ind + width, t_fast_mean, width, color='g', yerr=t_fast_std) # plot spent time for precomputed of x_inf and T
ax.set_ylim(0, 12)
ax.set_ylabel('simulation time, sec')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('FS', 'RS', 'LTS', 'IB'))
ax.legend((rects1[0], rects2[0]), ('During simulation', 'Precomputed'), fontsize=14)

fig2.savefig("figure_2.png", dpi=100)
