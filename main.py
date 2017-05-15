#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main script
"""
import numpy as np
import HH_classes as HH
import matplotlib.pyplot as plt
import timeit

def get_fs_neuron():
    
    neurons_params = {
        "V0" : -60, 
        "C"  : 1,   
    }
    
    Vt = -62
    
    external_current = {
        "Iext" : 0.7,        
    }
    
    leak_current = {
        "Erev" : -70.0,
        "gmax" : 0.015
    }
    
    sodium_current_params = {
        "Erev" : 50.0,
        "gmax" : 50.0,
        "Vt"   : Vt,
    }
    
    potassium_current_params = {
        "Erev" : -90.0,
        "gmax" : 10.0,
        "Vt"   : Vt,
    }

    neuron = HH.Neuron(neurons_params)
    neuron.addCurrent( HH.Current(external_current) )
    neuron.addCurrent( HH.LeakCurrent(leak_current) )
    neuron.addCurrent( HH.SodiumCurrent(sodium_current_params) )
    neuron.addCurrent( HH.PotassiumCurrent(potassium_current_params) )
    
    return neuron

def get_rs_neuron():
    
    neurons_params = {
        "V0" : -60, 
        "C"  : 1,   
    }
    
    Vt = -56.2
    
    external_current = {
        "Iext" : 0.9,        
    }
    
    leak_current = {
        "Erev" : -70.3,
        "gmax" : 0.02
    }
    
    sodium_current_params = {
        "Erev" : 50.0,
        "gmax" : 56.0,
        "Vt"   : Vt,
    }
    
    potassium_current_params = {
        "Erev" : -90.0,
        "gmax" : 6.0,
        "Vt"   : Vt,
    }
    
    slow_potassium_current_params = {
        "Erev" : -90.0,
        "gmax" : 0.07,     
        "tau_max" : 608,
    }

    neuron = HH.Neuron(neurons_params)
    neuron.addCurrent( HH.Current(external_current) )
    neuron.addCurrent( HH.LeakCurrent(leak_current) )
    neuron.addCurrent( HH.SodiumCurrent(sodium_current_params) )
    neuron.addCurrent( HH.PotassiumCurrent(potassium_current_params) )
    neuron.addCurrent( HH.SlowPotassiumCurrent(slow_potassium_current_params) )
    
    return neuron

def get_LTS_neuron():
    
    neurons_params = {
        "V0" : -60, 
        "C"  : 1,   
    }

    Vt = -50.0
    
    external_current = {
        "Iext" : 0.3,        
    }
    
    leak_current = {
        "Erev" : -50.0,
        "gmax" : 0.019
    }
    
    sodium_current_params = {
        "Erev" : 50.0,
        "gmax" : 50.0,
        "Vt"   : Vt,
    }
    
    potassium_current_params = {
        "Erev" : -90.0,
        "gmax" : 4.0,
        "Vt"   : Vt,
    }
    
    slow_potassium_current_params = {
        "Erev" : -90.0,
        "gmax" : 0.028,     
        "tau_max" : 4000,
    }
    
    t_type_calcium_current_params = {
        "Erev" : 120.0,
        "gmax" : 0.4,     
        "Vx"   : -7,
    }
    
    neuron = HH.Neuron(neurons_params)
    neuron.addCurrent( HH.Current(external_current) )
    neuron.addCurrent( HH.LeakCurrent(leak_current) )
    neuron.addCurrent( HH.SodiumCurrent(sodium_current_params) )
    neuron.addCurrent( HH.PotassiumCurrent(potassium_current_params) )
    neuron.addCurrent( HH.SlowPotassiumCurrent(slow_potassium_current_params) )
    neuron.addCurrent( HH.CalciumCurrentTType(t_type_calcium_current_params) )
    
    return neuron

def get_bursting_neuron():
    
    neurons_params = {
        "V0" : -60, 
        "C"  : 1,   
    }
    
    Vt = -58
    
    external_current = {
        "Iext" : 0.5,        
    }
        
    leak_current = {
        "Erev" : -75.0,
        "gmax" : 0.01
    }
    
    sodium_current_params = {
        "Erev" : 50.0,
        "gmax" : 50.0,
        "Vt"   : Vt,
    }
    
    potassium_current_params = {
        "Erev" : -90.0,
        "gmax" : 4.2,
        "Vt"   : Vt,
    }
    
    slow_potassium_current_params = {
        "Erev" : -90.0,
        "gmax" : 0.042,     
        "tau_max" : 1000,
    }
    
    l_type_calcium_current_params = {
        "Erev" : 120.0,
        "gmax" : 0.12,     
    }
    
    neuron = HH.Neuron(neurons_params)
    neuron.addCurrent( HH.Current(external_current) )
    neuron.addCurrent( HH.LeakCurrent(leak_current) )
    neuron.addCurrent( HH.SodiumCurrent(sodium_current_params) )
    neuron.addCurrent( HH.PotassiumCurrent(potassium_current_params) )
    neuron.addCurrent( HH.SlowPotassiumCurrent(slow_potassium_current_params) )
    neuron.addCurrent( HH.CalciumCurrentLType(l_type_calcium_current_params) )
    
    return neuron


plt.rc('axes', linewidth=2)
plt.rc('axes', linewidth=2)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('lines', linewidth=3) 
plt.rc('lines', markersize=4)
plt.rc('lines', color="black")

fig1, axs = plt.subplots(nrows=8, ncols=2, sharey=True, tight_layout=True, figsize=(15, 15))

time4plots = np.linspace(0, 1000, 100002)

t_slow_mean = []
t_fast_mean = []

t_slow_std = []
t_fast_std = []


t_slow_fs = np.array([], dtype=float)
for idx in range(1):
    neuron = get_fs_neuron()
    tmp = timeit.timeit("neuron.run(0.01, 1000)", setup="", globals=globals(), number=1)
    t_slow_fs = np.append(t_slow_fs, tmp)

axs[0, 0].plot(time4plots, neuron.Vhist)
axs[0, 1].plot(time4plots, neuron.Vhist)

t_slow_mean.append(t_slow_fs.mean())
t_slow_std.append(t_slow_fs.std())

########################
t_fast_fs = np.array([], dtype=float)
for idx in range(1):
    neuron = get_fs_neuron()
    tmp = timeit.timeit("neuron.runfast(0.01, 1000)", setup="", globals=globals(), number=1)
    t_fast_fs = np.append(t_fast_fs, tmp)



axs[1, 0].plot(time4plots, neuron.Vhist)
axs[1, 1].plot(time4plots, neuron.Vhist)

t_fast_mean.append(t_fast_fs.mean())
t_fast_std.append(t_fast_fs.std())




#########################

t_slow_rs = np.array([], dtype=float)
for idx in range(10):
    neuron = get_rs_neuron()
    tmp = timeit.timeit("neuron.run(0.01, 1000)", setup="", globals=globals(), number=1)
    t_slow_rs = np.append(t_slow_rs, tmp)

axs[2, 0].plot(time4plots, neuron.Vhist)
axs[2, 1].plot(time4plots, neuron.Vhist)

t_slow_mean.append(t_slow_rs.mean())
t_slow_std.append(t_slow_rs.std())
########################
t_fast_rs = np.array([], dtype=float)
for idx in range(10):
    neuron = get_rs_neuron()
    tmp = timeit.timeit("neuron.runfast(0.01, 1000)", setup="", globals=globals(), number=1)
    t_fast_rs = np.append(t_fast_rs, tmp)

axs[3, 0].plot(time4plots, neuron.Vhist)
axs[3, 1].plot(time4plots, neuron.Vhist)

t_fast_mean.append(t_fast_rs.mean())
t_fast_std.append(t_fast_rs.std())


######################
t_slow_lts = np.array([], dtype=float)
for idx in range(10):
    neuron = get_LTS_neuron()
    tmp = timeit.timeit("neuron.run(0.01, 1000)", setup="", globals=globals(), number=1)
    t_slow_lts = np.append(t_slow_lts, tmp)

axs[4, 0].plot(time4plots, neuron.Vhist)
axs[4, 1].plot(time4plots, neuron.Vhist)

t_slow_mean.append(t_slow_lts.mean())
t_slow_std.append(t_slow_lts.std())


######################
t_fast_lts = np.array([], dtype=float)
for idx in range(10):
    neuron = get_LTS_neuron()
    tmp = timeit.timeit("neuron.runfast(0.01, 1000)", setup="", globals=globals(), number=1)
    t_fast_lts = np.append(t_fast_lts, tmp)

axs[5, 0].plot(time4plots, neuron.Vhist)
axs[5, 1].plot(time4plots, neuron.Vhist)

t_fast_mean.append(t_fast_lts.mean())
t_fast_std.append(t_fast_lts.std())

######################

t_slow_ib = np.array([], dtype=float)
for idx in range(10):
    neuron = get_bursting_neuron()
    tmp = timeit.timeit("neuron.run(0.01, 1000)", setup="", globals=globals(), number=1)
    t_slow_ib = np.append(t_slow_ib, tmp)

axs[6, 0].plot(time4plots, neuron.Vhist)
axs[6, 1].plot(time4plots, neuron.Vhist)

t_slow_mean.append(t_slow_ib.mean())
t_slow_std.append(t_slow_ib.std())

######################
t_fast_ib = np.array([], dtype=float)
for idx in range(10):
    neuron = get_bursting_neuron()
    tmp = timeit.timeit("neuron.runfast(0.01, 1000)", setup="", globals=globals(), number=1)
    t_fast_ib = np.append(t_fast_ib, tmp)

axs[7, 0].plot(time4plots, neuron.Vhist)
axs[7, 1].plot(time4plots, neuron.Vhist)


for ax in axs[:, 0]:
    ax.set_xlim(0, 1000)
    ax.set_ylim(-95, 65)

for ax in axs[:, 1]:
    ax.set_xlim(0, 100)
    ax.set_ylim(-95, 65)

fig1.savefig("figure_1.png")



t_fast_mean.append(t_fast_ib.mean())
t_fast_std.append(t_fast_ib.std())


fig2, ax = plt.subplots()
width = 0.3
ind = np.arange(len(t_slow_mean))
rects1 = ax.bar(ind, t_slow_mean, width, color='b', yerr=t_slow_std)
rects2 = ax.bar(ind + width, t_fast_mean, width, color='g', yerr=t_fast_std)
ax.set_ylim(0, 12)
ax.set_ylabel('simulation time, sec')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('FS', 'RS', 'LTS', 'IB'))
ax.legend((rects1[0], rects2[0]), ('During simulation', 'Precomputed'), fontsize=14)

fig2.savefig("figure_2.png", dpi=100)
