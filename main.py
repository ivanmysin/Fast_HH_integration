#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main script
"""

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


fig, axs = plt.subplots(nrows=2, ncols=1)


# neuron = get_fs_neuron()
# neuron = get_rs_neuron()
# neuron = get_LTS_neuron()
neuron = get_bursting_neuron()

neuron.runfast(0.01, 1500)
axs[0].plot(neuron.Vhist)


#t_slow = timeit.timeit("neuron1.run(0.01, 1500)", setup="", globals=globals(), number=1)
#axs[0].plot(neuron1.Vhist)
#
#
#
#neuron2 = Neuron(neurons_params)
#neuron2.addCurrent( Current(leak_current) )
#neuron2.addCurrent( SodiumCurrent(sodium_current_params) )
#neuron2.addCurrent( PotassiumCurrent(potassium_current_params) )
#neuron2.addCurrent( SlowPotassiumCurrent(slow_potassium_current_params) )
#neuron2.addCurrent( CalciumCurrentLType(l_type_calcium_current_params) )
## neuron2.runfast(0.01, 1500)
#
#t_fast= timeit.timeit("neuron2.runfast(0.01, 1500)", setup=setup, globals=globals(), number=1)
#axs[1].plot(neuron2.Vhist)
#
#print (t_slow, t_fast)