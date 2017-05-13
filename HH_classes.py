#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fast integration
"""

import numpy as np
exp = np.exp
import matplotlib.pyplot as plt


class Current:
    def __init__ (self, params):
        self.I = -params["Iext"]
        
    def updateI(self, V, dt):
        pass
    
    def init_gate_vars(self, V):
        pass
    
    def precumpute_funtions(self, Vmin, Vmax, step, dt):
        pass
    
    def updateIfast(self, V, dt):
        pass
    
    def getI(self, V):
        return self.I

class LeakCurrent(Current):
    def __init__(self, params):
        self.Erev = params["Erev"]
        self.gmax = params["gmax"]
    
    def getI(self, V):
        return self.gmax * (V - self.Erev)
    

class SodiumCurrent(LeakCurrent):
    
    m_inf_fast = 0
    h_inf_fast = 0
    h_T_fast = 0
    
    def __init__(self, params):
        super().__init__(params)
        self.Vt = params["Vt"]
        self.m = 0
        self.h = 0
        
        
    
    def get_alpha_m(self, V):
        x = V - self.Vt - 13
        if ( np.any(x == 0) ):
            x += 0.0000001
        alpha_m = -0.32 * x / (exp( -0.25 * x ) - 1)
        return alpha_m
    
    def get_beta_m(self, V):
        x =  V - self.Vt - 40
        if ( np.any(x == 0) ):
            x += 0.0000001
        beta_m = 0.28 * x / ( exp( 0.2 * x ) - 1 )
        return beta_m

    def get_alpha_h(self, V):
        alpha_h = 0.128 * exp(-(V - self.Vt - 17) / 18)
        return alpha_h
    
    def get_beta_h(self, V):
        beta_h = 4 / ( 1 + exp( -0.2*(V - self.Vt - 40) ) )
        return beta_h

    def init_gate_vars(self, V):
        alpha_m = self.get_alpha_m(V)
        beta_m = self.get_beta_m(V)
        self.m = alpha_m / (alpha_m + beta_m)
        
        alpha_h = self.get_alpha_h(V)
        beta_h = self.get_beta_h(V)
        self.h = alpha_h / (alpha_h + beta_h)
    
    
    def updateI(self, V, dt):
        alpha_m = self.get_alpha_m(V)
        beta_m = self.get_beta_m(V)
        self.m = alpha_m / (alpha_m + beta_m)
        
        alpha_h = self.get_alpha_h(V)
        beta_h = self.get_beta_h(V)
        h_inf = alpha_h / (alpha_h + beta_h)
        tau_inf = 1 / (alpha_h + beta_h)
        
        self.h = h_inf - (h_inf - self.h) * exp(-dt/tau_inf)
    
    
    def precumpute_funtions(self, Vmin, Vmax, step, dt):
        
        if ( np.any(self.__class__.m_inf_fast > 0) ):
            return
        
        self.__class__.Vmin = Vmin
        self.__class__.step = step
        Vrange = np.arange(Vmin, Vmax, step)
        
        self.__class__.m_inf_fast = self.get_alpha_m(Vrange) / (self.get_alpha_m(Vrange) + self.get_beta_m(Vrange))
        
        
        tau_h = 1 / (self.get_alpha_h(Vrange) + self.get_beta_h(Vrange))
    
        self.__class__.h_inf_fast = self.get_alpha_h(Vrange) * tau_h 
    
        self.__class__.h_T_fast = exp(-dt / tau_h)
    
    
    def updateIfast(self, V, dt):
        
        idx = int( (V - self.__class__.Vmin) / self.__class__.step )

        self.m = self.__class__.m_inf_fast[idx]

        h_inf = self.__class__.h_inf_fast[idx]
        T_h = self.__class__.h_T_fast[idx]
        
        self.h = h_inf - (h_inf - self.h) * T_h
    
    
    
    def getI(self, V):
        return self.gmax * self.m * self.m * self.m * self.h * (V - self.Erev)


class PotassiumCurrent(LeakCurrent):
    n_inf_fast = 0
    n_T_fast = 0
    
    def __init__(self, params):
        super().__init__(params)
        self.Vt = params["Vt"]
        self.n =  0
        
    
    def get_alpha_n(self, V):
        x = V - self.Vt - 15
        if (np.any(x == 0) ):
            x += 0.0000001
        
        alpha_n = -0.032 * x / (exp(-0.2 * x) - 1 )
        return alpha_n
    
    def get_beta_n(self, V):
        beta_n = 0.5 * exp(-(V - self.Vt - 10)/40)
        return beta_n

    
    def updateI(self, V, dt):
       
        alpha_n = self.get_alpha_n(V)
        beta_n = self.get_beta_n(V)
        n_inf = alpha_n / (alpha_n + beta_n)
        tau_inf = 1 / (alpha_n + beta_n)
        
        self.n = n_inf - (n_inf - self.n) * exp(-dt/tau_inf)
    
    def init_gate_vars(self, V):
        alpha_n = self.get_alpha_n(V)
        beta_n = self.get_beta_n(V)
        self.n = alpha_n / (alpha_n + beta_n)
    
    def getI(self, V):
        return self.gmax * self.n * self.n * self.n * self.n * (V - self.Erev) 



    def precumpute_funtions(self, Vmin, Vmax, step, dt):
        
        if ( np.any(self.__class__.n_inf_fast > 0) ):
            return
        
        self.__class__.Vmin = Vmin
        self.__class__.step = step
        Vrange = np.arange(Vmin, Vmax, step)
        
        tau_n = 1 / (self.get_alpha_n(Vrange) + self.get_beta_n(Vrange))
        
        
        self.__class__.n_inf_fast = self.get_alpha_n(Vrange) * tau_n
    
        self.__class__.n_T_fast = exp(-dt / tau_n)
    
    
    def updateIfast(self, V, dt):
        
        idx = int( (V - self.__class__.Vmin) / self.__class__.step )

        n_inf = self.__class__.n_inf_fast[idx]
        T_n = self.__class__.n_T_fast[idx]
        
        self.n = n_inf - (n_inf - self.n) * T_n



class SlowPotassiumCurrent(LeakCurrent):
    p_inf_fast = 0
    p_T_fast   = 0
    def __init__(self, params):
        super().__init__(params)
        self.tau_max = params["tau_max"]
        self.p = 0
        
    def get_p_inf(self, V):

        p_inf =  1 / ( 1 + exp( -0.1*(V + 35) ) )
        return p_inf
    
    def get_tau_inf(self, V):
        
        
        tau = self.tau_max / ( 3.3*exp( (V + 35)/20 ) + exp( -(V + 35)/20) )
        return tau
    
    def init_gate_vars(self, V):
        self.p = self.get_p_inf(V)
    
    
    def updateI(self, V, dt):
       
        p_inf = self.get_p_inf(V)
        tau_inf = self.get_tau_inf(V)
        self.p = p_inf - (p_inf - self.p) * exp(-dt/tau_inf)
    
    def getI(self, V):
        return self.gmax * self.p * (V - self.Erev)

    
    def precumpute_funtions(self, Vmin, Vmax, step, dt):
        
        if ( np.any(self.__class__.p_inf_fast > 0) ):
            return
        
        self.__class__.Vmin = Vmin
        self.__class__.step = step
        Vrange = np.arange(Vmin, Vmax, step)
        
        tau_p = self.get_tau_inf(Vrange)
       
        self.__class__.p_inf_fast = self.get_p_inf(Vrange)
    
        self.__class__.p_T_fast = exp(-dt / tau_p)
    
    
    def updateIfast(self, V, dt):
        
        idx = int( (V - self.__class__.Vmin) / self.__class__.step )

        p_inf = self.__class__.p_inf_fast[idx]
        T_p = self.__class__.p_T_fast[idx]
        
        self.p = p_inf - (p_inf - self.p) * T_p
        
  
        
        
        
    
class CalciumCurrentLType(LeakCurrent):
    q_inf_fast = 0
    r_inf_fast = 0
    q_T_fast = 0
    r_T_fast = 0
    
    
    def __init__(self, params):
        super().__init__(params)
        self.q = 0
        self.r = 0
    
    def get_alpha_q(self, V):
        x = -27 - V
        if ( np.any(x == 0) ):
            x += 0.0000001
        
        alpha = 0.055 * x / (exp(x/3.8) - 1)
        return alpha
    
    def get_beta_q(self, V):
         beta = 0.94 * exp( (-75 - V )/17 )
         return beta
    
    def get_alpha_r(self, V):
        alpha = 0.000457 * exp( (-13 - V )/50 )
        return alpha
    
    def get_beta_r(self, V):
        beta = 0.0065 / ( exp((-15 - V )/28) + 1 )
        return beta

    def init_gate_vars(self, V):
        alpha_q = self.get_alpha_q(V)
        beta_q = self.get_beta_q(V)
        self.q = alpha_q / (alpha_q + beta_q)
        
        alpha_r = self.get_alpha_r(V)
        beta_r = self.get_beta_r(V)
        self.r = alpha_r / (alpha_r + beta_r)

    def updateI(self, V, dt):
        alpha_q = self.get_alpha_q(V)
        beta_q = self.get_beta_q(V)
        tau_q_inf = 1 / (alpha_q + beta_q)
        q_inf = alpha_q * tau_q_inf
        
        self.q = q_inf - (q_inf - self.q) * exp(-dt/tau_q_inf)
        
        alpha_r = self.get_alpha_r(V)
        beta_r = self.get_beta_r(V)
        tau_r_inf = 1 / (alpha_r + beta_r)
        r_inf = alpha_r * tau_r_inf
        
        self.r = r_inf - (r_inf - self.r) * exp(-dt/tau_r_inf)
    
    def getI(self, V):
        return self.gmax * self.q * self.q * self.r * (V - self.Erev)
    
    
    def precumpute_funtions(self, Vmin, Vmax, step, dt):
        
        if ( np.any(self.__class__.q_inf_fast > 0) ):
            return
        
        self.__class__.Vmin = Vmin
        self.__class__.step = step
        Vrange = np.arange(Vmin, Vmax, step)
        
        tau_q = 1 / (self.get_alpha_q(Vrange) + self.get_beta_q(Vrange))

        self.__class__.q_inf_fast = self.get_alpha_q(Vrange) * tau_q
        self.__class__.q_T_fast = exp(-dt / tau_q)

        tau_r = 1 / (self.get_alpha_r(Vrange) + self.get_beta_r(Vrange))

        self.__class__.r_inf_fast = self.get_alpha_r(Vrange) * tau_r
        self.__class__.r_T_fast = exp(-dt / tau_r)

    
    
    def updateIfast(self, V, dt):
        
        idx = int( (V - self.__class__.Vmin) / self.__class__.step )

        q_inf = self.__class__.q_inf_fast[idx]
        T_q = self.__class__.q_T_fast[idx]
        
        self.q = q_inf - (q_inf - self.q) * T_q
        

        r_inf = self.__class__.r_inf_fast[idx]
        T_r = self.__class__.r_T_fast[idx]
        
        self.r = r_inf - (r_inf - self.r) * T_r  
    
    

class CalciumCurrentTType(LeakCurrent):
    s_inf_fast = 0
    u_inf_fast = 0
    u_T_fast   = 0
    
    
    def __init__(self, params):
        super().__init__(params)
        self.Vx = params["Vx"]
        self.s = 0
        self.u = 0
    
    def get_s_inf(self, V):
        s_inf = 1 / ( 1 + exp( -(V + self.Vx + 57)/6.2) )
        return s_inf
    
    def get_u_inf(self, V):
        u_inf = 1 / ( 1 + exp( (V + self.Vx + 81)/4 ) )
        return u_inf
    
    def get_tau_u_inf(self, V):
        tau = (30.8 + ( 211.4 + exp( (V + self.Vx + 113.2)/5 ) ) ) / ( 3.7*(1 + exp( (V + self.Vx + 84)/3.2) ) )
        return tau

        
    def init_gate_vars(self, V):
        self.s = self.get_s_inf(V)
        self.u = self.get_u_inf(V)    
        
    def updateI(self, V, dt):
        self.s = self.get_s_inf(V)
        u_inf = self.get_u_inf(V)  
        tau_u_inf = self.get_tau_u_inf(V)
        self.u = u_inf - (u_inf - self.u) * exp(-dt/tau_u_inf)
    
    def getI(self, V):
        return self.gmax * self.s * self.s * self.u * (V - self.Erev)     
  
    def precumpute_funtions(self, Vmin, Vmax, step, dt):
        
        if ( np.any(self.__class__.s_inf_fast > 0) ):
            return
        
        self.__class__.Vmin = Vmin
        self.__class__.step = step
        
        Vrange = np.arange(Vmin, Vmax, step)
        
        self.__class__.s_inf_fast = self.get_s_inf(Vrange)
        
        self.__class__.u_inf_fast = self.get_u_inf(Vrange)
        
        self.__class__.u_T_fast = exp( -dt / self.get_tau_u_inf(Vrange) )
        

    
    
    def updateIfast(self, V, dt):
        
        idx = int( (V - self.__class__.Vmin) / self.__class__.step )

        
        self.s = self.__class__.s_inf_fast[idx]
        

        u_inf = self.__class__.u_inf_fast[idx]
        T_u = self.__class__.u_T_fast[idx]
        
        self.u = u_inf - (u_inf - self.u) * T_u  

class Neuron:
    
    def __init__(self, params):
        self.currents = []
        
        self.V = params["V0"]
        self.C = params["C"]
        self.Vhist = [self.V]

    def addCurrent(self, current):
        current.init_gate_vars(self.V)
        self.currents.append(current)
        
    
    def updateV(self, dt):
        I = 0
        for i in self.currents:
            i.updateI(self.V, dt)
            
            
            I -= i.getI(self.V)
        
        self.V += dt * (I / self.C)
        
        self.Vhist.append(self.V)
        
    def updateVfast(self, dt):
        I = 0
        for i in self.currents:
            i.updateIfast(self.V, dt)
            I -= i.getI(self.V)
            
        
        self.V += dt * (I / self.C)
        
        self.Vhist.append(self.V)
    
    def run(self, dt=0.01, duration=200):
        
   
       
        t = 0
        while(t <= duration):
            self.updateV(dt)
            
            t += dt
            
            
    def runfast(self, dt=0.01, duration=200):
        
        Vmin = -100.0
        Vmax = 60.0
        step = 0.01
        
        for i in self.currents:
            i.precumpute_funtions(Vmin, Vmax, step, dt)
        
       
        t = 0
        while(t <= duration):
            self.updateVfast(dt)
            
            t += dt

###############################################################################











