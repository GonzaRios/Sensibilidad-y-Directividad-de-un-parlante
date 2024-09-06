# -*- coding: utf-8 -*-

import numpy as np

# =============================================================================
# # F debe ser una lista de las frecuencias entregadas por la FFT de menor a mayor
# # AMP debe ser una lista de las amplitudes en dB de la FFT
# # OCT debe ser la subdivisión de octava (ej. OCT=3 -> Suavizado por tercios de octava)
# # OCT=0 es un caso especial donde no se realiza el suavizado y se devuelve la curva intacta
# # La función devuelve una lista con la curva suavizada de amplitudes en dB.
# 
# =============================================================================
    
def suavizado(F,AMP,OCT):
    ampsmooth=AMP
    if OCT!=0:
        for n in range(1,len(F)):
            fsup=F[n]*pow(2,1/(2*OCT))  #calcula el corte superior del promedio
            finf=F[n]*pow(2,1/(2*OCT))  #calcula el corte inferior del promedio

            if F[-1]<=fsup:
                idxsup=len(F)-n
            else:
                idxsup=np.argmin(abs(F[n:]-fsup))   #busca el índice de fsup
                
            if F[1]<=finf:
                idxinf=np.argmin(abs(F[0:n+1]-finf))    #busca el ínfice de finf
            else:
                idxinf=0
                
            if idxsup!=idxinf+n:
                temp=pow(10,AMP[idxinf:idxsup+n-1]*0.1)
                ampsmooth[n]=10*np.log10(sum(temp)/(idxsup+n-idxinf))
    return ampsmooth
