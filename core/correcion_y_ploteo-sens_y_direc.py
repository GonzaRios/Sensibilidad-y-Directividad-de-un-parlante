# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 01:00:29 2021

@author: NPass
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from suavizado import suavizado

np.seterr(divide = 'ignore') 

sens_df = pd.read_excel('Sensibilidad y Directividad - Curvas Smart.xlsx', sheet_name='Sensibilidad')
sens_values = sens_df.to_numpy()

direc_df = pd.read_excel('Sensibilidad y Directividad - Curvas Smart.xlsx', sheet_name='Directividad')
direc_values = direc_df.to_numpy()

imp_df = pd.read_excel('Sensibilidad y Directividad - Curvas Smart.xlsx', sheet_name='Impedancia')
imp_values = imp_df.to_numpy()

# =============================================================================
# SENSIBILIDAD
# =============================================================================
class Sens:
    freq = sens_values[2:,0]
    class sound:
        pink = sens_values[2:,1]
        f_100_500 = sens_values[2:,4]
        f_171 = sens_values[2:,7]
        
# =============================================================================
# DIRECTIVIDAD
# =============================================================================
class Direc:
    freq = direc_values[1:,0]
    class gr: # Grados
        _0 = direc_values[1:,1]
        _15 = direc_values[1:,6]
        _30 = direc_values[1:,11]
        _45 = direc_values[1:,16]
        _60 = direc_values[1:,21]
        _75 = direc_values[1:,26]
        _90 = direc_values[1:,31]
    class coh: # Coherencia
        _0 = direc_values[1:,3]
        _15 = direc_values[1:,8]
        _30 = direc_values[1:,13]
        _45 = direc_values[1:,18]
        _60 = direc_values[1:,23]
        _75 = direc_values[1:,28]
        _90 = direc_values[1:,33]

# =============================================================================
# IMPEDANCIA
# =============================================================================
class Imp:
    freq = imp_values[1:,0]
    class aire: #Si tiene o no masa agregada
        _sin_masa= imp_values[1:,1]
        _con_masa = imp_values[1:,5]
    class gab: # En el gabinete con o sin masa agregada
        _sin_masa= imp_values[1:,9]
        _con_masa= imp_values[1:,13]
# =============================================================================
# MEDICIONES DE REFERENCIA
# =============================================================================
        
class Ref:
    pink = 84
    f_171 = 94.8
    f_100_500 = 87.3

sens = Sens()
direc = Direc()
ref = Ref()

# =============================================================================
# CORRECCION DE LAS CURVAS DE SENSIBILIDAD y DIRECTIVIDAD
# =============================================================================

    # Valor mas cercano en frecuencia a 171 Hz
dif = np.abs(sens.freq - 171)
index_closest_171 = dif.argmin()

    # Diferencia entre la referencia y la curva del Smaart
default_dBSPL = sens.sound.f_171[index_closest_171]
ref_dBSPL = ref.f_171
dif_dBSPL = ref_dBSPL - default_dBSPL

## Corrección de sensibilidad tomando como referencia los dBSPL a 171 Hz
    # Correción de calibración
sens.sound.f_171 = sens.sound.f_171 + dif_dBSPL
sens.sound.f_100_500 = sens.sound.f_100_500 + dif_dBSPL
sens.sound.pink = sens.sound.pink + dif_dBSPL

    # Correción energetica
dif_sens_pink = sens.sound.f_171[index_closest_171] - sens.sound.pink[index_closest_171]
sens.sound.pink = sens.sound.pink + dif_sens_pink

dif_sens_100_500 = sens.sound.f_171[index_closest_171] - sens.sound.f_100_500[index_closest_171]
sens.sound.f_100_500 = sens.sound.f_100_500 + dif_sens_100_500

## Correción de directividad tomando como referencia los dBSPL a 171 Hz
direc.gr._0 = direc.gr._0 + dif_dBSPL 
# Normalización con respecto a la directividad en 0º
direc.gr._15 = (direc.gr._15 + dif_dBSPL) - direc.gr._0
direc.gr._30 = (direc.gr._30 + dif_dBSPL) - direc.gr._0
direc.gr._45 = (direc.gr._45 + dif_dBSPL) - direc.gr._0
direc.gr._60 = (direc.gr._60 + dif_dBSPL) - direc.gr._0
direc.gr._75 = (direc.gr._75 + dif_dBSPL) - direc.gr._0
direc.gr._90 = (direc.gr._90 + dif_dBSPL) - direc.gr._0
direc.gr._0 = direc.gr._0 - direc.gr._0

#Sonograma
direc_matrix = np.array([direc.gr._90,
                         direc.gr._75,
                         direc.gr._60,
                         direc.gr._45,
                         direc.gr._30,
                         direc.gr._15,
                         direc.gr._0,
                         direc.gr._15,
                         direc.gr._30,
                         direc.gr._45,
                         direc.gr._60,
                         direc.gr._75,
                         direc.gr._90])

def promedio_por_octava(GR):
    oct_bands = np.array([63,125,250,500,1000,2000,4000,8000])
    prom_per_oct = np.empty(len(oct_bands))
    for i in oct_bands:
        finf = i * 2**(-0.5)
        fsup = i * 2**(0.5)
        dif_freq_sup = np.abs(direc.freq - fsup)
        dif_freq_inf = np.abs(direc.freq - finf)
        index_fsup=np.argmin(dif_freq_sup)
        index_finf = np.argmin(dif_freq_inf)
        prom_i = np.mean(GR[index_finf:index_fsup])
        prom_per_oct[np.where(oct_bands==i)[0][0]] = prom_i 
        
    return prom_per_oct

direc_oct_0 = promedio_por_octava(direc.gr._0)
direc_oct_15 = promedio_por_octava(direc.gr._15)
direc_oct_30 = promedio_por_octava(direc.gr._30)
direc_oct_45 = promedio_por_octava(direc.gr._45)
direc_oct_60 = promedio_por_octava(direc.gr._60)
direc_oct_75 = promedio_por_octava(direc.gr._75)
direc_oct_90 = promedio_por_octava(direc.gr._90)

direc_matrix_oct = np.array([direc_oct_90,
                             direc_oct_75,
                             direc_oct_60,
                             direc_oct_45,
                             direc_oct_30,
                             direc_oct_15,
                             direc_oct_0,
                             direc_oct_15,
                             direc_oct_30,
                             direc_oct_45,
                             direc_oct_60,
                             direc_oct_75,
                             direc_oct_90])


# =============================================================================
# GRÁFICO DE SENSIBILIDAD
# =============================================================================

fig1 = plt.figure(1,[10,5])
sens_suav = suavizado(sens.freq,sens.sound.pink,3)
plt.semilogx(sens.freq,sens_suav)
# plt.xticks(np.array([500,8000]))
plt.grid()
plt.xlabel('Frecuencia [Hz]', fontsize = 18)
x_label1 = [r"$63$",r"$125$",r"$250$",r"$500$",r"$1 k$",r"$2 k$",r"$4 k$",r"$8 k$"]
plt.xticks(np.array([63,125,250,500,1000,2000,4000,8000]),x_label1, fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([44,12000])
plt.ylabel('Nivel de presión [dB SPL]', fontsize = 18)
# plt.title('Sensibilidad',fontsize=20)
plt.savefig('Sensibilidad.png')

index_100 = np.argmin(np.abs(100-sens.freq))
index_500 = np.argmin(np.abs(500-sens.freq))
sens_100_500 = np.mean(sens.sound.pink[index_100:index_500])
sens_pink = np.mean(sens.sound.pink)



# =============================================================================
# GRAFICO DE DIRECTIVIDAD (SONOGRAMA)
# =============================================================================
fig2 = plt.figure(2,[10,5])
ax2 = plt.subplot(111)
ax2.set_xscale("log")
plot = plt.contourf(direc.freq,np.arange(13),(direc_matrix.astype('float64'))) # Lista de colores de cmap: https://matplotlib.org/stable/tutorials/colors/colormaps.html
x_label1 = [r"$63$",r"$125$",r"$250$",r"$500$",r"$1 k$",r"$2 k$",r"$4 k$",r"$8 k$"]
plt.xticks(np.array([63,125,250,500,1000,2000,4000,8000]),x_label1,fontsize=18)
plt.xlim([44,12000])
y_label = [r"$90^o$",r"$45^o$", r"$0^o$", r"$45^o$",r"$90^o$"]
plt.yticks(np.array([0,3,6,9,12]),y_label, fontsize=18)
plt.xlabel('Frecuencia [Hz]', fontsize = 18)
plt.ylabel('Ángulo [$^o$]', fontsize = 18)
cbar = fig2.colorbar(plot)
cbar.ax.set_ylabel('Nivel relativo [$dB_{re}$]', fontsize = 18)
cbar.ax.tick_params(labelsize=14)
# plt.title('Sonograma',fontsize=20)
plt.savefig('Sonograma.png')

# =============================================================================
# PATRÓN POLAR
# =============================================================================

# Datos a plotear    
polar_63= direc_matrix_oct[:,0]
polar_125 = direc_matrix_oct[:,1]
polar_250 = direc_matrix_oct[:,2]
polar_500 = direc_matrix_oct[:,3]
polar_1k = direc_matrix_oct[:,4]
polar_2k = direc_matrix_oct[:,5]
polar_4k = direc_matrix_oct[:,6]
polar_8k = direc_matrix_oct[:,7]

#############################
### Directividad para 63, 250, 1k, 4k
s = pd.Series(np.arange(1))
theta=np.arange(-np.pi/2,np.pi/2+0.1,np.pi/12)

fig3 = plt.figure(figsize=(16,8))
ax3 = plt.subplot(111, projection = 'polar')
ax3.plot(theta,polar_63,linestyle = '--',lw=2,label='63 Hz')  
ax3.plot(theta,polar_250,linestyle = '--',lw=2,label='250 Hz')  
ax3.plot(theta,polar_1k,linestyle = '--',lw=2,label='1 kHz')  
ax3.plot(theta,polar_4k,linestyle = '--',lw=2,label='4 kHz')  


#Configuración del ploteo
ax3.set_theta_zero_location( 'N')
ax3.set_rorigin(-30)
ax3.set_thetamin(-90)
ax3.set_thetamax(90)

gr_label = [r"$90^o$",r"$30^o$",r"$60^o$", r"$0^o$", r"$60^o$",r"$30^o$",r"$90^o$"]
plt.xticks(np.array([ -np.pi/2 , -np.pi/6 , -np.pi/3 , 0 , np.pi/3 , np.pi/6 , np.pi/2 ]),gr_label,fontsize=14)
dB_label = [r"$-25.0$",r"$-20.0$", r"$-15.0$", r"$-10.0$",r"$-5.0$",r"$0.0$"]
plt.yticks( np.array([-25,-20,-15,-10,-5,0]),dB_label,fontsize=18)
plt.legend(loc='upper right', fontsize=14)
# plt.title('Patrón polar - 63 Hz, 250 Hz, 1 kHz, 4 kHz',fontsize=20)
plt.savefig('Patrón polar - 63 Hz, 250 Hz, 1 kHz, 4 kHz.png')

#############################
### Directividad para 125, 500, 2k, 8k
s = pd.Series(np.arange(1))
theta=np.arange(-np.pi/2,np.pi/2+0.1,np.pi/12)

fig4 = plt.figure(figsize=(16,8))
ax4 = plt.subplot(111, projection = 'polar')
ax4.plot(theta,polar_125,linestyle = '--',lw=2,label='125 Hz')  
ax4.plot(theta,polar_500,linestyle = '--',lw=2,label='500 Hz')  
ax4.plot(theta,polar_2k,linestyle = '--',lw=2,label='2 kHz')  
ax4.plot(theta,polar_8k,linestyle = '--',lw=2,label='8 kHz')  

#Configuración del ploteo
ax4.set_theta_zero_location( 'N')
ax4.set_rorigin(-30)
ax4.set_thetamin(-90)
ax4.set_thetamax(90)

gr_label = [r"$90^o$",r"$30^o$",r"$60^o$", r"$0^o$", r"$60^o$",r"$30^o$",r"$90^o$"]
plt.xticks(np.array([ -np.pi/2 , -np.pi/6 , -np.pi/3 , 0 , np.pi/3 , np.pi/6 , np.pi/2 ]),gr_label,fontsize=14)
dB_label = [r"$-25.0$",r"$-20.0$", r"$-15.0$", r"$-10.0$",r"$-5.0$",r"$0.0$"]
plt.yticks( np.array([-25,-20,-15,-10,-5,0]),dB_label,fontsize=18)
plt.legend(loc='upper right', fontsize=14)
# plt.title('Patrón polar - 125 Hz, 500 Hz, 2 kHz, 8 kHz',fontsize=20)
plt.savefig('Patrón polar - 125 Hz, 500 Hz, 2 kHz, 8 kHz.png')

# =============================================================================
# IMPEDANCIA
# =============================================================================

############# AL AIRE
eje_x = Imp.freq
asm_y1 = Imp.aire._sin_masa
acm_y2 = Imp.aire._con_masa



fig5= plt.figure(figsize=(10,5))
ax5= plt.subplot(111)
ax5.semilogx(eje_x,asm_y1, linestyle = '-',lw=2,label='Sin masa agregada')
ax5.semilogx(eje_x,acm_y2, linestyle = '--',lw=2,label='Con masa agregada')
x_label1 = [r"$31.5$",r"$63$",r"$125$",r"$250$",r"$500$",r"$1 k$",r"$2 k$",r"$4 k$",r"$8 k$",r"$12 k$"]
plt.xticks(np.array([31.5,63,125,250,500,1000,2000,4000,8000,12000]),x_label1,fontsize=18)
plt.xlim([20,10000])
plt.ylim([0,35])
y_label=[r"$0$",r"$5$",r"$10$",r"$15$",r"$20$",r"$25$",r"$30$",r"$35$",r"$40$"]
plt.yticks(np.array([0,5,10,15,20,25,30,35,40]),y_label,fontsize=18)
plt.xlabel('Frecuencia [Hz]', fontsize=18)
plt.ylabel('Impedancia [Ohm]', fontsize=18)
plt.grid()
plt.legend(loc='upper right', fontsize=16)
# plt.title('Curvas de Impedancia con y sin masa agregada al aire libre',fontsize=20)
plt.savefig('Impedancia - Aire libre.png')

########## EN GABINETE
gsm_y1 = Imp.gab._sin_masa 
gcm_y2 =Imp.gab._con_masa


fig6= plt.figure(figsize=(10,5))
ax6= plt.subplot(111)
ax6.semilogx(eje_x,gsm_y1, linestyle = '-',lw=2,label='Sin masa agregada')
ax6.semilogx(eje_x,gcm_y2, linestyle = '--',lw=2,label='Con masa agregada')
x_label1 = [r"$31.5$",r"$63$",r"$125$",r"$250$",r"$500$",r"$1 k$",r"$2 k$",r"$4 k$",r"$8 k$",r"$12 k$"]
plt.xticks(np.array([31.5,63,125,250,500,1000,2000,4000,8000,12000]),x_label1,fontsize=18)
plt.xlim([20,10000])
plt.ylim([0,35])
y_label=[r"$0$",r"$5$",r"$10$",r"$15$",r"$20$",r"$25$",r"$30$",r"$35$",r"$40$"]
plt.yticks(np.array([0,5,10,15,20,25,30,35,40]),y_label,fontsize=18)
plt.xlabel('Frecuencia [Hz]', fontsize=18)
plt.ylabel('Impedancia [Ohm]', fontsize=18)
plt.grid()
plt.legend(loc='upper right', fontsize=16)
# plt.title('Curvas de Impedancia con y sin masa agregada en gabinete',fontsize=20)
plt.savefig('Impedancia - Gabinete.png')

