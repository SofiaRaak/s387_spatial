import numpy as np
import cma
import numba
import os

amp, centre, sigma = (0.6969641043116005, 1.4904524578404597, 1.3102271681828053)

x_data = [0, 3, 6, 10, 20]

Aktp_data = [1.6, 2.3, 1.87, 1.8, 0.66]
Agop_data = [1.5, 2.6, 2.9, 2.1, 0.79]
Agopgw_data = [1.2, 1.5, 2.85, 2.15, 1.07]

Aktp_data_SEM = [0.17, 0.19, 0.26, 0.7, 0.27]
Agop_data_SEM = [0.22, 0.57, 0.23, 0.17, 0.42]
Agopgw_data_SEM = [0.0, 0.11, 0.21, 0.18, 0.17]

data = np.array([[Aktp_data], [Agop_data], [Agopgw_data]])
data_sem = np.array([[Aktp_data_SEM], [Agop_data_SEM], [Agopgw_data_SEM]])

nmda_x = [108.77762801045833-120, 125.66705100930517-120, 147.06140279649867-120, 168.61165694983492-120, 188.7003264205789-120, 208.8734430063173-120, 224.78198028484644-120, 252.21430079412767-120, 266.70672491352275-120, 292.67421277425-120, 308.54702242720464-120, 327.3170177176539-120, 350.40031180473215-120, 369.15731523133604-120,]
nmda_y = [0.2911963882618509, 0.5530474040632054, 0.6839729119638827, 0.7065462753950338, 0.7449209932279908, 0.7246049661399547, 0.6681715575620766, 0.6027088036117381, 0.5304740406320541, 0.4830699774266365, 0.4514672686230248, 0.4063205417607221, 0.3634311512415347, 0.3273137697516928,]
nmda_sem = [0.3702031602708802, 0.6817155756207673, 0.8600451467268622, 0.8893905191873588, 0.9638826185101579, 0.9480812641083518, 0.8781038374717831, 0.8239277652370203, 0.7223476297968396, 0.6681715575620766, 0.6094808126410833, 0.5643340857787809, 0.5079006772009027, 0.46275395033860045]
nmda_sem = np.array(nmda_sem) - np.array(nmda_y)
nmda_x = np.array(nmda_x) / 60

nmda_y = np.array(nmda_y)

##others
dt = 0.01
minutes = 20
numtimesteps = minutes/dt #100

#rate constants
k1 = 1# 0.4513
k_1 = 1# 0.1
k2 = 1# 0.1332
k_2 = 1# 6.65
k3 = 1# 0.3925
k_3 = 1# 0.2
k4 = 1# 0.017 #* 10
k_4 = 1# 0.017 #* 0
k5 = 1# 0.35 #* 0
k_5 = 1# 0.017 #* 0

Akt_init = 1# 0.686
Aktpb_init = 1# 0.01
Aktpf_init = 1# 0.01
Ago_init = 1# 78
Agop_init = 1# 0.01
Gw_init = 1# 0.71
Agopgw_init = 1# 0.01

rates = [k1, k_1, k2, k_2, k3, k_3]
inits = [Akt_init, Aktpb_init,  Ago_init, Agop_init, Gw_init, Agopgw_init]

theta = np.array(rates + inits)

beg = 0

@numba.jit(target_backend = 'cuda')
def model1(theta):
   
    def gauss(x, amp, centre, sigma):
        return amp*np.exp(-(x - centre)**2 / (2*sigma)**2)
   
    k1, k_1, k2, k_2, k3, k_3, Akt0, Aktp0, Ago0, Agop0, Gw0, Agopgw0 = np.exp(theta)
   
    Akt = np.zeros(int((minutes - beg)/dt))
    Aktp = np.zeros(int((minutes - beg)/dt))
    Ago = np.zeros(int((minutes - beg)/dt))
    Agop = np.zeros(int((minutes - beg)/dt))
    Gw = np.zeros(int((minutes - beg)/dt))
    Agopgw = np.zeros(int((minutes - beg)/dt))
    time = np.linspace(int(beg), int(minutes), int((minutes-beg)/dt))
   
    NMDA = gauss(time, amp, centre, sigma)
   
    Akt[0] = Akt0
    Aktp[0] = Aktp0
    Ago[0] = Ago0
    Agop[0] = Agop0
    Gw[0] = Gw0
    Agopgw[0] = Agopgw0
   
    for i in range(1, int((minutes-beg)/dt)):
        Akt[i] = Akt[i-1] + dt*(Aktp[i-1]*k_1 - Akt[i-1]*k1*NMDA[i])
        Aktp[i] = Aktp[i-1] + dt*(Akt[i-1]*k1*NMDA[i] - Aktp[i-1]*k_1)
        Ago[i] = Ago[i-1] + dt*(Agop[i-1]*k_2 - Aktp[i-1]*Ago[i-1]*k2)
        Agop[i] = Agop[i-1] + dt*(Aktp[i-1]*Ago[i-1]*k2 + Agopgw[i-1]*k_3 - Agop[i-1]*k_2 - Agop[i-1]*Gw[i-1]*k3)
        Gw[i] = Gw[i-1] + dt*(Agopgw[i-1]*k_3 - Agop[i-1]*Gw[i-1]*k3)
        Agopgw[i] = Agopgw[i-1] + dt*(Agop[i-1]*Gw[i-1]*k3 - Agopgw[i-1]*k_3)
       
    return Akt, Aktp, Ago, Agop, Gw, Agopgw, NMDA, time

#first simple model function fc calculation
def foldChange1(theta):
    Akt, Aktp, Ago, Agop, Gw, Agopgw, NMDA, time = model1(theta)
   
    Aktp_fold = np.zeros(len(Akt))
    Agop_fold = np.zeros(len(Akt))
    Agopgw_fold = np.zeros(len(Akt))
   
    for i in range(len(Akt)):
            Aktp_fold[i] = Aktp[i] / ( Aktp[0] )#+ Akt[0]) / akcont
            Agop_fold[i] = (Agop[i] + Agopgw[i]) / ( Agop[0]  + Agopgw[0] )#+ Ago[0] ) / agcont
            Agopgw_fold[i] = Agopgw[i] / ( Agopgw[0] )#+ Gw[0]) / gwcont
           
    return Aktp_fold, Agop_fold, Agopgw_fold, time

def error1(theta):
    Aktp_fold, Agop_fold, Agopgw_fold, time = foldChange1(theta)
   
    aktp_model = list(np.interp(x_data, time, Aktp_fold))
    agop_model = list(np.interp(x_data, time, Agop_fold))
    agopgw_model = list(np.interp(x_data, time, Agopgw_fold))
   
    model = np.array([[aktp_model], [agop_model], [agopgw_model]])
   
    return np.sum((data - model)**2)

cma.fmin(error1, theta, 2)

os.renames('outcmaes', 'outcmaes_all-optim_gpu2')