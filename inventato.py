# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:34:08 2020

@author: franc
"""

from random import *
#import random
from math import *
import numpy as np
import time 
import matplotlib.pyplot as plt
import csv
import pandas as pd

# constant:
T_0=1
q=3/5
E_Y=10
a=1/2 # =>gamma(2+1)=2!=2
N=20


def beta(ro, N=N, t0=T_0, q=q, EY=E_Y):
    return (ro*N*(t0+(1-q)*EY))/2

def sample_inter_arrival(T0=T_0, Y_mean=E_Y):
    r1, r2 = random(), random()
    T=None
    if r1<q:
        T=T0
    else:
        T=round(T0-Y_mean*log(r2), 0)
    return T

def sample_processing_time(bet, a=a):
    r3=random()
    return max(1, min(100*2*bet, round(bet*(-log(r3))**(1/a), 0)))


def longest_residual_time(servers):
    vuota = []
    for i in range(N):
        vuota.append(sum(servers[i][1]))
    return max(vuota)

def time_goes_on_server(server, dt):
    #print(server[1][0], dt)
    try:
        #print(server[1][0], dt)
        if server[1][0] <= dt:
         #   print('e')
            residual = dt - server[1][0]
          #  print('a')
            del server[0][0]
            del server[1][0]
            if residual > 0:
           #     print('b')
                time_goes_on_server(server, residual)
        else:
            #print('c')
            #server1 = server
            server[1][0] -= dt
    except:
      #  print('d')
        pass
    return


# our
def shorter_queue_time_inventato(servers):
    vuota = []
    for i in range(N):
        vuota.append(sum(servers[i][1]))
    return vuota.index(min(vuota))

##### prove jsq
def choose_with_jsq(servers):
    vuota = []
    for i in range(N):
        vuota.append(len(servers[i][0]))
    return vuota.index(min(vuota))   


##### prove pod
def choose_with_pod_d(servers):
    vuota = []
    a = sample(list_servers, 3)
    for i in a:
        vuota.append(len(servers[i][0]))
    return a[vuota.index(min(vuota))]


##### prove JBT-d
def threshold_with_jbt_d(servers):
    vuota = []
    for i in sample(list_servers, 3):
        vuota.append(len(servers[i][0]))
    threshold = min(vuota)
    return threshold

def build_ones__with_jbt_d(servers, threshold):
    ones = []
    for i in range(N):
        if len(servers[i][0])>=threshold:
            ones.append(i)
    return ones

def choose_with_jbt_d(servers, ones):
    if len(ones) != 0:
        out  = sample(ones, 1)
        ones.remove(out[0])
    else:
        out  = sample(list_servers, 1)
    return out[0]





#%%


  
inter_iter = 20
rhos = np.append(np.arange(0.8, 1, 0.02, dtype = float), 0.99).round(2)
print('#ro', len(rhos), 'rhos',  rhos)
iterazioni=11000
scarto = 1000


list_servers = [i for i in range(N)]

#%%

start=time.time()

E_D_rho = []
real_rho = []
sd = []
ci = []
for rho in rhos:
    print(rho, time.time()-start)
    # evaluate beta
    bet = beta(rho)
    E_D = []
    before_mean_rhos = []
    for _ in range(inter_iter):         # this loop is for the mean
        arrivals_samples=[int(round(sample_inter_arrival(), 0)) for i in range(iterazioni)]   
        arrival_times=list(np.cumsum(arrivals_samples))
        arrival_times.insert(0,0)
        E_T = np.mean(arrivals_samples)

        '''
        only for JBT
        '''
        jbt_sample = np.diff(np.array(arrival_times)%1000)

        # for each server i built a list  [[task_1, .., task_n], [time_task_1, .., time_task_n]]
        servers = {i:[[], []] for i in range(N)}
        time_inside_servers = []
        # sample processing_time w.r.t. beta(rho)
        processing_samples=[int(round(sample_processing_time(bet), 0)) for i in range(iterazioni)]
        E_X = np.mean(processing_samples)
        before_mean_rhos.append(E_X/(N*E_T))
        #print(processing_samples)
        for i in range(iterazioni):
            t = arrival_times[i+1] # this time
            dt = t - arrival_times[i] # this - previous 
            for j in range(N):
                time_goes_on_server(servers[j], dt)
            '''
            nostro inventato
            '''
            #shorter = shorter_queue_time_inventato(servers)
            
            '''
            jsq
            '''
            #shorter = choose_with_jsq(servers)
            
            
            '''
            Pod_d
            '''
            #shorter = choose_with_pod_d(servers)
            

            '''
            JBT_d
            '''
            if arrival_times[i] == 0 or jbt_sample[i] <0:
              #print(arrival_times[i], jbt_sample[i], i)
              threshold = threshold_with_jbt_d(servers)
              ones = build_ones__with_jbt_d(servers, threshold)
            shorter = choose_with_jbt_d(servers, ones)




            servers[shorter][0].append(i)
            servers[shorter][1].append(processing_samples[i])
            time_inside_servers.append(sum(servers[shorter][1]))
        
        E_D.append(np.mean(time_inside_servers[scarto:]))
        #tempo_fin = arrival_times[-1] + longest_residual_time(servers)
    E_D_rho.append(np.mean(E_D))
    real_rho.append(np.mean(before_mean_rhos))
    sd.append(np.std(E_D))
    ci.append((np.std(E_D)*1.96)/np.sqrt(len(real_rho)))
            
            
print(time.time()-start)       


#%%

# save file output



#mean msg
mean_msg = [0 for i in range(len(real_rho))]


d  = {'Rho' : rhos, 'Mean_Time':E_D_rho,'Mean_Msg': mean_msg, 'Rho_Emp':real_rho,'CI': ci}
df = pd.DataFrame(data=d)    
    
df.to_csv('nostro.csv', index = False)





#%%
# plot
vuota = []
for i in range(N):
    vuota.append(sum(servers[i][1]))


print(np.array(vuota)/ sum(vuota))


plt.plot(real_rho,E_D_rho)


