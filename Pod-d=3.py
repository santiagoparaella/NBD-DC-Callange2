from random import *
from math import *
import numpy as np
import time as tempo
import matplotlib.pyplot as plt
import csv

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

def min_queue(servers, server_queue):
    m = len(server_queue[servers[0]])
    s = servers[0]
    for server in servers[1:]:
        if len(server_queue[server]) < m:
            m = len(server_queue[server])
            s = server
    if len(server_queue[s])>0:
        print('Coda del server minore non vuota!!!!!!!!!!!!!')
    return s

def check(server_queue, server_in, dispatcher_queue, general_task):
    #ritorna true se deve continuare
    
    #controllo coda disptcher
    if len(dispatcher_queue)!=0:
        return True
    #controllo code dei server
    for v in server_queue.values():
        if len(v)!=0:
            return True
    #controllo servers
    if len(list(server_in.values()))!=0:
        return True

    #controllo sui task generale
    for v in general_task.values():
        if v==0:
            return True
    

    #altrimenti continua    
    return False

'''
# test di convergenza E ~ Empirical E

b=beta(0.8) 

print('Beta = ', b)

for iter in [1000, 10000, 100000, 1000000]:
    print('Sample size : ', iter, '\n')
    s1=0
    for i in range(iter):
        s1+=round(sample_processing_time(b), 0)

    print('E[X] = ', 2*b, '\nEmp E[X] = ', s1/iter )

    print('differenza = ', abs(2*b - s1/iter))

    #test sample inter arrival 

    s2=0
    for i in range(iter):
        s2+=round(sample_inter_arrival(), 0)
    print('E[T] = ', T_0+((1-q)*E_Y), '\n Emp E[T] = ',s2/iter)

    print('differenza = ', abs( (T_0+((1-q)*E_Y)) -  s2/iter), '\n\n')

'''
with open('pod.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Ro', 'Mean_Time', 'Mean_Msg'])


d=3
inter_iter=20
rhos = np.append(np.arange(0.8, 1, 0.02, dtype = float), 0.99).round(2)
#rhos = np.arange(0.95, 0.99, 0.01, dtype=float)
print('#ro', len(rhos), 'rhos',  rhos)
iterazioni=10000
#generiamo inter-arrival e processing times
arrivals_samples=[int(round(sample_inter_arrival(), 0)) for i in range(iterazioni)]   
arrival_times=list(np.cumsum(arrivals_samples))
start=tempo.time()
for rho in rhos:
    rho=round(rho, 2)
    # calcoliamo beta
    bet=beta(rho)
    inter_time=[]
    inter_msg=[]
    for _ in range(inter_iter):
        processing_samples=[int(round(sample_processing_time(bet), 0)) for i in range(iterazioni)]
        at = list(arrival_times)
        processing_times={at[i]:processing_samples[i] for i in range(iterazioni)}
        dispatcher_queue, task_in = [], []
        server_queue, server_in, general_task = {i: [] for i in range(N)}, {}, {i: -1 for i in arrival_times}
        

        i=-1
        while check(server_queue, server_in, dispatcher_queue, general_task) or (arrival_times[-1] +processing_times[arrival_times[-1]])>i:
            i+=1
            if len(at) > 0: # se ci sono task che devono ancora arrivare
                if i == at[0]: #se siamo nello slot di tempo in cui arriva un task

                    # pod implementation
                    current_task = at.pop(0)
                    task_in.append(current_task)
                    servers = sample(range(N), d)
                    #servers=list(range(N))
                    min_server = min_queue(servers, server_queue)
                    server_queue[min_server].append(current_task)
            for task in task_in:
                general_task[task] = general_task[task] + 1
            
            for j in range(N):
                #print('i:\t', i, 'server in:\t', server_in)
                if server_in.get(j) == None: #il server j non sta alcun task
                    if len(server_queue[j]) > 0:# ma ha qualcosa in coda
                        server_in[j] = (server_queue[j].pop(0), 0)
            
                else:# il server sta eseguendo un task
                    task = server_in[j][0] # il processo in esecuzione
                    time = server_in[j][1] # tempo già processato nel server
                    # aggiornamento
                    server_in[j] = (task, time + 1)
                    # se la task ha esaurito il tempo di processing time
                    if processing_times[task] == time + 1: # la task ha esaurito il tempo
                        #lo rimuoviamo sia dal server che dal sistema
                        del task_in[task_in.index(task)] # rimuove dalla lista di task in esecuzione
                        del server_in[j]
                        # controlliamo se ci sono tasks in coda
                        
                        if len(server_queue[j]) > 0: # se la coda di quel server non è vuota
                            server_in[j] = (server_queue[j].pop(0), 0) # rimuove dalla coda ed assegna il task successivo
                        
                            
        #togliamo il processing time
        #for k, v in general_task.items():
        #    general_task[k]=v-processing_times[k]
    
        #mean system time
        mean_time=sum(general_task.values())/len(general_task)
        #mean msg
        mean_msg=2*d #iterazioni*2*d/iterazioni => 2*d (come da teoria) => 2*3=6

        inter_time.append(mean_time)
        inter_msg.append(mean_msg)
        #print('general task post ', general_task)
    print('inter time: ', inter_time)



    with open('pod.csv', mode='a+') as csv_file:
        csv_writer = csv.writer(csv_file)
        #calcoli mean system time e mean message-per-task-arrival
        
        #mean system time
        mean_time=sum(inter_time)/inter_iter
        #mean msg
        mean_msg=sum(inter_msg)/inter_iter
        csv_writer.writerow([round(rho, 2), mean_time, mean_msg])

    print('ro: ', rho)
    print('task_in \n', task_in)
    #'''
    print('server_in:', server_in)
    print('arrival_times \n', arrival_times)
    print('processing_times \n', processing_times)
    print('general_task \n', general_task, '\t', sum(general_task.values()), '\n \n \n')
    #'''


print('overall time in minutes:\t', str((tempo.time() - start)/60))


