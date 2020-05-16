from random import *
from math import *
import numpy as np
import time as tempo
import matplotlib.pyplot as plt
import csv
from multiprocessing import Process
import math

# constant:
T_0=1
q=3/5
E_Y=10
a=1/2 # =>gamma(2+1)=2!=2
N=20

def beta(ro, N=N, t0=T_0, q=q, EY=E_Y):
    return (ro*N*(t0+(1-q)*EY))/2 # ottenuto invertendo la formula di rho:= rho=E[X]/(N*E[T])

def sample_inter_arrival(T0=T_0, Y_mean=E_Y): # secondo le disposizioni sulle slide del prof
    r1, r2 = random(), random()
    T=None
    if r1<q:
        T=T0
    else:
        T=round(T0-Y_mean*log(r2), 0)
    return T

def sample_processing_time(bet, a=a): # secondo le disposizioni sulle slide del prof
    r3=random()
    return max(1, min(100*2*bet, round(bet*(-log(r3))**(1/a), 0)))

def min_queue(servers, server_queue): # versione incrementale per calcolare la coda minima
    m = len(server_queue[servers[0]])
    s = servers[0]
    for server in servers[1:]:
        if len(server_queue[server]) < m:
            m = len(server_queue[server])
            s = server
    return s

def check(server_queue, server_in, dispatcher_queue, general_task): # controllo generale che sia tutto finito nel sistema
                                                                    # oneroso ma incrementale
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


def make_simulation_with_rho(rho, arrival_times): # una simulazione su un determinato rho consiste nella ripetzione della simulazione
                                                # dato los teso ro per n volte, (n=inter_iter)
    rho=round(rho, 2)
    # calcoliamo beta
    bet=beta(rho) # otteniamo il beta dato il rho
    inter_time=[] # lista dei tempi medi per il calcolo della media delle medie
    inter_msg=[] # lista dei messaggi per il caldolo della media delle medie
    inter_rho=[] # lista dei rho empirici per il calcolo della media delle medie
    for _ in range(inter_iter): # inizio dei calcoli per la media delle medie
        processing_samples=[int(round(sample_processing_time(bet), 0)) for i in range(iterazioni)]

        mean_processing=sum(processing_samples[scarto:])/len(processing_samples[scarto:]) # empirical mean processing time

        at = list(arrival_times) # copia degli arrivals_time per poterla modificare
        processing_times={at[i]:processing_samples[i] for i in range(iterazioni)} # dizionario:= task:tempo_necessario
        dispatcher_queue, task_in = [], [] # coda del dispatcher e lista dei task presenti nel sistema al variare del tempo
        server_queue, server_in, general_task = {i: [] for i in range(N)}, {}, {i: 0 for i in arrival_times} 
        # dizionario:= server:coda_del_server; dizionario:= server:(task, tempo_passato_nel_server) [rappresenta i server attivi]
        # dizionario:= task: tempo_già_trascorso_nel_sistema dal suo arrivo all'uscita
        

        i=-1
        while check(server_queue, server_in, dispatcher_queue, general_task) or (arrival_times[-1] +processing_times[arrival_times[-1]])>i:
            i+=1
            if len(at) > 0: # se ci sono task che devono ancora arrivare
                if i == at[0]: #se siamo nello slot di tempo in cui arriva un task

                    # pod implementation
                    current_task = at.pop(0) # preleva il task
                    task_in.append(current_task) #aggiungilo alla lista dei task arrivi (già arrivati e non ancora usciti dal sistema)
                    servers = sample(range(N), d) # pesca d server a random
                    
                    min_server = min_queue(servers, server_queue) # ottieni il server con la coda minore
                    server_queue[min_server].append(current_task) # aggiungi il task alla coda server con la coda minore

            for task in task_in: # per ogni task nel sistema
                general_task[task] = general_task[task] + 1 # aumenta il suo tempo di una unità
            
            for j in range(N): # per ogni server
                if server_in.get(j) == None: #se il server j non sta lavorando
                    if len(server_queue[j]) > 0:# ma ha qualcosa in coda
                        server_in[j] = (server_queue[j].pop(0), 0) # prendi quel qualcosa e mettilo nel server
            
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
          
                            
    
        #mean system time
        mean_time=(sum(list(general_task.values())[scarto:]))/(len(general_task)-scarto)
        #mean msg
        mean_msg=2*d #iterazioni*2*d/iterazioni => 2*d (come da teoria) => 2*3=6
        #mean rho
        print('m_ar', mean_arrival, 'm_pr', mean_processing)
        mean_rho=mean_processing/(N*mean_arrival)
        

        inter_time.append(mean_time)
        inter_msg.append(mean_msg)
        inter_rho.append(mean_rho)
        #print('general task post ', general_task)
        print('partial inter time\n', inter_time)


    print('inter time: ', inter_time)

    #scritta dei file csv
    with open(file_name, mode='a+') as csv_file:
        csv_writer = csv.writer(csv_file)
        #calcoli mean system time e mean message-per-task-arrival
        
        #mean system time
        mean_time=sum(inter_time)/inter_iter
        #mean msg
        mean_msg=sum(inter_msg)/inter_iter
        #mean rho empirico
        mean_rho=sum(inter_rho)/inter_iter

        # confidence interval
        sd=np.std(inter_time) # standard deviation
        ci=(sd*1.96)/math.sqrt(len(inter_time))

        csv_writer.writerow([round(rho, 2), mean_time, mean_msg, round(mean_rho, 2), ci])

    print('ro: ', rho)
    print('task_in \n', task_in)
    '''
    print('server_in:', server_in)
    print('arrival_times \n', arrival_times)
    print('processing_times \n', processing_times)
    print('general_task \n', general_task, '\t', sum(general_task.values()), '\n \n \n')
    '''

file_name='pod.csv' # nome del file in cui srivere
with open(file_name, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Rho', 'Mean_Time', 'Mean_Msg', 'Rho_Emp', 'CI'])

d=N
scarto=5000 # numero di task da non considerare affinchè i calcoli siano fatti a regime scartando la fase di start-up iniziale
inter_iter=30 # numero di inter iterazioni per la media delle medie
rhos = [0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1]#np.append(np.arange(0.8, 1.01, 0.02, dtype = float), 1.01).round(2)
print('#ro', len(rhos), 'rhos',  rhos)
iterazioni=25000 # numero di task che si vuole processare (100.000 non è fattibile)
#generiamo inter-arrival e processing times
arrivals_samples=[int(round(sample_inter_arrival(), 0)) for i in range(iterazioni)] # generazione degli arrival
arrival_times=list(np.cumsum(arrivals_samples)) # cumulazione degli arrival per ottenere i tempi assoluti di arrivo

mean_arrival=sum(arrivals_samples[scarto:])/len(arrivals_samples[scarto:]) # empirical mean arrival time togliendo lo scarto

start=tempo.time()

processi=[Process(target=make_simulation_with_rho, args=(rho, arrival_times)) for rho in rhos] # creazione dei processi uno per ogni rho


######################## ATTENZIONE!!!! POTREBBE SATURARE IL PROCESSORE E LA RAM ############################

[p.start() for p in processi]# avvio dei processi
[p.join() for p in processi]# join per aspettare che tutti finiscano
    

print('overall time in minutes:\t', str((tempo.time() - start)/60))


