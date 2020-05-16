# versione 1: come da slide... ma il professore ha suggerito una modifica che implemento nella versione 2

from random import *
from math import *
import numpy as np
import time as tempo
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
    #if len(server_queue[s])>0:
    #    print('Coda del server minore non vuota!!!!!!!!!!!!!')
    return m

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

def make_simulation_with_rho(rho, arrival_times):
    # calcoliamo beta
    bet=beta(rho)
    inter_time=[]
    inter_msg=[]
    inter_rho=[]
    for _ in range(inter_iter):
        number_of_msg=0 #numero di messaggi
        r=0 #threshold
        processing_samples=[int(round(sample_processing_time(bet), 0)) for i in range(iterazioni)]
        at = list(arrival_times)
        processing_times={at[i]:processing_samples[i] for i in range(iterazioni)}

        mean_processing=sum(processing_samples[scarto:])/len(processing_samples[scarto:]) # empirical mean processing time

        dispatcher_queue, task_in, dispatcher_memory = [], [], list(range(N))
        server_queue, server_in, general_task = {i: [] for i in range(N)}, {}, {i: -1 for i in arrival_times}
        

        i=-1
        while check(server_queue, server_in, dispatcher_queue, general_task) or (arrival_times[-1] +processing_times[arrival_times[-1]])>i:
            i+=1
            if len(at) > 0: # se ci sono task che devono ancora arrivare
                if i == at[0]: #se siamo nello slot di tempo in cui arriva un task
                    #JBT implementation
                    current_task = at.pop(0)
                    task_in.append(current_task)
                    # anche quella che arriva sullo scadere dello slot viene inclusa in quello slot

                    #scelta del server
                    if len(dispatcher_memory)>0:# se vi sono server con bit = 1
                        server = sample(dispatcher_memory, 1)[0] #prendi un server tra quelli che hanno il bit = 1 a caso
                        dispatcher_memory.remove(server) #setta il bit di quel server a zero
                    else: # nessun server ha il bit = 1
                        server=sample(range(N), 1)[0]#prendine uno random tra tutti i server
                    
                    #manda il task a quel server
                    server_queue[server].append(current_task)

            if int(str(i)[-practical_discrete:])==0: #se lo slot di tempo è scaduto
                #aggiornamento della treshold
                servers=sample(range(N), d) # peschiamo d random servers
                number_of_msg+=2*d # mandati e ricevuto 2*d messaggi
                coda_minima = min_queue(servers, server_queue) # e prendiamo la coda minima tra i d
                r=coda_minima
                number_of_msg+=N # mandati N messaggi di aggiornamento coda
                    
            for task in task_in:
                general_task[task] = general_task[task] + 1
            
            for j in range(N):
                coda_pre=len(server_queue[j]) #lunghezza della coda all'inizio

                if server_in.get(j) == None: #il server j non ha alcun task
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

                #monitoraggio coda: 
                if coda_pre>len(server_queue[j]) and coda_pre==r:#se la coda è diminuita di 1, allora ha fatto una transizione
                #da r a r-1 e quindi deve mandare un messaggio al server
                    number_of_msg+=1 #aggiornamento numero di messagi
                    dispatcher_memory.append(j)#aggiornamento " pratico"
            #togliamo i messaggi dei primi 5000 task
            if i<scarto: # per scartare i primi messaggi che saranno sicuramente di meno
                number_of_msg=0      
                        
                            
        #togliamo il processing time
        #for k, v in general_task.items():
        #    general_task[k]=v-processing_times[k]
    
        #mean system time
        mean_time=(sum(list(general_task.values())[scarto:]))/(len(general_task)-scarto)
        #mean msg
        mean_msg=number_of_msg/(iterazioni-scarto) 
        #mean rho
        mean_rho=mean_arrival/(N*mean_processing)

        inter_time.append(mean_time)
        inter_msg.append(mean_msg)
        inter_rho.append(mean_rho)
        #print('general task post ', general_task)
        print('partial inter time\n', inter_time)

        
    print('total inter time: ', inter_time)



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
        sd=np.std(inter_time)
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

file_name='jbt.csv'


with open(file_name, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Rho', 'Mean_Time', 'Mean_Msg', 'Rho_Emp', 'CI'])

scarto=5000
d=3
discrete=1000 #modalità di discretizzazione
practical_discrete=3 # numeo di zeri del discrete
inter_iter=30
rhos = [0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1]#rhos = np.append(np.arange(0.8, 1, 0.02, dtype = float), 0.99).round(2)
#rhos = np.arange(0.95, 0.99, 0.01, dtype=float)
print('#ro', len(rhos), 'rhos',  rhos)
iterazioni=55000
#generiamo inter-arrival e processing times
arrivals_samples=[int(round(sample_inter_arrival(), 0)) for i in range(iterazioni)]   
arrival_times=list(np.cumsum(arrivals_samples))

mean_arrival=sum(arrivals_samples[scarto:])/len(arrivals_samples[scarto:])

start=tempo.time()

processi=[Process(target=make_simulation_with_rho, args=(rho, arrival_times)) for rho in rhos]
[p.start() for p in processi]
[p.join() for p in processi]
#for rho in rhos:
#    make_simulation_with_rho(rho, arrival_times)
    

print('overall time in minutes:\t', str((tempo.time() - start)/60))

