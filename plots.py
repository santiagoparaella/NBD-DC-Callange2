import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate the df for the plot:

jbt = pd.read_csv('jbt.csv')
jbt.columns = ['rho', 'mean_time', 'mean_msg', 'rho_emp', 'ci']
jbt['type'] = ['jbt'] * len(jbt)
jbt['rho_emp'] = jbt[jbt['rho_emp'] != 1]

jsq = pd.read_csv('jsq.csv')
jsq.columns = ['rho', 'mean_time', 'mean_msg', 'rho_emp', 'ci']
jsq['type'] = ['jsq'] * len(jsq)

pod = pd.read_csv('pod.csv')
pod.columns = ['rho', 'mean_time', 'mean_msg', 'rho_emp', 'ci']
pod['type'] = ['pod'] * len(pod)

df = pd.concat([jbt, jsq, pod], ignore_index = True, axis = 0)

# Let's plot the mean response time

plt.figure(figsize = (16, 16), facecolor = ('whitesmoke'), )

sns.lineplot(x = 'rho_emp', y = 'mean_time', hue = 'type', data = df,
            markers = ["o", "o", "o"], style = "type",  markersize = 10, lw = 2, 
            palette = ['mediumblue' ,'darkorange', 'green'])

plt.errorbar(jbt['rho_emp'], jbt['mean_time'], yerr = jbt['ci'], fmt = 'none', 
             capsize = 6, alpha = .6, ecolor = 'mediumblue')
plt.errorbar(jsq['rho_emp'], jsq['mean_time'], yerr = jsq['ci'], fmt = 'none', 
             capsize = 6, alpha = .6, ecolor = 'darkorange')
plt.errorbar(pod['rho_emp'], pod['mean_time'], yerr = pod['ci'], fmt = 'none', 
             capsize = 6, alpha = .8, ecolor = 'green')
plt.axvline(x = 1, ls = '--', c = 'r', lw = 1.5)

plt.legend(['jbt', 'jsq', 'pod'], loc='upper left', fontsize = 22, frameon = False)

plt.xticks(np.linspace(0.8, 1.04, num=13), fontsize = 16)
plt.yticks(np.arange(0, 5500, 500), fontsize = 16)

plt.ylim(ymin = 0, ymax = 5000)

plt.xlabel('empiricall ρ', fontsize = 22, labelpad = 14)
plt.ylabel('mean response time', fontsize = 22, labelpad = 18)

plt.title('Mean Response Time in Function of ρ', fontsize = 24, pad = 16)

plt.grid(ls = '--', c = 'lightgrey')
plt.show()

# Let's plot the mean rmessage time

plt.figure(figsize = (16, 16), facecolor = ('whitesmoke'))
sns.lineplot(x = 'rho_emp', y = 'mean_msg', hue = 'type', data = df,
            markers = ["o", "o", "o"], style = "type",  markersize = 10, lw = 2, 
            palette = ['mediumblue' ,'darkorange', 'green'])

plt.legend(['jbt', 'jsq', 'pod'], loc='upper left', fontsize = 22, frameon = False)

plt.xticks(np.linspace(0.8, 1, num=11), fontsize = 16)
plt.yticks(np.arange(-10, 60, 10), 
           ['', 0, 10, 20, 30, 40, ''],
           fontsize = 16)

plt.xlabel('empiricall ρ', fontsize = 22, labelpad = 14)
plt.ylabel('mean messages', fontsize = 22, labelpad = 18)

plt.title('Mean Messages  in Function of ρ', fontsize = 24, pad = 16)

plt.ylim(ymin = -10, ymax = 50)
plt.grid(ls = '--', c = 'lightgrey')
plt.show()