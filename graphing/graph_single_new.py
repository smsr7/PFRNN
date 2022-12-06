import joypy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

name = "_12_county_no_ent"#_no_ent"

#rl = pd.read_csv("output/"+str(name)+".csv", header=None).iloc[1:]*100
rl = pd.read_csv("output/prediction_rl"+str(name)+".csv", header=None).iloc[1:]*100
true = pd.read_csv("gidi_env/gidi_sim/out0/29.csv").iloc[1:] * 100#"gidi_env/gidi_sim/abr/prog48.csv", header=None) * 100#pd.read_csv("output/prog_true.csv", header=None)*100
action = pd.read_csv("output/action"+str(name)+".csv", header=None).iloc[1:]*100
tests = pd.read_csv("output/progress_12_county_no_act.csv", header=None)*100
#tests = pd.read_csv("output/progress_12_county.csv", header=None)*100
#tests = pd.read_csv("output/_12_county.csv", header=None)*100

#rl = pd.read_csv("output/prediction_rl_no_act.csv", header=None).iloc[1:]*100
#true = pd.read_csv("gidi_env/gidi_sim/abr/prog48.csv", header=None) * 100#pd.read_csv("output/prog_true.csv", header=None)*100
#action = pd.read_csv("output/action_no_act.csv", header=None).iloc[1:]*100
#tests = pd.read_csv("output/progress_single_traj.csv", header=None)*100

fig, axs = plt.subplots(3,4,sharex=True, sharey=True)
fig.suptitle('PF-RNN prediction and actions vs ground truth (Individual Trajectories)')
action = np.ones((12,185)) * 10
x = 0
for i in range(3):
    for ii in range(4):
        name = x
        rl = pd.read_csv("output/prediction_rl_single"+str(name)+".csv", header=None).iloc[1:]*100
        action = pd.read_csv("output/action_single"+str(name)+".csv", header=None).iloc[1:]*100
        tests = pd.read_csv("output/progress_single"+str(name)+".csv", header=None)*100

        green = axs[i,ii].plot(tests, color="green", label='Positive test %')
        orange = axs[i,ii].plot(rl, color="orange",label='PF-RNN prediction')
        blue = axs[i,ii].plot(true.iloc[:,x], color="blue", label = 'True %')
        #print(true.iloc[:,x])
        red = axs[i,ii].plot(action, color="red", label = 'Tests')#np.ones(len(true[x])) * .1
        #red = axs[i,ii].plot(np.ones(len(true[x])), color="red", label = 'Tests')#np.ones(len(true[x])) * .1
        
        axs[i,ii].set_xlim(0,180)
        axs[i,ii].set_ylim(0,30)
        axs[i,ii].set_xticks(np.arange(0, 180, 50))
        x +=1

#fig.set_ylim(0,100)
fig.supxlabel('Tick (day)')
fig.supylabel('Infection Rate (% infected)')
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center', ncol=4,mode="expand",bbox_to_anchor=(.112, .4, .8, .55))
#plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75)
#fig.legend([green,orange,blue,red],['Positive test %', 'PF-RNN prediction', 'True %', 'Tests'], loc = (0.5, 0), ncol=5 )

fig.savefig("out_single.png", dpi=300)

'''

acts = pd.read_csv("output/acts_all.csv", header=None)
rwd = pd.read_csv("output/rewards_all.csv", header=None)
import re
n = re.compile(r'\((.*),')
rwd1 = []
for i in rwd[0]:
    rwd1.append(float(n.findall(i)[0]))    
rwd2 = []
for i in rwd[1]:
    rwd2.append(float(n.findall(i)[0]))    


fig,axs = plt.subplots(1,3)
axs[0].plot(acts,color='orange')
axs[0].set_title('tests')#['Tests','elbo','mse']
axs[1].plot(rwd1,color='orange')
axs[1].set_title('elbo')
axs[2].plot(rwd2,color='orange')
axs[2].set_title('mse')

fig.savefig("out_rwd"+str(name)+".png", dpi=300)
#var = pd.DataFrame()

#var['variance'] = np.sqrt(pd.read_csv("output/var_201.csv"))

#plot = var.plot()
#plot.get_figure().savefig("var_out_201.png")
'''