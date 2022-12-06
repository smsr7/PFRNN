import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#name = "_12_county"#_no_ent"

#rl = pd.read_csv("output/"+str(name)+".csv", header=None).iloc[1:]*100
rl = pd.read_csv("output/LSTM.csv", header=None).iloc[1:]*100
true = pd.read_csv("gidi_env/gidi_sim/out0/29.csv").iloc[1:] * 100#"gidi_env/gidi_sim/abr/prog48.csv", header=None) * 100#pd.read_csv("output/prog_true.csv", header=None)*100
action = np.ones((12,185)) * 10
#action = pd.read_csv("output/action"+str(name)+".csv", header=None).iloc[1:]*100

tests = pd.read_csv("output/_12_county.csv", header=None)*100
#tests = pd.read_csv("output/progress_12_county.csv", header=None)*100
#tests = pd.read_csv("output/_12_county.csv", header=None)*100

#rl = pd.read_csv("output/prediction_rl_no_act.csv", header=None).iloc[1:]*100
#true = pd.read_csv("gidi_env/gidi_sim/abr/prog48.csv", header=None) * 100#pd.read_csv("output/prog_true.csv", header=None)*100
#action = pd.read_csv("output/action_no_act.csv", header=None).iloc[1:]*100
#tests = pd.read_csv("output/progress_single_traj.csv", header=None)*100

fig, axs = plt.subplots(3,4,sharex=True, sharey=True)
fig.suptitle('LSTM prediction and actions vs ground truth (No Action)')

x = 0
for i in range(3):
    for ii in range(4):
        green = axs[i,ii].plot(tests[x], color="green", label='Positive test %')
        orange = axs[i,ii].plot(rl[x], color="orange",label='PF-RNN prediction')
        blue = axs[i,ii].plot(true.iloc[:,x], color="blue", label = 'True %')
        #print(true.iloc[:,x])
        red = axs[i,ii].plot(action[x], color="red", label = 'Tests')#np.ones(len(true[x])) * .1
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

fig.savefig("out_LSTM.png", dpi=300)
