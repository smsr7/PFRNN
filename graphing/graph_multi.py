import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


true = pd.read_csv("gidi_env/gidi_sim/abr/prog48.csv", header=None) * 100#pd.read_csv("output/prog_true.csv", header=None)*100

#rl['predicted'] = pd.read_csv("output/pf_201.csv")#pd.read_csv("output/predicted.csv")
#rl['pf_predicted'] = pd.read_csv("output/pf_201.csv")*100
#rl['actual'] = pd.read_csv("output/prog_true0.csv")*100

#rl['actions'] = pd.read_csv("output/action.csv") * 100
#print(rl['actions'].sum() * 100)

#rl['equal'] = 100*np.asarray(pd.read_csv("output/test.csv"))[15:,11]/50
#rl['diff_rnn'] = (rl['actual'] - rl['predicted'])
#rl['diff_eq'] = (rl['actual'] - rl['equal'])

#plot = rl.plot(subplots=True)
fig, axs = plt.subplots(3,4,sharex=True, sharey=True)
fig.suptitle('RL prediction and actions vs ground truth (single county)')

error = []
x = 0
for i in range(3):
    for ii in range(4):
        action = pd.read_csv("output/action_single"+str(x)+".csv", header=None)*100
        tests = pd.read_csv("output/progress_single"+str(x)+".csv", header=None)*100
        rl = pd.read_csv("output/prediction_rl_single"+str(x)+".csv", header=None)*100
        axs[i,ii].plot(tests[0], color="green", label='Positive test %')
        axs[i,ii].plot(rl[0][1:], color="orange",label='PF-RNN prediction')
        axs[i,ii].plot(true[x], color="blue", label = 'True %')
        axs[i,ii].plot(action[0][1:], color="red", label = 'Tests')
        #axs[i,ii].plot(abs(np.asarray(rl[0][1:]) - np.asarray(true[x])) / (np.asarray(true[x])))#((np.asarray(rl[0][1:]) - np.asarray(true[x])/100)**2) / len(rl[0][1:]))
        #print(len(rl[0][1:]) , len(true[x]))
        #error.append(sum((np.asarray(rl[0][1:]) - np.asarray(true[x])/100)**2) / len(rl[0][1:]))
        axs[i,ii].set_xlim(0,150)
        axs[i,ii].set_xticks(np.arange(0, len(true[x]), 50))
        x +=1
fig.supxlabel('Tick (day)')
fig.supylabel('Infection Rate (% infected)')
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center', ncol=4,mode="expand",bbox_to_anchor=(.112, .4, .8, .55))

#fig.set_ylim(0,100)
fig.savefig("out_single_new_test.png", dpi=300)
