import joypy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

name = "_12_county_multi"#_multi_org"#_no_ent"
name_2 = "_12_county_multi_org"#_multi_org"#_no_ent"
name_3 = "_12_county_action"
name_4 = "_12_county"

names = ["Hanover", "Henrico", "New Kent", "Powhatan", "Richmond", "Charles City", "Chesterfield", "Amelia", "Nottoway", "Dinwiddie", "Prince George", "Sussex"]

#rl = pd.read_csv("output/"+str(name)+".csv", header=None).iloc[1:]*100
rl = pd.read_csv("output/prediction_rl"+str(name)+".csv", header=None,names=names).iloc[1:]*100
rl_2 = pd.read_csv("output/prediction_rl"+str(name_2)+".csv", header=None,names=names).iloc[1:]*100
rl_3 = pd.read_csv("output/prediction_rl"+str(name_3)+".csv", header=None,names=names).iloc[1:]*100
rl_4 = pd.read_csv("output/prediction_rl"+str(name_4)+".csv", header=None,names=names).iloc[1:]*100
true = pd.read_csv("gidi_env/gidi_sim/out0/29.csv", names=names).iloc[1:] * 100#"gidi_env/gidi_sim/abr/prog48.csv", header=None) * 100#pd.read_csv("output/prog_true.csv", header=None)*100
action = pd.read_csv("output/action"+str(name_4)+".csv", header=None,names=names).iloc[1:]*175
#action = np.ones((185,12)) * 10
#action for 4 is ent

fig, axs = plt.subplots(3,4,sharex=True, sharey=True)
#fig.suptitle('PF-RNN prediction and actions vs ground truth (No Entropy)')

x = 0
for i in range(3):
    for ii in range(4):
        orange = axs[i,ii].plot((-true+rl_2).iloc[:,x], color="orange", label='Prediction Error')
        #blue = axs[i,ii].plot((-true+rl_2).iloc[:,x], color="blue", label='No Entropy')
        #red = axs[i,ii].plot((-true+rl_3).iloc[:,x], color="red", label='No Action')
        #.twinx()
        #red = axs[i,ii].plot(action.iloc[:,x], color="red",label='PF-RNN prediction')
       # action = action[:60]
        act = action.iloc[:,x]
        red = axs[i,ii].plot(act, color="red", label = 'Tests')#np.ones(len(true[x])) * .1
        #blue = axs[i,ii].plot(true.iloc[:,x], color="blue", label = 'True %')
        #print(true.iloc[:,x])
        #red = axs[i,ii].plot(action[x], color="red", label = 'Tests')#np.ones(len(true[x])) * .1
        #red = axs[i,ii].plot(np.ones(len(true[x])), color="red", label = 'Tests')#np.ones(len(true[x])) * .1
        
        axs[i,ii].set_xlim(0,180)
        axs[i,ii].set_ylim(-12,12)
        axs[i,ii].set_xticks(np.arange(0, 180, 50))
        x +=1

fig.supxlabel('Tick (day)')
fig.supylabel('Difference (%)')
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines[:2], labels[:2], loc = 'upper center', ncol=2,mode='expand',bbox_to_anchor=(.112, .4, .8, .55))
#plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75)
#print(red)
#fig.legend([red[0], orange],['Tests','Prediction Error'], loc = (0.5, 0), ncol=2,bbox_to_anchor=(.112, .4, .8, .55) )

fig.savefig("out_a_no_ent.png", dpi=300)

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