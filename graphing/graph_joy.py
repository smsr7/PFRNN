from cProfile import label
import joypy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


exp = "_12_county_multi"

name=['Hanover', 'Henrico', 'New Kent','Powatan','Richmond','Chesterfield','Charles City','Amelia','Nottoway','Dinwiddle','Sussex','Prince George']
true = pd.read_csv("gidi_env/gidi_sim/out0/29.csv", names=name).iloc[1:] * 100#"gidi_env/gidi_sim/abr/prog48.csv", header=None) * 100#pd.read_csv("output/prog_true.csv", header=None)*100

pred = pd.read_csv("output/prediction_rl"+str(exp)+".csv", names=name).iloc[1:] * 100#.to_numpy()


df = pd.DataFrame({"idx":[],"county":[],"rate":[]})

#for idx, i in true.iterrows():
    #print(i.to_numpy())
 #   print(idx)
  #  for e, ii in enumerate(i.to_numpy()):
   #     df = df.append(pd.DataFrame({"idx":[idx],"county":[name[e]],"rate":[ii]}))

#for i in range(len(true)):
  #  for ii in range(12):
 #       df = df.append(pd.DataFrame({"idx":[i], 'county':[name[ii]], 'rate':[true.iloc[i,ii]], 'pred':[pred.iloc[i,ii]]}))
#print(df.loc[:, df.columns!='idx'][50:100])

fig, axes = joypy.joyplot(true, kind="values",fade=True, title="Epidemic Trajectories of 12 Virginia Counties",x_range=(0,150))#,color='orange')
#fig, axes = joypy.joyplot(df.loc[:, df.columns!='idx'], by='county',alpha=.5,legend=True, kind='values',title="Epidemic Trajectories of 12 Virginia Counties",x_range=(0,150))#,color='orange')

#var = pd.DataFrame()

#var['variance'] = np.sqrt(pd.read_csv("output/var_201.csv"))

#plot = var.plot()
#plot.get_figure()
fig.savefig("out_new.png")
