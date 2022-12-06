import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rl = pd.read_csv("output/prediction_rl_single.csv", header=None)*100
true = pd.read_csv("gidi_env/gidi_sim/abr/prog48.csv", header=None) * 100#pd.read_csv("output/prog_true.csv", header=None)*100
action = pd.read_csv("output/action_single.csv", header=None)*100
tests = pd.read_csv("output/progress_single.csv", header=None)*100
#rl['predicted'] = pd.read_csv("output/pf_201.csv")#pd.read_csv("output/predicted.csv")
#rl['pf_predicted'] = pd.read_csv("output/pf_201.csv")*100
#rl['actual'] = pd.read_csv("output/prog_true0.csv")*100

#rl['actions'] = pd.read_csv("output/action.csv") * 100
#print(rl['actions'].sum() * 100)

#rl['equal'] = 100*np.asarray(pd.read_csv("output/test.csv"))[15:,11]/50
#rl['diff_rnn'] = (rl['actual'] - rl['predicted'])
#rl['diff_eq'] = (rl['actual'] - rl['equal'])

plot = true[1].plot(color="blue")
plot.plot(tests[0], color="green")
plot.plot(rl[0][1:], color="orange")
plot.plot(action[0][1:], color="red")
plot.get_figure().savefig("out_single.png")
