import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

true = pd.read_csv("gidi_env/gidi_sim/out0/29.csv").iloc[1:].to_numpy()#"gidi_env/gidi_sim/abr/prog48.csv", header=None) * 100#pd.read_csv("output/prog_true.csv", header=None)*100



xx = []
yy = []

for ii in range(10):
    x = []
    y = []
    for i in range(1,200):
        tests = pd.read_csv("output/baseline/"+str(ii)+"_"+str(i)+".csv", header=None).iloc[1:].to_numpy()
        #print(np.nan_to_num(tests - true))
        
        x.append(i)
        error = np.sqrt((np.nan_to_num(tests - true)**2).sum().sum())
        #print(error)
        y.append(error)
    yy.append(y)
yy = np.array(yy)
mean = yy.mean(axis=0)
minimum = yy.min(axis=1)
maximum = yy.max(axis=1)
print(mean)


#ax.plot(x, y)
fig, ax = plt.subplots(figsize=(8,8))

x = np.array(x)
X_Y_Spline = make_interp_spline(x[3:150], mean[3:150])
X = np.linspace(3,149, 500)
Y = X_Y_Spline(X)


plt.plot(X,Y)#sns.lmplot(data=pd.DataFrame({"Number of Tests":x,"MSE":mean}), x='Number of Tests', y='MSE', order=5,ci=None)
#ax.plot()



name = "_12_county"#_no_ent"
rl = pd.read_csv("output/prediction_rl"+str(name)+".csv", header=None).iloc[2:].to_numpy()
action = pd.read_csv("output/action"+str(name)+".csv", header=None).iloc[1:].to_numpy() * 100

error = np.sqrt((np.nan_to_num(rl - true)**2).sum().sum())
test = action.sum().sum()/180

print(error, test)

#ax.set_ylabel('Number of Tests')
#fig = plot.fig#get_figure()
plt.scatter(6.2, .335, marker='o', color='orange', s=50)


name = "_12_county_multi_org"#_no_ent"
rl = pd.read_csv("output/prediction_rl"+str(name)+".csv", header=None).iloc[2:].to_numpy()
action = pd.read_csv("output/action"+str(name)+".csv", header=None).iloc[1:].to_numpy() * 100

error = np.sqrt((np.nan_to_num(rl - true)**2).sum().sum())
test = action.sum().sum()/180

print(error, test)

#ax.set_ylabel('Number of Tests')
#fig = plot.fig#get_figure()
plt.scatter(test, error, marker='o', color='purple', s=50)


name = "_12_county_no_act"#_no_ent"
rl = pd.read_csv("output/prediction_rl"+str(name)+".csv", header=None).iloc[2:].to_numpy()
action = pd.read_csv("output/action"+str(name)+".csv", header=None).iloc[1:].to_numpy() * 100

error = np.sqrt((np.nan_to_num(rl - true)**2).sum().sum())
test = action.sum().sum()/180

print(error, test)

#ax.set_ylabel('Number of Tests')
#fig = plot.fig#get_figure()
plt.scatter(10, error, marker='o', color='red', s=50)



name = "_12_county_no_ent"#_no_ent"
rl = pd.read_csv("output/prediction_rl"+str(name)+".csv", header=None).iloc[2:].to_numpy()
action = pd.read_csv("output/action"+str(name)+".csv", header=None).iloc[1:].to_numpy() * 100

error = np.sqrt((np.nan_to_num(rl - true)**2).sum().sum())
test = action.sum().sum()/180

print(error, test)

#ax.set_ylabel('Number of Tests')
#fig = plot.fig#get_figure()
plt.scatter(15, .232, marker='o', color='blue', s=50)

rl = pd.read_csv("output/LSTM.csv", header=None).iloc[2:].to_numpy()
#action = pd.read_csv("output/action"+str(name)+".csv", header=None).iloc[1:].to_numpy() * 100

error = np.sqrt((np.nan_to_num(rl - true)**2).sum().sum())
test = 10

print(error, test)

#ax.set_ylabel('Number of Tests')
#fig = plot.fig#get_figure()
plt.scatter(test, error, marker='o', color='green', s=50)

plt.legend(['Error',  'PF-RNN with action', 'PF-RNN multi trajectory','PF-RNN no action', 'PF-RNN no entropy'])
#ax.set_xlabel('MSE')
#print(plot.fig)
plt.xlabel("Number of Tests")  # add X-axis label
plt.ylabel("MSE")  # adsd Y-axis label
plt.title("Error Comparison of Models and Prevalence Testing")

plt.tight_layout()
plt.savefig("out_baseline_new.png", dpi=300)
