import pandas as pd
import numpy as np

x = np.array([7,  36,  41,  53,  85,  87, 127, 135, 145, 149, 183, 760]) + 51000
for locator in range(1,5):
    for i in range(30):
        out = np.zeros((12,181))
        data = pd.read_csv("gidi_env/gidi_sim/out" + str(locator)+ "/"+str(i)+"_final.csv",index_col=None)

        for idx, ii in data.iterrows():
            #print([ii['tick']],[np.where(ii['county'] == x)[0][0]],ii['rate'] )
            out[np.where(ii['county'] == x)[0][0],int(ii['tick'])] = ii['rate']



        out = np.asarray(out).T
        pd.DataFrame(out).to_csv('gidi_env/gidi_sim/out'+str(locator)+'/'+str(i)+'.csv',index=False)