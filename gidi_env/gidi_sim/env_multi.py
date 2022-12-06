#try:
from gidi_env.gidi_sim.utils import DataLoader, PickleLoader
#except:
#from utils import DataLoader, PickleLoader

import numpy as np
import pandas as pd
import pickle
import random
import time
import math
import csv
import torch

class env():
    def __init__(self, profile = False):

        self.PROFILE = profile
        if self.PROFILE:
            print("INIT")
            start = time.time()

        self.epoch = 0
        #hyper params
        self.FIDS       = np.sort(np.array([87,41,127,135,760,145,7,53,149,183,36,85]))
        self.PPL        = PickleLoader("gidi_env/gidi_sim/Data/ppl")
        self.TESTS      = 100

        self.POP        = []

        for i in range(len(self.FIDS)):
            self.POP.append(len(self.PPL[i]))


        if self.PROFILE:
            end = time.time()
            print("Load Dataset:",end - start)

        #episode spefic parameters
        FIDS = len(self.FIDS)

        self.last_action = np.ones(FIDS)        #action at previous timestep
        self.action      = np.ones(FIDS)        #current action
        self.true_num    = np.zeros(FIDS)       #array of the true prop for each area
        self.test_num    = np.zeros(FIDS)       #tested number of positives
        self.cur_step    = 0                    #current step
        self.done        = False                #eps completed
        self.tests       = np.zeros(FIDS)       #number of tests assigned to each county
        self.iter        = -1
        self.output = ""

        if self.PROFILE:
            end = time.time()
            print("Init:",end - start)
            print()


    def reset(self, init_time = 0, env_step = None, testing = False, max_tick =5000):
        '''
        reset enviornment - return initial state
        env_step define which simulation iteration to use
        '''
        if self.PROFILE:
            start = time.time()

        if(env_step == None):
            env_num = random.randint(1,4)
            env_step = random.randint(0,29)

        
        if(testing):
            env_step = 29
            env_num = 0
            self.test = True
            pd.DataFrame([]).to_csv('output/progress'+str(self.output)+'.csv',header=False,index=False)
         
        else:
            self.test = False

        self.DATASET  = pd.read_csv("gidi_env/gidi_sim/out"+str(env_num)+"/" + str(env_step) + ".csv", index_col=None, header=None).iloc[1:]
        #print(self.DATASET)
        #try:
        #    self.DATASET  = pd.read_csv("gidi_env/gidi_sim/abr/prog" + str(env_step) + ".csv", index_col=None, header=None)
         #   self.DATASET  = pd.read_csv("gidi_env/gidi_sim/out" + str(env_step) + ".csv", index_col=None, header=None)
        #except:
        #    self.reset()
        #print(self.DATASET)
        self.done = False
        self.cur_step = 0 + init_time
        self.max_tick = max_tick

        self.state       = np.zeros(12)
        self.tested      = []
        self.true_pop    = []
        self.testing     = []
        self.tests       = np.zeros(len(self.FIDS))


        self.epoch += 1
        
        #pd.DataFrame([]).to_csv('output/'+str(self.output)+'.csv',header=False,index=False)
            
        return self.state


    def step(self, action):
        '''
        progress enviornment by tick - return state done
        '''
        action[np.where(action < .01)] = .01
        if self.cur_step == len(self.DATASET) - 1 or self.cur_step > self.max_tick:
            self.done = True

        
        tests = np.floor(action * self.TESTS)
        #print(tests, action, self.TESTS)
        #print(self.DATASET.iloc[self.cur_step])
        
        #print(tests.astype(int))


        s_new = np.random.binomial(tests.astype(int), self.DATASET.iloc[self.cur_step].astype(float))

        #print(s_new, self.DATASET.iloc[self.cur_step].astype(float))

        self.state = s_new / tests
        #print(np.array(self.DATASET.iloc[self.cur_step]) - self.state)
        self.cur_step += 1
        
        if self.test:
            #print(s_new, tests)
            
            pd.DataFrame([np.asarray(s_new/tests)]).to_csv('output/progress'+str(self.output)+'.csv', mode='a',header=False,index=False)
            
        
        return self.state, self.done


    def getStateSize(self):
        """
        Return the size the state
        """
        return len(self.FIDS) * 1


    def get_total_actions(self):
        """
        Return the size action vector
        """
        return len(self.FIDS)

    def get_env_info(self):
        """
        Return basic infos about the environment
        """
        env_info = {"state_shape": self.getStateSize(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": 1,
                    "episode_limit": 180}
        return env_info

    def close(self):
        """
        Close the environment
        """
        return

    def seed(self, seed_num):
        np.random.seed(seed_num)

    def scale(self,X):
        X = np.asarray(X)
        X_std = (X - np.amin(X)) / (np.amax(X) - np.amin(X) + 1e-9)
        return(np.nan_to_num(X_std))


if __name__ == "__main__":
    profile = True
    if profile:
        start = time.time()
    env = env(profile = False)
    env.TESTS = 0
    env.output = str(env.TESTS)

    for i in range(1,20):
        env.TESTS = i*10
        env.output = str(env.TESTS)
        env.reset(testing=True)
        np.set_printoptions(suppress=True)
        
        done = False
        while not done:
            #print(i)
            s,done = (env.step(np.ones(12)))
        
            #env.save()

    if profile:
        end = time.time()
        print("SIM time:", end - start)
