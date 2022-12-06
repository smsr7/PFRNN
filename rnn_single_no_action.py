import numpy as np
import pandas as pd
import signal
import multiprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from gidi_env.gidi_sim.utils import DataLoader, PickleLoader
import tensorflow as tf

from gidi_env.gidi_sim.env_rnn import env as envs


global rnn
global STEP_SIZE
STEP_SIZE   = 20


def BUILD_RNN(STEP_SIZE):
    regressor = Sequential()
    regressor.add(LSTM(units = 240, return_sequences = True, input_shape = (12,STEP_SIZE)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 240, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 120, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 64))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 12))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return regressor

rnn = BUILD_RNN(STEP_SIZE)

from keras.callbacks import EarlyStopping, ModelCheckpoint


def train_rnn(STEP_SIZE, EPOCHS, env):
    state = np.zeros((20,12))
    actions = np.ones((20,12)) * 1
    s_1 = env.reset()

    while env.epoch < EPOCHS:
        s_1 = env.reset()
     
        done = False
        x = state
        y = np.array([s_1])
        while not done:
            #c = np.array(rnn.predict_on_batch(X))
            act = np.ones(12) * .1
            
            x = np.dstack((x,state))
            y = np.dstack((y, np.array([s_1])))
            
            state = np.append(state[1:],np.array([s_1]), axis=0)
            s_1, done = env.step(act)
            actions = np.append(actions[1:],np.reshape(act, (1,12)),axis=0)

        x = x.T
        y = y.T
        #print(x.shape,y.shape)

        #x = tf.expand_dims(x, axis=-1)
        history = rnn.fit(x,  y,
                    epochs=5, verbose=0)

        print(env.epoch)
    return

def main():
    env = envs()
    EPOCHS      = 500
    train_rnn(STEP_SIZE, EPOCHS, env)
    #X = env.reset(testing=True)

    #X = tf.expand_dims(X, axis=-1)
    state = np.zeros((20,12))
    actions = np.ones((20,12)) * 1

    s_1 = env.reset()
    
    done = False
    x = state
    y = np.array([s_1])
    while not done:
        #c = np.array(rnn.predict_on_batch(X))
        act = np.ones(12) * .1
        
        x = np.dstack((x,state))
        y = np.dstack((y, np.array([s_1])))
        
        state = np.append(state[1:],np.array([s_1]), axis=0)
        s_1, done = env.step(act)
        actions = np.append(actions[1:],np.reshape(act, (1,12)),axis=0)

    x = x.T
    y = y.T
        
    rnn.evaluate(x,y)
    #print(np.array(rnn.predict_on_batch(x)))
    pd.DataFrame((np.array(rnn.predict_on_batch(x)))).to_csv('output/LSTM.csv',header=False,index=False)
    #np.concatenate
    #pd.DataFrame(Y).to_csv('output/actual.csv', header=False,index=False)

if __name__ == '__main__':
    main()
    #data(STEP_SIZE).reset(testing=True)
