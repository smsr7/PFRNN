from pfrnn import PFLSTMCell
from gidi_env.gidi_sim.env_multi import env as envs
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

class pfrnn(nn.Module):
    def __init__(self):

        super(pfrnn, self).__init__()
        INPUT_SIZE = 40*12
        HIDDEN_SIZE = 64*12

        EXT_ACT = 20*12
        EXT_OBS = 20*12
        NUM_PARTICLES = 32*12
        RESAMP_ALPHA = 0.6
        DROPOUT_RATE = .4

        self.num_particles = NUM_PARTICLES
        self.hidden_dim = HIDDEN_SIZE

        self.model = PFLSTMCell(NUM_PARTICLES, INPUT_SIZE, HIDDEN_SIZE, EXT_OBS, EXT_ACT, RESAMP_ALPHA)

        self.hnn_dropout = nn.Dropout(DROPOUT_RATE)
        self.hidden2label = nn.Linear(self.hidden_dim, 24)


        self.obs_embedding = nn.Linear(EXT_ACT,EXT_OBS)
        self.act_embedding = nn.Linear(240, EXT_ACT)
    #def encode(self):

    def detach_hidden(self, hidden):
        if isinstance(hidden, tuple):
            return tuple([h.detach() for h in hidden])
        else:
            return hidden.detach()

    def forward(self, x, act, batch_size = 1):
        seq_len = len(list(x))
        #batch_size = len(list(x))
        initializer = torch.zeros
        h0 = initializer(batch_size * self.num_particles, self.hidden_dim)
        c0 = initializer(batch_size * self.num_particles, self.hidden_dim)
        p0 = torch.ones(batch_size * self.num_particles, 1) * np.log(1 / self.num_particles)
        hidden = (h0, c0, p0)

        obs_input = torch.from_numpy(x.flatten()).float()

        #act_input = torch.from_numpy(np.full((240,1), act)).float()
        act_input = torch.from_numpy(act.flatten()).float()

        emb_obs = torch.relu(self.obs_embedding(obs_input)).view([1,240])
        emb_act = torch.relu(self.act_embedding(act_input)).view([1,240])

        embedding = torch.cat((emb_obs, emb_act), dim=1)
        embedding = embedding.repeat(self.num_particles,1,1)
        seq_len = embedding.size(1)
        hidden_states = []
        probs = []
        #print(embedding.shape)#,hidden.shape)

        for i in range(seq_len):
            hidden = self.model.forward(embedding[:, i, :],hidden)
            hidden_states.append(hidden[0])
            probs.append(hidden[-1])
            #if i % 10 == 0:
            #    hidden = self.detach_hidden(hidden)

        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = self.hnn_dropout(hidden_states)

        probs = torch.stack(probs, dim=0)
        prob_reshape = probs.view([seq_len, self.num_particles, -1, 1])
        out_reshape = hidden_states.view([seq_len, self.num_particles, -1, self.hidden_dim])
        #print(out_reshape[0], prob_reshape[0])
        y = out_reshape * torch.exp(prob_reshape)
        y = torch.sum(y, dim=1)
        y = self.hidden2label(y)

        pf_labels = self.hidden2label(hidden_states)

        #y_out_xy = torch.sigmoid(y[:, :, :2])

        y_out = torch.sigmoid(y[:, :, :])
        #y_out = torch.cat([y_out_xy, y_out_h], dim=2)

        #pf_out_xy = torch.sigmoid(pf_labels[:, :, :2])
        pf_out = torch.sigmoid(pf_labels[:, :, :])
        #pf_out = torch.cat([pf_out_xy, pf_out_h], dim=2)
        #print(y_out.size(), pf_out.size())
        #print(pf_out)
        return y_out, pf_out


    def step(self,x,y, pred=None, particle_pred=None):
        bpdecay = .1
        #if pred is None:
        #pred, particle_pred = self.forward(x)
        #else:
        #pred, particle_pred = torch.autograd.Variable(torch.from_numpy(pred).view([len(y),1,1]), requires_grad = True).float() * 1, torch.autograd.Variable(torch.from_numpy(particle_pred).view([len(y),self.num_particles,1]), requires_grad = True).float() * 1
        #print(x[0][0][100], pred[0][0])
    
        #print(pred, y)
        #print(y, pred[100])
        
        batch_size = int(y.shape[2])

        gt_normalized = torch.from_numpy(np.array(y)).float().T.transpose(1,2)#.view([1,24,batch_size])#torch.stack(, dim=0)

        
        #print(pred[100], gt_normalized[100])
        sl = batch_size
        bpdecay_params = np.exp(bpdecay * np.arange(sl))
        bpdecay_params = bpdecay_params / np.sum(bpdecay_params)

        if torch.cuda.is_available():
            bpdecay_params = torch.FloatTensor(bpdecay_params).cuda()
        else:
            bpdecay_params = torch.FloatTensor(bpdecay_params)

        bpdecay_params = bpdecay_params.unsqueeze(0)
        bpdecay_params = bpdecay_params.unsqueeze(2)

        h_weight = 20
        b_weight = .05
        l1_weight = 0.5
        l2_weight = .5
        elbo_weight = 1

        l2_pred_loss = torch.nn.functional.mse_loss(pred, gt_normalized, reduction='none') * bpdecay_params
        l1_pred_loss = torch.nn.functional.l1_loss(pred, gt_normalized, reduction='none') * bpdecay_params

        l2_a_loss = torch.sum(l2_pred_loss[:, :, :12])
        l2_b_loss = torch.sum(l2_pred_loss[:, :, 12:])

        l2_loss =  h_weight * l2_a_loss + b_weight * l2_b_loss

        l1_a_loss = torch.mean(l1_pred_loss[:, :, :12])
        l1_b_loss = torch.mean(l1_pred_loss[:, :, 12:])
        l1_loss =  h_weight * l1_a_loss + b_weight * l1_b_loss


        #print(l2_weight , l2_loss , l1_weight ,l1_loss)
        pred_loss = l2_weight * l2_loss + l1_weight * l1_loss

        total_loss = pred_loss
        #print(total_loss)
        particle_pred = particle_pred.transpose(0, 1).contiguous()
        particle_gt = gt_normalized.repeat(1, self.num_particles, 1).transpose(0, 1).contiguous()

        l2_particle_loss = torch.nn.functional.mse_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params
        l1_particle_loss = torch.nn.functional.l1_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params


        #print(len(l1_particle_loss[0]))
        # p(y_t| \tau_{1:t}, x_{1:t}, \theta) is assumed to be a Gaussian with variance = 1.
        # other more complicated distributions could be used to improve the performance
        y_prob_l2 = torch.exp(-l2_particle_loss).view(self.num_particles, -1, sl, 24)
        l2_particle_loss = - y_prob_l2.mean(dim=1).log()

        y_prob_l1 = torch.exp(-l1_particle_loss).view(self.num_particles, -1, sl, 24)
        l1_particle_loss = - y_prob_l1.mean(dim=1).log()
        #print(y_prob_l1, l2_particle_loss)
        xy_l2_particle_loss = torch.mean(l2_particle_loss[:, :, :12])
        h_l2_particle_loss = torch.mean(l2_particle_loss[:,:,12:])
        
        '''
        remove *
        '''


        l2_particle_loss =  h_l2_particle_loss * h_weight + b_weight * xy_l2_particle_loss

        xy_l1_particle_loss = torch.mean(l1_particle_loss[:, :, :12])
        h_l1_particle_loss = torch.mean(l1_particle_loss[:,:,12:])
        l1_particle_loss =  h_l1_particle_loss * h_weight + b_weight * xy_l1_particle_loss

        belief_loss = l2_weight * l2_particle_loss + l1_weight * l1_particle_loss
        total_loss = total_loss + elbo_weight * belief_loss


        z = torch.from_numpy(np.array(y)).float().view([batch_size,1,24])

        #for i in z[:,0,12:]:
         #   print(i)#print(z[1,0,12:])# z[150,1,12:])
        loss_last = torch.nn.functional.mse_loss(pred[:,:,:12], z[:,:,:12]) * h_weight
        loss_last += torch.nn.functional.mse_loss(pred[:,:,12:], z[:,:,12:]) * b_weight
        particle_pred = particle_pred.view(self.num_particles, 24, batch_size)

        #obj_loss = -torch.sum(pred[:,1])

        return total_loss, loss_last, pred, particle_pred

def train_pfrnn(model,STEP_SIZE, EPOCHS):
    env = envs()
    lr = 5e-4
    optim = "Adam"
    optimizer = get_optim(lr, optim, model)

    # training - channging number of tests (normalize by constant)
    # Simplify the error - mse first
    # 
    rwd = []
    acts = []
    while env.epoch < EPOCHS:
        
        model.zero_grad()
        state = np.zeros((20,12))
        actions = np.random.rand(20,12)
        s_1 = env.reset()

        done = False

        pred, pred_pf = torch.empty((1,1,24)), torch.empty((1,384,24))
        x,y = np.zeros((20,12)), np.zeros((20, 12))


        x = state#np.append(x,state, axis=0)
        y = np.array([s_1])#np.append(y, np.array([s_1]), axis=0)
        a = actions
        while not done:
            c, c_pf = model.forward(state,actions)
            
            act = np.array([c.detach().numpy()[0][0][12:]])[0]
            #print(pred.shape, c.shape)
            
            #x = np.column_stack((x,np.array([state])))

            x = np.dstack((x,state))
            a = np.dstack((a,actions))
            y = np.dstack((y, np.array([s_1])))
            #y = np.column_stack((y,s_1))

            pred = torch.cat((pred, c))
            #print(pred.shape, pred_pf.shape, c.shape, c_pf.shape)
            pred_pf = torch.cat((pred_pf, c_pf))
            state = np.append(state[1:],np.array([s_1]), axis=0)
            s_1, done = env.step(act)
            actions = np.append(actions[1:],np.reshape(act, (1,12)),axis=0)
            #act = np.random.rand(1,1)
            
        print(env.epoch)


        #x = x.flatten()
        #a = a.flatten()
        #y = y.flatten()
        
        

        #y = np.append(y,y*0)
        y = np.hstack((y,y*0 + .01))
        x = np.hstack((x,a))
        #print(x.shape,a.shape,y.shape)
        #y = np.concatenate(y.flatten(), np.zeros((len(y.flatten()))))
        #print(x.shape, y.shape)
        loss,log_loss, c, c_pf = model.step(x, y, pred, pred_pf)

        print(loss,log_loss)#,obj_loss)
        rwd.append([loss,log_loss])
        acts.append([a.sum() / len(a)])
        log_loss.backward(retain_graph=True)
        loss.backward()
        #obj_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #torch.nn.utils.clip_grad_norm_(pfrnn.model.parameters(), 2)
        optimizer.step()

        if env.epoch % 500 == 0:
            prog = pred.detach().numpy()[:,:,:12]
            action = pred.detach().numpy()[:,:,12:]
            prog = prog.reshape((len(prog), 12))
            action = action.reshape((len(prog), 12))
            pd.DataFrame(prog).to_csv("output/full/"+str(env.epoch)+".csv",header=False,index=False)
            pd.DataFrame(action).to_csv("output/full/a"+str(env.epoch)+".csv",header=False,index=False)

    i = "_12_county_multi"
    pd.DataFrame(acts).to_csv('output/acts_all'+str(i)+'.csv',header=False,index=False)
    pd.DataFrame(rwd).to_csv('output/rewards_all'+str(i)+'.csv',header=False,index=False)

    #for i in range(3):
    

    env.output = i
    model.eval()
    with torch.no_grad():
        model.zero_grad()
        state = np.zeros((20,12))
        actions = np.random.rand(20,12)
        s_1 = env.reset(testing=True)
        

        done = False

        pred, pred_pf = torch.empty((1,1,24)), torch.empty((1,384,24))
        x,y = np.zeros((20,12)), np.zeros((20, 12))


        x = state#np.append(x,state, axis=0)
        y = np.array([s_1])#np.append(y, np.array([s_1]), axis=0)
        while not done:
            c, c_pf = model.forward(state,actions)

            act = np.array([c.detach().numpy()[0][0][12:]])[0]
            #print(pred.shape, c.shape)

            #x = np.column_stack((x,np.array([state])))

            x = np.append(x,state, axis=0)
            y = np.append(y, np.array([s_1]), axis=0)
            #y = np.column_stack((y,s_1))

            pred = torch.cat((pred, c))
            #print(pred.shape, pred_pf.shape, c.shape, c_pf.shape)
            pred_pf = torch.cat((pred_pf, c_pf))
            state = np.append(state[1:],np.array([s_1]), axis=0)
            s_1, done = env.step(act)
            actions = np.append(actions[1:],np.reshape(act, (1,12)),axis=0)

        prog = pred.detach().numpy()[:,:,:12]
        act = pred.detach().numpy()[:,:,12:]
        prog = prog.reshape((len(prog), 12))
        act = act.reshape((len(prog), 12))
        
        pd.DataFrame(prog).to_csv("output/prediction_rl"+str(i)+".csv",header=False,index=False)
        pd.DataFrame(act).to_csv("output/action"+str(i)+".csv",header=False,index=False)

    return

def get_optim(lr, optim, model):
    if optim == 'RMSProp':
        optim = torch.optim.RMSprop(
            model.parameters(), lr=lr)
    elif optim == 'Adam':
        optim = torch.optim.Adam(
            model.parameters(), lr=lr)
    else:
        raise NotImplementedError

    return optim


def main():
    STEP_SIZE = 20
    EPOCHS = 1000



    m = pfrnn()
    train_pfrnn(m,STEP_SIZE,EPOCHS)

if __name__ == '__main__':
    main()
