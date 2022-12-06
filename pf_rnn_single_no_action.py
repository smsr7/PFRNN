from pfrnn import PFLSTMCell
from gidi_env.gidi_sim.env_single import env as envs
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

class pfrnn(nn.Module):
    def __init__(self):

        super(pfrnn, self).__init__()
        INPUT_SIZE = 40
        HIDDEN_SIZE = 64

        EXT_ACT = 20
        EXT_OBS = 20
        NUM_PARTICLES = 10
        RESAMP_ALPHA = 0.5
        DROPOUT_RATE = .3

        self.num_particles = NUM_PARTICLES
        self.hidden_dim = HIDDEN_SIZE

        self.model = PFLSTMCell(NUM_PARTICLES, INPUT_SIZE, HIDDEN_SIZE, EXT_OBS, EXT_ACT, RESAMP_ALPHA)

        self.hnn_dropout = nn.Dropout(DROPOUT_RATE)
        self.hidden2label = nn.Linear(self.hidden_dim, 1)


        self.obs_embedding = nn.Linear(EXT_ACT,EXT_OBS)
        self.act_embedding = nn.Linear(1, EXT_ACT)
    #def encode(self):

    def detach_hidden(self, hidden):
        if isinstance(hidden, tuple):
            return tuple([h.detach() for h in hidden])
        else:
            return hidden.detach()

    def forward(self, x, batch_size = 1):
        seq_len = len(list(x))
        #batch_size = len(list(x))
        initializer = torch.zeros
        h0 = initializer(batch_size * self.num_particles, self.hidden_dim)
        c0 = initializer(batch_size * self.num_particles, self.hidden_dim)
        p0 = torch.ones(batch_size * self.num_particles, 1) * np.log(1 / self.num_particles)
        hidden = (h0, c0, p0)

        obs_input = torch.from_numpy(x).float()
        #print(obs_input.shape)
        act_input = torch.from_numpy(np.full((len(list(x)),1), .5)).float()
        #print(act_input.shape)
        emb_obs = torch.relu(self.obs_embedding(obs_input))
        emb_act = torch.relu(self.act_embedding(act_input))
        #print(emb_act.size(),emb_obs.size())
        #print(emb_act, emb_obs)
        embedding = torch.cat((emb_obs, emb_act), dim=1)
        #print(embedding)
        #embedding = torch.cat((emb_obs, emb_act), dim=1)
        #print(embedding.shape)
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


    def step(self,x,y):
        bpdecay = .08
        pred, particle_pred = self.forward(x)
        #print(particle_pred)
        pred = pred * 100
        gt_normalized = torch.from_numpy(y).float().view([len(y),1,1])#torch.stack(, dim=0)
        batch_size = pred.size(1)
        sl = pred.size(0)
        bpdecay_params = np.exp(bpdecay * np.arange(sl))
        bpdecay_params = bpdecay_params / np.sum(bpdecay_params)
        if torch.cuda.is_available():
            bpdecay_params = torch.FloatTensor(bpdecay_params).cuda()
        else:
            bpdecay_params = torch.FloatTensor(bpdecay_params)

        bpdecay_params = bpdecay_params.unsqueeze(0)
        bpdecay_params = bpdecay_params.unsqueeze(2)
        #pred = pred.transpose(0, 1).contiguous()

        h_weight = .5
        l1_weight = 1.0
        l2_weight = 1.0
        elbo_weight = 1.0
        #pred = pred.transpose(0, 1).contiguous()
        l2_pred_loss = torch.nn.functional.mse_loss(pred, gt_normalized, reduction='none') * bpdecay_params
        l1_pred_loss = torch.nn.functional.l1_loss(pred, gt_normalized, reduction='none') * bpdecay_params
        #print(l2_pred_loss,"bbbbbbbbb")
        #print(l2_pred_loss)
        #print(len(l2_pred_loss[0]))
        #l2_xy_loss = torch.sum(l2_pred_loss[:, :, :2])
        l2_h_loss = torch.sum(l2_pred_loss[:, 0, :])

        l2_loss =  h_weight * l2_h_loss

        #l1_xy_loss = torch.mean(l1_pred_loss[:, :, :2])
        l1_h_loss = torch.mean(l1_pred_loss[:, 0, :])
        l1_loss =  h_weight * l1_h_loss

        pred_loss = l2_weight * l2_loss + l1_weight * l1_loss

        total_loss = pred_loss

        particle_pred = particle_pred.transpose(0, 1).contiguous()
        particle_gt = gt_normalized.repeat(1, self.num_particles, 1).transpose(0, 1).contiguous()


        l2_particle_loss = torch.nn.functional.mse_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params
        l1_particle_loss = torch.nn.functional.l1_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params


        #print(len(l1_particle_loss[0]))
        # p(y_t| \tau_{1:t}, x_{1:t}, \theta) is assumed to be a Gaussian with variance = 1.
        # other more complicated distributions could be used to improve the performance
        y_prob_l2 = torch.exp(-l2_particle_loss).view(self.num_particles, -1, sl, 1)
        l2_particle_loss = - y_prob_l2.mean(dim=1).log()

        y_prob_l1 = torch.exp(-l1_particle_loss).view(self.num_particles, -1, sl, 1)
        l1_particle_loss = - y_prob_l1.mean(dim=1).log()
        #print(y_prob_l1, l2_particle_loss)
        #xy_l2_particle_loss = torch.mean(l2_particle_loss[:, :, :2])
        h_l2_particle_loss = torch.mean(l2_particle_loss)
        l2_particle_loss =  h_weight * h_l2_particle_loss

        #xy_l1_particle_loss = torch.mean(l1_particle_loss[:, :, :2])
        h_l1_particle_loss = torch.mean(l1_particle_loss)
        l1_particle_loss = h_weight * h_l1_particle_loss

        belief_loss = l2_weight * l2_particle_loss + l1_weight * l1_particle_loss
        total_loss = total_loss + elbo_weight * belief_loss


        z = torch.from_numpy(y).float().view([len(y),1])#.type(torch.DoubleTensor)
        #print(z)
        loss_last = torch.nn.functional.mse_loss(pred[:,-1], z)

        #loss_last = loss_last.todouble()
        #print(particle_pred)
        particle_pred = particle_pred.view(self.num_particles, batch_size, sl)
        #print(loss_last,"qqqqqxxxxxxx")

        #obj_loss = (-1*(abs(y - pred) + abs(pred_pf.var())))

        return total_loss, loss_last, pred, particle_pred

def train_pfrnn(model, iteration, STEP_SIZE, EPOCHS):
    env = envs(STEP_SIZE)
    lr = 8e-4
    optim = "Adam"
    optimizer = get_optim(lr, optim, model)
    while env.epoch < EPOCHS:

        model.zero_grad()
        state = np.zeros(STEP_SIZE)
        act= .1
        
        env.iteration = iteration
        s_1 = env.reset()
        


        done = False

        pred, pred_pf = torch.empty((1,1,1)), torch.empty((1,10,1))
        x,y = np.empty((1,0)), np.empty((1,0))


        x = np.column_stack((x,np.array([state])))
        y = np.column_stack((y,s_1))
        while not done:
            c, c_pf = model.forward(np.array([state]))

            #print(pred.shape, c_pf.shape)

            x = np.column_stack((x,np.array([state])))
            y = np.column_stack((y,s_1))
            pred = torch.cat((pred, c))
            pred_pf = torch.cat((pred_pf, c_pf))

            state = np.append(state[1:],s_1)
            s_1, done = env.step(act)
            act =.1

        #print(env.epoch)

        x = np.reshape(x, (len(y[0]),20))
        y = np.reshape(y, (len(y[0]),1))

        #print(x,y,pred,pred_pf)
        loss,log_loss, c, c_pf = model.step(x, y)#, pred, pred_pf)

        print(loss,log_loss)#,obj_loss)
        log_loss.backward(retain_graph=True)
        loss.backward()
        #obj_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #torch.nn.utils.clip_grad_norm_(pfrnn.model.parameters(), 2)
        optimizer.step()

        print(env.epoch)
    model.eval()
    with torch.no_grad():
        s_1 = env.reset(testing=True)
        model.zero_grad()
        done = False
        act = .1
        prog = np.array([])
        prog_pf = np.array([])


        while not done:
            loss,log_loss, c, c_pf = model.step(np.array([state]),s_1)

            print(loss,log_loss)

            c = torch.reshape(c,(-1,)).detach().numpy()
            c_pf = c_pf.view(10,len(c)).detach().numpy() * 100
            prog = np.append(prog, c)
            prog_pf = np.append(prog_pf, c_pf)


            state = np.append(state[1:],s_1)
            s_1, done = env.step(np.array([act]))
        
            #c = torch.reshape(c,(-1,)).detach().numpy()
        #c_pf = c_pf.view(10,len(c)).detach().numpy() * 100
        
        pd.DataFrame(prog).mean(axis=1).to_csv("output/prediction_rl_single_"+ str(env.iteration)+".csv",header=False,index=False)
        #pd.DataFrame(c_pf).to_csv("output/pf_"+str(env.epoch)+".csv",header=False,index=False)
        #pd.DataFrame(c_pf).var(axis=0).to_csv("output/var_"+str(env.epoch)+".csv",header=False,index=False)
        #pd.DataFrame(c_pf).mean(axis=1).to_csv("output/"+str(env.epoch)+".csv",header=False,index=False)
        #pd.DataFrame(c).mean(axis=1).to_csv("output/"+str(env.epoch)+".csv",header=False,index=False)

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


def main(i):
    STEP_SIZE = 20
    EPOCHS = 200


    m = pfrnn()
    train_pfrnn(m,i,STEP_SIZE,EPOCHS)

if __name__ == '__main__':
    for i in range(12):
        main(i)
