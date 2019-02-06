import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp


class ActorCriticNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64]):
        super(ActorCriticNet, self).__init__()
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer
        self.p_fcs = nn.ModuleList()
        self.v_fcs = nn.ModuleList()
        self.hidden_layer_v = [256, 256]
        p_fc = nn.Linear(num_inputs, self.hidden_layer[0])
        v_fc = nn.Linear(num_inputs, self.hidden_layer_v[0])
        self.p_fcs.append(p_fc)
        self.v_fcs.append(v_fc)
        for i in range(len(self.hidden_layer)-1):
            p_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i+1])
            v_fc = nn.Linear(self.hidden_layer_v[i], self.hidden_layer_v[i+1])
            self.p_fcs.append(p_fc)
            self.v_fcs.append(v_fc)
        self.mu = nn.Linear(self.hidden_layer[-1], num_outputs)
        self.log_std = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)
        self.v = nn.Linear(self.hidden_layer_v[-1],1)
        self.noise = 0
        #self.train()

    def forward(self, inputs):
        # actor
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = F.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return mu, log_std, v

    def set_noise(self, noise):
        self.noise = noise

class ActorCriticNetWithHeightMap(ActorCriticNet):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64]):
        super(ActorCriticNetWithHeightMap, self).__init__()
        self.height_map_dim = [512, 512]
        self.height_map = np.zeros((512, 512))

        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 11, stride=3, padding=0),nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 9, stride=3, padding=0),nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=3, padding=1),nn.ReLU())
        self.fc = nn.Linear(5*5*32, 128)
        self.p_fcs[0] = nn.Linear(num_inputs + 128, self.hidden_layer[0])
        self.v_fcs[0] = nn.Linear(num_inputs + 128, self.hidden_layer[0])

    def set_height_map(self, height_map):
        self.height_map = np.copy(height_map)

    def forward(self, input):
        hegiht_map = Variable(torch.Tensor(self.hegiht_map).unsqueeze(0))
        height_map_output = self.conv1(height_map)
        height_map_output = self.conv2(height_map_output)
        height_map_output = self.conv3(height_map_output)
        height_map_output = self.out_view(out.size(0), -1)
        height_map_output = self.fc(height_map_output)
        x = F.relu(self.p_fc[0](torch.cat([input, height_map_output])))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.p_fcs[i+1](x))
        mu = F.tanh(self.mu(x))
        log_std = Variable(self.noise*torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](torch.cat([inputs2, height_map_output])))
        for i in range(len(self.hidden_layer)-1):
            x = F.relu(self.v_fcs[i+1](x))
        v = self.v(x)
        #print(mu)
        return mu, log_std, v


class Shared_grad_buffers():
    def __init__(self, model):
        self.grads = {}
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] = torch.ones(p.size()).share_memory_()

    def add_gradient(self, model):
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] += p.grad.data

    def reset(self):
        for name,grad in self.grads.items():
            self.grads[name].fill_(0)

class Shared_obs_stats():
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs).share_memory_()
        self.mean = torch.zeros(num_inputs).share_memory_()
        self.mean_diff = torch.zeros(num_inputs).share_memory_()
        self.std = torch.zeros(num_inputs).share_memory_()
        self.num_inputs = num_inputs
        self.sum = torch.zeros(num_inputs).share_memory_()
        self.sum_sqr = torch.zeros(num_inputs).share_memory_()

    def observes(self, obs):
        # observation mean var updates
        x = obs.data.squeeze()
        if True:
            self.n += 1.
            last_mean = self.mean.clone()
            self.sum = self.sum + x
            self.sum_sqr += x.pow(2)
            self.mean = self.sum / self.n
            self.std = (self.sum_sqr / self.n - self.mean.pow(2)).clamp(1e-2,1e9).sqrt()
            self.mean = self.mean.float()
            self.std = self.std.float()
        #self.mean = (self.mean * self.n + x) / self.
            #self.mean += (x-self.mean)/self.n
            #self.mean_diff += (x-last_mean)*(x-self.mean)
            #self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_mean = Variable(self.mean.unsqueeze(0).expand_as(inputs))
        obs_std = Variable(self.std.unsqueeze(0).expand_as(inputs))
        obs_mean = ((inputs - obs_mean) / obs_std)
        #obs_std = Variable(torch.sqrt(self.var).unsqueeze(0).expand_as(inputs))
        return torch.clamp(obs_mean, -10.0, 10.0)

    def reset(self):
        self.n = torch.zeros(self.num_inputs).share_memory_()
        self.mean = torch.zeros(self.num_inputs).share_memory_()
        self.mean_diff = torch.zeros(self.num_inputs).share_memory_()
        self.var = torch.zeros(self.num_inputs).share_memory_()