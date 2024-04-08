import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='models'):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.fc1.weight.data.uniform_(-f1, f1)
        #self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        #f2 = 0.002
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)
        #self.q.weight.data.uniform_(-f3, f3)
        #self.q.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    
    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file), strict=True)


class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, allow_x_movement, name,
                 chkpt_dir='models'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.allow_x_movement = allow_x_movement
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.fc1.weight.data.uniform_(-f1, f1)
        #self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #f2 = 0.002
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        #f3 = 0.004
        f3 = 0.003
        
        self.mu_z = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.mu_z.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu_z.bias.data,   -f3, f3)
        
        if self.allow_x_movement:
            self.mu_x = nn.Linear(self.fc2_dims, 1)
            T.nn.init.uniform_(self.mu_x.weight.data, -f3, f3)
            T.nn.init.uniform_(self.mu_x.bias.data,   -f3, f3)
            
        self.mu_a = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.mu_a.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu_a.bias.data,   -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    
    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        action_z = T.sigmoid(self.mu_z(x))     # action_z are in range [ 0., 1.]
        if self.allow_x_movement:
            action_x = T.tanh(self.mu_x(x))    # action_x are in range [-1., 1.]
        action_a = T.tanh(self.mu_a(x))        # action_a are in range [-1., 1.]
        if self.allow_x_movement:
            actions = T.cat([action_z, action_x, action_a], dim=-1)
            return actions
        else:
            actions = T.cat([action_z, action_a], dim=-1)
            return actions

    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file), strict=True)
