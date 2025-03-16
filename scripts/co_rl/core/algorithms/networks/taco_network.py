import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from scripts.co_rl.core.utils.utils import weight_init, TruncatedNormal

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
        
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        self.apply(weight_init)
        
    def forward(self, obs):
        states = obs[:, :10]
        height_maps = obs[:, 10:].view(-1, 23, 11)
        h = self.convnet(height_maps)
        h = h.view(h.shape[0], -1)
        return torch.cat([states, h], dim=1)
    
class Twin_Q_net(nn.Module):
    def __init__(self, projection_dim, feature_dim, latent_a_dim, device, hidden_dims=(256, 256), activation_fc=F.relu):
        super(Twin_Q_net, self).__init__()
        self.device = device

        self.activation_fc = activation_fc

        self.trunk = nn.Sequential(nn.Linear(projection_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.input_layer_A = nn.Linear(feature_dim + latent_a_dim, hidden_dims[0])
        self.hidden_layers_A = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_A = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_A.append(hidden_layer_A)
        self.output_layer_A = nn.Linear(hidden_dims[-1], 1)

        self.input_layer_B = nn.Linear(feature_dim + latent_a_dim, hidden_dims[0])
        self.hidden_layers_B = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_B = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_B.append(hidden_layer_B)
        self.output_layer_B = nn.Linear(hidden_dims[-1], 1)
        self.apply(weight_init)

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action, act_tok=None):
        x, u = self._format(state, action)
        if act_tok is not None:
            action = act_tok(action)
        h = self.trunk(x)        
        
        h_action = torch.cat([h, u], dim=1)

        x_A = self.activation_fc(self.input_layer_A(h_action))
        for i, hidden_layer_A in enumerate(self.hidden_layers_A):
            x_A = self.activation_fc(hidden_layer_A(x_A))
        x_A = self.output_layer_A(x_A)

        x_B = self.activation_fc(self.input_layer_B(h_action))
        for i, hidden_layer_B in enumerate(self.hidden_layers_B):
            x_B = self.activation_fc(hidden_layer_B(x_B))
        x_B = self.output_layer_B(x_B)

        return x_A, x_B


class Actor(nn.Module):
    def __init__(self, projection_dim, feature_dim, action_dim, action_bound,
                 hidden_dims=(256, 256), activation_fc=F.relu, device='cuda'):
        super(Actor, self).__init__()
        self.device = device

        self.trunk = nn.Sequential(nn.Linear(projection_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(feature_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)

        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        self.action_rescale = torch.as_tensor((action_bound[1] - action_bound[0]) / 2., dtype=torch.float32)
        self.action_rescale_bias = torch.as_tensor((action_bound[1] + action_bound[0]) / 2., dtype=torch.float32)

        self.apply(weight_init)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state, std=None):
        x = self._format(state)
        x = self.trunk(x)
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        mean = self.mean_layer(x)
        mean = torch.tanh(mean) * self.action_rescale + self.action_rescale_bias
        
        if std == None:
            return mean
        else:
            std = torch.ones_like(mean) * std
            dist = TruncatedNormal(mean, std)
            return dist

class ActionEncoding(nn.Module):
    def __init__(self, action_dim, hidden_dim, latent_action_dim, multistep):
        super().__init__()
        self.action_dim = action_dim
        self.action_tokenizer = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_action_dim)
        )

        self.action_seq_tokenizer = nn.Sequential(
            nn.Linear(latent_action_dim*multistep, latent_action_dim*multistep),
            nn.LayerNorm(latent_action_dim*multistep), nn.Tanh()
        )
        self.apply(weight_init)
        
    def forward(self, action, seq=False):
        if seq:
            batch_size = action.shape[0]    
            action = self.action_tokenizer(action) #(batch_size, length_action_dim)
            action = action.reshape(batch_size, -1)
            return self.action_seq_tokenizer(action)
        else:
            return self.action_tokenizer(action)   
        
class TACO_model(nn.Module):
    def __init__(self, projection_dim, feature_dim, latent_a_dim, hidden_dim, act_tok, encoder, multistep=3, device='cuda'):
        super(TACO_model, self).__init__()

        self.multistep = multistep
        self.encoder = encoder
        self.device = device

        self.proj_sa = nn.Sequential(
            nn.Linear(feature_dim + latent_a_dim*self.multistep, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.act_tok = act_tok
        self.proj_s = nn.Sequential(nn.Linear(projection_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.reward = nn.Sequential(
            nn.Linear(feature_dim+latent_a_dim*multistep, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.apply(weight_init)
    
    def encode(self, x, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.proj_s(self.encoder(x)[0])
        else:
            z_out = self.proj_s(self.encoder(x)[0])
        return z_out
    
    def project_sa(self, s, a):
        x = torch.concat([s,a], dim=-1)
        return self.proj_sa(x)
    
    def compute_logits(self, z_a, z_pos):        
        Wz = torch.matmul(self.W, z_pos.T)
        logits = torch.matmul(z_a, Wz) 
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


