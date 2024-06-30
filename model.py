import torch
from torch import nn
from torch.distributions import Normal,Bernoulli
import math
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size
    def forward(self, tensor):
        return tensor.view(self.size)
    
class Encoder(nn.Module):
    def __init__(self, channel_input: int, latent_dim: int):
        super(Encoder, self).__init__()

        self.channel_input = channel_input
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),#[b,16,14,14]
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),#[b,32,7,7]
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),#[b,32,4,4]
            )
        self.output_layers = nn.Sequential(nn.Linear(512, 256),#[b,256]
                                            nn.ReLU(),
                                            nn.Linear(256, 2*self.latent_dim))#[b,2*latent_dim]

    def forward(self, x, K=20):
        _out = self.feature_extractor(x)#[b,32,4,4]
        _out = _out.view(-1, 4*4*32)
        _out = self.output_layers(_out)#[b,2*latent_dim]
        _mu = _out[:, 0:self.latent_dim]#[b,latent_dim]
        _log_var = _out[:, self.latent_dim:]#[b,latent_dim]       
        dist = Normal(_mu, (0.5 * _log_var).exp())#[b,latent_dim]维的正态分布
        _z = dist.rsample([K])#[K,b,latent_dim]
        return _mu, _log_var, _z, dist

class BernoulliDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_channel: int):
        super(BernoulliDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_channel = output_channel
        self.conv_layers = nn.Sequential(nn.Linear(latent_dim, 256),#[L,b,256]
                                        nn.ReLU(),
                                        nn.Linear(256, 512),#[L,b,512]
                                        nn.ReLU(),
                                        View((-1, 32, 4, 4)),#[L*b,32,4,4]
                                        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                                        nn.ReLU(),#[L*b,32,7,7]
                                        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),
                                        nn.ReLU(),#[L*b,16,14,14]
                                        nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),
                                        #[L*b,1,28,28]
                                    )
    def forward(self, latent, K=20): #latent dim :[K,b,latent_dim]
        _logits =  self.conv_layers(latent) #[K*b,1,28,28]
        _logits = _logits.view(K, latent.shape[1], 28, 28) #[K,b,28,28]
        output_dist = Bernoulli(logits=_logits) #[K,b,28,28]
        return _logits, output_dist

class ConvVAE(nn.Module):
    def __init__(self, input_channel: int, latent_dim: int):
        super(ConvVAE, self).__init__()
        self.input_channel = input_channel#1
        self.latent_dim = latent_dim#40
        self.encoder = Encoder(self.input_channel, self.latent_dim)
        self.decoder = BernoulliDecoder(self.latent_dim, self.input_channel)
        self.prior = Normal(0,1)

    def forward(self, input, K=20, mask = None,beta = 1):#input: [b,1,28,28]
        _mu, _log_var, _z, qzgivenx = self.encoder(input, K)#[K,b,latent_dim]
        log_pz = self.prior.log_prob(_z)#[K,b,latent_dim]
        log_qzgivenx = qzgivenx.log_prob(_z)
        kl = log_qzgivenx - log_pz  #[K,b,latent_dim]
        _logits, pxgivenz = self.decoder(_z, K) #[K,b,28,28]
        log_pxgivenz = pxgivenz.log_prob(input.squeeze())  #[b,28,28]-->log_prob-->[K,b,28,28]
        batch_kl = torch.sum(kl, axis=-1)#.permute(1,0) #[K,b]
        masked_log_pxgivenz = log_pxgivenz * mask.squeeze()#[K,b,28,28]
        batch_log_pxgivenz = torch.sum(masked_log_pxgivenz, axis=[2, 3]) # [K,b]
        
        _bound = batch_log_pxgivenz - beta * batch_kl
        bound = torch.logsumexp(_bound, axis=0) - math.log(K)#由于前面对P和Q取了对数，故这里先取exp再相加再取log，这里是对K取平均，bound维数为b
        avg_iwae_bound = torch.mean(bound)#这里是对(batch)取平均

        _output_dict = {'q_mean': _mu,
                        'q_log_var': _log_var,
                        'latents': _z,
                       # 'q_dist': q_dist,
                       # 'logits': _logits,
                        'output_dist': pxgivenz,
                        'kl': batch_kl,
                        'likelihood': batch_log_pxgivenz,#这个是[K,b]维度
                        'iwae_bound': avg_iwae_bound#这个是取batch似然取平均
                        }
        return _output_dict