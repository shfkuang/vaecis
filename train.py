#%%
import numpy as np
import torch
from data import get_binarized_MNIST
from torch.utils.data import TensorDataset, DataLoader
from model import ConvVAE
from math import ceil
from tqdm import tqdm
from data import create_mcar_mask, create_mar_mask, get_imputation
from torchvision import utils

missing_pattern = 'MCAR'
device = torch.device('cuda')
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data,data_valid,_ = get_binarized_MNIST()
missing_rate = 0.5
if missing_pattern == 'MCAR':
    mask_train = create_mcar_mask(50000, 28*28, missing_rate)
    mask_valid = create_mcar_mask(10000, 28*28, missing_rate)

elif missing_pattern == 'MAR':
    mask_train = create_mar_mask(data)
    mask_valid = create_mar_mask(data_valid)

missing_data_mask = np.ones_like(mask_train) - mask_train

# 0 imputation
zero_imputed_train = get_imputation(data, mask_train, missing_data_mask, 0)

latent_dim = 40
learning_rate = 0.0005
n_epoch = 100
batch_size = 100
K_training = 50
n_iwae_testing = 50

train_dataset = TensorDataset(torch.from_numpy(zero_imputed_train).float(), torch.from_numpy(mask_train))
valid_dataset = TensorDataset(torch.from_numpy(data_valid).float(), torch.from_numpy(mask_valid))
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=True)

model = ConvVAE(input_channel=1, latent_dim=latent_dim)
model.to(device)


observation_dim = 1
latent_dim = 50
learning_rate = 3e-4
n_epoch = 100
batch_size = 64
n_iwae_training = 100
n_iwae_testing = 100
beta = 0.95

train_dataset = TensorDataset(torch.from_numpy(zero_imputed_train).float(), torch.from_numpy(mask_train))
valid_dataset = TensorDataset(torch.from_numpy(data_valid).float(), torch.from_numpy(mask_valid))

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=True


model = ConvVAE(input_channel=1, latent_dim=latent_dim)
model.to(device)
print(mode

#%%
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())

# print(count_parameters(model))
# print(count_parameters(model.encoder))
# print(count_parameters(model.decoder)

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
avgbound = []
best_valid_log_likelihood = -10000000000
best_valid_log_kl = -10000000000
for epoch in tqdm(range(n_epoch)):
    tmp_kl = 0
    tmp_likelihood = 0
    tmp_vae_elbo = 0
    tmp_iwae_elbo = 0
    obs_in_epoch = 0

    for i, (batch_img, batch_mask) in enumerate(train_loader):
        optim.zero_grad()

        obs_in_epoch += len(batch_img)

        batch_mask = batch_mask.to(device)
        batch_mask = torch.unsqueeze(batch_mask,1)
        batch_img = torch.unsqueeze(batch_img, 1)
        batch_img = batch_img.to(device)

        _output_dict_ = model(batch_img, n_iwae_training, mask=batch_mask,beta = beta)

        if n_iwae_training is not None:
            loss = -_output_dict_['iwae_bound']
        else:
            loss = -_output_dict_['vae_bound']

        # print(loss)
        loss.backward()
        optim.step()

        # if n_iwae_training is not None:
        #     tmp_kl += torch.sum(torch.mean(_output_dict_['kl'],dim=0)).item()
        #     tmp_likelihood += torch.sum(torch.mean(_output_dict_['likelihood'], dim=0)).item()
        # else:
        #     tmp_kl += torch.sum(_output_dict_['kl']).item()
        #     tmp_likelihood += torch.sum(_output_dict_['likelihood']).item()

        # print(_output_dict_['likelihood'].shape)
        # print(torch.mean(_output_dict_['likelihood'], dim=0).shape)
        # print(torch.mean(_output_dict_['likelihood'], dim=1).shape)
        # print('----')
        # tmp_likelihood += torch.sum(torch.mean(_output_dict_['likelihood'], dim=1)).item()
        # print('2: ', loss.item() * len(obs))
        # print(len(batch_img))
        # print(loss)
        # print(-loss.item() * len(batch_img))
        # print('-----')
        # tmp_vae_elbo += _output_dict_['vae_bound'].item() * len(batch_img)
        # tmp_iwae_elbo += _output_dict_['iwae_bound'].item() * len(batch_img)

    # print(tmp_vae_elbo)
    # print(
    #     "epoch {0}/{1}, train VAE ELBO: {2:.2f}, train IWAE bound: {3:.2f}, train likelihod: {4:-2f}, train KL: {5:.2f}"
    #         .format(epoch + 1, n_epoch, tmp_vae_elbo / obs_in_epoch, tmp_iwae_elbo / obs_in_epoch,
    #                 tmp_likelihood / obs_in_epoch, tmp_kl / obs_in_epoch))
    with torch.no_grad():
        model.eval()
        valid_log_like = 0
        valid_log_kl = 0
        valid_obs = 0
        for j, (valid_batch_img, valid_batch_mask) in enumerate(valid_loader):
            valid_batch_img = torch.unsqueeze(valid_batch_img, 1)
            valid_batch_img = valid_batch_img.to(device)
            valid_obs += len(valid_batch_img)
            valid_batch_mask = valid_batch_mask.to(device)
            valid_batch_mask = torch.unsqueeze(valid_batch_mask,1)
            _output_dict_ = model(valid_batch_img, n_iwae_testing, mask=valid_batch_mask,beta = beta)
            valid_elbo = _output_dict_['iwae_bound'] * len(valid_batch_img)
            valid_log_like += valid_elbo
            
            valid_kl = torch.mean(torch.logsumexp(_output_dict_['kl'], axis=0)) * len(valid_batch_img)
            valid_log_kl += valid_kl

        avg_valid = valid_log_like / valid_obs
        avg_valid_kl = valid_log_kl / valid_obs
        print('Validation log p(x): ', avg_valid)
        avgbound.append(avg_valid.cpu().numpy())
        if avg_valid > best_valid_log_likelihood:
            best_valid_log_likelihood = avg_valid
            best_valid_log_kl = avg_valid_kl
            # print('BEST',best_valid_log_likelihood)
            #
            # # save model
            # torch.save(model.state_dict(),
            #            saving_directory + 'models/best_baseline_model_based_on_validation_log_like' + str(seed) + '_iwae_' + str(n_iwae_training) + '_mar.pt')

        model.train()
print('BEST_log_like: ',best_valid_log_likelihood)
print('BEST_kl: ',best_valid_log_kl)
#torch.save(model.state_dict(),f'./model_100_{beta}.pt')