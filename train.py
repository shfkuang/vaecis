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
optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
best_valid_log_likelihood = -10000000000

#%%
for epoch in tqdm(range(n_epoch)):
    tmp_kl = 0
    tmp_likelihood = 0
    tmp_iwae_elbo = 0
    obs_in_epoch = 0

    for i, (batch_img, batch_mask) in enumerate(train_loader):
        optim.zero_grad()
        obs_in_epoch += len(batch_img)
        batch_mask = batch_mask.to(device)
        batch_mask = torch.unsqueeze(batch_mask,1)#[b,1,28,28]
        batch_img = torch.unsqueeze(batch_img, 1)#[b,1,28,28]
        batch_img = batch_img.to(device)
        _output_dict_ = model(batch_img, K_training, mask=batch_mask)
        loss = -_output_dict_['iwae_bound']
        loss.backward()
        optim.step()
        tmp_kl += torch.sum(torch.mean(_output_dict_['kl'],dim=0)).item()
        tmp_likelihood += torch.sum(torch.mean(_output_dict_['likelihood'], dim=0)).item()
        tmp_iwae_elbo += _output_dict_['iwae_bound'].item() * len(batch_img)
    print(
        "epoch {0}/{1}, train IWAE bound: {2:.2f}, train likelihod: {3:-2f}, train KL: {4:.2f}"
            .format(epoch + 1, n_epoch, tmp_iwae_elbo / obs_in_epoch,
                    tmp_likelihood / obs_in_epoch, tmp_kl / obs_in_epoch))

    # with torch.no_grad():
    #     model.eval()
    #     valid_log_like = 0
    #     valid_obs = 0
    #     for j, (valid_batch_img, valid_batch_mask) in enumerate(valid_loader):
    #         valid_batch_img = torch.unsqueeze(valid_batch_img, 1)
    #         valid_batch_img = valid_batch_img.to(device)
    #         valid_obs += len(valid_batch_img)
    #         _output_dict_ = model(valid_batch_img, n_iwae_testing)
    #         valid_elbo = _output_dict_['iwae_bound'] * len(valid_batch_img)
    #         valid_log_like += valid_elbo

    #     avg_valid = valid_log_like / valid_obs
    #     print(f'epoch: {epoch}, Validation log p(x): {avg_valid}')
    #     if avg_valid > best_valid_log_likelihood:
    #         best_valid_log_likelihood = avg_valid
    #         torch.save(model.state_dict(),f'./mcar_epoch{epoch+1}.pt')
    #     model.train()
torch.save(model.state_dict(),f'./mcar_epoch{epoch+1}.pt')

# %%
