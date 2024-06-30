#%% 对测试集进行缺失值填补
import numpy as np
import torch
from data import get_binarized_MNIST
from torch.utils.data import TensorDataset, DataLoader
from model import ConvVAE
from data import create_mcar_mask, create_mar_mask, get_imputation

device = torch.device('cuda')
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_,_,data_test = get_binarized_MNIST()

missing_rate = 0.3
missing_pattern = 'MCAR'
if missing_pattern == 'MCAR':
    mask_test = create_mcar_mask(10000, 28*28, missing_rate)
elif missing_pattern == 'MAR':
    mask_test = create_mar_mask(data_test)

missing_data_mask = np.ones_like(mask_test) - mask_test
# 0 imputation
zero_imputed_test = get_imputation(data_test, mask_test, missing_data_mask, 0)

latent_dim = 40
learning_rate = 0.0005
n_epoch = 100
batch_size = 8

test_dataset = TensorDataset(torch.from_numpy(zero_imputed_test).float(), torch.from_numpy(mask_test))
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

model = ConvVAE(input_channel=1, latent_dim=latent_dim)
model.load_state_dict(torch.load('./mcar_epoch100.pt'))
model.to(device)
model.eval()
K = 50 # imputation samples
whole_dataset_imputations = []
T = 3

batch_img, batch_mask = next(iter(test_loader))
batch_img = torch.unsqueeze(batch_img, 1)#[b,1,28,28]
img = batch_img.to(device)
mask = batch_mask.to(device)

_mu, _log_var, _z, qzgivenx = model.encoder(img, K)
xmis0, _ = model.decoder(_z, K)
xhat = img.squeeze()*mask+(1-mask)*xmis0#[K,b,28,28]
xhat = xhat.type(torch.cuda.FloatTensor).view(-1,1,28,28)#[K*b,1,28,28]
#%%
ztkall=torch.Tensor().cuda()
wtkall=torch.Tensor().cuda()
with torch.no_grad():
    for t in range(1,T+1):
        _mu, _log_var, _z, qzgivenx = model.encoder(xhat, 1)#z:[1,K*b,40] qz
        logqzgivenx = qzgivenx.log_prob(_z) #[1, K*b, 40]
        logqzgivenxobs = torch.sum(logqzgivenx, dim=-1).view(K,-1) #[K,b]
        _z = _z.view(K,batch_size, latent_dim)#[K,b,40]
        logpx = model.prior.log_prob(_z)  #[K, batch_size, latent_dim]
        logpx = torch.sum(logpx, dim=-1).view(K,-1) #[K, batch_size]
        logits, pxgivenz = model.decoder(_z, K)# [K, batch_size, 28, 28]
        logpxgivenz = pxgivenz.log_prob(img.squeeze(1))
        logpxobsgivenz = torch.sum(logpxgivenz*mask, dim=[2, 3]) #[K, batch_size]
        log_unnormalized_importance_weights = logpxobsgivenz + logpx - logqzgivenxobs
        _w = torch.softmax(log_unnormalized_importance_weights,0, dtype=torch.float32)#[K,b] normalized w
        ztk_dist =  torch.distributions.Categorical(logits=_w.transpose(0,1))#[b,K]
        idx = ztk_dist.sample([K]).transpose(0,1) #[b,K]个位置
        #分别处理每个batch的重加权采样
        _zt = _z.view(batch_size,K,latent_dim)
        ztk=torch.Tensor().cuda()
        for i in range(batch_size):
            ztk = torch.cat((ztk, _zt[i][idx[i]]), dim=0)
        #ztk--[b,K,40]
        
        ztk = ztk.view(batch_size,K, latent_dim)#[b,K,40]
        ztk = ztk.transpose(0,1)#[K,b,40]

        xhat, _ = model.decoder(ztk, K)# [K, batch_size, 28, 28]
        xhat = img.squeeze()*mask+(1-mask)*xhat#[K,b,28,28]
        xhat = xhat.type(torch.cuda.FloatTensor).view(-1,1,28,28)#[K*b,1,28,28]
        xhat = torch.sigmoid(xhat)
        ztkall =torch.cat((ztkall, ztk), dim=0)
        wtkall =torch.cat((wtkall, _w), dim=0)
    #ztkall--->[t,K,b,40]
    #wtkall--->[t,K,b]
    ztkall = ztkall.view(T*K,batch_size,latent_dim)
    wtkall = wtkall.view(T*K,batch_size)
    _wall = torch.softmax(wtkall,0, dtype=torch.float32)#[T*K,b]
    ztkall_dist =  torch.distributions.Categorical(logits=_wall.transpose(0,1))#[b,T*K]
    sample_tk_w = ztkall_dist.sample([T*K])#[T*K,b]

    ztkallt = ztkall.view(batch_size,T*K,latent_dim)
    sample_tk_wt = sample_tk_w.transpose(0,1)#[b,T*K]
    
    zi=torch.Tensor().cuda()
    for i in range(batch_size):
        zi=torch.cat((zi, ztkallt[i][sample_tk_wt[i]]), dim=0)#[b*TK,40]
    zi = zi.view(T*K,batch_size, latent_dim)#[T*K,b,40]
    xhatTK, _ = model.decoder(zi, T*K)#[T*K,b,28,28]

    xhatTK = torch.sigmoid(xhatTK)
    
#%%
    xhatTK = xhatTK.view(T*K,batch_size,-1)
    imputations = torch.einsum('ij,ijk->jk', _wall, xhatTK) # batch_size * 784
    final_imputations = torch.round(imputations)
    final_imputations = final_imputations.reshape(-1,28,28)
    xfullfinal = img.squeeze()*mask+(1-mask)*final_imputations
    recon = xfullfinal.cpu().numpy()

    # whole_dataset_imputations.extend(final_imputations)

#%%
whole_dataset_imputations = np.array(whole_dataset_imputations)
print(whole_dataset_imputations.shape) # same shape as the original flatten vector

# print(mask_train.flatten().shape)
# missing_idx_trainset = np.where(mask_train.flatten()==0.)[0]
# print(missing_idx_trainset[0:10])
# true_values_missing_data = data.flatten()[missing_idx_trainset]
# imputed_values = whole_dataset_imputations.flatten()[missing_idx_trainset]
# print(missing_idx_trainset.shape)
# print(true_values_missing_data.shape)
# print(imputed_values.shape)
# true_values_missing_data = torch.from_numpy(true_values_missing_data).int()
# imputed_values = torch.from_numpy(imputed_values).int()

# # now I can compute the accuracy
# print('Accuracy')
# print((true_values_missing_data == imputed_values).shape)
# print(torch.sum(true_values_missing_data == imputed_values)/len(missing_idx_trainset))

# print('F1 score')
# from sklearn.metrics import f1_score
# print(f1_score(true_values_missing_data, imputed_values))

# #
# #
# # I can plot some imputations
# n_examples = 64
# zero_imputed_train = zero_imputed_train.flatten()
# zero_imputed_train[missing_idx_trainset] = imputed_values
# imputed_train = zero_imputed_train.reshape(50000,1,28,28)

# # get some data
# dataset_display = get_imputation(data, mask_train, missing_data_mask, 0.5)

# dataset_display = dataset_display.reshape(-1,1,28,28)
# data_with_missing_values = dataset_display[0:n_examples,:,:,:]
# imputed_examples = imputed_train[0:n_examples,:,:,:]
# data = data.reshape(-1,1,28,28)
# true_examples = data[0:n_examples,:,:,:]

# data_with_missing_values = torch.from_numpy(data_with_missing_values).float()
# imputed_examples = torch.from_numpy(imputed_examples).float()
# true_examples = torch.from_numpy(true_examples).float()

# data_with_missing_values = utils.make_grid(data_with_missing_values)
# imputed_examples = utils.make_grid(imputed_examples)
# true_examples = utils.make_grid(true_examples)
#%%
import matplotlib.pyplot as plt
# plt.imshow(whole_dataset_imputations[10].reshape(28,28), cmap='gray')
plt.imshow(recon[3], cmap='gray')
plt.title('Images with missing data')
plt.show()
plt.close()

# plt.imshow(imputed_examples[0], cmap='gray')
# plt.title('Images imputed with single imputation')
# # plt.show()
# plt.close()

# plt.imshow(true_examples[0], cmap='gray')
# plt.title('True images in the training set')
# # plt.show()
# plt.close()


# whole_dataset_imputations = torch.from_numpy(whole_dataset_imputations).float()
# whole_dataset_imputations = whole_dataset_imputations.reshape(-1,1,28,28)
# reconstruction = whole_dataset_imputations[0:n_examples,:,:,:]
# reconstruction = utils.make_grid(reconstruction)

# plt.imshow(reconstruction[0], cmap='gray')
# plt.title('Reconstructions (Imputation of the whole images)')
# plt.show()
# plt.close()

# %%
