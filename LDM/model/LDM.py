import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from kmeans_pytorch import kmeans, kmeans_predict


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

clean_path = '../dataset/wsdream/wsdream_train.dat'
num_user = n_user
num_item = n_item
data = pd.read_csv(clean_path, sep="\t", names=['uid', 'iid', 'rating'])
row, col, rating = [], [], []
for line in data.itertuples():
    uid, iid, r = list(line)[1:]
    row.append(uid)
    col.append(iid)
    rating.append(r)
matrix = csr_matrix((rating, (row, col)), shape=(num_user, num_item))
train_matrix = matrix.toarray()

batch_mask = train_matrix.copy()
batch_mask[batch_mask > 0] = 1

dataset = torch.Tensor(train_matrix).float()

matrix_re = csr_matrix((rating, (col, row)), shape=(num_item, num_user))
train_matrix_re = matrix_re.toarray()
dataset_re = torch.Tensor(train_matrix_re).float()

seeds = [5586, 4810, 577, 1945, 74]
num_cluster = 4
np.random.seed(seeds[num_cluster - 2])

# k-means
cluster_ids_x, cluster_centers = kmeans(
    X=dataset_re, num_clusters=num_cluster, distance='euclidean', device=torch.device("cuda:0")
)
assign_list = []
for i in range(num_cluster):
    assign_list.append(np.where(cluster_ids_x == i)[0])

target_num = 20
targets_list = []
targets = targets_list[:target_num]
target_qos = 0.001

# Encoder
class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(self.linear2(x))
        return x

# Decoder
class Decoder(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(self.linear2(x))
        return x

class AE(torch.nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(AE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, output_size)
    def forward(self, x):
        feat = self.encoder(x)
        re_x = self.decoder(feat)
        return re_x

loss_BCE = torch.nn.BCELoss(reduction = 'sum')
loss_MSE = torch.nn.MSELoss(reduction = 'sum')

latent_d = 128
vaes = []
latent_matrix = []
for i in range(num_cluster):
    api_list = assign_list[i]
    input_size = output_size = len(api_list)
    latent_size = int(latent_d / num_cluster)
    if(i == num_cluster - 1 and latent_d % num_cluster != 0):
        latent_size = latent_d - latent_size * (num_cluster - 1)
    ae_hidden_size = 256

    ae_epochs = 500
    ae_batch_size = 32
    ae_learning_rate = 1e-3
    modelname_i = 'vaes_' + str(i) + '.pth'
    ae_model = AE(input_size, output_size, latent_size, ae_hidden_size)
    optimizer = optim.Adam(ae_model.parameters(), lr=ae_learning_rate)
    vaes.append(ae_model)
    flag = False
    # try:
    #     ae_model.load_state_dict(torch.load(modelname_i))
    #     print('[INFO] Load AutoEncoder Model complete')
    #     flag = True
    # except:
    #     pass
    matrix_tmp = train_matrix[:, api_list]
    target_idx_list = np.where(np.isin(api_list, targets))[0]
    dataset_tmp = torch.Tensor(matrix_tmp).float()
    ae_dataloader = torch.utils.data.DataLoader(dataset_tmp, batch_size=ae_batch_size, shuffle=True)
    if (flag == False):
        loss_history = {'train': [], 'eval': []}
        for epoch in range(ae_epochs):
            # train
            ae_model.train()
            train_loss = 0
            train_nsample = 0
            for idx, imgs in enumerate(ae_dataloader):
                re_imgs = ae_model(imgs)
                target_imgs = re_imgs.detach().clone()
                n = len(target_imgs)
                for item_id in target_idx_list:
                    for user_id in range(n):
                        target_imgs[user_id, item_id] = target_qos
                loss_atk = loss_BCE(re_imgs, target_imgs)
                loss = loss_MSE(re_imgs, imgs) + loss_atk
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            torch.save(ae_model.state_dict(), modelname_i)
    for j in range(num_user):
        raw = dataset_tmp[j]
        latent_raw = ae_model.encoder(raw)
        if(i == 0):
            latent_matrix.append(latent_raw.detach().numpy().tolist())
        else:
            latent_matrix[j].extend(latent_raw.detach().numpy().tolist())
latent_matrix = np.array(latent_matrix)

num_steps = 500
input_size_diff = output_size_diff = latent_d
#beta，beta
betas = torch.linspace(-6,6,num_steps)
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

#alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt
alphas = 1-betas
alphas_prod = torch.cumprod(alphas,0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)

alphas_bar_sqrt = torch.sqrt(alphas_prod)

one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
==one_minus_alphas_bar_sqrt.shape


def q_x(x_0, t):
    noise = torch.randn_like(x_0)
    noise = abs(noise)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise)



class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units = latent_d):
        super(MLPDiffusion, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(input_size_diff, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, output_size_diff),
                nn.ReLU(),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x, t):
        #         x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-2](x)
        x = self.linears[-1](x)
        return x


def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]

    if(batch_size % 2 == 0):
        t = torch.randint(0, n_steps, size=(batch_size // 2,))
        t = torch.cat([t, n_steps - 1 - t], dim=0)
    else:
        t = torch.randint(0, n_steps, size=(batch_size,))
    t = t.unsqueeze(-1)

    a = alphas_bar_sqrt[t]
    aml = one_minus_alphas_bar_sqrt[t]

    e = torch.randn_like(x_0)
    x = x_0 * a + e * aml

    output = model(x, t.squeeze(-1))

    return (e - output).square().mean()

def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    t = torch.tensor([t])
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x, t)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z

    return (sample)

seed = 1234


class EMA():

    def __init__(self, mu=0.01):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


print('Training model...')
batch_size = 32
diff_dataset = torch.Tensor(latent_matrix).float()
dataloader = torch.utils.data.DataLoader(diff_dataset, batch_size=batch_size, shuffle=True)
# epoch
num_epoch = 1000

model = MLPDiffusion(num_steps)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(num_epoch):

    for idx, batch_x in enumerate(dataloader):
        loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    if (step % 100 == 0):
        print(loss)
        x_seq = p_sample_loop(model, diff_dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)

forward = []
for i in range(num_steps):
    q_i = q_x(diff_dataset, torch.tensor([i]))
    forward.append(q_i)

reverse = []
for i in range(num_steps):
    cur_x = x_seq[i].detach()
    reverse.append(cur_x)

ans = []
for aaa in reverse:
    ans.append(aaa.numpy())
latent_ans = ans[num_steps-1]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

latent_ans = sigmoid(latent_ans)

ratings = np.zeros((num_user, num_item))
idx = 0
for latent_user in latent_ans:
    start = 0
    end = int(latent_d / num_cluster)
    for i in range(num_cluster):
        api_list = assign_list[i]
        if(i == num_cluster - 1):
            latent_user_i = latent_user[start:]
        else:
            latent_user_i = latent_user[start:end]
        start = end
        end += int(latent_d / num_cluster)
        latent_ratings = torch.Tensor(latent_user_i)
        re_ratings = vaes[i].decoder(latent_ratings).detach().numpy()
        new_ratings = re_ratings
        for j in range(len(new_ratings)):
            print(j)
            ratings[idx][api_list[j]] = new_ratings[j]
    idx += 1

for i in range(len(ratings)):
    arr = ratings[i]
    ratings[i] = [max(0.001, min(x, 19.99)) for x in arr]
ratings_mask = ratings * batch_mask

n_users = num_user
fake_users = []
for idx in range(n_users):
    profile_sample = ratings_mask[idx]
    profile_sample_1 = [round(x, 5) for x in profile_sample]
    fake_users.append(profile_sample_1)
data_to_write = ""
attacked_path = '../result/attacked/wsdream_diffusion_' + str(num_steps) + '.dat'
index = 0

for fake_profile in fake_users:
    injected_iid = [i for i, x in enumerate(fake_profile) if x != 0]
    injected_rating = [fake_profile[i] for i in injected_iid]
    data_to_write += ('\n'.join(
        map(lambda x: '\t'.join(map(str, [index] + list(x))),
            zip(injected_iid, injected_rating))) + '\n')
    index += 1
if os.path.exists(attacked_path): os.remove(attacked_path)
with open(attacked_path, 'a+') as fout:
    fout.write(data_to_write)
