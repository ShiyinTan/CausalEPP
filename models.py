import cppimport
import torch
import torch.nn as nn
from config import opt
import sys
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
sys.path.append('cppcode')


def min_max_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def z_score_normalization(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    normalized_data = (data - mean_val) / std_val
    return normalized_data

def max_abs_normalization(data):
    max_abs_val = np.max(np.abs(data))
    normalized_data = data / max_abs_val
    return normalized_data

def robust_scaler(data):
    median_val = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    normalized_data = (data - median_val) / iqr
    return normalized_data



class Models(nn.Module):
    def __init__(self, user_num, item_num, factor_num, min_time, max_time, device='cpu'):
        super(Models, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        self.n_users = user_num
        self.m_items = item_num
        self.embed_size = factor_num
        self.q = nn.Parameter(torch.ones(item_num, ) * opt.q) # quality
        self.b = nn.Parameter(torch.ones(item_num, ) * opt.b)
        self.tau = torch.ones(item_num, ) * opt.tau  # 100000000
        self.pbd_import()
        self.device = device

        if opt.method == 'TIDE' or opt.method == 'TIDE-noq' or opt.method == 'TIDE-fixq':
            self.pbd.load_popularity(self.tau.cpu().detach().numpy())
        elif opt.method in ['CausalEPP']:
            if opt.dataset in ['Amazon-Music', 'Amazon-CDs_and_Vinyl', 'Douban-movie']:
                self.pbd.load_popularity(self.tau.cpu().detach().numpy())
            save_file_name_item_freq = "data/{}/item_frequency_all_times.csv".format(opt.dataset)
            save_file_name_user_freq = "data/{}/user_frequency_all_times.csv".format(opt.dataset)
            save_file_name_user_freq_high_popu = "data/{}/user_frequency_high_popularity_all_times.csv".format(opt.dataset)
            save_file_all_times = "data/{}/all_times.npy".format(opt.dataset)
            save_file_item_freq_overall = 'data/{}/item_frequency.csv'.format(opt.dataset)

            self.item_frequency_overall = pd.read_csv(save_file_item_freq_overall)['count'].values
            self.item_frequency_overall = robust_scaler(self.item_frequency_overall)
            self.item_frequency_overall = np.clip(self.item_frequency_overall, 1e-3, 1.0)
            self.item_frequency_all_times = pd.read_csv(save_file_name_item_freq, index_col=0).fillna(0)+1
            self.user_frequency_all_times = pd.read_csv(save_file_name_user_freq, index_col=0).fillna(0)+1
            
            self.user_frequency_high_popularity_all_times = pd.read_csv(save_file_name_user_freq_high_popu, index_col=0)
            self.all_times = np.load(save_file_all_times) # [0, 1] [1, 2] [2, 3]
            self.user_sensitivity_all_times = self.user_frequency_high_popularity_all_times.fillna(0)/(opt.smooth_parameter+self.user_frequency_all_times.fillna(0)+1)
            self.user_sensitivity_all_times = min_max_normalization(self.user_sensitivity_all_times)
            series_data = self.user_sensitivity_all_times\
                                .replace(0, np.nan)\
                                .interpolate(method='linear', axis=1).fillna(0).clip(1e-3, 1.0)
            self.user_sensitivity_rolling_mean_all_times = series_data.T.rolling(window=opt.rolling_window, min_periods=1).mean().T
            self.user_sensitivity_gradients_all_times = pd.DataFrame(np.gradient(self.user_sensitivity_rolling_mean_all_times, axis=1), 
                                                                     index=self.user_sensitivity_rolling_mean_all_times.index, 
                                                                     columns=self.user_sensitivity_rolling_mean_all_times.columns)
            self.user_sensitivity_all_times = np.clip(self.user_sensitivity_all_times, 1e-3, 1.0)

            item_frequency_all_times_dict = {}
            for i in range(self.item_frequency_all_times.shape[1]):
                item_frequency_all_times_dict[i] = min_max_normalization(self.item_frequency_all_times.iloc[:,i])
            self.item_frequency_all_times_norm = pd.DataFrame(item_frequency_all_times_dict)

            self.item_frequency_rolling_mean_all_times = self.item_frequency_all_times_norm.T.rolling(window=opt.rolling_window, min_periods=1).mean().T
            self.item_frequency_gradients_all_times = pd.DataFrame(np.gradient(self.item_frequency_rolling_mean_all_times, axis=1), 
                                                                    index=self.item_frequency_rolling_mean_all_times.index, 
                                                                    columns=self.item_frequency_rolling_mean_all_times.columns)


        self.max_time = max_time
        self.min_time = min_time
        PDA_popularity = pd.read_csv(opt.PDA_popularity_path, sep='\t')
        self.PDA_array = PDA_popularity.values.reshape(-1)
        self.DICE_pop = np.load(opt.DICE_popularity_path)
        if opt.backbone == 'LightGCN':
            self.get_graph()
            self.embed_user_0 = nn.Embedding(user_num, factor_num)
            self.embed_item_0 = nn.Embedding(item_num, factor_num)
            nn.init.normal_(self.embed_user_0.weight, std=0.1)
            nn.init.normal_(self.embed_item_0.weight, std=0.1)
            self.get_emb()
            if not opt.test_only:
                pass
            # print(self.embed_user, type(self.embed_user))
        else:
            self.embed_user = nn.Embedding(user_num, factor_num)
            self.embed_item = nn.Embedding(item_num, factor_num)
            nn.init.normal_(self.embed_user.weight, std=0.01)
            nn.init.normal_(self.embed_item.weight, std=0.01)

    def pbd_import(self):
        if opt.dataset == 'Douban-movie':
            self.pbd = cppimport.imp("pybind_douban")
        elif opt.dataset == 'Amazon-CDs_and_Vinyl':
            self.pbd = cppimport.imp("pybind_amazon_cd")
        elif opt.dataset == 'Amazon-Music':
            self.pbd = cppimport.imp("pybind_amazon_music")
        elif opt.dataset == 'Ciao':
            self.pbd = cppimport.imp("pybind_ciao")
        elif opt.dataset == 'gowalla':
            self.pbd = cppimport.imp("pybind_gowalla")

    def get_graph(self):
        # print('Creating GCN graph')
        if os.path.exists(opt.graph_data_path) and os.path.exists(opt.graph_index_path):
            graph_data = np.load(opt.graph_data_path)
            graph_index = np.load(opt.graph_index_path)
            graph_data = torch.from_numpy(graph_data)
            graph_index = torch.from_numpy(graph_index)

            graph_size = torch.torch.Size(
                [self.n_users + self.m_items, self.n_users + self.m_items])
            self.Graph = torch.sparse.FloatTensor(graph_index, graph_data, graph_size)
            # print('loaded graph')
            self.Graph = self.Graph.coalesce().to(self.device)
        else:
            train_data = pd.read_csv(opt.train_data, sep='\t')
            trainUser = train_data['user'].values
            trainItem = train_data['item'].values
            user_dim = torch.LongTensor(trainUser)
            item_dim = torch.LongTensor(trainItem)
            # print(user_dim)
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size(
                [self.n_users + self.m_items, self.n_users + self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            index = dense.nonzero()
            data = dense[dense >= 1e-9]
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size(
                [self.n_users + self.m_items, self.n_users + self.m_items]))
            np.save(opt.graph_index_path, index.t().numpy())
            np.save(opt.graph_data_path, data.numpy())
            # print('saved graph')
            self.Graph = self.Graph.coalesce().to(self.device)
            # print('Created GCN graph')
            # print(self.Graph, self.Graph.shape)

    def get_emb(self, is_training=True):
        if opt.backbone == 'MF':
            embed_user, embed_item = self.embed_user.weight, self.embed_item.weight
            return embed_user, embed_item
        else:
            users_emb = self.embed_user_0.weight
            items_emb = self.embed_item_0.weight
            all_emb = torch.cat([users_emb, items_emb])
            all_emb = all_emb.to(self.device)
            #   torch.split(all_emb , [self.num_users, self.num_items])
            embs = [all_emb]
            # print(self.Graph)
            # print(all_emb)
            for layer in range(opt.n_layers):
                all_emb = torch.sparse.mm(self.Graph, all_emb)
                embs.append(all_emb)

            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            self.embed_user, self.embed_item = torch.split(light_out, [self.n_users, self.m_items])
            # print(self.embed_item)
            # self.embed_user, self.embed_item = torch.split(all_emb, [self.n_users, self.m_items])
            return self.embed_user, self.embed_item

    def reg_loss(self, user, item_i, item_j):
        if opt.backbone == 'LightGCN':
            user_embedding_0 = self.embed_user_0.weight[user]
            item_i_embedding_0 = self.embed_item_0.weight[item_i]
            item_j_embedding_0 = self.embed_item_0.weight[item_j]
        else:
            user_embedding_0 = self.embed_user.weight[user]
            item_i_embedding_0 = self.embed_item.weight[item_i]
            item_j_embedding_0 = self.embed_item.weight[item_j]

        reg_loss = (1 / 2) * (user_embedding_0.norm(2).pow(2) +
                              item_i_embedding_0.norm(2).pow(2) +
                              item_j_embedding_0.norm(2).pow(2)) / float(len(user))

        return reg_loss

    def forward(self, user, item_i, item_j, timestamp, split_idx):
        # t0 = time.time()
        embed_user, embed_item = self.get_emb()
        # print(time.time() - t0)

        user_embedding = embed_user[user]
        item_i_embedding = embed_item[item_i]
        item_j_embedding = embed_item[item_j]

        reg_loss = self.reg_loss(user, item_i, item_j)

        if opt.method == "base":  # MF模型，不用计算popularity，直接输出结果
            prediction_i = (user_embedding * item_i_embedding).sum(
                dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(
                dim=-1)
            return prediction_i, prediction_j, reg_loss
        elif opt.method == 'IPS':
            item_i = item_i.cpu().numpy().astype(int)
            prediction_i = (user_embedding * item_i_embedding).sum(
                dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(
                dim=-1)
            IPS_c = torch.from_numpy(
                np.minimum(
                    1 / self.DICE_pop[item_i],
                    opt.IPS_lambda)).to(self.device) / opt.IPS_lambda
            ips_loss = -1 * \
                       (IPS_c * (prediction_i - prediction_j).sigmoid().log()).sum() + opt.lamb * reg_loss
            return ips_loss
        elif opt.method == 'DICE':
            DICE_size = int(self.embed_size / 2)
            prediction_i = (user_embedding * item_i_embedding).sum(
                dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(
                dim=-1)
            # print('click', prediction_i.shape)
            loss_click = -(prediction_i - prediction_j).sigmoid().log().sum()

            if opt.backbone == "MF":
                user_embedding_1 = self.embed_user(user.unique())[:, 0:DICE_size]
                item_i_embedding_1 = self.embed_item(item_i.unique())[:, 0:DICE_size]
                item_j_embedding_1 = self.embed_item(item_j.unique())[:, 0:DICE_size]
                user_embedding_2 = self.embed_user(user.unique())[:, DICE_size:]
                item_i_embedding_2 = self.embed_item(item_i.unique())[:, DICE_size:]
                item_j_embedding_2 = self.embed_item(item_j.unique())[:, DICE_size:]
            else:
                user_embedding_1 = self.embed_user[user.unique()][:, 0:DICE_size]
                item_i_embedding_1 = self.embed_item[item_i.unique()][:, 0:DICE_size]
                item_j_embedding_1 = self.embed_item[item_j.unique()][:, 0:DICE_size]
                user_embedding_2 = self.embed_user[user.unique()][:, DICE_size:]
                item_i_embedding_2 = self.embed_item[item_i.unique()][:, DICE_size:]
                item_j_embedding_2 = self.embed_item[item_j.unique()][:, DICE_size:]
            loss_discrepancy = -1 * ((user_embedding_1 - user_embedding_2).sum() +
                                     (item_i_embedding_1 - item_i_embedding_2).sum() +
                                     (item_j_embedding_1 - item_j_embedding_2).sum())

            item_i_np = item_i.cpu().numpy().astype(int)
            item_j_np = item_j.cpu().numpy().astype(int)
            pop_relation = self.DICE_pop[item_i_np] > self.DICE_pop[item_j_np]
            user_O1 = user[pop_relation]
            user_O2 = user[~pop_relation]
            item_i_O1 = item_i[pop_relation]
            item_j_O1 = item_j[pop_relation]
            item_i_O2 = item_i[~pop_relation]
            item_j_O2 = item_j[~pop_relation]

            if opt.backbone == "MF":
                user_embedding = self.embed_user(user_O1)[:, 0:DICE_size]
                item_i_embedding = self.embed_item(item_i_O1)[:, 0:DICE_size]
                item_j_embedding = self.embed_item(item_j_O1)[:, 0:DICE_size]
            else:
                user_embedding = self.embed_user[user_O1][:, 0:DICE_size]
                item_i_embedding = self.embed_item[item_i_O1][:, 0:DICE_size]
                item_j_embedding = self.embed_item[item_j_O1][:, 0:DICE_size]
            prediction_i = (user_embedding * item_i_embedding).sum(
                dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(
                dim=-1)
            # print('interest', prediction_i.shape)
            loss_interest = - \
                (prediction_i - prediction_j).sigmoid().log().sum()

            if opt.backbone == "MF":
                user_embedding = self.embed_user(user_O1)[:, DICE_size:]
                item_i_embedding = self.embed_item(item_i_O1)[:, DICE_size:]
                item_j_embedding = self.embed_item(item_j_O1)[:, DICE_size:]
            else:
                user_embedding = self.embed_user[user_O1][:, DICE_size:]
                item_i_embedding = self.embed_item[item_i_O1][:, DICE_size:]
                item_j_embedding = self.embed_item[item_j_O1][:, DICE_size:]
            prediction_i = (user_embedding * item_i_embedding).sum(
                dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(
                dim=-1)
            # print('p1', prediction_i.shape)
            loss_popularity_1 = - \
                (prediction_j - prediction_i).sigmoid().log().sum()

            if opt.backbone == "MF":
                user_embedding = self.embed_user(user_O2)[:, DICE_size:]
                item_i_embedding = self.embed_item(item_i_O2)[:, DICE_size:]
                item_j_embedding = self.embed_item(item_j_O2)[:, DICE_size:]
            else:
                user_embedding = self.embed_user[user_O2][:, DICE_size:]
                item_i_embedding = self.embed_item[item_i_O2][:, DICE_size:]
                item_j_embedding = self.embed_item[item_j_O2][:, DICE_size:]
            prediction_i = (user_embedding * item_i_embedding).sum(
                dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(
                dim=-1)
            # print('p2', prediction_i.shape)
            loss_popularity_2 = - \
                (prediction_i - prediction_j).sigmoid().log().sum()
            # print(loss_click, loss_interest, loss_popularity_1, loss_popularity_2)

            # dice_loss = loss_click + 0.1*(loss_interest + loss_popularity_1 + loss_popularity_2)
            return loss_click, loss_interest, loss_popularity_1, loss_popularity_2, loss_discrepancy, reg_loss
        elif opt.method == "PDA" or opt.method == 'PD':
            item_i = item_i.cpu().numpy()
            item_j = item_j.cpu().numpy()
            PDA_idx_i = (item_i * 10 + split_idx).astype(int)
            PDA_popularity_i = self.PDA_array[PDA_idx_i]
            PDA_popularity_i = torch.from_numpy(PDA_popularity_i).to(self.device)
            PDA_idx_j = (item_j * 10 + split_idx).astype(int)
            PDA_popularity_j = self.PDA_array[PDA_idx_j]
            PDA_popularity_j = torch.from_numpy(PDA_popularity_j).to(self.device)
            prediction_i = (F.elu((user_embedding * item_i_embedding).sum(dim=-1)
                                  ) + 1) * (PDA_popularity_i ** opt.PDA_gamma)
            prediction_j = (F.elu((user_embedding * item_j_embedding).sum(dim=-1)
                                  ) + 1) * (PDA_popularity_j ** opt.PDA_gamma)
            # print(PDA_popularity_i ** opt.PDA_gamma, PDA_popularity_j ** opt.PDA_gamma)
            return prediction_i, prediction_j, reg_loss
        elif opt.method == 'TIDE-noc':
            self.popularity_i = F.softplus(self.q[item_i])
            self.popularity_j = F.softplus(self.q[item_j])
            self.prediction_i = F.softplus((user_embedding * item_i_embedding).sum(
                dim=-1)) * torch.tanh(self.popularity_i)
            self.prediction_j = F.softplus((user_embedding * item_j_embedding).sum(
                dim=-1)) * torch.tanh(self.popularity_j)
            # print(torch.tanh(self.popularity_i), torch.tanh(self.popularity_j))
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
        elif opt.method == 'TIDE-noq':
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            # t0 = time.time()
            popularity_i_s = torch.from_numpy(
                self.pbd.popularity(item_i_np, timestamp)).to(self.device)
            self.popularity_i = F.softplus(
                self.b[item_i]) * popularity_i_s  # 先计算正样本的流行度

            self.popularity_j = torch.from_numpy(
                self.pbd.popularity(item_j_np, timestamp)).to(self.device)
            self.popularity_j = F.softplus(self.b[item_j]) * self.popularity_j

            self.prediction_i = F.softplus((user_embedding * item_i_embedding).sum(
                dim=-1)) * torch.tanh(self.popularity_i)
            self.prediction_j = F.softplus((user_embedding * item_j_embedding).sum(
                dim=-1)) * torch.tanh(self.popularity_j)
            # print(torch.tanh(self.popularity_i), torch.tanh(self.popularity_j))
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
        elif opt.method == 'TIDE' or opt.method == 'TIDE-int' or opt.method == 'TIDE-e' or opt.method == 'TIDE-fixq':
            # self.pbd.load_popularity(self.tau.cpu().detach().numpy())
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            # t0 = time.time()
            popularity_i_s = torch.from_numpy(
                self.pbd.popularity(item_i_np, timestamp)).to(self.device)
            self.popularity_i = F.softplus(
                self.q[item_i]) + F.softplus(self.b[item_i]) * popularity_i_s  # 先计算正样本的流行度

            self.popularity_j = torch.from_numpy(
                self.pbd.popularity(item_j_np, timestamp)).to(self.device)
            self.popularity_j = F.softplus(self.q[item_j]) + F.softplus(
                self.b[item_j]) * self.popularity_j

            self.prediction_i = F.softplus((user_embedding * item_i_embedding).sum(
                dim=-1)) * torch.tanh(self.popularity_i)
            self.prediction_j = F.softplus((user_embedding * item_j_embedding).sum(
                dim=-1)) * torch.tanh(self.popularity_j)
            # print(torch.tanh(self.popularity_i), torch.tanh(self.popularity_j))
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
        
        
        elif opt.method in ['CausalEPP']:
            user_u_np = user.cpu().numpy()
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()

            time_block = np.clip(np.searchsorted(self.all_times, timestamp), 0, 100)
            sensitivity_u = torch.from_numpy(self.user_sensitivity_all_times.iloc[user_u_np, time_block].values).to(self.device)
            frequency_i = torch.from_numpy(self.item_frequency_overall[item_i_np]).to(self.device)
            frequency_j = torch.from_numpy(self.item_frequency_overall[item_j_np]).to(self.device)

            
            short_term_popularity_norm_i = torch.from_numpy(self.item_frequency_all_times_norm.iloc[item_i_np, time_block].values).to(self.device)
            short_term_popularity_norm_j = torch.from_numpy(self.item_frequency_all_times_norm.iloc[item_j_np, time_block].values).to(self.device)
            popularity_i_s = torch.from_numpy(
                self.pbd.popularity(item_i_np, timestamp)).to(self.device)
            popularity_j_s = torch.from_numpy(
                self.pbd.popularity(item_j_np, timestamp)).to(self.device)
            
            alignment_p_i_s_u = torch.exp(-opt.smooth_parm_alpha * torch.abs(short_term_popularity_norm_i-sensitivity_u))
            alignment_p_j_s_u = torch.exp(-opt.smooth_parm_alpha * torch.abs(short_term_popularity_norm_j-sensitivity_u))

            self.popularity_i = F.softplus(self.q[item_i]) * frequency_i + \
                F.softplus(self.b[item_i]) * popularity_i_s * alignment_p_i_s_u # 先计算正样本的流行度
            self.popularity_j = F.softplus(self.q[item_j]) * frequency_j + \
                F.softplus(self.b[item_j]) * popularity_j_s * alignment_p_j_s_u
            
            self.prediction_i = F.softplus((user_embedding * item_i_embedding).sum(dim=-1)) * \
                torch.tanh(self.popularity_i)
            self.prediction_j = F.softplus((user_embedding * item_j_embedding).sum(dim=-1)) * \
                torch.tanh(self.popularity_j)
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()

            return self.prediction_i, self.prediction_j, reg_loss