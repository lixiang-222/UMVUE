# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb
import time

from tqdm import tqdm

from utils import ndcg_func,  recall_func, precision_func
acc_func = lambda x,y: np.sum(x == y) / len(x)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import expected_calibration_error

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class MF(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
           
    def fit(self, x, y, 
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                # print(selected_idx)
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                xent_loss = self.xent_func(pred,sub_y)

                #loss = xent_loss

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def predict(self, x, is_training=False):
        if is_training:
            pred, u_emb, i_emb = self.forward(x, is_training)
            return pred.detach().cpu().numpy(), u_emb.detach().cpu().numpy(), i_emb.detach().cpu().numpy()
        else:
            pred = self.forward(x, is_training)
            return pred.detach().cpu().numpy()

class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x, is_training=False):
        if is_training:
            pred, u_emb, i_emb = self.forward(x, is_training)
            return pred.detach().cpu().numpy(), u_emb.detach().cpu().numpy(), i_emb.detach().cpu().numpy()
        else:
            pred = self.forward(x, is_training)
            return pred.detach().cpu().numpy()

class NCF_BaseModel(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        out = self.sigmoid(self.linear_1(z_emb))
        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)        

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()


class Embedding_Sharing(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(Embedding_Sharing, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        if is_training:
            return torch.squeeze(z_emb), U_emb, V_emb
        else:
            return torch.squeeze(z_emb)        
    
    
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_1 = torch.nn.Linear(input_size, input_size // 2, bias = False)
        self.linear_2 = torch.nn.Linear(input_size // 2, 1, bias = True)
        self.xent_func = torch.nn.BCELoss()        
        
    def forward(self, x):
        
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.sigmoid(x)

        return torch.squeeze(x)

    def predict(self, x, u_emb_test = None, i_emb_test = None):
        if u_emb_test is not None:
            obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
                                shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
            obs_u = u_emb_test[obs == 1]
            obs_i = i_emb_test[obs == 1]
            obs_x = torch.cat([obs_u, obs_i], axis=1)
            # print(obs_x.shape)
            pred = self.forward(obs_x)
            return pred.detach().cpu().numpy()
        else:
            pred = self.forward(x)
            return pred.detach()



class Base(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size, batch_size_prop,
                    embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, num_epoch = 1000, stop = 5, lr=0.05, lamb=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 

        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9
        # generate all counterfactuals and factuals
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model.forward(sub_x)

                xent_loss = F.binary_cross_entropy(pred, sub_y, reduction="mean") # o*eui/pui  

                optimizer_pred.zero_grad()
                xent_loss.backward()
                optimizer_pred.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]
        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]
        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()

class MF_DR_JL_ECE(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, n_bins = 15, calib_lamb = 1, calib_epoch_freq=2,
        num_epoch=1000, lr1=0.05, lr2=0.05, lr3=0.05, gamma = 0.1, lamb_prop = 0, lamb_pred = 0, lamb_imp = 0,
        tol=1e-4, G=1, option = 'ce', verbose=True): 

        optimizer_prop = torch.optim.Adam(
            self.propensity_model.parameters(), lr=lr1, weight_decay=lamb_prop)
        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr2, weight_decay=lamb_pred)
        optimizer_imp = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr3, weight_decay=lamb_imp)

        last_loss = 1e9
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                prop = self.propensity_model.forward(sub_x)
                inv_prop = 1/torch.clip(prop, gamma, 1)
                
                sub_y = torch.Tensor(sub_y).cuda()
                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x).cuda()                
                
                x_all_idx = ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.forward(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction="sum")
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction="sum")
                        
                ips_loss = (xent_loss - imputation_loss)
                
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction="sum")
                
                dr_loss = (ips_loss + direct_loss)/x_sampled.shape[0]
                optimizer_pred.zero_grad()
                dr_loss.backward()
                optimizer_pred.step()

                pred_detached = pred.detach()
                e_loss = F.binary_cross_entropy(pred_detached, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred_detached, reduction="none")
                
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop.detach()).sum()
                optimizer_imp.zero_grad()
                imp_loss.backward()
                optimizer_imp.step()
                
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
                prop_all = self.propensity_model.forward(x_sampled)

                if option == 'mse':
                    prop_loss = nn.MSELoss()(prop_all, sub_obs)
                else:
                    prop_loss = F.binary_cross_entropy(prop_all, sub_obs)
                # prop_loss = nn.MSELoss()(prop_all, sub_obs)
                
                loss = prop_loss

                if idx % int(calib_epoch_freq * total_batch) == 0:
                    calib_loss, boundaries = expected_calibration_error(sub_obs, prop_all, n_bins, return_boundaries=True)
                    loss += calib_lamb * calib_loss
                elif boundaries is not None:
                    calib_loss = expected_calibration_error(sub_obs, prop_all, n_bins, boundaries=boundaries)
                    loss += calib_lamb * calib_loss

                optimizer_prop.zero_grad()
                loss.backward()
                optimizer_prop.step()

                epoch_loss += dr_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-JL-ECE] epoch:{}, loss:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL-ECE] epoch:{}, loss:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL-ECE] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred
    
class MF_IPS(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size, batch_size_prop,
                    embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        # self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        # self.epsilon = torch.nn.Parameter(torch.Tensor([0]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, num_epoch = 1000, stop = 5, lr=0.05, lamb=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 
        
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9
        # generate all counterfactuals and factuals
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()
                
                pred = self.prediction_model.forward(sub_x)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="mean") # o*eui/pui  
                
                optimizer_pred.zero_grad()
                xent_loss.backward()
                optimizer_pred.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]
        
        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]
        
        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x_all)
        total_batch = num_sample // self.batch_size_prop
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(num_sample) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.") 
            
class MF_ASIPS(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size, batch_size_prop,
                    embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction1_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        # self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.prediction2_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x_all)
        total_batch = num_sample // self.batch_size_prop
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(num_sample) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.") 

    def fit(self, x, y, tao, num_epoch = 1000, stop = 5, lr=0.05, lamb=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 
        
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]
        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_prediction1 = torch.optim.Adam(
            self.prediction1_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prediction2 = torch.optim.Adam(
            self.prediction2_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        
        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9
        # generate all counterfactuals and factuals
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction1_model.forward(sub_x)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y,
                    weight=inv_prop.detach())

                loss = xent_loss

                optimizer_prediction1.zero_grad()
                loss.backward()
                optimizer_prediction1.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 10:
                    print("[MF-IPS-Pred1] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-Pred1] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-Pred1] Reach preset epochs, it seems does not converge.")

        early_stop = 0
        for epoch in range(num_epoch):                   
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction2_model.forward(sub_x)
                
                xent_loss = F.binary_cross_entropy(pred, sub_y,
                    weight=inv_prop.detach())

                loss = xent_loss

                optimizer_prediction2.zero_grad()
                loss.backward()
                optimizer_prediction2.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 10:
                    print("[MF-IPS-Pred2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-Pred2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-Pred2] Reach preset epochs, it seems does not converge.")
        
        early_stop = 0
        x_all = generate_total_sample(self.num_users, self.num_items)
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):                
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                
                pred_u1 = self.prediction1_model.forward(x_sampled)
                pred_u2 = self.prediction2_model.forward(x_sampled)
                x_sampled_common = x_sampled[(pred_u1.detach().cpu().numpy() - pred_u2.detach().cpu().numpy()) < tao]

                pred_u3 = self.prediction_model.forward(x_sampled_common)

                sub_y = self.prediction1_model.forward(x_sampled_common)

                xent_loss = F.binary_cross_entropy(pred_u3, sub_y.detach())

                loss = xent_loss

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-ASIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ASIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ASIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]
        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()
    
class MF_SNIPS(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size, batch_size_prop,
                    embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        # self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        # self.epsilon = torch.nn.Parameter(torch.Tensor([0]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, num_epoch = 1000, stop = 5, lr=0.05, lamb=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 
        
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9
        # generate all counterfactuals and factuals
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()
                
                pred = self.prediction_model.forward(sub_x)
                
                xent_loss = 100*F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") / torch.sum(inv_prop) # o*eui/pui  
                
                optimizer_pred.zero_grad()
                xent_loss.backward()
                optimizer_pred.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]
        
        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]
        
        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x_all)
        total_batch = num_sample // self.batch_size_prop
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(num_sample) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.") 
    

class MF_DR(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([0]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5,
        num_epoch=1000, lr=0.05, lamb1=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 
        
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb1)        

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9

        y_mean = np.mean(y)
        # generate all counterfactuals and factuals
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()
                
                pred = self.prediction_model.forward(sub_x)
                imputation_y = torch.Tensor([y_mean] * self.batch_size).cuda()
                
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)


                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = torch.Tensor([y_mean] * G * self.batch_size).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui  
                # imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")              

                imputation_loss = nn.MSELoss(reduction="sum")(pred, imputation_y)                              
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                # ips_loss = xent_loss/selected_idx.shape[0]
                                
                # direct loss
                # direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = nn.MSELoss(reduction="sum")(pred_u, imputation_y1)
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss
                                
                optimizer_pred.zero_grad()
                loss.backward()
                optimizer_pred.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x_all)
        total_batch = num_sample // self.batch_size_prop
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(num_sample) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.")      
    

class MF_CVIB(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size, batch_size_prop,
                    embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, num_epoch = 1000, stop = 5, lr=0.05, lamb=0, alpha = 0.1, gamma = 0.01,
        tol=1e-4, G=1, verbose=True): 

        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        self.alpha = alpha
        self.gamma = gamma

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals for info reg
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size
        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model.forward(sub_x)
                xent_loss = self.xent_func(pred,sub_y)

                # pair wise loss
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)

                pred_ul = self.prediction_model.forward(x_sampled)

                logp_hat = pred.log()

                pred_avg = pred.mean()
                pred_ul_avg = pred_ul.mean()

                info_loss = self.alpha * (- pred_avg * pred_ul_avg.log() - (1-pred_avg) * (1-pred_ul_avg).log()) + self.gamma* torch.mean(pred * logp_hat)

                loss = xent_loss + info_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-CVIB] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-CVIB] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-CVIB] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()


class MF_DR_JL(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, 
        num_epoch=1000, lr=0.05, lamb1=0, lamb2=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 
        
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb1)
        
        optimizer_impu = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb2)

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9
        # generate all counterfactuals and factuals
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()
                
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).cuda()
                
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)

                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = self.imputation_model.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui  
                # imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")              

                imputation_loss = nn.MSELoss(reduction="sum")(pred, imputation_y)                              
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                # ips_loss = xent_loss/selected_idx.shape[0]
                                
                # direct loss
                # direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = nn.MSELoss(reduction="sum")(pred_u, imputation_y1)
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss
                                
                optimizer_pred.zero_grad()
                loss.backward()
                optimizer_pred.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")

                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).mean()
                

                optimizer_impu.zero_grad()
                
                imp_loss.backward()

                optimizer_impu.step()                

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x_all)
        total_batch = num_sample // self.batch_size_prop
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(num_sample) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.")    
        
    
class MF_TDR_JL(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, 
        num_epoch=1000, lr=0.05, lamb1=0, lamb2=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb1)
        
        optimizer_impu = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb2)

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9
        # generate all counterfactuals and factuals
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()
                
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).cuda()
                
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)

                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = self.imputation_model.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui  
                # imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")              

                imputation_loss = nn.MSELoss(reduction="sum")(pred, imputation_y)                              
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                # ips_loss = xent_loss/selected_idx.shape[0]
                                
                # direct loss
                # direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = nn.MSELoss(reduction="sum")(pred_u, imputation_y1)
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss
                                
                optimizer_pred.zero_grad()
                loss.backward()
                optimizer_pred.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")

                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).mean()
                

                optimizer_impu.zero_grad()
                
                imp_loss.backward()

                optimizer_impu.step()                

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-TDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))                    
                    
                    e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                    e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")   
                    
                    TMLE_beta = inv_prop-1
                    TMLE_alpha = e_loss - e_hat_loss
                    TMLE_epsilon = ((TMLE_alpha * TMLE_beta).sum()/(TMLE_beta * TMLE_beta).sum())
                
                #     sub_x = torch.cat([obs_u, obs_i], axis=1)
                # x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                # x_sampled = x_all[x_all_idx]

                    u_sampled = self.u[x_all[:, 0]]
                    i_sampled = self.i[x_all[:, 1]]
                    sub_x = torch.cat([u_sampled, i_sampled], axis=1)
                
                    one_over_zl = self.propensity_model.forward(sub_x).detach()
                    e_hat_TMLE = (TMLE_epsilon.item() * (one_over_zl.float()- torch.tensor([1.]).cuda()))
                    e_hat_TMLE_obs = e_hat_TMLE[torch.where(torch.Tensor(obs).cuda() == 1)]

                    # np.random.shuffle(x_all)
                    # np.random.shuffle(all_idx)
                    
                    # print(x[:, 0])
                    # print(self.u)
                    # print(len(self.u))
                    obs_u = self.u[x[:, 0]]
                    obs_i = self.i[x[:, 1]]

                    selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                    sub_u = obs_u[selected_idx]
                    sub_i = obs_i[selected_idx]
                    sub_x = torch.cat([sub_u, sub_i], axis=1)

                    # sub_x = x[selected_idx] 
                    sub_y = y[selected_idx]
                    
                    one_over_zl_obs = 1 / torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)
                    # inv_prop = one_over_zl_obs[selected_idx].detach()                
                
                    sub_y = torch.Tensor(sub_y).cuda()
                       
                    pred = self.prediction_model.forward(sub_x)
                    imputation_y = self.imputation_model.predict(sub_x)

                    x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                    x_sampled1 = x_all[x_all_idx]

                    u_sampled = self.u[x_sampled1[:, 0]]
                    i_sampled = self.i[x_sampled1[:, 1]]
                    x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                                       
                    pred_u = self.prediction_model.forward(x_sampled) 
                    imputation_y1 = self.imputation_model.predict(x_sampled)
                                       
                    xent_loss = ((F.binary_cross_entropy(pred, sub_y, reduction ="none") ** 2) * one_over_zl_obs).sum()
                    imputation_loss = ((F.binary_cross_entropy(pred, imputation_y, reduction="none")
                                        + e_hat_TMLE_obs[selected_idx].squeeze().detach()) ** 2).sum()
                        
                    ips_loss = xent_loss - imputation_loss
                    
                    # direct loss
                    sub_x_sampled_number = []
                    for i in x_sampled1:
                        sub_x_sampled_number.append((self.num_items * i[0] + i[1]))
                    sub_x_sampled_number = np.array(sub_x_sampled_number)                 
                
                    direct_loss = ((F.binary_cross_entropy(pred_u, imputation_y1, reduction="none") + e_hat_TMLE[sub_x_sampled_number].squeeze().detach()) ** 2).sum()
                    
                    loss = (ips_loss + direct_loss)/sub_x_sampled_number.shape[0]
                    
                    optimizer_pred.zero_grad()
                    loss.backward()
                    optimizer_pred.step()
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x_all)
        total_batch = num_sample // self.batch_size_prop
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(num_sample) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.")


class MF_MRDR_JL(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, 
        num_epoch=1000, lr=0.05, lamb1=0, lamb2=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 
        
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb1)
        
        optimizer_impu = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb2)

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9
        # generate all counterfactuals and factuals
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()
                
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).cuda()
                
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)

                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = self.imputation_model.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui  
                # imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")              

                imputation_loss = nn.MSELoss(reduction="sum")(pred, imputation_y)                              
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                # ips_loss = xent_loss/selected_idx.shape[0]
                                
                # direct loss
                # direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = nn.MSELoss(reduction="sum")(pred_u, imputation_y1)
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss
                                
                optimizer_pred.zero_grad()
                loss.backward()
                optimizer_pred.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")

                imp_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop ** 2) * (1 - 1/inv_prop)).mean()
                
                optimizer_impu.zero_grad()
                
                imp_loss.backward()

                optimizer_impu.step()                

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x_all)
        total_batch = num_sample // self.batch_size_prop
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(num_sample) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.")


def one_hot(x):
    out = torch.cat([torch.unsqueeze(1-x,1),torch.unsqueeze(x,1)],axis=1)
    return out

def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(1, keepdim=True)

    
class MF_ours_JL_nb(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([0]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, lr2 = 1,
        num_epoch=1000, lr=0.05, lamb1=0, lamb2=0, lamb3=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 
        
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb1)
        
        optimizer_impu = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb2)
        
        optimizer_epo = torch.optim.Adam(
            [self.epsilon], lr=lr2, weight_decay=lamb3)

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9
        # generate all counterfactuals and factuals
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).cuda()
                
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).cuda()
                
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)

                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = self.imputation_model.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui  
                # imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")              

                imputation_loss = nn.MSELoss(reduction="sum")(pred, imputation_y)                              
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                # ips_loss = xent_loss/selected_idx.shape[0]
                                
                # direct loss
                # direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = nn.MSELoss(reduction="sum")(pred_u, imputation_y1)
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss
                                
                optimizer_pred.zero_grad()
                loss.backward()
                optimizer_pred.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")

                imputation_loss_all = e_hat_loss + self.epsilon * (torch.ones(sub_x.shape[0]).cuda() - 1/inv_prop)
                imp_loss = (((e_loss - imputation_loss_all) ** 2) * inv_prop).mean()
                

                optimizer_impu.zero_grad()
                optimizer_epo.zero_grad()
                
                imp_loss.backward()

                optimizer_impu.step()                
                optimizer_epo.step()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x_all)
        total_batch = num_sample // self.batch_size_prop
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(num_sample) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.")        


class MF_DR_BIAS(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([0]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5,
        num_epoch=1000, lr=0.05, lamb1=0, lamb2=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 

        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb1)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb2)

        # last_loss = 1e9

            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model(sub_x).detach(), gamma, 1)             

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).cuda()
                
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)

                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = self.imputation_model.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = nn.MSELoss(reduction="sum")(pred, imputation_y)                 
                
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                
                
                # direct loss
                direct_loss = nn.MSELoss(reduction="sum")(pred_u, imputation_y1)
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss              
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 3 ) * ((1 - 1 / inv_prop.detach()) ** 2)).mean()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x_all)
        total_batch = num_sample // self.batch_size_prop
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(num_sample) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.")
    
    
class MF_DR_MSE(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([0]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5,
        num_epoch=1000, lr=0.05, lamb1=0, lamb2=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 

        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb1)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb2)

        # last_loss = 1e9

            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model(sub_x).detach(), gamma, 1)             

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).cuda()
                
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)

                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = self.imputation_model.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = nn.MSELoss(reduction="sum")(pred, imputation_y)                 
                
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                
                
                # direct loss
                direct_loss = nn.MSELoss(reduction="sum")(pred_u, imputation_y1)
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss              
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")

                imp_bias_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 3 ) * ((1 - 1 / inv_prop.detach()) ** 2)).mean()
                imp_mrdr_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 2 ) * (1 - 1 / inv_prop.detach())).mean()
                imp_loss = gamma * imp_bias_loss + (1-gamma) * imp_mrdr_loss
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()
    
    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x_all)
        total_batch = num_sample // self.batch_size_prop
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(num_sample) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.")
           

class MF_IPS_V2(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size, batch_size_prop,
                    embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        # self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        # self.epsilon = torch.nn.Parameter(torch.Tensor([0]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, alpha = 1, beta = 1, theta = 1, eta = 1,
            num_epoch = 1000, stop = 5, lr=0.05, lamb=0, gamma = 0.05,
        tol=1e-4, G=1, verbose=True): 
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_pred = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        # obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y_entire = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        #得到观测6960个prop当作inv_prop
        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        for epoch in range(num_epoch):
            # sampling counterfactuals
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model(sub_x).detach(), gamma, 1)                
                sub_y = torch.Tensor(sub_y).cuda()
                
                pred = self.prediction_model.forward(sub_x)        
                
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                                       

                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)            
                ips_loss = xent_loss 
                             
                # ctr loss
                
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
                # print(sub_obs)
                sub_entire_y = torch.Tensor(y_entire[x_all_idx]).cuda()

                inv_prop_all = 1/torch.clip(self.propensity_model.forward(x_sampled), gamma, 1)
                # print(inv_prop_all.shape)
                prop_loss = F.binary_cross_entropy(1/inv_prop_all, sub_obs)                                    
                # print(prop_loss.shape)
                pred = self.prediction_model.forward(x_sampled)
                
                pred_loss = F.binary_cross_entropy(1/inv_prop_all * pred, sub_entire_y)

                ones_all = torch.ones(len(inv_prop_all)).cuda()
                w_all = torch.divide(sub_obs,1/inv_prop_all)-torch.divide((ones_all-sub_obs),(ones_all-(1/inv_prop_all)))
                bmse_loss = (torch.mean(w_all * pred))**2
                
                loss = alpha * prop_loss + beta * pred_loss + ips_loss +  eta * bmse_loss

                optimizer_pred.zero_grad()
                loss.backward()
                optimizer_pred.step()
                                     
                epoch_loss += xent_loss.detach().cpu().numpy()                         
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ESCM2] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()


class MF_DR_V2(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([0]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, alpha = 1, beta = 1, theta = 1, eta = 1,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=False): 
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]
        
        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        # obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y_entire = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        #得到观测6960个prop当作inv_prop
        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        for epoch in range(num_epoch):
            # sampling counterfactuals
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x), gamma, 1)
                
                sub_y = torch.Tensor(sub_y).cuda()
                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).cuda()               
                
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                                       
                pred_u = torch.clip(self.prediction_model.forward(x_sampled), 0, 1) 
                imputation_y1 = torch.clip(self.imputation_model.forward(x_sampled), 0, 1)             
                
                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
                imputation_loss = -torch.sum(imputation_y * torch.log(pred + 1e-6) + (1-imputation_y) * torch.log(1 - pred + 1e-6))
                        
                ips_loss = (xent_loss - imputation_loss) # batch size
                
                # direct loss
                                
                direct_loss = -torch.sum(imputation_y1 * torch.log(pred_u + 1e-6) + (1-imputation_y1) * torch.log(1 - pred_u + 1e-6))
                
                dr_loss = (ips_loss + direct_loss)/x_sampled.shape[0]
                                                  
                # pred = torch.clip(self.prediction_model.forward(sub_x).detach(), 0, 1)
                # imputation_y = torch.clip(self.imputation_model.forward(sub_x), 0, 1)
                
                e_loss = -sub_y * torch.log(pred + 1e-6) - (1-sub_y) * torch.log(1 - pred + 1e-6)
                e_hat_loss = -imputation_y * torch.log(pred + 1e-6) - (1-imputation_y) * torch.log(1 - pred + 1e-6)
                
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                # ctr loss
                
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
                # print(sub_obs)
                sub_entire_y = torch.Tensor(y_entire[x_all_idx]).cuda()

                inv_prop_all = 1/torch.clip(self.propensity_model.forward(x_sampled), gamma, 1)
                # print(inv_prop_all.shape)
                prop_loss = F.binary_cross_entropy(1/inv_prop_all, sub_obs)                                    
                # print(prop_loss.shape)
                pred = self.prediction_model.forward(x_sampled)
                
                pred_loss = F.binary_cross_entropy(1/inv_prop_all * pred, sub_entire_y)

                ones_all = torch.ones(len(inv_prop_all)).cuda()
                w_all = torch.divide(sub_obs,1/inv_prop_all)-torch.divide((ones_all-sub_obs),(ones_all-(1/inv_prop_all)))
                bmse_loss = (torch.mean(w_all * pred))**2
                
                loss = alpha * prop_loss + beta * pred_loss + theta * imp_loss + dr_loss + eta * bmse_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                                     
                epoch_loss += xent_loss.detach().cpu().numpy()                         
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-ESCM2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-ESCM2] Reach preset epochs, it seems does not converge.")
        
        torch.save(self.propensity_model.state_dict(), 'weight_model0.pth')
    
    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()



class MF_SDR(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.imputation_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([0]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()


    def fit(self, x, y, eta = 1, gamma = 0.05,
        num_epoch=1000, lr=0.05, lamb1 = 0, lamb2 = 0, lamb3 = 0,
        tol=1e-4, G=1, verbose = False): 
        
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb1)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb2)
        optimizer_propensity = torch.optim.Adam(
            self.propensity_model.parameters(), lr=lr, weight_decay = lamb3)
        
        last_loss = 1e9

        observation = sps.csr_matrix((np.ones(len(x)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # print(observation.shape)
               
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)
                
        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        # one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)].detach().cuda()

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]
            # propensity score
                inv_prop = 1 / torch.clip(self.propensity_model.forward(sub_x), gamma, 1) 

                sub_y = torch.Tensor(sub_y).cuda()
                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction="none")
                e_hat_loss = nn.MSELoss(reduction="none")(imputation_y, pred.detach())
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop.detach()).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()  
                
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)           

                imputation_y1 = self.imputation_model.predict(x_sampled).cuda()
                
                prop_loss = F.binary_cross_entropy(self.propensity_model.forward(x_sampled), torch.Tensor(observation[x_all_idx]).cuda(), reduction="sum")
                pred_y1 = self.prediction_model.predict(x_sampled).cuda()

                imputation_loss = F.binary_cross_entropy(imputation_y1, pred_y1, reduction = "none")

                loss = prop_loss + eta * ((1 - torch.Tensor(observation[x_all_idx]).cuda() * 1 / torch.clip(self.propensity_model.forward(x_sampled), gamma, 1)) * (imputation_loss - imputation_loss.mean())).sum() ** 2
                
                # print(((1 - torch.Tensor(observation[x_all_idx]).cuda() * 1 / torch.clip(self.propensity_model.forward(x_sampled), 0.02, 1)) * (imputation_loss - imputation_loss.mean())).sum() ** 2) #都是0.1 - 10
                # print('p', prop_loss) #0.1 - 10
                
                optimizer_propensity.zero_grad()
                loss.backward()
                optimizer_propensity.step()

                pred = self.prediction_model.forward(sub_x)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight = inv_prop.detach(), reduction="sum")
                xent_loss = (xent_loss)/(inv_prop.detach().sum())
                # print(inv_prop.detach())
                optimizer_prediction.zero_grad()
                xent_loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()      
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-Stable-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-Stable-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-Stable-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu() 



class DA_MF(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model_mcar = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.prediction_model_mnar = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        # self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([0]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, alpha = 1,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, G=1, verbose=True): 
        
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_prediction_model_mnar = torch.optim.Adam(
            self.prediction_model_mnar.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prediction_model_mcar = torch.optim.Adam(
            self.prediction_model_mcar.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)

                # propensity score            

                sub_y = torch.Tensor(sub_y).cuda()

                pred_mnar_naive = self.prediction_model_mnar.forward(sub_x)
                pred_mcar_naive = self.prediction_model_mcar.forward(sub_x)
                
                pred_mnar_all = self.prediction_model_mnar.forward(x_sampled)
                pred_mcar_all = self.prediction_model_mcar.forward(x_sampled)
                
                Loss_naive = F.binary_cross_entropy(pred_mnar_naive, sub_y, reduction = 'mean')
                Loss_dif_all = nn.MSELoss()(pred_mnar_all, pred_mcar_all.detach())
                Loss_dif_mnar = nn.MSELoss()(pred_mnar_naive, pred_mcar_naive.detach())
                                       
                Loss = Loss_naive + alpha * (Loss_dif_all - Loss_dif_mnar)
                
                epoch_loss += Loss_naive.detach().cpu().numpy()
                
                optimizer_prediction_model_mnar.zero_grad()
                Loss.backward()
                optimizer_prediction_model_mnar.step()                
                    
                pred_mnar_naive = self.prediction_model_mnar.forward(sub_x)
                pred_mcar_naive = self.prediction_model_mcar.forward(sub_x)
                
                pred_mnar_all = self.prediction_model_mnar.forward(x_sampled)
                pred_mcar_all = self.prediction_model_mcar.forward(x_sampled)

                Loss_dif_all = nn.MSELoss()(pred_mcar_all, pred_mnar_all.detach())
                Loss_dif_mnar = nn.MSELoss()(pred_mcar_naive, pred_mnar_naive.detach())
                                       
                Loss = Loss_dif_all - Loss_dif_mnar                
                
                optimizer_prediction_model_mcar.zero_grad()
                Loss.backward()
                optimizer_prediction_model_mcar.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL] Reach preset epochs, it seems does not converge.")


    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model_mnar.predict(obs_x)
        return pred.cpu()
    
class MF_MR(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.imputation_model1 = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.imputation_model2 = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.propensity_model = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([0]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()


    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x_all)
        total_batch = num_sample // self.batch_size_prop
        early_stop = 0

        for epoch in range(num_epoch):

            ul_idxs = np.arange(num_sample) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).cuda()
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-PS] Reach preset epochs, it seems does not converge.")

        
    def fit(self, x, y, unlabel_x = None, stab = 0, gamma = 0.05,
        num_epoch=1000, lr=0.05, lamb1=0, lamb2 = 0,
        tol=1e-4, G=2, verbose=True): 

        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u[obs == 1]
        # obs_i = self.i[obs == 1]

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb1)
        optimizer_imputation1 = torch.optim.Adam(
            self.imputation_model1.parameters(), lr=lr, weight_decay=lamb2)
        optimizer_imputation2 = torch.optim.Adam(
            self.imputation_model2.parameters(), lr=lr, weight_decay=lamb2)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)

                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1 / torch.clip(self.propensity_model.forward(sub_x), gamma, 1).detach()               

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model1.forward(sub_x)
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = nn.MSELoss(reduction="none")(imputation_y, pred)
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                optimizer_imputation1.zero_grad()
                imp_loss.backward()
                optimizer_imputation1.step()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model2.forward(sub_x)
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = nn.MSELoss(reduction="none")(imputation_y, pred)
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                optimizer_imputation2.zero_grad()
                imp_loss.backward()
                optimizer_imputation2.step()                  
                
                pred = self.prediction_model.forward(sub_x)
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                u = torch.Tensor(np.c_[inv_prop.cpu().numpy(), self.imputation_model1.predict(sub_x).cpu().numpy(), self.imputation_model2.predict(sub_x).cpu().numpy()])
                
                # print(u.shape)
                # print(u)
                # print(u.T.matmul(u))
                
                # print((u.T.matmul(u) + stab * torch.eye(3)).shape)
                # print((u.T.matmul(e_loss)).shape)
                # print(u.T.matmul(e_loss))
                # print((torch.sum(u * e_loss, axis = 1)))
                
                matrix_inv = torch.Tensor(np.linalg.inv(u.T.matmul(u) + stab * torch.eye(3))).cuda()
    
                eta = torch.squeeze(matrix_inv.matmul(u.T.cuda().matmul(e_loss)))
                
                # print(eta.shape)
                # print(eta)
                
                # x_sampled = unlabel_x[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)

                inv_prop_all = 1 / torch.clip(self.propensity_model.forward(x_sampled), gamma, 1).detach().cpu().numpy() 
            
                u = torch.Tensor(np.c_[inv_prop_all, self.imputation_model1.predict(x_sampled).cpu().numpy(), self.imputation_model2.predict(x_sampled).cpu().numpy()]).cuda()
                
                loss = torch.mean(u * eta)
                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()                   
                
                epoch_loss += loss.detach().cpu().numpy()
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MR] Reach preset epochs, it seems does not converge.")
        
    def predict(self, x):
        # obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
        #                      shape=(self.num_users_test, self.num_items_test), dtype=np.float32).toarray().reshape(-1)
        # obs_u = self.u_test[obs == 1]
        # obs_i = self.i_test[obs == 1]

        obs_u = self.u_test[x[:, 0]]
        obs_i = self.i_test[x[:, 1]]

        obs_x = torch.cat([obs_u, obs_i], axis=1)
        # print(obs_x.shape)
        pred = self.prediction_model.predict(obs_x)
        return pred.cpu()
    

class mf(nn.Module):    
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.user_emb_table = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.item_emb_table = torch.nn.Embedding(self.num_items, self.embedding_k)
        
        print('mf initialized')

    def forward(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])


        out = torch.sigmoid((user_emb * item_emb).sum(dim=1))

        return out  


    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            
            return pred 



class mf_add_bias(nn.Module):    
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.user_emb_table = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.item_emb_table = torch.nn.Embedding(self.num_items, self.embedding_k)
        
        self.user_bias = nn.Parameter(torch.zeros(self.num_users))
        self.item_bias = nn.Parameter(torch.zeros(self.num_items))
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        print('mf_add_bias initialized')

    def forward(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])

        rating = (user_emb * item_emb).sum(dim=1) + self.user_bias[x[:, 0]] + self.item_bias[x[:, 1]] + self.global_bias
        
        out = torch.sigmoid(rating)
        
        return out  
    
         

    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            
            return pred 
    


class logistic_regression_all(nn.Module):    
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.user_emb_table = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.item_emb_table = torch.nn.Embedding(self.num_items, self.embedding_k)
        
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias = True)

        print('logistic_regression_all initialized')

    def forward(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])

        # concat
        z_emb = torch.cat([user_emb, item_emb], axis=1)

        out = torch.sigmoid(self.linear_1(z_emb))

        return torch.squeeze(out)        
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            
            return pred


    def get_all(self):
        user_logit = F.linear(self.user_emb_table.weight, self.linear_1.weight[:, :self.embedding_k])
        item_logit = F.linear(self.item_emb_table.weight, self.linear_1.weight[:, self.embedding_k:])
        
        out = torch.sigmoid(user_logit + item_logit.reshape(1, -1) + self.linear_1.bias)

        return out.reshape(-1)      
        
    def get_all_predict(self):
        with torch.no_grad():
            
            return self.get_all()



class logistic_regression(nn.Module):    
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.user_emb_table = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.item_emb_table = torch.nn.Embedding(self.num_items, self.embedding_k)
        
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias=True)

        print('logistic_regression initialized')

    def forward(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])

        # concat
        z_emb = torch.cat([user_emb, item_emb], axis=1)

        out = torch.sigmoid(self.linear_1(z_emb))

        return torch.squeeze(out)        
        

        
    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            
            return pred 
    


class ncf(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.user_emb_table = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.item_emb_table = torch.nn.Embedding(self.num_items, self.embedding_k)
        
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k, bias=True)
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=True)    

        self.relu = torch.nn.ReLU()
        
        print('ncf initialized')

    def forward(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])
        
        z_emb = torch.cat([user_emb, item_emb], axis=1)
 
        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)            
        
        out = torch.sigmoid(self.linear_2(h1))  
        
        return torch.squeeze(out)      
        
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            
            return pred 




class ncf_2_layer(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.user_emb_table = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.item_emb_table = torch.nn.Embedding(self.num_items, self.embedding_k)
        
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k, bias=True)
        self.linear_2 = torch.nn.Linear(self.embedding_k, self.embedding_k, bias=True)    
        self.linear_3 = torch.nn.Linear(self.embedding_k, 1, bias=True)

        self.relu = torch.nn.ReLU()
        
        print('ncf_2_layer initialized')

    def forward(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])
        
        z_emb = torch.cat([user_emb, item_emb], axis=1)
 
        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)            
        h2 = self.linear_2(h1)
        h2 = self.relu(h2)
        
        out = torch.sigmoid(self.linear_3(h2))  
        
        return torch.squeeze(out)      
        
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            
            return pred 

class dr_mse_jl_gpl(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.model_pred = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.model_impu = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.model_prop = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()


    def fit(self, x_all, obs, x, y, num_epochs=100, alpha=1.0, beta=0.5, theta=0.5, gamma=0.05, G=4, pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, stop=5, tol=1e-4): 
        # print('fit', grad_type, num_epochs, alpha, beta, theta, gamma, G, pred_lr, impu_lr, prop_lr, pred_lamb, impu_lamb, prop_lamb, stop)    
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0

        y = torch.Tensor(y).cuda()
        obs = torch.Tensor(obs).cuda()
        # knn_matrix = torch.Tensor(knn_matrix).cuda()        
        for epoch in range(num_epochs):
            all_idx = torch.randperm(num_samples)
            ul_idxs = torch.randperm(x_all.shape[0])

            epoch_loss = 0
            for idx in range(total_batch):   
                # propensity
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                
                sub_obs = obs[x_all_idx]
                
                prop_all = torch.clip(self.model_prop(x_sampled), gamma, 1.0)

                prop_loss = F.binary_cross_entropy(prop_all, sub_obs, reduction='sum')   
                
                       
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                # sub_x = x[selected_idx] 
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)
                sub_y = y[selected_idx]
                
                pred = self.model_pred(sub_x)  
                imputation_y = self.model_impu(sub_x)              

                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y.detach(), pred.detach(), reduction='none')
                
                prop = torch.clip(self.model_prop(sub_x) , gamma, 1.0)
                bias_loss = torch.sum(((1.0 - 2 * prop + prop ** 2) / (prop ** 2)) * ((e_loss - e_hat_loss) ** 2))
                var_loss = torch.sum(((e_loss - e_hat_loss) ** 2) / (prop ** 2))
                
                gpl_loss = (prop_loss + alpha * (beta * bias_loss + (1.0 - beta) * var_loss)) / float(x_sampled.shape[0])

                
                optimizer_propensity.zero_grad()
                gpl_loss.backward()
                optimizer_propensity.step()
                
                
                # impu
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x), gamma, 1.0)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')


                imp_bias_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 3) * ((1.0 - 1.0 / inv_prop.detach()) ** 2)).sum() / float(x_sampled.shape[0])
                imp_mrdr_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 2) * (1.0 - 1.0 / inv_prop.detach())).sum() / float(x_sampled.shape[0])

                imp_loss = (1.0 - theta) * imp_bias_loss + theta * imp_mrdr_loss
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
                # pred
                imputation_y = self.model_impu.predict(sub_x)              

                pred_u = self.model_pred(x_sampled) 
                imputation_y1 = self.model_impu.predict(x_sampled)             

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                            
                epoch_loss += xent_loss.detach()
            
            
            # if self.is_tensorboard:
            #     tb_log.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                                        
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
                

        return epoch
        
    def predict(self, x):
        with torch.no_grad():
            obs_u = self.u_test[x[:, 0]]
            obs_i = self.i_test[x[:, 1]]
            
            obs_x = torch.cat([obs_u, obs_i], axis=1)
            # print(obs_x.shape)
            pred = self.model_pred.predict(obs_x)
            return pred.cpu()

class ours_icdmw_DR(nn.Module):
    def __init__(self, num_users, num_items, u_emb, i_emb, 
                 num_users_test, num_items_test, u_emb_test, i_emb_test, batch_size,
                 batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users_test = num_users_test
        self.num_items_test = num_items_test
        self.num_users = num_users
        self.num_items = num_items
        self.u = torch.Tensor(u_emb).cuda()
        self.i = torch.Tensor(i_emb).cuda()
        self.u_test = torch.Tensor(u_emb_test).cuda()
        self.i_test = torch.Tensor(i_emb_test).cuda()
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.model_pred = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))
        self.model_impu = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1])) 
        self.model_prop = MLP(input_size=int(u_emb.shape[1]+i_emb.shape[1]))

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()


    def fit(self, x_all, obs, x, y, knn_matrix, k, cont = 1, num_epochs=100, alpha=1.0, beta=0.5, theta=0.5, gamma=0.05,
             G=4, pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, stop=5, tol=1e-4): 
        # print('fit', grad_type, num_epochs, alpha, beta, theta, gamma, G, pred_lr, impu_lr, prop_lr, pred_lamb, impu_lamb, prop_lamb, stop)    
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        obs_u = self.u[x[:, 0]]
        obs_i = self.i[x[:, 1]]

        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        y = torch.Tensor(y).cuda()
        obs = torch.Tensor(obs).cuda()
        knn_matrix = torch.Tensor(knn_matrix).cuda()
        for epoch in range(num_epochs):
            all_idx = torch.randperm(num_samples)
            ul_idxs = torch.randperm(x_all.shape[0])

            epoch_loss = 0
            for idx in range(total_batch):   
                # propensity
                x_all_idx = ul_idxs[self.batch_size*G*idx:(idx+1)*G*self.batch_size]
                x_sampled = x_all[x_all_idx]

                u_sampled = self.u[x_sampled[:, 0]]
                i_sampled = self.i[x_sampled[:, 1]]
                x_sampled = torch.cat([u_sampled, i_sampled], axis=1)
                
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                # sub_x = x[selected_idx] 
                sub_u = obs_u[selected_idx]
                sub_i = obs_i[selected_idx]
                sub_x = torch.cat([sub_u, sub_i], axis=1)
                sub_y = y[selected_idx]

                sub_obs = obs[x_all_idx]
                sub_knn = knn_matrix[selected_idx]

                prop_all = torch.clip(self.model_prop(x_sampled), gamma, 1.0)

                prop_loss = F.binary_cross_entropy(prop_all, sub_obs, reduction='sum')   


                pred = self.model_pred(sub_x)  
                imputation_y = self.model_impu(sub_x)              

                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y.detach(), pred.detach(), reduction='none')

                prop = torch.clip(self.model_prop(sub_x), gamma, 1.0)
                if cont:
                    bias_loss = torch.sum(((k / (k-1) * sub_knn**2 - sub_knn/(k-1) - 2 * sub_knn * prop + prop ** 2) / (prop ** 2)) * ((e_loss - e_hat_loss) ** 2))
                    var_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * (sub_knn - k / (k-1) * sub_knn**2 + sub_knn/(k-1))/ (prop ** 2))
                else:
                    # bias_loss = torch.sum(((k / (k-1) * sub_knn**2 - sub_knn/(k-1) - 2 * prop + prop ** 2) / (prop ** 2)) * ((e_loss - e_hat_loss) ** 2))
                    # var_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * (1 - k / (k-1) * sub_knn**2 + sub_knn/(k-1))/ (prop ** 2))
                    bias_loss = torch.sum(((k / (k-1) * sub_knn**2 - sub_knn/(k-1) - 2 * sub_knn * prop + prop ** 2) / (prop ** 2)) * ((e_loss - e_hat_loss) ** 2))
                    var_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * (sub_knn - k / (k-1) * sub_knn**2 + sub_knn/(k-1))/ (prop ** 2))
                gpl_loss = (prop_loss + alpha * (beta * bias_loss + (1.0 - beta) * var_loss)) / float(x_sampled.shape[0])

                
                optimizer_propensity.zero_grad()
                gpl_loss.backward()
                optimizer_propensity.step()

                # impu
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x), gamma, 1.0)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')


                # imp_bias_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 3) * ((1.0 - 1.0 / inv_prop.detach()) ** 2)).sum() / float(x_sampled.shape[0])
                # imp_mrdr_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 2) * (1.0 - 1.0 / inv_prop.detach())).sum() / float(x_sampled.shape[0])

                # imp_loss = (1.0 - theta) * imp_bias_loss + theta * imp_mrdr_loss
                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * (inv_prop.detach()))  / float(x_sampled.shape[0])
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()

                # pred
                imputation_y = self.model_impu.predict(sub_x)              

                pred_u = self.model_pred(x_sampled) 
                imputation_y1 = self.model_impu.predict(x_sampled)             

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')

                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])

                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach()
            
            
            # if self.is_tensorboard:
            #     tb_log.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                                        
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
        return epoch
        
    def predict(self, x):
        with torch.no_grad():
            obs_u = self.u_test[x[:, 0]]
            obs_i = self.i_test[x[:, 1]]
            
            obs_x = torch.cat([obs_u, obs_i], axis=1)
            # print(obs_x.shape)
            pred = self.model_pred.predict(obs_x)
            return pred.cpu().numpy()

class UMVUE_DR(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, knn_matrix, k, cont=1, num_epochs=100, 
            alpha=1.0, beta=0.5, gamma=0.05, G=4, 
            pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, 
            pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, 
            stop=5, tol=1e-4, verbose=True): 
        """
        训练 UMVUE_DR 模型
        
        Args:
            x: 观测到的 user-item 对 (num_samples, 2)
            y: 对应的标签 (num_samples,)
            knn_matrix: KNN 矩阵，用于计算倾向性偏差 (num_samples,)
            k: KNN 中的邻居数量
            cont: 是否使用连续版本的 GPL loss (0 或 1)
            num_epochs: 训练轮数
            alpha: GPL loss 的权重
            beta: 偏差-方差权衡参数
            theta: 未使用（保留参数）
            gamma: 倾向性分数的下界
            G: 采样倍数（每个 batch 采样 G * batch_size 个样本）
            pred_lr: 预测模型学习率
            impu_lr: 插补模型学习率
            prop_lr: 倾向性模型学习率
            pred_lamb: 预测模型正则化系数
            impu_lamb: 插补模型正则化系数
            prop_lamb: 倾向性模型正则化系数
            stop: 早停的容忍轮数
            tol: 早停的损失变化阈值
            verbose: 是否打印训练信息
        """
        
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(
            self.propensity_model.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        # 生成所有可能的 user-item 对
        x_all = generate_total_sample(self.num_users, self.num_items)
        
        # 构建观测矩阵（用于倾向性模型训练）
        obs = sps.csr_matrix(
            (np.ones(len(y)), (x[:, 0], x[:, 1])), 
            shape=(self.num_users, self.num_items), 
            dtype=np.float32
        ).toarray().reshape(-1)
        
        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        
        # 转换为 Tensor
        y = torch.Tensor(y).cuda()
        obs = torch.Tensor(obs).cuda()
        knn_matrix = torch.Tensor(knn_matrix).cuda()
        
        for epoch in range(num_epochs):
            # 随机打乱索引
            all_idx = np.arange(num_samples)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in tqdm(range(total_batch), desc=f'Epoch {epoch+1}/{num_epochs}'):
            # for idx in range(total_batch):   
                # ========== 采样数据 ==========
                # 从观测数据中采样
                selected_idx = all_idx[idx * self.batch_size:(idx + 1) * self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_knn = knn_matrix[selected_idx]
                
                # 从所有 user-item 对中采样（用于倾向性模型和直接估计）
                x_all_idx = ul_idxs[self.batch_size * G * idx:(idx + 1) * G * self.batch_size]
                x_sampled = x_all[x_all_idx]
                sub_obs = obs[x_all_idx]

                # ========== Propensity Model Training ==========
                # 倾向性预测（在采样的所有 user-item 对上）
                prop_all = torch.clip(self.propensity_model.forward(x_sampled), gamma, 1.0)
                prop_loss = F.binary_cross_entropy(prop_all, sub_obs, reduction='sum')

                # 预测和插补（在观测数据上）
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x)

                # 计算误差项
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(
                    imputation_y.detach(), pred.detach(), reduction='none')

                # 观测样本的倾向性分数
                prop = torch.clip(self.propensity_model.forward(sub_x), gamma, 1.0)
                
                # GPL loss (偏差和方差)
                if cont:
                    bias_loss = torch.sum(
                        ((k / (k - 1) * sub_knn**2 - sub_knn / (k - 1) - 2 * sub_knn * prop + prop**2) / (prop**2)) * 
                        ((e_loss - e_hat_loss)**2))
                    var_loss = torch.sum(
                        ((e_loss - e_hat_loss)**2) * 
                        (sub_knn - k / (k - 1) * sub_knn**2 + sub_knn / (k - 1)) / (prop**2))
                else:
                    bias_loss = torch.sum(
                        ((k / (k - 1) * sub_knn**2 - sub_knn / (k - 1) - 2 * sub_knn * prop + prop**2) / (prop**2)) * 
                        ((e_loss - e_hat_loss)**2))
                    var_loss = torch.sum(
                        ((e_loss - e_hat_loss)**2) * 
                        (sub_knn - k / (k - 1) * sub_knn**2 + sub_knn / (k - 1)) / (prop**2))
                
                gpl_loss = (prop_loss + alpha * (beta * bias_loss + (1.0 - beta) * var_loss)) / float(x_sampled.shape[0])

                optimizer_propensity.zero_grad()
                gpl_loss.backward()
                optimizer_propensity.step()

                # ========== Imputation Model Training ==========
                inv_prop = 1.0 / torch.clip(self.propensity_model.forward(sub_x), gamma, 1.0)
                
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                imp_loss = torch.sum(((e_loss - e_hat_loss)**2) * inv_prop.detach()) / float(x_sampled.shape[0])
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()

                # ========== Prediction Model Training ==========
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x)
                
                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = self.imputation_model.forward(x_sampled)

                # IPS loss
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')
                ips_loss = xent_loss - imputation_loss

                # Direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')

                # DR loss
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])

                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()
            
            # 早停检查
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > stop:
                    if verbose:
                        print(f"[UMVUE-DR] epoch:{epoch}, loss:{epoch_loss}")
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
            
            if epoch % 10 == 0 and verbose:
                print(f"[UMVUE-DR] epoch:{epoch}, loss:{epoch_loss}")
        
        if verbose:
            print("[UMVUE-DR] Reach preset epochs, it seems does not converge.")
        
        return epoch
        
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred
    


