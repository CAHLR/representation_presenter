__author__ = 'jwj'
import torch
import numpy as np
import pandas as pd
import pickle
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR


sampled_data = 'sampled_data.pkl'


class Net(torch.nn.Module):
    def __init__(self, idim, vector_dim, odim):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(idim, vector_dim, bias=False).cuda()
        self.l2 = torch.nn.Linear(vector_dim, odim, bias=False).cuda()

    def forward(self, x):
        h = self.l1(x)
        y_pred = self.l2(h)
        return y_pred


def generate_batch_data_random(x, y, batch_size, indim):
    ylen = len(y)
    index = np.random.randint(ylen, size=batch_size)
    target_batch = x[index]
    context_batch = y[index]
    target = np.zeros((batch_size, indim))
    for i in range(batch_size):
        target[i, target_batch[i]] = 1
    context_course = zip(*context_batch)
    context = list(context_course)[0]
    return target, context

loss_ls = []
def train_mini_batch(word_target, word_context, batch_s, optimizer, model, loss_fn, iter, indim, epoch):
    for t in range(epoch):  # epoch
        for i in range(iter):
            target, context = generate_batch_data_random(word_target, word_context, batch_s, indim)
            target = torch.FloatTensor(target)
            context = torch.LongTensor(context)
            target = Variable(target, requires_grad=False)
            target = target.cuda()
            context = Variable(context, requires_grad=False)
            context = context.cuda()
            optimizer.zero_grad()
            y_pred = model(target).cuda()
            loss = loss_fn(y_pred, context)
            loss_fn.cuda()
            loss_ls.append('epoch' + str(t+1)+':' + 'The'+str(i+1)+'-th interation: training loss'+str(loss.item()))
            print('epoch' + str(t+1)+':' + 'The'+str(i+1)+'-th interation: training loss'+str(loss.data.item())+'\n')
            loss.backward()
            optimizer.step()
    with open('./log_windowsize_10_mincount_5_embs_300.pkl', 'wb') as f:
        pickle.dump(loss_ls, f)


def train(batch_s, token, factors, vec_dim, epoch):

    token_file = open(token+'_id.pkl', 'rb')
    token_dic = pickle.load(token_file)
    token_id = token_dic[token+'_id']
    id_token = token_dic['id_'+token]
    token_file.close()
    vocab_size = len(token_id)

    factor_dic = {}
    factor_dic_reverse = {}
    for factor in factors:
        file = open(factor+'_id.pkl', 'rb')
        factor_file = pickle.load(file)
        factor_dic[factor] = factor_file[factor+'_id']
        factor_dic_reverse[factor] = factor_file['id_'+factor]

    f = open(sampled_data, 'rb')
    data = pickle.load(f)
    f.close()
    word_target = list(data['token_target'])
    word_context = list(data['token_context'])
    word_target = np.array(word_target)
    word_context = np.array(word_context)
    print("construct model")
    indim = len(token_id)
    outdim = len(token_id)
    for i in factors:
        indim += len(factor_dic[i])

    iter = len(word_target)//batch_s

    loss_fn = torch.nn.CrossEntropyLoss()
    model = Net(indim, vec_dim, outdim)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('train on mini_batch')
    train_mini_batch(word_target, word_context, batch_s, optimizer, model, loss_fn, iter, indim, epoch)
    torch.save(model, 'torch_model.pkl')

    a = torch.load('torch_model.pkl')

    param = a.state_dict()

    token2vec = param['l1.weight'].cpu().numpy()[:, :vocab_size].T
    token_weight_df = pd.core.frame.DataFrame({'weight': pd.Series(list(token2vec))}).reset_index()
    token_id_df = pd.DataFrame(list(id_token.items()), columns=['id', 'name'])
    token_weight_df.columns = ['id', 'weight']
    token_tsv = pd.merge(token_id_df, token_weight_df, on='id')
    token_tsv.to_csv(token+'_embeddings.tsv', sep='\t')

    k = 0
    for i in factors:
        factor2vec = param['l1.weight'].cpu().numpy()[:, vocab_size+k:vocab_size+k+len(factor_dic[i])].T
        k += len(factor_dic[i])
        factor_weight_df = pd.core.frame.DataFrame({'weight': pd.Series(list(factor2vec))}).reset_index()
        factor_id_df = pd.DataFrame(list(factor_dic_reverse[i].items()), columns=['id', 'name'])
        factor_weight_df.columns = ['id', 'weight']
        factor_tsv = pd.merge(factor_id_df, factor_weight_df, on='id')
        factor_tsv.to_csv(i + '_embeddings.tsv', sep='\t')
