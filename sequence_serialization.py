__author__ = 'jwj'
import pandas as pd
import pickle
import numpy as np


def set_token_id(data, token):
    count = data.groupby(token).size()
    count = pd.core.frame.DataFrame({'count': count}).reset_index()
    count.sort_values(by=['count'], ascending=False, inplace=True)
    count.reset_index(inplace=True)
    count = count.iloc[:, [1, 2]]
    # get token_id to dictionary
    dic = count.to_dict('dict')
    dic1 = dic[token]
    reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
    alldata = {token+'_id': reversed_dic1, 'id_'+token: dic1}
    f = open(token+'_id.pkl', 'wb')
    pickle.dump(alldata, f)
    f.close()


def set_sort_key_id(data, sort_key):
    sort = data[sort_key].drop_duplicates().sort_values(ascending=True)
    dic1 = dict()
    for i in sort:
        dic1[i] = len(dic1)
    reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
    alldata = {sort_key+'_id': dic1, 'id_'+sort_key: reversed_dic1}
    f = open(sort_key+'_id.pkl', 'wb')
    pickle.dump(alldata, f)
    f.close()


# get group_key id, filter out those with enrollment records less than 5
def set_group_id(data, group_key):
    count = data.groupby(group_key).size()
    count = pd.core.frame.DataFrame({'count': count}).reset_index()
    count.sort_values(by=['count'], ascending=False, inplace=True)
    count.reset_index(inplace=True)
    count = count.loc[:, [group_key, 'count']]
    count = count.loc[count['count'] >= 5]
    dic1 = dict()
    li = list(count[group_key])
    for i in li:
        dic1[i] = len(dic1)
    reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
    alldata = {group_key+'_id': dic1, 'id_'+group_key: reversed_dic1}
    f = open(group_key+'_id.pkl', 'wb')
    pickle.dump(alldata, f)
    f.close()
    return count


# get the whole data
# goes after get_token()
def get_seq(data, count, group_key, token, sort_key, sort_key_length):
    data = data.loc[data[group_key].isin(count[group_key]), [token, sort_key, group_key]]
    # transfer to np matrix
    mat_data = [[None for i in range(sort_key_length)] for j in range(len(count[group_key]))]
    f = open(group_key+'_id.pkl', 'rb')
    group_key_id = pickle.load(f)[group_key+'_id']
    f = open(sort_key+'_id.pkl', 'rb')
    sort_key_id = pickle.load(f)[sort_key+'_id']
    f = open(token+'_id.pkl', 'rb')
    token_id = pickle.load(f)[token+'_id']
    f.close()
    data = data.groupby([group_key, sort_key])
    for keys in data.groups.keys():
        print(data.get_group(keys))
        courses = tuple()
        for i in data.get_group(keys)['Num_subject']:
            courses = courses + (token_id[i],)
        print(courses)
        mat_data[group_key_id[keys[0]]][sort_key_id[keys[1]]] = courses
    mat_file = {'data': mat_data}
    f = open('data_matrix.pkl', 'wb')
    pickle.dump(mat_file, f)
    f.close()
    return mat_data


def get_factor(data, group_key_count, factor, group_key, sort_key):
    data = data.loc[data[group_key].isin(group_key_count[group_key])]
    print(data)
    data = data.loc[data[factor].notnull()].reset_index()
    print(data)
    s = data[factor].str.split('; ').apply(pd.Series, 1).stack()
    print(s)
    s.index = s.index.droplevel(-1)
    print(s)
    s.name = factor
    del data[factor]
    data = data.join(s)
    sem_instr = data.loc[:, [factor, sort_key]]
    sem_instr.drop_duplicates(inplace=True)
    count1 = sem_instr.groupby(factor).size()
    count1 = pd.core.frame.DataFrame({'count_sem': count1}).reset_index()
    count1.sort_values(by=['count_sem'], ascending=False, inplace=True)
    count1.reset_index(inplace=True)
    count1 = count1.loc[:, [factor, 'count_sem']]
    dic = dict()
    li = list(count1[factor])
    for i in li:
        dic[i] = len(dic)
    reversed_dic = dict(zip(dic.values(), dic.keys()))
    alldata = {factor + '_id': dic, 'id_' + factor: reversed_dic}
    f = open(factor + '_id.pkl', 'wb')
    pickle.dump(alldata, f)
    f.close()


def get_factors_seq(data, count, group_key, token, sort_key, sort_key_length, factors):
    data = data.loc[data[group_key].isin(count[group_key])]
    for factor in factors:
        data = data.loc[data[factor].notnull()]
    f = open(group_key+'_id.pkl', 'rb')
    group_key_id = pickle.load(f)[group_key+'_id']
    mat_data = [[None for i in range(sort_key_length)] for j in range(len(group_key_id.keys()))]
    f = open(sort_key+'_id.pkl', 'rb')
    sort_key_id = pickle.load(f)[sort_key+'_id']
    f = open(token+'_id.pkl', 'rb')
    token_id = pickle.load(f)[token+'_id']
    factor_dic = {}
    for factor in factors:
        f = open(factor+'_id.pkl', 'rb')
        factor_dic[factor] = pickle.load(f)[factor+'_id']
    f.close()
    num_token = len(token_id)
    data = data.groupby([group_key, sort_key])
    for keys in data.groups.keys():
        tokens = list()
        for index, rows in data.get_group(keys).iterrows():
            i = rows[token]
            token_per = list()
            k = 0
            token_per += [token_id[i]]
            for j in factors:
                multi_values = rows[j].split('; ')
                #print(multi_values)
                for m in multi_values:
                    token_per += [factor_dic[j][m]+num_token+k]
                k = len(factor_dic[j])
            tokens += [token_per]
            print(token_per)
        print(tokens)
        mat_data[group_key_id[keys[0]]][sort_key_id[keys[1]]] = tokens
    mat_file = {'data': mat_data}
    f = open('data_matrix.pkl', 'wb')
    pickle.dump(mat_file, f)
    f.close()
    return mat_data







