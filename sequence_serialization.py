__author__ = 'jwj'
import pandas as pd
import pickle
import numpy as np


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
    return count


def set_sort_key_id(data, sort_key):
    sort = data[sort_key].sort_values(ascending=True)
    dic1 = dict()
    for i in sort:
        dic1[i] = len(dic1)
    reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
    alldata = {sort_key+'_id': dic1, 'id_'+sort_key: reversed_dic1}
    f = open(sort_key+'_id.pkl', 'wb')
    pickle.dump(alldata, f)
    f.close()


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


def get_factor(data, token, group_key_count, factor, group_key):
    data = data.loc[data[group_key].isin(group_key_count[group_key])]
    count = data.groupby(factor).size()
    count = pd.core.frame.DataFrame({'count_factor': count}).reset_index()
    count.sort_values(by=['count_factor'], ascending=False, inplace=True)
    count.reset_index(inplace=True)
    count = count.loc[:, [factor, 'count_factor']]
    #count.to_csv(factor+'_id.csv')
    dic = dict()
    li = list(count[factor])
    for i in li:
        dic[i] = len(dic)
    reversed_dic = dict(zip(dic.values(), dic.keys()))
    alldata = {factor+'_id': dic, 'id_'+factor: reversed_dic}
    f = open(factor+'_id.pkl', 'wb')
    pickle.dump(alldata, f)


def get_factors_seq(data, count, group_key, token, sort_key, sort_key_length, factors):
    data = data.loc[data[group_key].isin(count[group_key])]
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
            factor_list = list()
            k = 0
            for j in factors:
                factor_list += rows[j]
                tokens = tokens + [[token_id[i], factor_dic[j][rows[j]]+num_token+k]]
                k = len(factor_dic[j])
        #print(tokens)
        mat_data[group_key_id[keys[0]]][sort_key_id[keys[1]]] = tokens
    mat_file = {'data': mat_data}
    f = open('data_matrix.pkl', 'wb')
    pickle.dump(mat_file, f)
    f.close()
    return mat_data







