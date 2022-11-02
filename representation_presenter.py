import os
import sys
import getopt
import time
import pandas as pd
import numpy as np
import gensim
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.models import FastText

time0 = time.time()
#argument initialization
# epoch = 50
epoch = 50
negative_num = 20
vec_size = 100
window_size = 10
min_count_num = 3
# currently hardcoding sg=1 (use skip gram)
Using_tsne = False
loss_flag = False
percentage = 0
process_type = 0
tsne_path = ''
tsne_dim = 2
eps = 1e-8
model_used = 1
model_choice_dic = {1:'word2vec', 2:'fasttext'}
pca_dim = 50
#add new parameter
# -g grouped by
group_key = ''
# -s sorted by
sort_key = ''
feature = False
# -k token
token_key = ''
# dup
removedup = True
# outputfile type: 1 for pre-processing, 2 for vector tsv file, others for 2 dims tsv file
out_type = 0
# inputfile data type: 1 for pre-preprocessing file, 2 for vector tsv file, others for raw file
in_type = 0
sep = '\t'



try:
    opts, args = getopt.getopt(sys.argv[1:],'hi:o:x:e:n:v:w:c:t:lp:d:g:s:k:m:f',)
except getopt.GetoptError:
    print('\nxxx.py -i <inputfile,inputType> -o <outputfile,outputType>  -x <model_used> -e <epoch num> -n <negative> -v <vector size> -w <window size> -c <min count> -t <tsne_path>  -l (if calculating val loss) -g <group_by> -k <token_key> -s <sort_by> \nDefault parameter: epoch=50, negative=10, vector size=100, window size=10, min count=5\n')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('\nxxx.py -i <inputfile,inputType> -o <outputfile,outputType> -x <model_used> -e <epoch num> -n <negative> -v <vector size> -w <window size> -c <min count> -t <tsne_path>  -l (if calculating val loss) -g <group_by> -k <token_key> -s <sort_by> \nDefault parameter: epoch=50, negative=10, vector size=100, window size=10, min count=5\n Attention:\nWhen input type = 1, -k must be set, and output type shouldn\'t be 1\nWhen input type = 2, -t must be set, and output type must be 3\nWhen input type = 3, -k -t -g must be set\nWhen output type = 1, -k -t -g must be set, and input type must be 3\nWhen output type = 2, -k must be set, and input type shouldn\'t be 2\nWhen output type = 3, -t must be set\n')
        sys.exit()
    if opt in ("-i"):
        if len(arg.split(',')) == 1:
            inputfile = arg
            in_type = 3
            input_type = 'Raw'
        else:
            inputfile = arg.split(',')[0]
            in_type = int(arg.split(',')[1])
            if in_type == 1:
                input_type = 'Pre-Processing'
            elif in_type == 2:
                input_type = 'Vector'
            else:
                input_type = 'Raw'
            
    if opt in ("-o"):
        if len(arg.split(',')) == 1:
            outputfile = arg
            out_type = 3
            output_type = '2D'
        else:
            outputfile = arg.split(',')[0]
            out_type = int(arg.split(',')[1])
            if out_type == 1:
                output_type = 'Pre-Processing'
            elif out_type == 2:
                output_type = 'Vector'
            elif out_type == 3:
                output_type = '2D'
            else:
                output_type = 'All Type'
                
    if opt in ("-x"):
        ## w2v(1) or ft(2)
        model_used = int(arg)
    if opt in ("-e"):
        epoch = int(arg)
    if opt in ("-n"):
        negative_num = int(arg)
    if opt in ("-v"):
        vec_size = int(arg)
        pca_dim = min(50, vec_size)
    if opt in ("-w"):
        window_size = int(arg)
    if opt in ("-c"):
        min_count_num = int(arg)
    if opt in ("-t"):
        Using_tsne = True
        tsne_path = arg
    if opt in ("-l"):
        loss_flag = True
        #percentage = float(arg)
    if opt in ("-d"):
        if arg == 'd':
            removedup = False
    if opt in ("-g"):
        group_key = arg
    if opt in ("-k"):
        token_key = arg
        token_list = token_key.split(',')
    if opt in ("-s"):
        sort_key = arg
    if opt in ("-m"):
        tsne_dim = int(arg)
    if opt in ("-f"):
        feature = True
        feature_name = arg.split(',')[0]
        merge_type = arg.split(',')[1]
    
# Start
if tsne_dim != 2:
    output_type = str(tsne_dim)+'D'
    
    
if in_type == 2 and out_type < 3 or in_type==1 and out_type==1:
    print('Type error:\n')
    sys.exit()
    
if group_key == '':
    if in_type == 3 or out_type == 1:
        print('option [-g] must be set\n')
        sys.exit()

if sort_key == '':
    if in_type == 3 or out_type == 1:
        print('option [-s] must be set\n')
        sys.exit()


print(opts)
print('Input file: '+inputfile)
print('Input type: '+input_type)
print('Output file: '+outputfile)
print('Output type: '+output_type)
print('Group by: '+group_key)
print('Sort  by: '+sort_key)
print('Token: '+token_key)
print('Model: '+model_choice_dic[model_used])
print('Duplication Removal: ',removedup)

def read_big_csv(inputfile):
    import pandas as pd
    with open(inputfile,'r') as f:
        a = f.readline()
    csvlist = a.split(',')
    tsvlist = a.split('\t')
    if len(csvlist)>len(tsvlist):
        sep = ','
    else:
        sep = '\t'
    print('sep:' , sep)
    reader = pd.read_csv(inputfile, iterator=True, low_memory = False,delimiter=sep)
    loop = True
    chunkSize = 100000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
    df = pd.concat(chunks, ignore_index=True)
    return df

def train_model(datalist):
    
    if model_used == 1:
        print('training Word2Vec')
        model = Word2Vec( negative = negative_num, vector_size = vec_size, window = window_size, min_count = min_count_num, workers = 20, sg=1)
    else:
        print('training FastText')
        model = FastText( negative = negative_num, vector_size = vec_size, window = window_size, min_count = min_count_num, workers = 20, sg=1)
    
    model.build_vocab(datalist, progress_per=10000 )
    model.train(datalist, total_examples=len(datalist), epochs=epoch,compute_loss=True)
    
    word_vectors = model.wv

    namelist = word_vectors.index_to_key

    veclist = word_vectors.vectors.tolist()
    print('Effective Vectors: ', len(namelist))
    vecframe = pd.DataFrame(veclist,index=namelist)
    print('Model training process completed')
    print('Training loss',model.running_training_loss)
    return vecframe





def norep(datalist):
    newlist = []
    for item in datalist:
        smallitem = []
        for word in item:
            if len(smallitem)==0 or smallitem[len(smallitem)-1]!=word:
                smallitem.append(word)
        newlist.append(smallitem)
    return newlist


def prepareData(inputfile, sort_key, group_key, token_list):
    #dataflow = pd.read_csv(inputfile,delimiter='\t', low_memory = False)
    dataflow = read_big_csv(inputfile)
    i = 0
    for item in token_list:
        i += 1
        dataflow[item] = dataflow[item].astype('str')
        if i == 1:
            dataflow['token_key']  = dataflow[item]
        else:
            dataflow['token_key'] += '_' + dataflow[item]
    data_flow = pd.DataFrame.sort_values(dataflow, by = sort_key, ascending=True)
    frame = data_flow[['token_key', group_key]]
    frame = frame.dropna(axis=0, how = 'any')
    group_frame = frame.groupby(group_key)
    group_list = list(group_frame)
    train_list = []
    for item in group_list:
        if item[1].shape[0] > 5:
            train_list.append(list(item[1]['token_key']))
    if removedup == True:
        train_list = norep(train_list)
    return train_list


def get_tsne_df(vec_frame, tsne_dim):
    array_embs = vec_frame.iloc[:,1:].values
    tsne = TSNE(random_state=1, n_components=tsne_dim, metric="cosine")
    frame= pd.DataFrame(tsne.fit_transform(array_embs))
    frame[vec_frame.columns[0]] = vec_frame.iloc[:,0]
    cols = list(frame.columns)
    cols = [cols[-1]] + cols[:-1]
    frame = frame[cols]
    return frame
# main

timebf = time.time()
# data pre-processing
if in_type == 2:
    print('Input type: Vector File')
    print('Output type: 2D Vector File')
    vec_frame = read_big_csv(inputfile)
    
    frame = get_tsne_df(vec_frame,2)
    print('Output type: ', tsne_dim, 'D Vector File')

else:
    if in_type == 1:
        # input type: corpus for training
        print('Input type: Pre-processing File for Model Training')
        train_list = []
        with open(inputfile, 'r') as f:
            for line in f.readlines():
                train_list.append(line[:-1].split(' '))
                
    else:
        # input file: raw file
        print('Input type: Raw File')
        train_list = prepareData(inputfile, sort_key, group_key, token_list)
    # train word2vec model
    if out_type == 1:
        # save pre-processing file for outputfile(.txt)
        print('Output type: Pre-processing File for Model Training')
        outlist = []
        for line in train_list:
            outlist.append(' '.join(line))
        with open(outputfile, 'w') as f:
            f.write('\n'.join(outlist)+'\n')
    else:
        vec_frame = train_model(train_list)
        head = [token_key]
        head.extend(list(map(str,range(1,vec_size+1))))
        vec_frame = vec_frame.reset_index()
        vec_frame.columns = head
        
        if out_type == 2:
            # save vector file for outputfile(.tsv)
            print('Output type: Vector File')
            vec_frame.to_csv(outputfile, sep='\t', index=False)
        
        elif out_type == 3:
            frame = get_tsne_df(vec_frame,tsne_dim)
            print('Output type: ', tsne_dim, 'D Vector File')
            print(frame.shape)
            new_outputfile_name = outputfile.replace('.tsv', '') + '_tsne_dim_' + str(tsne_dim) + '.tsv'
            frame.to_csv(new_outputfile_name, sep = '\t', index = False)
            
    if out_type == 4:
        frame = get_tsne_df(vec_frame,tsne_dim)
        
        print('Output type: ', tsne_dim, 'D Vector File')
        print(frame.shape)
        new_outputfile_name = outputfile.replace('.tsv', '') + '_tsne_dim_' + str(tsne_dim)+ '.tsv'
        frame.to_csv(new_outputfile_name, sep = '\t', index = False)
        
        print('Output type: full vector File')
        vec_frame.to_csv(outputfile, sep='\t', index=False)

        
timeaf = time.time()
print('TIME: ',timeaf-timebf)

if feature == True:
    frame = read_big_csv(outputfile)
    feature_frame = read_big_csv(feature_name)
    frame = pd.merge(frame,  feature_frame, how = 'left', on = frame.columns[0])
    frame.to_csv(outputfile, sep = '\t', index = False)
    