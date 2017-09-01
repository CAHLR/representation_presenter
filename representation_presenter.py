import os
import sys
import getopt
import time
import pandas as pd
import numpy as np
import gensim
import random

time0 = time.time()
#argument initialization
epoch = 50
negative_num = 20
vec_size = 100
window_size = 10
min_count_num = 3
Using_tsne = False
loss_flag = False
percentage = 0
process_type = 0
tsne_path = ''
tsne_dim = 2
eps = 1e-8
pca_dim = 50
#add new parameter
# -g grouped by
group_key = ''
# -s sorted by
sort_key = ''
# -k token
token_key = ''
# dup
removedup = True
# outputfile type: 1 for pre-processing, 2 for vector tsv file, others for 2 dims tsv file
out_type = 0
# inputfile data type: 1 for pre-preprocessing file, 2 for vector tsv file, others for raw file
in_type = 0

try:
    opts, args = getopt.getopt(sys.argv[1:],'hi:o:e:n:v:w:c:t:lp:d:g:s:k:m:',)
except getopt.GetoptError:
    print('\nxxx.py -i <inputfile,inputType> -o <outputfile,outputType> -e <epoch num> -n <negative> -v <vector size> -w <window size> -c <min count> -t <tsne_path>  -l (if calculating val loss) -g <group_by> -k <token_key> -s <sort_by> \nDefault parameter: epoch=50, negative=10, vector size=100, window size=10, min count=5\n')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('\nxxx.py -i <inputfile,inputType> -o <outputfile,outputType> -e <epoch num> -n <negative> -v <vector size> -w <window size> -c <min count> -t <tsne_path>  -l (if calculating val loss) -g <group_by> -k <token_key> -s <sort_by> \nDefault parameter: epoch=50, negative=10, vector size=100, window size=10, min count=5\n Attention:\nWhen input type = 1, -k must be set, and output type shouldn\'t be 1\nWhen input type = 2, -t must be set, and output type must be 3\nWhen input type = 3, -k -t -g must be set\nWhen output type = 1, -k -t -g must be set, and input type must be 3\nWhen output type = 2, -k must be set, and input type shouldn\'t be 2\nWhen output type = 3, -t must be set\n')
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
        tsne_dim = 3

# Start
if tsne_dim == 3:
    output_type = '3D'
    
    
if in_type == 2 and out_type < 3 or in_type==1 and out_type==1:
    print('Type error:\n')
    sys.exit()
    
if group_key == '':
    if in_type == 3 or out_type == 1:
        print('option [-g] must be set\n')
        sys.exit()
if token_key == '':
    if in_type != 2 or out_type != 3:
        print('option [-t] must be set\n')
        sys.exit()
if sort_key == '':
    if in_type == 3 or out_type == 1:
        print('option [-s] must be set\n')
        sys.exit()
if tsne_path == '':
    if in_type == 2 or out_type == 3:
        print('option [-t] must be set\n')
        sys.exit()
        
print(opts)
print('Input file: '+inputfile)
print('Input type: '+input_type)
print('Output file: '+outputfile)
print('Output type: '+output_type)
print('Group by: '+group_key)
print('Sort  by: '+sort_key)
print('Token: '+token_key)
print('Duplication Removal: ',removedup)

def read_big_csv(inputfile):
    import pandas as pd
    reader = pd.read_csv(inputfile, iterator=True, low_memory = False,delimiter='\t')
    loop = True
    chunkSize = 100000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    df = pd.concat(chunks, ignore_index=True)
    return df

def train_model(datalist):
    from gensim.models import Word2Vec
    model = Word2Vec(datalist, negative = negative_num, size = vec_size, window = window_size, min_count = min_count_num, workers = 20, compute_loss = True)
    word_vectors = model.wv
    namelist = word_vectors.index2word
    veclist = word_vectors.syn0.tolist()
    print('Effective Vectors: ', len(namelist))
    vecframe = pd.DataFrame(veclist,index=namelist)
    print('Model training process completed')
    print('Training loss',model.running_training_loss)
    return vecframe


def do_tsne(inputfile, path):
    tmpfile = inputfile+'tmp.csv'
    command = "addpath(\'"+tsne_path+"\');"
    command += "inputdata = importdata(\'"+inputfile+"\');"
    command += "data = inputdata.data;"
    command += "data = data(2:size(data,1),:);"
    command += "numDims = "+str(tsne_dim)+"; pcaDims = "+str(pca_dim)+"; perplexity = 20; theta = .5; alg = 'svd';"
    command += "map = fast_tsne(data, numDims, pcaDims, perplexity, theta, alg);"
    command += "csvwrite(\'"+tmpfile+"\',map);"
    os.system('matlab -nodisplay -nosplash -nojvm -r \"'+command+'\"'+'quit;')
    outputframe = pd.read_csv(tmpfile, header = None)
    os.system('rm -r '+tmpfile)
    return outputframe


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



# main

timebf = time.time()
# data pre-processing
if in_type == 2:
    print('Input type: Vector File')
    print('Output type: 2D Vector File')
    vec_frame = read_big_csv(inputfile)
    frame = do_tsne(inputfile, tsne_path)
    frame = pd.concat([vec_frame[[vec_frame.columns[0]]],frame[[0,1]]],axis=1)
    frame.columns = [vec_frame.columns[0], 'x', 'y']
    frame.to_csv(outputfile, sep = '\t', index = False)
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
            # save 2 dims tsv file
            vec_frame.to_csv(outputfile, sep='\t', index=False)
            frame = do_tsne(outputfile, tsne_path)
            if tsne_dim == 2:
                frame = pd.concat([vec_frame[[vec_frame.columns[0]]],frame[[0,1]]],axis=1)
                print('Output type: 2D Vector File')
                frame.columns = [vec_frame.columns[0], 'x', 'y']
            if tsne_dim == 3:
                frame = pd.concat([vec_frame[[vec_frame.columns[0]]],frame[[0,1,2]]],axis=1)
                print('Output type: 3D Vector File')
                frame.columns = [vec_frame.columns[0], 'x', 'y', 'z']
            os.system('rm -r '+outputfile)
            frame.to_csv(outputfile, sep = '\t', index = False)
            
    if out_type == 4:
        outlist = []
        for line in train_list:
            outlist.append(' '.join(line))
        with open('PP-'+outputfile, 'w') as f:
            f.write('\n'.join(outlist)+'\n')
        vec_frame = train_model(train_list)
        head = [token_key]
        head.extend(list(map(str,range(1,vec_size+1))))
        vec_frame = vec_frame.reset_index()
        vec_frame.columns = head
        # save vector file for outputfile(.tsv)
        vec_frame.to_csv('VS-'+outputfile, sep='\t', index=False)
        frame = do_tsne('VS-'+outputfile, tsne_path)
        frame = pd.concat([vec_frame[[vec_frame.columns[0]]],frame[[0,1]]],axis=1)
        if tsne_dim == 2:
            frame = pd.concat([vec_frame[[vec_frame.columns[0]]],frame[[0,1]]],axis=1)
            print('Output type: 2D Vector File')
            frame.columns = [vec_frame.columns[0], 'x', 'y']
        if tsne_dim == 3:
            frame = pd.concat([vec_frame[[vec_frame.columns[0]]],frame[[0,1,2]]],axis=1)
            print('Output type: 3D Vector File')
            frame.columns = [vec_frame.columns[0], 'x', 'y', 'z']
            
        os.system('rm -r '+outputfile)
        frame.to_csv(outputfile, sep = '\t', index = False)
        
timeaf = time.time()
print('TIME: ',timeaf-timebf)