_author__ = 'jwj'
from tensorflow.keras.preprocessing.sequence import skipgrams
# from keras.preprocessing import sequence
import numpy as np
import pickle


def sampling(window_size, token):
    token_file = open(token+'_id.pkl', 'rb')
    token_dic = pickle.load(token_file)
    token_id = token_dic[token+'_id']
    vocab_size = len(token_id)
    id_token = token_dic['id_'+token]
    token_file.close()
    f = open('data_matrix.pkl', 'rb')
    data = pickle.load(f)['data']
    data = np.array(data)
    couples = []
    labels = []
    print('Start sampling')
    for i in range(data.shape[0]):
        print("this is"+str(i))
        a = data[i][data[i] != np.array(None)]
        if a.size==0:
            continue
        seq = np.sum(a)
        couple, label = skipgrams(seq, vocab_size, window_size=window_size, negative_samples=0)
        couples.extend(couple)
        print(len(couples))
        labels.extend(label)
    course_target, course_context = zip(*couples)  # zip(*list[[]]) = unzip to tuple
    print('Finish sampling')

    all_data = {'token_target': course_target, 'token_context': course_context, 'labels': labels}
    f = open('sampled_data.pkl', 'wb')
    pickle.dump(all_data, f)
    f.close()
