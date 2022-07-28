_author__ = 'jwj'
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.preprocessing import sequence
import numpy as np
import pickle
from tqdm import tqdm

args = {
    # 'institution': 'broome',
    'institution': 'cortland',
    # 'institution': 'broome_cortland'
    'window_size': 5, # the window of words around the target word that will be used to draw the context words from
    'min_count': 5
}

window_size = args['window_size']

course_file = open('/home/yueqi/multi_inst_plan/course2vec_suny/course_preprocess/{}/course_id_mincount_{}.pkl'.format(args['institution'], args['min_count']), 'rb')
course = pickle.load(course_file)
course_file.close()
course_id = course['course_id']
vocab_size = len(course_id)
id_course = course['id_course']
f = open('/home/yueqi/multi_inst_plan/course2vec_suny/enroll_preprocess/{}/stu_sem_course&grade&subject_mincount_{}.pkl'.format(args['institution'], args['min_count']), 'rb')
data = pickle.load(f)['stu_sem_course&grade&subject']
f.close()
data = np.array(data)
couples = []
labels = []
print('Start sampling')
for i in tqdm(range(data.shape[0])):
    #i = 102742
    # print("this is"+str(i))
    a = data[i][data[i] != np.array(None)]
    if a.size==0:
        continue
    seq = np.sum(a)
    couple, label = skipgrams(seq, vocab_size, window_size=window_size, negative_samples=0)
    couples.extend(couple)
    #print(couples)
    # print(len(couples))
    #print(couples)
    labels.extend(label)
course_target, course_context = zip(*couples)  # zip(*list[[]]) = unzip to tuple
print('Finish sampling')

all_data = {'course_target': course_target, 'course_context': course_context, 'labels': labels}
f = open('/home/yueqi/multi_inst_plan/course2vec_suny/course2vec_grade_subject/data/{}/sampled_data_windowsize{}_mincount_{}.pkl'.format(args['institution'], args['window_size'], args['min_count']),'wb')
pickle.dump(all_data, f)
f.close()
