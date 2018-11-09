__author__ = 'jwj'
import pandas as pd
import pickle


def gene_token_id(data):
    count = data.groupby('Num_subject').size()
    count = pd.core.frame.DataFrame({'count': count}).reset_index()
    count.sort_values(by=['count'], ascending=False, inplace=True)
    count.reset_index(inplace=True)
    count = count.iloc[:,[1,2]]
    #print(count)
    # get course id to dictionary
    dic = count.to_dict('dict')
    dic1 = dic['Num_subject']
    reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
    alldata = {'course_id': reversed_dic1, 'id_course':dic1}
    f = open('course_id.pkl', 'wb')
    pickle.dump(alldata, f)
    f.close()


