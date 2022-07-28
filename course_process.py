__author__ = 'jwj'
import pandas as pd
import pickle

args={
    'institution': 'cortland',
    'min_count': 5,
}

df_broome = pd.read_csv('/home/yueqi/multi_inst_plan/suny_preprocess/preprocessed_data/Broome/enrollment_Broome_TERM_SUNY_CSID_feats.csv',
                        header=0,
                        dtype={'OSKI_COURSE_SUBJECT_SHORT_NM': str, 'OSKI_GRADE_NM': str})
df_cortland = pd.read_csv('/home/yueqi/multi_inst_plan/suny_preprocess/preprocessed_data/Cortland/enrollment_Cortland_TERM_SUNY_CSID_feats.csv',
                        header=0,
                        dtype={'OSKI_COURSE_SUBJECT_SHORT_NM': str, 'OSKI_GRADE_NM': str})
if args['institution'] == 'broome':
    data = df_broome
if args['institution'] == 'cortland':
    data = df_cortland
if args['institution'] == 'broome_cortland':
    data = pd.concat((df_broome, df_cortland))

print(data.shape)
data.drop_duplicates(inplace=True)
print(data.shape)
data.dropna(inplace=True)
print(data.shape)
#print(data.loc[data['Course Number']=='-'])
# data = data.loc[(data['Offering Type Desc']=='Primary')&(data['Grade Subtype Desc']!='Administrative Code')&(data['Grade Subtype Desc']!='Unknown')&(data['Semester Year Name Concat']!='2008 Spring')&(data['Semester Year Name Concat']!='2008 Summer')]
# data['Num_subject'] = data['Course Number'] + ' '+data['Course Subject Short Nm']
# data.drop(columns=['Course Number', 'Course Subject Short Nm'], inplace=True)

count = data.groupby('TERM_SUNY_CSID')['TERM_SUNY_CSID', 'OSKI_GRADE_NM'].count()
count = count.loc[count['TERM_SUNY_CSID']>=args['min_count']]
# print(count)


data = data.loc[data['TERM_SUNY_CSID'].isin(count.index), ['TERM_SUNY_CSID', 'OSKI_COURSE_SUBJECT_SHORT_NM']]
print(data.shape)
count = data.groupby('TERM_SUNY_CSID').size()
#print(type(count))
count = pd.core.frame.DataFrame({'count' :count}).reset_index()
count.sort_values(by=['count'], ascending=False, inplace=True)
count.reset_index(inplace=True)
count = count.iloc[:,[1,2]]
#print(count)
# get course id to dictionary
dic = count.to_dict('dict')
dic1 = dic['TERM_SUNY_CSID']
reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
alldata = {'course_id': reversed_dic1, 'id_course':dic1}
f = open('/home/yueqi/multi_inst_plan/course2vec_suny/course_preprocess/{}/course_id_mincount_{}.pkl'.format(args['institution'], args['min_count']), 'wb')
pickle.dump(alldata, f)
f.close()
# testify whether Num_subject can be the key
data.drop_duplicates(subset={'TERM_SUNY_CSID', 'OSKI_COURSE_SUBJECT_SHORT_NM'}, inplace=True)
#print(data)
data.to_csv('/home/yueqi/multi_inst_plan/course2vec_suny/course_preprocess/{}/course_dept_mincount_{}.csv'.format(args['institution'], args['min_count']), index=False)
count.to_csv('/home/yueqi/multi_inst_plan/course2vec_suny/course_preprocess/{}/course_enroll_num_mincount_{}.csv'.format(args['institution'], args['min_count']))

