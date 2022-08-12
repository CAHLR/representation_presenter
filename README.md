## Updates Aug 2022
###### To train suny dataset
NOTE: the following data is pseudo data

	python representation_presenter -i your_data_path/enrollment_Broome_TERM_SUNY_CSID_feats_multi_c2v_input.csv -g ANON_ID -s t -k TERM_SUNY_CSID -r OSKI_COURSE_SUBJECT_SHORT_NM

| Stuid  | Semester_year_name_concat | Course_id | Subject_short_name |
| ------------- | ------------- | ------------- | ------------- |
| 664564  | Fall 2024  | 32541 | COM |
| 453656  | Spring 2025  | 56423 | SOC |

## Previous specification
Tool to learn representations for and visualize interactions from event-log files, such as log data from course clickstream, arbitrary website event logs,  netflows, and other temporal sequences. 

SYNOPSIS
	python representation_presenter.py [options]


OPTIONS
	INPUT AND OUTPUT:
	-i inputfile_name,input_type
		type '1': Pre-processing file, train list for word2vec(txt)
		type '2': Vector file after word2vec with key for token(tsv)
		type '3': Raw file, event log file(tsv)
		Default: type '3'

	-o outputfile_name,output_type
		type '1': Pre-processing file, train list for word2vec(txt)
		type '2': Vector file after word2vec with key for token(tsv)
		type '3': 2D Vector file
		type '4': All three types above
		Default: type '3'

	There should be no space between file name and file type.


	PARAMETERS FOR WORD2VEC
	-n negative
   		Negative sampling when using word2vec. Default 10.

	-v vector size
   		Vector size when using word2vec(size). Default 100.

	-w window size
		Window size when using word2vec(window). Default 10.

	-c min count
		Min count number when using word2vec(min_count).

	-e epoch num
		Iteration times for word2vec. Default 50.


	KEYS
	-g group_by_key
		Group by. 
	
	-s sort_by_key
		Sort by.

	-k token1,token2,token3...(no space)
		Tokens for analysis

	-m dim
		Dimension of data points in output file(after t-sne)


	OTHERS
	-t t-sne_path
   		When using t-sne, choose this command. And there should be files 'fast_t-sne.m' and 'bh_t-sne' in t-sne_path.

   	-dd
   		If you use this option, duplications won't be removed when generate training list

   	-f featurefile,type
   	    type determines how to merge feature file and vector file
   		type 'inner': intersection
   		type 'left' : save rows in vector file only
   		type 'right': save rows in feature file only 


Example: Input file are row event log file and output are all files, then use command:

	python/python3 representation_presenter.py -i input.tsv,3 -o n10_vs50_ws2.tsv,4 -g basic_type -s time -k username -t /home/xingzb14/tsne -n 10 -v 50 -c 5 -w 2 -e 30



