#
# Q3.  Semantic similarity 
# a)  Using the output from Q2, identify semantically similar job ads and group them 
# together. 
#

# Load the data
import pandas as pd
data_df = pd.read_csv('Q2_data.csv')


# remove the NAN rows
mask = pd.isna( data_df.job_responsibility2 )
data2_df = data_df[~mask]
# data2_df

#
# Convert the string to list. 
#
import ast
data2_df['job_responsibility3'] = data2_df['job_responsibility2'].map(lambda x: ast.literal_eval(x) )
# data2_df[['job_responsibility2', 'job_responsibility3']]



# from gensim.models import Word2Vec
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Train model
model = gensim.models.Word2Vec(
        data2_df['job_responsibility3'].tolist(),
        size=150,
        window=10,
        min_count=2,
        workers=10)

model.train(data2_df['job_responsibility3'], total_examples=len(data2_df['job_responsibility3']), epochs=30)

# Filter out those word that are not in the vocab
data2_df['job_responsibility4'] = data2_df['job_responsibility3'].map(lambda x_list : [x for x in x_list if x in model.wv.vocab] )



#
# Compare the all the similarity
#
from itertools import combinations
couple_list = list( combinations(range(100), 2) )
# Note 100 choose 2 samples contributes to 4,950 combinations.
# Due to lack of time, I will try all the combination for a small samples , i.e. 100 rows first.
# Theoretically, the following algorithm should list the top 30 groups that produces the best simlarity.
#


result_list =[]
for couple in couple_list:
    vec1=data2_df['job_responsibility4'].iloc[couple[0]]
    vec2=data2_df['job_responsibility4'].iloc[couple[1]]
    result = model.wmdistance(vec1, vec2)
    result_list.append(result)



top_list = sorted( zip(result_list, couple_list))[:30]
for i, tops in enumerate(top_list):
    score  = tops[0]
    num_list = tops[1]
    print ('%d. Job posts of simliar requirements are : %s, score=%f' % (i, str( num_list), score) )
    for num in num_list:
        print('[ %d] %s' % (num, data2_df['job_title'].iloc[num]) )
    #
    print()


#
# Result (partial)
#

# 0. Job posts of simliar requirements are : (6, 69), score=0.000000
# [ 6] Community Development, Capacity Building and Conflict Management Consultant
# [ 69] Community Development, Capacity Building and Conflict Management Consultant

# 1. Job posts of simliar requirements are : (7, 30), score=0.000000
# [ 7] General Manager
# [ 30] General Manager

# 2. Job posts of simliar requirements are : (15, 27), score=0.000000
# [ 15] Chief/ Supervisor of Programs Department
# [ 27] Chief/ Supervisor of Programs Department

# 3. Job posts of simliar requirements are : (20, 22), score=0.000000
# [ 20] Receptionist
# [ 22] Receptionist

# 4. Job posts of simliar requirements are : (65, 85), score=0.000000
# [ 65] Intern
# [ 85] Intern

# 5. Job posts of simliar requirements are : (81, 83), score=0.738664
# [ 81] Environmental Education (EE) Technical Coordinator (short tirm) - Pre-Service Training (PST) of Volunteers
# [ 83] Community Health Education (CHE) Technical Coordinator (short tirm) - Pre-Service Training (PST) of Volunteers

# 6. Job posts of simliar requirements are : (82, 83), score=1.110092
# [ 82] BECD Technical Coordinator (short tirm) - Pre-Service Training (PST) of Volunteers
# [ 83] Community Health Education (CHE) Technical Coordinator (short tirm) - Pre-Service Training (PST) of Volunteers

# 7. Job posts of simliar requirements are : (81, 82), score=1.265750
# [ 81] Environmental Education (EE) Technical Coordinator (short tirm) - Pre-Service Training (PST) of Volunteers
# [ 82] BECD Technical Coordinator (short tirm) - Pre-Service Training (PST) of Volunteers

# 8. Job posts of simliar requirements are : (83, 84), score=1.724409
# [ 83] Community Health Education (CHE) Technical Coordinator (short tirm) - Pre-Service Training (PST) of Volunteers
# [ 84] TEFL Technical Coordinator (short tirm) - Pre-Service Training (PST) of Volunteers

# 9. Job posts of simliar requirements are : (81, 84), score=2.083384
# [ 81] Environmental Education (EE) Technical Coordinator (short tirm) - Pre-Service Training (PST) of Volunteers
# [ 84] TEFL Technical Coordinator (short tirm) - Pre-Service Training (PST) of Volunteers

# 10. Job posts of simliar requirements are : (82, 84), score=2.213471
# [ 82] BECD Technical Coordinator (short tirm) - Pre-Service Training (PST) of Volunteers
# [ 84] TEFL Technical Coordinator (short tirm) - Pre-Service Training (PST) of Volunteers

# 11. Job posts of simliar requirements are : (62, 98), score=6.154613
# [ 62] Language and Administrative Assistant
# [ 98] Project Assistant

# 12. Job posts of simliar requirements are : (89, 98), score=9.234989
# [ 89] Administrative Assistant
# [ 98] Project Assistant

# 13. Job posts of simliar requirements are : (45, 68), score=9.781793
# [ 45] Assistant to Director/ Translator
# [ 68] Assistant to Director, Translator/ Interpreter

# 14. Job posts of simliar requirements are : (62, 89), score=9.984749
# [ 62] Language and Administrative Assistant
# [ 89] Administrative Assistant

# 15. Job posts of simliar requirements are : (48, 49), score=13.222682
# [ 48] MMT Project Manager
# [ 49] Health Coordinators (two positions are open)

# 16. Job posts of simliar requirements are : (52, 56), score=14.137726
# [ 52] Project Assistant
# [ 56] Project Manager

# 17. Job posts of simliar requirements are : (23, 74), score=16.375717
# [ 23] Quality Assurance/ Health Systems Management Advisor
# [ 74] Project Deputy Director

# 18. Job posts of simliar requirements are : (23, 48), score=16.925303
# [ 23] Quality Assurance/ Health Systems Management Advisor
# [ 48] MMT Project Manager

# 19. Job posts of simliar requirements are : (47, 49), score=17.194048
# [ 47] Project Assistant
# [ 49] Health Coordinators (two positions are open)

# 20. Job posts of simliar requirements are : (16, 74), score=17.385211
# [ 16] Deputy Program Director
# [ 74] Project Deputy Director

# 21. Job posts of simliar requirements are : (23, 49), score=17.560958
# [ 23] Quality Assurance/ Health Systems Management Advisor
# [ 49] Health Coordinators (two positions are open)

# 22. Job posts of simliar requirements are : (49, 74), score=17.573700
# [ 49] Health Coordinators (two positions are open)
# [ 74] Project Deputy Director

# 23. Job posts of simliar requirements are : (48, 74), score=17.599045
# [ 48] MMT Project Manager
# [ 74] Project Deputy Director

# 24. Job posts of simliar requirements are : (12, 68), score=17.663057
# [ 12] Administrative Assistant
# [ 68] Assistant to Director, Translator/ Interpreter

# 25. Job posts of simliar requirements are : (1, 98), score=17.700399
# [ 1] Full-time Community Connections Intern (paid internship)
# [ 98] Project Assistant

# 26. Job posts of simliar requirements are : (66, 92), score=17.787907
# [ 66] Executive Director
# [ 92] Latin America Manager

# 27. Job posts of simliar requirements are : (32, 66), score=17.999146
# [ 32] Country Director
# [ 66] Executive Director

# 28. Job posts of simliar requirements are : (57, 74), score=18.223902
# [ 57] Office Manager (AVET1)
# [ 74] Project Deputy Director

# 29. Job posts of simliar requirements are : (48, 95), score=18.261683
# [ 48] MMT Project Manager
# [ 95] Program Officer 

#
# To do next (Idea)
#
# Due to lack of time, I am not able to further complete the assignment.
# However, the basica idea is, once we have all the score of each pair (non-repetitive), 
# we can further build a tree and decide the threshold of grouping.
#
# For example, assume that we have score of (A,B) =10,  (B, C) = 11, but (A, C)= 20. If we
# set the threshold to be 15, then we can group (A,B,C) as one group.
#



