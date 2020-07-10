import pandas as pd
from matplotlib import pyplot as plt

patient = pd.read_csv('PatientInfo.csv')
patient_route = pd.read_csv('PatientRoute.csv')
# PatientInfo.csv와 PatientRoute.csv를 사용한다.

matrix = [[] for i in range(len(patient))]
for i in range(len(patient)):
    matrix[i].append(patient.loc[i].patient_id)
    matrix[i].append(patient.loc[i].age)
    
cnt = 0
count = 0

for i in range(len(patient_route)):
    if matrix[cnt][0] == patient_route.loc[i].patient_id:
        count = count+1
    else:
        matrix[cnt].append(count)
        cnt = cnt +1
        count = 1

for i in range(5):
    print(matrix[i])


old = [['0s',0,0],['10s',0,0],['20s',0,0],['30s',0,0],['40s',0,0],['50s',0,0],['60s',0,0],['70s',0,0],['80s',0,0],['90s',0,0]]
    
for i in range(len(matrix)):
    for j in range(len(old)):
        if matrix[i][1] == old[j][0]:
            old[j][1] = old[j][1] + 1
            if(len(matrix[i])==3):
                old[j][2] = old[j][2] + matrix[i][2]

for i in range(5):
    print(old[i])

for i in range(len(old)):
    print("나이: ", old[i][0], "||  감염자 수: " ,old[i][1], "||  이동한 평균 횟수: ", old[i][2]/old[i][1])


olds = []
olds_how_many = []
olds_avg = []

for i in range(len(old)):
    olds.append(old[i][0])
    olds_how_many.append(old[i][1])
    olds_avg.append(old[i][2]/old[i][1])

plt.bar(olds, olds_how_many, color = 'orangered')
plt.xlabel('Age group')
plt.ylabel('Number')
plt.title('Number of confirmers')
plt.show()

plt.bar(olds, olds_avg, color = 'blueviolet')
plt.xlabel('Age group')
plt.ylabel('Average')
plt.title('Average number of moves')
plt.show()