# AI-X-Deeplearning

# **Title**: 다양한 머신러닝 기법을 활용한 심장병 발병 예측

# **Members**:
<br>도은찬, 서울 기계공학부, dec1995@hanyang.ac.kr
<br>정다연, 서울 기계공학부, jdd2635@hanyang.ac.kr
         
# **Ⅰ. Proposal(Option A)**

  - Motivation:
  <br> 심장질환은 국내 사망원인 1위 질환인 암 다음으로 우리의 목숨을 위협하는 가장 무서운 질병이다. 이는 돌연사의 원인이 되기도 하며 고령 환자의 주요 사망 원인이다. 국내에서는 매년 20만명이 넘는 환자들이 심부전으로 병원을 찾는다. 우리나라 뿐만 아니라 다른 여러 국가에서도 사망 원인이 되는 심각한 질환이고 특히 미국 내에서는 전 인종이 영향을 받은 주된 사망원인 중 하나이다. 미국 CDC의 통계자료에 따르면 미국인들의 절반이 심장병의 세 가지 주요 원인인 고혈압, 고 콜레스테롤, 흡연 중 적어도 한가지를 가지고 있다고 한다. 이 외에도 심장질환을 발병시킬 수 있는 요인에는 육체 및 정신의 건강한 정도, 뇌졸중 경험 여부 그리고 음주 등이 있다. 따라서 이러한 원인들을 분석하여 심장병 발생 여부를 예측하는 것은 인류의 중대한 과업이라고 할 수 있다. 그러한 과업에 동참하기위해 우리 팀은 미국 CDC의 방대한 심장 질환 지표 데이터를 활용하여 그것을 예측할 수 있는 모델을 만들고 어떤 모델이 최적의 결과를 제시할 수 있는지 분석할 것이다. 
  
  - What do you want to see at the end?
  <br> 심장 질환은 예방 가능성이 높은 질환이다. 위에서 언급한 것처럼 주로 흡연, 운동 부족, 비만 등이 주요 원인이며, 이를 약물치료나 생활 습관의 교정함으로서 심장질환 발병을 예방할 수 있다. 따라서 환자의 성별, 나이, BMI 지수, 흡연 여부 등을 특징인자로 하는 머신러닝 기법들을 적용하여 심장질환 발생 가능성을 예측하고 사망률이 높은 질병 중 하나인 심장질환을 예측하고 이러한 환자들에게 약물치료, 생활 습관 교정 등의 예방법을 제시하여 심장질환으로 인한 사망률을 낮출 수 있는 모델을 제시하고자 한다.
  
# **Ⅱ. Datasets**

  - https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease (데이터셋 출처)
  
  - Describing dataset
  <br> ![캡처](https://user-images.githubusercontent.com/116618556/199441411-adf3fa21-bd9d-46f7-aa4d-a78fad62c8e8.JPG)
  <br>
  <br> 본 데이터셋은 미국 질병통제예방센터(CDC)가 2020년에 분석한 성인 40만명의 건강 지표로 이루어져 있다. 총 18개의 변수(9개의 불리언, 5개의 문자열, 4개의 정수형)로 이루어져 있으며, 각 feature별 자세한 내용은 하기와 같다.
  <br>
  <br> 1. HeartDisease: 심장질환 발병 여부, 불리언 자료형(Yes or No)
  <br> 2. BMI: Body Mass Index(BMI, 체질량지수), 정수형 자료형
  <br> 3. Smoking: 흡연 여부, 일생동안 100개비 이상 흡연하였으면 Yes에 해당, 불리언 자료형(Yes or No)
  <br> 4. AlcholDrinking: 음주 여부, 남성의 경우 일주일에 14병 이상의 음주, 여성의 경우 일주일에 7병 이상의 음주 시 Yes에 해당, 불리언 자료형(Yes or No)
  <br> 5. Stroke: 뇌졸중 발병 여부, 불리언 자료형(Yes or No)
  <br> 6. PhysicalHealth: 한 달(30일) 중 신체적으로 아프거나 다친 날이 얼마인가요?, 정수형 자료형
  <br> 7. MentalHealth: 지난 한 달(30일) 중 정신적으로 안 좋은 날이 얼마인가요?, 벙수형 자료형
  <br> 8. DiffWalking: 걷거나 계단을 오르내리는 데에 어려움이 있나요?, 불리언 자료형(Yes or No)
  <br> 9. Sex: 성별, 문자형 자료형(Male, Female)
  <br> 10. AgeCategory: 연령 범주를 14개로 카테고리화, 문자형 자료형(10-74, 65-69, 60-64 등)
  <br> 11. Race: 인종, 문자형 자료형(White, Hispanic, Balck 등)
  <br> 12. Diabetic: 당뇨병 발병 여부, 불리언 자료형(Yes or No)
  <br> 13. PhysicalActivity: 지난 한 달(30일) 중 직업으로 인한 활동을 제외한 운동을 한 적이 있습니까?, 불리언 자료형(Yes or No)
  <br> 14. GenHealth: 전반적으로 자신의 건강이 좋다고 생각하십니까?, 정수형 자료형(Excellent, Very Good, Good, Fair, Poor)
  <br> 15. SleepTime: 평균적으로 하루에 얼마만큼의 수면 시간을 가지시나요?, 정수형 자료형
  <br> 16. Asthma: 천식 발병 여부, 불리언 자료형(Yes or No)
  <br> 17. KidneyDisease: 신장 결석, 방광염, 요실금을 제외한 신장 질환 발병 여부, 불리언 자료형(Yes or No)
  <br> 18. SkinCancer: 피부암 발병 여부, 불리언 자료형(Yes or No)
  <br>
  
# **Ⅲ. Methodology**

  - Explaining your choice of algorithms
  <br> SVM, logistic regression, random forest ...
  - Explaining features

## 데이터 분석을 위한 라이브러리 불러오기
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
```
## 데이터 셋 불러오기
```python
df = pd.read_csv('/content/drive/MyDrive/AIX_deeplearning data/heart_2020_cleaned.csv')
df.head()
```
|index|HeartDisease|BMI|Smoking|AlcoholDrinking|Stroke|PhysicalHealth|MentalHealth|DiffWalking|Sex|AgeCategory|Race|Diabetic|PhysicalActivity|GenHealth|SleepTime|Asthma|KidneyDisease|SkinCancer|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|No|16\.6|Yes|No|No|3\.0|30\.0|No|Female|55-59|White|Yes|Yes|Very good|5\.0|Yes|No|Yes|
|1|No|20\.34|No|No|Yes|0\.0|0\.0|No|Female|80 or older|White|No|Yes|Very good|7\.0|No|No|No|
|2|No|26\.58|Yes|No|No|20\.0|30\.0|No|Male|65-69|White|Yes|Yes|Fair|8\.0|Yes|No|No|
|3|No|24\.21|No|No|No|0\.0|0\.0|No|Female|75-79|White|No|No|Good|6\.0|No|No|Yes|
|4|No|23\.71|No|No|No|28\.0|0\.0|Yes|Female|40-44|White|No|Yes|Very good|8\.0|No|No|No|

## 데이터 셋의 결측치 여부 확인
```python
df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 319795 entries, 0 to 319794
Data columns (total 18 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   HeartDisease      319795 non-null  object 
 1   BMI               319795 non-null  float64
 2   Smoking           319795 non-null  object 
 3   AlcoholDrinking   319795 non-null  object 
 4   Stroke            319795 non-null  object 
 5   PhysicalHealth    319795 non-null  float64
 6   MentalHealth      319795 non-null  float64
 7   DiffWalking       319795 non-null  object 
 8   Sex               319795 non-null  object 
 9   AgeCategory       319795 non-null  object 
 10  Race              319795 non-null  object 
 11  Diabetic          319795 non-null  object 
 12  PhysicalActivity  319795 non-null  object 
 13  GenHealth         319795 non-null  object 
 14  SleepTime         319795 non-null  float64
 15  Asthma            319795 non-null  object 
 16  KidneyDisease     319795 non-null  object 
 17  SkinCancer        319795 non-null  object 
dtypes: float64(4), object(14)
memory usage: 43.9+ MB
```
확인 결과 결측지는 존재하지 않음.

## 데이터 분석

### 심장병 발병 여부
```python
f, ax = plt.subplots(1,2,figsize=(18,8))
df['HeartDisease'].value_counts().plot.pie(explode=[0, 0.1],
                                             autopct='%1.1f%%', ax=ax[0],shadow=True)

ax[0].set_title('PiePlot - HeartDisease', fontsize=20)
ax[0].set_ylabel('')

sns.countplot('HeartDisease', data=df, ax=ax[1])
ax[1].set_title('CountPlot - HeartDisease', fontsize=20)
ax[1].set_xlabel('HeartDisease', fontsize=16)
ax[1].set_ylabel('Count', fontsize=16)
plt.show()
```
![image](https://user-images.githubusercontent.com/116618571/202108277-6fad8ddc-478e-43d8-96ef-8f011650217b.png)

### 성별에 따른 심장병 발병 여부
```python
f, ax = plt.subplots(1,2,figsize=(18,8))

sns.countplot('Sex', data=df, ax=ax[0])
ax[0].set_title('Sex Ratio', fontsize=20)
ax[0].set_ylabel('')
ax[0].set_xlabel('Sex', fontsize=16)

sns.countplot('Sex',hue='HeartDisease', data=df, ax=ax[1])
ax[1].set_title('Sex: HeartDisease and Healthy', fontsize=20)
ax[1].set_ylabel('Count', fontsize=16)
ax[1].set_xlabel('Sex', fontsize=16)
```
```python
m_df=df[df['Sex']=='Male']
fm_df=df[df['Sex']=='Female']
```
```python
f, ax = plt.subplots(1,2,figsize=(18,8))
m_df['HeartDisease'].value_counts().plot.pie(explode=[0, 0.1],
                                             autopct='%1.1f%%', ax=ax[0],shadow=True)

ax[0].set_title('PiePlot - Male\'s HeartDisease', fontsize=20)
ax[0].set_ylabel('')

fm_df['HeartDisease'].value_counts().plot.pie(explode=[0, 0.1],
                                             autopct='%1.1f%%', ax=ax[1],shadow=True)

ax[1].set_title('PiePlot - Female\'s HeartDisease', fontsize=20)
ax[1].set_ylabel('')
```
![image](https://user-images.githubusercontent.com/116618571/202109298-cfe63eb8-c71e-4bb4-bc2c-08abe157c7de.png)


# **Ⅳ. Evaluation & Analysis**
  
  ```python
  # Import libraries
  import pandas as pd
  from sklearn import preprocessing
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score
  import sklearn.svm as svm
  import sklearn.metrics as mt
  from sklearn.linear_model import LogisticRegression
  ```
  ```python
  # Read csv data
  df = pd.read_csv('/content/drive/MyDrive/heart_2020_cleaned.csv')
  print(df.head())
  ```
  |index|HeartDisease|BMI|Smoking|AlcoholDrinking|Stroke|PhysicalHealth|MentalHealth|DiffWalking|Sex|AgeCategory|Race|Diabetic|PhysicalActivity|GenHealth|SleepTime|Asthma|KidneyDisease|SkinCancer|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|No|16\.6|Yes|No|No|3\.0|30\.0|No|Female|55-59|White|Yes|Yes|Very good|5\.0|Yes|No|Yes|
|1|No|20\.34|No|No|Yes|0\.0|0\.0|No|Female|80 or older|White|No|Yes|Very good|7\.0|No|No|No|
|2|No|26\.58|Yes|No|No|20\.0|30\.0|No|Male|65-69|White|Yes|Yes|Fair|8\.0|Yes|No|No|
|3|No|24\.21|No|No|No|0\.0|0\.0|No|Female|75-79|White|No|No|Good|6\.0|No|No|Yes|
|4|No|23\.71|No|No|No|28\.0|0\.0|Yes|Female|40-44|White|No|Yes|Very good|8\.0|No|No|No|

## **- Feature Engineering**

  ```python
  # Extract String feature
  df_string = df[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']]
  df_string.head()
  ```
  |index|BMI|PhysicalHealth|MentalHealth|SleepTime|
|---|---|---|---|---|
|0|16\.6|3\.0|30\.0|5\.0|
|1|20\.34|0\.0|0\.0|7\.0|
|2|26\.58|20\.0|30\.0|8\.0|
|3|24\.21|0\.0|0\.0|6\.0|
|4|23\.71|28\.0|0\.0|8\.0|

  ```python
  # Extract Boolean feature & Switch it to string factor
  df_diabetic = df[['Diabetic']].replace({'Yes (during pregnancy)' : 1, 'No, borderline diabetes' : 0})
  df_bool_1 = df[['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']]
  df_bool = pd.concat([df_bool_1, df_diabetic], axis=1)
  df_bool = df_bool.replace({'No':0, 'Yes':1})
  df_bool.head()
  ```
    
  |index|HeartDisease|Smoking|AlcoholDrinking|Stroke|DiffWalking|PhysicalActivity|Asthma|KidneyDisease|SkinCancer|Diabetic|
|---|---|---|---|---|---|---|---|---|---|---|
|0|0|1|0|0|0|1|1|0|1|1|
|1|0|0|0|1|0|1|0|0|0|0|
|2|0|1|0|0|0|1|1|0|0|1|
|3|0|0|0|0|0|0|0|0|1|0|
|4|0|0|0|0|1|1|0|0|0|0|

  ```python
  # Extract categorical feature & Switch it to string factor
  df_sex = df[['Sex']].replace({'Female':0, 'Male':1})
  df_agecategory = df[['AgeCategory']].replace({'18-24':0, '25-29':1, '30-34':2, '35-39':3, '40-44':4, '45-49':5, '50-54':6, '55-59':7, '60-64':8, '65-69':9, '70-74':10, '75-79':11, '80 or older':12})
  df_race = df[['Race']].replace({'American Indian/Alaskan Native':0, 'Asian':1, 'Black':2, 'Hispanic':3, 'Other':4, 'White':5})
  df_genhealth = df[['GenHealth']].replace({'Excellent':4, 'Very good':3, 'Good':2, 'Fair':1, 'Poor':0})
  df_category = pd.concat([df_sex, df_agecategory, df_race, df_genhealth], axis=1)
  df_category.head()
  ```
  
|index|Sex|AgeCategory|Race|GenHealth|
|---|---|---|---|---|
|0|0|7|5|3|
|1|0|12|5|3|
|2|1|9|5|1|
|3|0|11|5|2|
|4|0|4|5|3|

  ```python
  # Concatenate separated columns into one data frame
  df_modified = pd.concat([df_string, df_bool, df_category], axis=1)
  df_modified.head(10)
  ```
  
  |index|BMI|PhysicalHealth|MentalHealth|SleepTime|HeartDisease|Smoking|AlcoholDrinking|Stroke|DiffWalking|PhysicalActivity|Asthma|KidneyDisease|SkinCancer|Diabetic|Sex|AgeCategory|Race|GenHealth|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|16\.6|3\.0|30\.0|5\.0|0|1|0|0|0|1|1|0|1|1|0|7|5|3|
|1|20\.34|0\.0|0\.0|7\.0|0|0|0|1|0|1|0|0|0|0|0|12|5|3|
|2|26\.58|20\.0|30\.0|8\.0|0|1|0|0|0|1|1|0|0|1|1|9|5|1|
|3|24\.21|0\.0|0\.0|6\.0|0|0|0|0|0|0|0|0|1|0|0|11|5|2|
|4|23\.71|28\.0|0\.0|8\.0|0|0|0|0|1|1|0|0|0|0|0|4|5|3|
|5|28\.87|6\.0|0\.0|12\.0|1|1|0|0|1|0|0|0|0|0|0|11|2|1|
|6|21\.63|15\.0|0\.0|4\.0|0|0|0|0|0|1|1|0|1|0|0|10|5|1|
|7|31\.64|5\.0|0\.0|9\.0|0|1|0|0|1|0|1|0|0|1|0|12|5|2|
|8|26\.45|0\.0|0\.0|5\.0|0|0|0|0|0|0|0|1|0|0|0|12|5|1|
|9|40\.69|0\.0|0\.0|10\.0|0|0|0|0|1|1|0|0|0|0|1|9|5|2|

  ```python
  # Min-Max normalization
  min_max_scaler = preprocessing.MinMaxScaler()
  df_scaled = min_max_scaler.fit_transform(df_modified)
  df_scaled = pd.DataFrame(df_scaled, columns=df_modified.columns)
  df_scaled.head(10)
  ```
   |index|BMI|PhysicalHealth|MentalHealth|SleepTime|HeartDisease|Smoking|AlcoholDrinking|Stroke|DiffWalking|PhysicalActivity|Asthma|KidneyDisease|SkinCancer|Diabetic|Sex|AgeCategory|Race|GenHealth|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|0\.055293975612700746|0\.1|1\.0|0\.17391304347826086|0\.0|1\.0|0\.0|0\.0|0\.0|1\.0|1\.0|0\.0|1\.0|1\.0|0\.0|0\.5833333333333333|1\.0|0\.75|
|1|0\.10044669805625983|0\.0|0\.0|0\.26086956521739135|0\.0|0\.0|0\.0|1\.0|0\.0|1\.0|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|1\.0|0\.75|
|2|0\.17578172159845468|0\.6666666666666666|1\.0|0\.30434782608695654|0\.0|1\.0|0\.0|0\.0|0\.0|1\.0|1\.0|0\.0|0\.0|1\.0|1\.0|0\.75|1\.0|0\.25|
|3|0\.147168900156948|0\.0|0\.0|0\.21739130434782608|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.0|0\.0|0\.9166666666666666|1\.0|0\.5|
|4|0\.14113243993722085|0\.9333333333333333|0\.0|0\.30434782608695654|0\.0|0\.0|0\.0|0\.0|1\.0|1\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.3333333333333333|1\.0|0\.75|
|5|0\.20342870940480506|0\.2|0\.0|0\.4782608695652174|1\.0|1\.0|0\.0|0\.0|1\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.9166666666666666|0\.4|0\.25|
|6|0\.11602076542315587|0\.5|0\.0|0\.13043478260869565|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|1\.0|0\.0|1\.0|0\.0|0\.0|0\.8333333333333333|1\.0|0\.25|
|7|0\.23687069902209346|0\.16666666666666666|0\.0|0\.34782608695652173|0\.0|1\.0|0\.0|0\.0|1\.0|0\.0|1\.0|0\.0|0\.0|1\.0|0\.0|1\.0|1\.0|0\.5|
|8|0\.17421224194132562|0\.0|0\.0|0\.17391304347826086|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.0|0\.0|0\.0|1\.0|1\.0|0\.25|
|9|0\.34613062899915487|0\.0|0\.0|0\.3913043478260869|0\.0|0\.0|0\.0|0\.0|1\.0|1\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.75|1\.0|0\.5|

  ```python
  # Correlation heatmap
  plt.figure(figsize=(12, 12))
  sns.heatmap(data=df_scaled.corr(), annot=True, fmt='.2f', linewidth=0.5, cmap='Blues')
  ```
  ![image](https://user-images.githubusercontent.com/116618556/202103973-f6103b27-c795-4aa7-836a-2ed9bf3455e8.png)

  <br>Corrleation coefficient의 절댓값이 0.1 이상인 값들은 HeartDiease와 상대적으로 강한 선형관계를 가지므로 이 feature들만 사용하는 경우와 모든 feature를 다 사용하는 경우의 정확도에 대한 비교를 진행하고자 한다. 
  
  ## **- Train/Test split**
  
  ```python
  df_yes = df_scaled[df['HeartDisease']=='Yes']
  df_no = df_scaled[df['HeartDisease']=='No']
  df_yes_train = df_yes.iloc[0:21898]
  df_yes_test = df_yes.iloc[21898:]
  df_no_train = df_no.iloc[0:233938]
  df_no_test = df_no.iloc[233938:]
  df_train = pd.concat([df_yes_train, df_no_train]).sample(frac=1).reset_index(drop=True)
  df_test = pd.concat([df_yes_test, df_no_test]).sample(frac=1).reset_index(drop=True)
  ```
  
  
  - Graphs, tables, any statistics
  
# **Ⅴ. Related Work**
   <br>> http://www.samsunghospital.com/dept/main/index.do?DP_CODE=XB301&MENU_ID=001002 (심장질환 예방)
   <br>> https://www.nhis.or.kr/magazin/mobile/201411/sub02_02.html
   <br>> https://www.korea.kr/news/healthView.do?newsId=148896724
   <br>> https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease (데이터셋)
   <br>> https://computer-science-student.tistory.com/113
   <br>> https://www.kaggle.com/code/mrisdal/exploring-survival-on-the-titanic
   <br>> https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks (Github Readme code block)
   
# **Ⅵ. Conclusion: Discussion**
