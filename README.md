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
  
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
```
```python
df = pd.read_csv('/content/drive/MyDrive/AIX_deeplearning data/heart_2020_cleaned.csv')
df.head()
```

# **Ⅳ. Evaluation & Analysis**
  
  ```python
  #Import libraries
  import pandas as pd
  from sklearn import preprocessing
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score
  from sklearn.svm import SVC
  import sklearn.metrics as mt
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import confusion_matrix
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
<br>Corrleation coefficient의 절댓값이 0.1 이상인 총 9개의 feature들(PhysicalHealth, Smoking, Stroke, DiffWalking, PhysicalActivity, KidneyDisease, Diabetic, AgeCategory, GenHealth)은 타 feature들에 비해 HeartDiease와 상대적으로 강한 선형관계를 가지므로 이 feature들만 사용하는 경우와 모든 feature를 다 사용하는 경우의 정확도에 대한 비교를 진행하고자 한다. 
  
  ## **- Train/Test split**
  
  ```python
  # Train/Test split
  df_yes = df_scaled[df['HeartDisease']=='Yes']
  df_no = df_scaled[df['HeartDisease']=='No']
  df_yes_train = df_yes.iloc[0:21898]
  df_yes_test = df_yes.iloc[21898:]
  df_no_train = df_no.iloc[0:233938]
  df_no_test = df_no.iloc[233938:]
  df_train = pd.concat([df_yes_train, df_no_train]).sample(frac=1).reset_index(drop=True)
  df_test = pd.concat([df_yes_test, df_no_test]).sample(frac=1).reset_index(drop=True)
  ```  
총 319,795개의 HeartDisease(심장병 발병) 데이터 중 Yes(발병)에 해당하는 데이터는 27,373개로 약 8.56% 정도로 적은 비율을 가지므로 학습/시험 데이터를 랜덤하게 분리하게 될 경우 Yes(발병)에 해당하는 데이터가 학습 데이터셋에 너무 적게 들어갈 우려가 있다고 판단하였다. 따라서 이번 프로젝트에서는 Yes(발병) 및 No(발병 X)에 해당하는 데이터를 각각 8/2 비율로 분리하고 이 비율을 유지하며 다시 합치는 과정을 통해 훈련/시험 데이터셋을 구성하였다. 이를 통해 데이터 불균형으로 인한 문제를 방지하고자 한다.

  ```python
  # train dataset check
  df_train.head(5)
  ```  
  |index|BMI|PhysicalHealth|MentalHealth|SleepTime|HeartDisease|Smoking|AlcoholDrinking|Stroke|DiffWalking|PhysicalActivity|Asthma|KidneyDisease|SkinCancer|Diabetic|Sex|AgeCategory|Race|GenHealth|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|0\.15368827719425332|0\.06666666666666667|0\.0|0\.34782608695652173|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.5|1\.0|0\.75|
|1|0\.27381383556682365|0\.0|0\.0|0\.30434782608695654|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|1\.0|0\.3333333333333333|1\.0|0\.5|
|2|0\.15694796571290598|0\.0|0\.0|0\.21739130434782608|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.5|1\.0|0\.75|
|3|0\.10817336713751058|0\.0|0\.0|0\.30434782608695654|0\.0|1\.0|0\.0|0\.0|0\.0|1\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.08333333333333333|1\.0|0\.25|
|4|0\.27465893999758545|0\.0|0\.0|0\.30434782608695654|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.0|0\.8|0\.5|
  
  ```python
  # test dataset check
  df_test.head(5)
  ```
  |index|BMI|PhysicalHealth|MentalHealth|SleepTime|HeartDisease|Smoking|AlcoholDrinking|Stroke|DiffWalking|PhysicalActivity|Asthma|KidneyDisease|SkinCancer|Diabetic|Sex|AgeCategory|Race|GenHealth|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|0\.24508028492092238|0\.0|0\.0|0\.21739130434782608|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.6666666666666666|1\.0|0\.75|
|1|0\.19703006157189423|0\.0|0\.1|0\.26086956521739135|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.5|0\.8|0\.75|
|2|0\.2113968368948449|0\.0|0\.0|0\.3913043478260869|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.5|0\.6000000000000001|0\.5|
|3|0\.1574308825304841|0\.0|0\.4|0\.30434782608695654|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.0|0\.0|1\.0|0\.0|0\.0|0\.8333333333333333|1\.0|1\.0|
|4|0\.15429192321622603|0\.0|0\.06666666666666667|0\.17391304347826086|0\.0|0\.0|0\.0|0\.0|0\.0|1\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.75|1\.0|0\.5|

이를 통해 255,836개의 train dataset과 63,959개의 test dataset을 구축한다.

  ## **- Feature Selection(correlation coefficient가 0.1이상인 9개의 feature에 대해서)**
  ```python
  # Extract 9 features
  train_x = df_train[['PhysicalHealth', 'Smoking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'KidneyDisease', 'Diabetic', 'AgeCategory', 'GenHealth']]
  train_y = df_train[['HeartDisease']]
  test_x = df_test[['PhysicalHealth', 'Smoking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'KidneyDisease', 'Diabetic', 'AgeCategory', 'GenHealth']]
  test_y = df_test[['HeartDisease']]
  print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
  ```
  
  ## **- Model training with selected features**
  ```python
  # RandomForest with selected features
  rf_model = RandomForestClassifier(n_estimators=500, random_state=0)
  rf_model.fit(train_x, train_y)
  score = rf_model.score(test_x, test_y)
  print(score)
  ```
  
  ```python
  # Confusion matrix for RandomForest model
  cm = pd.DataFrame(confusion_matrix(test_y, y_pred), columns=test_y, index=test_y)
  sns.heatmap(cm, annot=True)
  ```
  
  
  
  Random forest model에 선별된 feature들을 넣고 예측했을 때, **91.06%** 의 정확도로 예측하는 것을 확인할 수 있다.
  
  ```python
  
  
  ```python
  # Logistic Regression with selected features
  lr_model = LogisticRegression(random_state=0)
  lr_model.fit(train_x, train_y)
  score = lr_model.score(test_x, test_y.values.ravel())
  print(score*100)
  ```
  
  선별된 feature로 Logistic regression을 수행했을 때, **91.56%** 의 정확도로 예측하는 것을 확인할 수 있다.
  
  ## **- Select all features to compare**
  ```python
  train_x = df_train[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Diabetic', 'Sex', 'AgeCategory', 'Race', 'GenHealth']]
  train_y = df_train[['HeartDisease']]
  test_x = df_test[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Diabetic', 'Sex', 'AgeCategory', 'Race', 'GenHealth']]
  test_y = df_test[['HeartDisease']]
  print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
  ```
  
  ## **- Model training with all features**
  ```python
  # RandomForest with all features
  rf_model = RandomForestClassifier(n_estimators=500, random_state=0)
  rf_model.fit(train_x, train_y)
  score = rf_model.score(test_x, test_y)
  print(score*100)
  ```
  Random forest model에 모든 feature들을 넣고 예측했을 때, **90.67%** 의 정확도로 예측하는 것을 확인할 수 있다.
  
  ```python
  # Logistic Regression with all features
  lr_model = LogisticRegression(random_state=0)
  lr_model.fit(train_x, train_y)
  score = lr_model.score(test_x, test_y.values.ravel())
  print(score*100)
  ```
  모든 feature로 Logistic regression을 수행했을 때, **91.60%** 의 정확도로 예측하는 것을 확인할 수 있다.
  
# **Ⅴ. Related Work**
   <br>> http://www.samsunghospital.com/dept/main/index.do?DP_CODE=XB301&MENU_ID=001002 (심장질환 예방)
   <br>> https://www.nhis.or.kr/magazin/mobile/201411/sub02_02.html
   <br>> https://www.korea.kr/news/healthView.do?newsId=148896724 (심장질환)
   <br>> https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease (데이터셋)
   <br>> https://computer-science-student.tistory.com/113 (Titanic 생존자 예측)
   <br>> https://www.kaggle.com/code/mrisdal/exploring-survival-on-the-titanic (Exporling Survival on the Titanic)
   <br>> https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks (Github Readme code block)
   <br>> https://partrita.github.io/posts/random-forest-python/ (python random forest analysis)
   <br>> https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a (Logistric Regression)
   <br>> https://www.kaggle.com/code/prashant111/svm-classifier-tutorial/notebook (Support Vector Machine)
   <br>>
   
# **Ⅵ. Conclusion: Discussion**
