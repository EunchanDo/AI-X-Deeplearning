# AI-X-Deeplearning

# **Title**: 머신러닝 기법을 활용한 심장 질환 발병 예측

### **Members**:
도은찬, 서울 기계공학부, dec1995@hanyang.ac.kr
<br>정다연, 서울 기계공학부, jdd2635@hanyang.ac.kr

<br> **YouTube Link:**  
         
# **Ⅰ. Proposal(Option A)**

  **- Motivation:**
  <br> 심장질환은 국내 사망원인 1위 질환인 암 다음으로 우리의 목숨을 위협하는 가장 무서운 질병이다. 이는 돌연사의 원인이 되기도 하며 고령 환자의 주요 사망 원인이다. 국내에서는 매년 20만명이 넘는 환자들이 심부전으로 병원을 찾는다. 우리나라 뿐만 아니라 다른 여러 국가에서도 사망 원인이 되는 심각한 질환이고 특히 미국 내에서는 전 인종이 영향을 받은 주된 사망원인 중 하나이다. 미국 CDC의 통계자료에 따르면 미국인들의 절반이 심장병의 세 가지 주요 원인인 고혈압, 고 콜레스테롤, 흡연 중 적어도 한가지를 가지고 있다고 한다. 이 외에도 심장질환을 발병시킬 수 있는 요인에는 육체 및 정신의 건강한 정도, 뇌졸중 경험 여부 그리고 음주 등이 있다. 따라서 이러한 원인들을 분석하여 심장병 발생 여부를 예측하는 것은 인류의 중대한 과업이라고 할 수 있다. 그러한 과업에 동참하기위해 우리 팀은 미국 CDC의 방대한 심장 질환 지표 데이터를 활용하여 그것을 예측할 수 있는 모델을 만들고 어떤 모델이 최적의 결과를 제시할 수 있는지 분석할 것이다. 
  
  **- What do you want to see at the end?**
  <br> 심장 질환은 예방 가능성이 높은 질환이다. 위에서 언급한 것처럼 주로 흡연, 운동 부족, 비만 등이 주요 원인이며, 이를 약물치료나 생활 습관의 교정함으로서 심장질환 발병을 예방할 수 있다. 따라서 환자의 성별, 나이, BMI 지수, 흡연 여부 등을 특징인자로 하는 머신러닝 기법들을 적용하여 심장질환 발생 가능성을 예측하고 사망률이 높은 질병 중 하나인 심장질환을 예측하고 이러한 환자들에게 약물치료, 생활 습관 교정 등의 예방법을 제시하여 심장질환으로 인한 사망률을 낮출 수 있는 모델을 제시하고자 한다.
  
# **Ⅱ. Datasets**

  ### **- https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease (데이터셋 출처)**
  
  ### **- Describing dataset**
  <br> ![캡처](https://user-images.githubusercontent.com/116618556/199441411-adf3fa21-bd9d-46f7-aa4d-a78fad62c8e8.JPG)
  
  <br> 본 데이터셋은 미국 질병통제예방센터(CDC)가 2020년에 분석한 성인 약 40만명의 건강 지표로 이루어져 있다. 총 18개의 변수(9개의 불리언, 5개의 문자열, 4개의 정수형)로 이루어져 있으며, 각 feature별 자세한 내용은 하기와 같다.
  <br>
  <br> 1. HeartDisease: 심장질환 발병 여부, 불리언 자료형(Yes or No)
  <br> 2. BMI: Body Mass Index(BMI, 체질량지수), 정수형 자료형
  <br> 3. Smoking: 흡연 여부, 일생동안 100개비 이상 흡연하였으면 Yes에 해당, 불리언 자료형(Yes or No)
  <br> 4. AlcholDrinking: 음주 여부, 남성의 경우 일주일에 14병 이상의 음주, 여성의 경우 일주일에 7병 이상의 음주 시 Yes에 해당, 불리언 자료형(Yes or No)
  <br> 5. Stroke: 뇌졸중 발병 여부, 불리언 자료형(Yes or No)
  <br> 6. PhysicalHealth: 한 달(30일) 중 신체적으로 아프거나 다친 날이 얼마인가요?, 정수형 자료형
  <br> 7. MentalHealth: 지난 한 달(30일) 중 정신적으로 안 좋은 날이 얼마인가요?, 정수형 자료형
  <br> 8. DiffWalking: 걷거나 계단을 오르내리는 데에 어려움이 있나요?, 불리언 자료형(Yes or No)
  <br> 9. Sex: 성별, 범주형 자료형('Male', 'Female')
  <br> 10. AgeCategory: 연령 범주를 14개로 카테고리화, 범주형 자료형('18-24', '25-29', '30-34', '35-39' 등)
  <br> 11. Race: 인종, 범주형 자료형('White', 'Hispanic', 'Black' 등)
  <br> 12. Diabetic: 당뇨병 발병 여부, 불리언 자료형(Yes or No)
  <br> 13. PhysicalActivity: 지난 한 달(30일) 중 직업으로 인한 활동을 제외한 운동을 한 적이 있습니까?, 불리언 자료형(Yes or No)
  <br> 14. GenHealth: 전반적으로 자신의 건강이 좋다고 생각하십니까?, 범주형 자료형('Excellent', 'Very Good', 'Good', 'Fair', 'Poor')
  <br> 15. SleepTime: 평균적으로 하루에 얼마만큼의 수면 시간을 가지시나요?, 정수형 자료형
  <br> 16. Asthma: 천식 발병 여부, 불리언 자료형(Yes or No)
  <br> 17. KidneyDisease: 신장 결석, 방광염, 요실금을 제외한 신장 질환 발병 여부, 불리언 자료형(Yes or No)
  <br> 18. SkinCancer: 피부암 발병 여부, 불리언 자료형(Yes or No)
  
# **Ⅲ. Methodology**

## **Explaining your choice of algorithms**

### **Random Forest**
<br> 랜덤 포레스트(Random Forest)란 배깅(Bagging)과 의사결정나무 모델을 결합한 앙상블 모델이다[12]. 배깅은 부트스트랩(Bootstrap) 샘플링을 이용하여 주어진 하나의 데이터로 학습된 예측 모델보다 더 정확도가 높은 모델을 만들 수 있는 방법이다[13]. 아래의 그림은 배깅을 나타내는 그림이다.
![image](https://user-images.githubusercontent.com/116618571/203448843-ec316d2e-c1f7-4762-8830-72d369499837.png)
위의 그림에서 알 수 있듯이, L이라는 기존 학습 데이터가 있다면 L에 대해 부트스트랩 샘플링을 진행하여 데이터셋을 L1,L2 ... LB의 서브 데이터셋으로 나눈다. 그러면 나누어진 데이터 셋에 대해 학습한 모델인 $\phi(x,L1)$, $\phi(x,L2)$, ..., $\phi(x,LB)$를 얻게 되고 그 모델들의 결과를 종합하여 예측 정확도를 향상시키는 앙상블 방법이다. 의사결정나무 모델은 데이터를 분석하여 이들 사이에 존재하는 패턴을 예측 가능한 규칙(질문)들을 이용하여 만든 기계학습 모델 중 하나이다[14]. 만약 배깅 과정에서 각 서브 데이터 셋에 대한 학습 모델을 의사결정나무로 설정한다면 랜덤포레스트 모델이 되는 것이다. 요약하자면, 랜덤 포레스트 모델은 의사결정나무 모델의 장점을 취하고 배깅이라는 앙상블 기법을 활용하여 주어진 데이터 셋의 한계를 극복한 참신한 모델이다. 랜덤포레스트의 장점은 높은 정확도, 빠른 학습 속도 그리고 과적합의 감소 등이 있다. 

### **Logistic Regression**
<br> 로지스틱 회귀(Logistic Regression)란 독립 변수의 선형 결합을 이용하여 발생 가능성을 예측하는데 사용되는 통계 기법으로 데이터가 어떤 범주에 속할 확률을 0에서 1 사이의 값으로 예측하고 그 확률에 따라 이진 분류를 수행하는 지도 학습 알고리즘이다. 선형 회귀(Linear Regression)에서 발전된 기법으로 선형 회귀 모델은 연속 종속 변수와 독립 변수 간의 관계를 설명하는데 사용되는 모델인데 비해 로지스틱 회귀 모델은 범주형 종속 변수에 대한 예측을 수행하는 모델이라는 차이점이 있다. 이번 프로젝트에서 사용하는 로지스틱 회귀 모델은 9개, 17개의 feature로 심장 질환 발병을 예측하므로 여러 개의 독립 변수로 범주형 종속 변수를 예측하는 다항 로지스틱 회귀 모델이다. 로지스틱 회귀에서는 시그모이드 함수를 사용하는데, 시그모이드 함수의 형태는 아래와 같으며, 이는 독립 변수가 (-∞, ∞)의 어느 숫자이든 상관 없이 종속 변수 또는 결과 값이 항상 0과 1사이에 있도록 변환해준다. 시그모이드 함수를 사용하면 x값이 작을 때의 예측값은 0에 수렴하며, x값이 클 때의 예측값은 1에 수렴한다. 따라서 예측 값을 0과 1 사이의 값으로 추출할 수 있으며, 임계값(threshold)(기본적으로 0.5로 설정)을 넘을 경우를 1로, 넘지 않을 경우를 0으로 분류하는 이진 분류 수행 지도 학습 알고리즘이다. 
<br>
![image](https://user-images.githubusercontent.com/116618556/203255558-83cad801-0b1e-43bf-945f-b81e862cd437.png)
<br> Logistic regression은 Anomaly detection, Disease prediction 등에 사용하는 모델로 심장 질환 발병을 예측하는 이번 프로젝트에 적합한 모델이라고 생각되어 채택하게 되었다.

## **Explaining features**

## 데이터 불러오기
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
|index|HeartDisease|BMI|Smoking|AlcoholDrinking|Stroke|PhysicalHealth|MentalHealth|DiffWalking|Sex|AgeCategory|Race|Diabetic|PhysicalActivity|GenHealth|SleepTime|Asthma|KidneyDisease|SkinCancer|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|No|16\.6|Yes|No|No|3\.0|30\.0|No|Female|55-59|White|Yes|Yes|Very good|5\.0|Yes|No|Yes|
|1|No|20\.34|No|No|Yes|0\.0|0\.0|No|Female|80 or older|White|No|Yes|Very good|7\.0|No|No|No|
|2|No|26\.58|Yes|No|No|20\.0|30\.0|No|Male|65-69|White|Yes|Yes|Fair|8\.0|Yes|No|No|
|3|No|24\.21|No|No|No|0\.0|0\.0|No|Female|75-79|White|No|No|Good|6\.0|No|No|Yes|
|4|No|23\.71|No|No|No|28\.0|0\.0|Yes|Female|40-44|White|No|Yes|Very good|8\.0|No|No|No|
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
df.info() 함수를 활용하여 피쳐들의 결측치 여부를 확인한 결과 결측치는 존재하지 않음.

## 심장병 발병률 확인하기
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
![image](https://user-images.githubusercontent.com/116618571/203219636-e60a2b1b-50b4-4adc-ad96-a67003169c81.png)

위 그림은 전체 데이터 셋에대한 심장병 발병률과 발병 환자 수를 pieplot과 countplot으로 나타낸 그림이다. 

## 피쳐 분석
모델 학습 시 심장병 발병에 주로 영향을 주는 피쳐만 사용하면 모델 예측 정확도가 증가할 수 있고 학습 시간 또한 감소할 수 있다. 따라서 전체 피쳐들중, 3가지 피쳐를 선택해 분석하여 어떤 피쳐가 심장병 발병에 유의미한 영향을 미치는지 분석할 것이고 이는 이후에 있을 피쳐 선택에 도움을 줄 것이다.  

### GenHealth(건강에 대한 자신의 생각) 피쳐에 따른 심장병 발생 여부 분석 
GenHealth 피쳐는 총 5개의 범주로 구성돼 있는데, 다음과 같다: Excellent, Very good, Good, Fair, Poor.
```python
E=df[df['GenHealth']=='Excellent']
Vg=df[df['GenHealth']=='Very good']
G=df[df['GenHealth']=='Good']
F=df[df['GenHealth']=='Fair']
P=df[df['GenHealth']=='Poor']

f, ax = plt.subplots(1,5,figsize=(18,8))
E['HeartDisease'].value_counts().plot.pie(explode=[0, 0.1],
                                             autopct='%1.1f%%', ax=ax[0],shadow=True)

ax[0].set_title('Excellent', fontsize=10)
ax[0].set_ylabel('')

Vg['HeartDisease'].value_counts().plot.pie(explode=[0, 0.1],
                                             autopct='%1.1f%%', ax=ax[1],shadow=True)

ax[1].set_title('Very good', fontsize=10)
ax[1].set_ylabel('')

G['HeartDisease'].value_counts().plot.pie(explode=[0, 0.1],
                                             autopct='%1.1f%%', ax=ax[2],shadow=True)

ax[2].set_title('Good', fontsize=10)
ax[2].set_ylabel('')
F['HeartDisease'].value_counts().plot.pie(explode=[0, 0.1],
                                             autopct='%1.1f%%', ax=ax[3],shadow=True)

ax[3].set_title('Fair', fontsize=10)
ax[3].set_ylabel('')

P['HeartDisease'].value_counts().plot.pie(explode=[0, 0.1],
                                             autopct='%1.1f%%', ax=ax[4],shadow=True)

ax[4].set_title('Poor', fontsize=10)
ax[4].set_ylabel('')
```
![image](https://user-images.githubusercontent.com/116618571/203222334-365b282f-5a77-48de-8506-7b694f727cbf.png)
자신의 건강이 안좋다고 생각할 수록 심장병 발병률이 증가함을 알 수 있다. 

### AgeCategory(연령 범주) 피쳐에 따른 심장병 발생 여부 분석 
AgeCategory 피쳐는 18세부터 80세 이상까지의 연령대를 13개의 연령 구간으로 나눈 피쳐이다. 즉, 총 13개의 범주를 가진 피쳐이다.
```python
#연령순으로 데이터 정렬
age = dict(df['AgeCategory'].value_counts())
sorted_age = sorted(age.items())
```
```python
# AgeCategory피쳐와 심장병 여부 데이터 기준으로 grouping 진행
groups_age = df.groupby(['HeartDisease','AgeCategory'])

Yes_age= dict(groups_age.size()[13:])
sorted_Yes_age = sorted(Yes_age.items())
```
```python
# 심장병 발생률을 연령대를 기준으로 plot
ratio_age=[]
x_age=[]
for i in range(13):
  ratio_age.append(sorted_Yes_age[i][1]/sorted_age[i][1])
  x_age.append(sorted_age[i][0])

plt.figure(figsize=(18,8))
plt.plot(x_age,ratio_age,'-x')
plt.grid(True)
plt.ylim([0,1])
plt.xlabel('Age', fontsize=16)
plt.ylabel('Ratio', fontsize=16)
plt.title('Rate of heart disease by age', fontsize=20)
```
![image](https://user-images.githubusercontent.com/116618571/203223226-35fd0a6b-334d-4b01-aa5f-6c601fd4c4cf.png)

심장병 발생률이 연령대가 증가할 수록 확실히 커짐을 확인할 수 있다. 따라서 GenHealth 그리고 AgeCategory 피쳐는 모델 학습 시 포함시키면 정확도 향상에 영향을 줄 것이다.

### Race(인종) 피쳐에 따른 심장병 발생 여부 분석 
Race 피쳐는 6개의 범주로 구성돼 있다. 각각 American indian/Alaskan native, Asian, Black, Hispanic, white, Other이다.
```python
# 인종과 심장병 피쳐로 grouping
groups = df.groupby(['HeartDisease','Race'])
```
```python
# 인종을 알파벳 순으로 정렬
race = dict(df['Race'].value_counts())
sorted_race = sorted(race.items())
```
```python
# 심장병 발생을 알파벳 순으로 정렬
Yes = dict(groups.size()[6:])
sorted_Yes = sorted(Yes.items())
```
```python
# 인종별 심장병 발생률 plot
ratio=[]
x=[]
for i in range(6):
  ratio.append(sorted_Yes[i][1]/sorted_race[i][1])
  x.append(sorted_race[i][0])
plt.figure(figsize=(18,8))
plt.plot(x,ratio,'-x')
plt.grid(True)
plt.ylim([0,1])
plt.xlabel('Race', fontsize=16)
plt.ylabel('Ratio', fontsize=16)
plt.title('Rate of heart disease by race', fontsize=20)
```
![image](https://user-images.githubusercontent.com/116618571/203225511-bc4ae650-9ad4-4cfc-8201-7b47be1941a7.png)
인종별 심장병 발생률 그래프를 통해 심장병 발생 여부는 인종과는 크게 관련이 없음을 확인할 수 있다. 따라서 Race 피쳐는 학습 시 무시해도 될만한 피쳐라고 판단할 수 있다.

## 요약
위의 데이터 분석과정을 통해 심장병 발생에 영향을 주는 피쳐는 GenHealth, AgeCategory이고 영향을 주지 않는 피쳐는 Race임을 알게됐다. 모델 학습 시 전자의 피쳐를 사용하고 후자의 피쳐를 생략하면 모델 예측 정확도가 더욱 증가할 수 있을 것이다. 그런데, 지금까지는 총 17개의 피쳐 중 3가지 피쳐만 분석하여 어떤 피쳐를 학습 데이터로 이용할지 판단했다. 하지만, 전체 피쳐에 대한 분석이 필요하므로, 차후에 Confusion matrix를 통해 상관계수를 계산하여 0.1 이상의 피쳐만을 사용하여 모델을 학습할 것이다. 위의 GenHealth와 AgeCategory 피쳐는 전부 상관계수의 크기가 0.1 이상의 피쳐들임을 알 수 있을것이다.  
# **Ⅳ. Evaluation & Analysis**
## **- Import Libraries**
  
  ```python
  #Import Libraries
  import pandas as pd
  import numpy as np
  from sklearn import preprocessing
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score
  from sklearn.model_selection import train_test_split
  import sklearn.metrics as mt
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import confusion_matrix
  ```
  ## **- Data Load**
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

## **- Feature Engineering**

  ## **- Extract String feature**
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
  
  ## **- Extract Boolean feature**
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
  
  ## **- Extract Categorical feature**
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

  ## **- Concatenate into one data frame**
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

  ## **- Min-Max Normalization**
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
  
  ## **- Correlation coefficient heatmap**
  ```python
  # Correlation heatmap
  plt.figure(figsize=(12, 12))
  sns.heatmap(data=df_scaled.corr(), annot=True, fmt='.2f', linewidth=0.5, cmap='Blues')
  ```
  ![image](https://user-images.githubusercontent.com/116618556/202103973-f6103b27-c795-4aa7-836a-2ed9bf3455e8.png)
<br>Corrleation coefficient의 절댓값이 0.1 이상인 총 9개의 feature들(PhysicalHealth, Smoking, Stroke, DiffWalking, PhysicalActivity, KidneyDisease, Diabetic, AgeCategory, GenHealth)은 타 feature들에 비해 HeartDiease와 상대적으로 강한 선형관계를 가지므로 이 feature들만 사용하는 경우와 모든 feature를 다 사용하는 경우의 정확도에 대한 비교를 진행하고자 한다. 
  
  ## **- Train/Test split with selected features**
  
  ```python
  # Train/Test split with selected features
  selected_features = df_scaled[['PhysicalHealth', 'Smoking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'KidneyDisease', 'Diabetic', 'AgeCategory', 'GenHealth']]
  label = df_scaled['HeartDisease']
  x_train, x_test, y_train, y_test = train_test_split(selected_features, label, test_size=0.2)
  print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
  ```
  
  ## **- Random Forest with selected features**
  ```python
  # RandomForest with selected features
  rf_model = RandomForestClassifier(n_estimators=500, random_state=0)
  rf_model.fit(x_train, y_train)
  y_pred = rf_model.predict(x_test)
  score = accuracy_score(y_test, y_pred)
  print(score*100)
  ```
  <br>다음과 같이 **selected features만 사용** 하여 **Random Forest** 로 예측할 경우, **91.08%** 의 예측 정확도를 나타내는 것을 알 수 있다.
  
  ```python
  # Confusion matrix for RandomForest model
  cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['HeartDisease=no', 'HeartDisease=yes'], index=['HeartDisease=no', 'HeartDisease=yes'])
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=2).set_title('Random Forest w/ selected features', fontsize=15)
  plt.xlabel('Prediction', fontsize=13)
  plt.ylabel('Ground Truth', fontsize=13)
  ```
  ![image](https://user-images.githubusercontent.com/116618556/203228390-1bf606a6-37b6-41cf-b549-7a7273e78c6b.png)
  
  <br> selected feature만 사용하여 Random Forest로 예측할 경우의 confusion matrix.
  
  ## **- Logistic Regression with selected features**
  ```python
  # Logistic Regression with selected features
  lr_model = LogisticRegression(random_state=0)
  lr_model.fit(x_train, y_train)
  y_pred = lr_model.predict(x_test)
  score = accuracy_score(y_test, y_pred)
  print(score*100)
  ```
  <br> 다음과 같이 **selected features만 사용** 하여 **Logistic Regression** 으로 예측할 경우, **91.64%** 의 예측 정확도를 나타내는 것을 알 수 있다.
  
  ```python  
  # Confusion matrix for Logistic Regression model
  cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['HeartDisease=no', 'HeartDisease=yes'], index=['HeartDisease=no', 'HeartDisease=yes'])
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=2).set_title('Logistic Regression w/ selected features', fontsize=15)
  plt.xlabel('Prediction', fontsize=13)
  plt.ylabel('Ground Truth', fontsize=13)
  ```
  ![image](https://user-images.githubusercontent.com/116618556/203232074-4690016d-1368-42a5-9b7a-e3bc8b663a33.png)
  
  <br> selected feature만 사용하여 Logistic Regression으로 예측할 경우의 confusion matrix.  
  
  ## **- Train/Test split with all features**
  
  ```python
  # Train/Test split with all features
  all_features = df_scaled[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Diabetic', 'Sex', 'AgeCategory', 'Race', 'GenHealth']]
  label = df_scaled['HeartDisease']
  x_train, x_test, y_train, y_test = train_test_split(all_features, label, test_size=0.2)
  print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
  ```
  ## **- Random Forest with all features**
  ```python
  # RandomForest with all features
  rf_model = RandomForestClassifier(n_estimators=500, random_state=0)
  rf_model.fit(x_train, y_train)
  y_pred = rf_model.predict(x_test)
  score = accuracy_score(y_test, y_pred)
  print(score*100)
  ```
  <br> 다음과 같이 **feature를 전부 사용** 하여 **Random Forest** 로 예측할 경우, **90.66%** 의 예측 정확도를 나타내는 것을 알 수 있다.
  
  ```python
  # Confusion matrix for RandomForest model
  cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['HeartDisease=no', 'HeartDisease=yes'], index=['HeartDisease=no', 'HeartDisease=yes'])
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=2).set_title('Random Forest w/ all features', fontsize=15)
  plt.xlabel('Prediction', fontsize=13)
  plt.ylabel('Ground Truth', fontsize=13)
  ```
  ![image](https://user-images.githubusercontent.com/116618556/203232181-7cb6fb04-4dae-43e8-82c4-868c0c695a66.png)
  
  <br> feature를 전부 사용하여 Random Forest로 예측할 경우의 confusion matrix.

  ## **- Logistic Regression with all features**
  ```python
  # Logistic Regression with all features
  lr_model = LogisticRegression(random_state=0)
  lr_model.fit(x_train, y_train)
  y_pred = lr_model.predict(x_test)
  score = accuracy_score(y_test, y_pred)
  print(score*100)
  ```
  <br> 다음과 같이 **features를 전부 사용** 하여 **Logistic Regression** 으로 예측할 경우, **91.53%** 의 예측 정확도를 나타내는 것을 알 수 있다.
  
  ```python
  # Confusion matrix for Logistic Regression model
  cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['HeartDisease=no', 'HeartDisease=yes'], index=['HeartDisease=no', 'HeartDisease=yes'])
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=2).set_title('Logistic Regression w/ all features', fontsize=15)
  plt.xlabel('Prediction', fontsize=13)
  plt.ylabel('Ground Truth', fontsize=13)
  ```
  ![image](https://user-images.githubusercontent.com/116618556/203233166-df6cc2d0-8fb3-4f8e-bd4c-af62628ad425.png)
  
  <br> feature를 전부 사용하여 Logistic Regression으로 예측할 경우의 confusion matrix.
  
# **Ⅴ. Related Work**
   <br>> [1]http://www.samsunghospital.com/dept/main/index.do?DP_CODE=XB301&MENU_ID=001002 (심장질환 예방)
   <br>> [2]https://www.nhis.or.kr/magazin/mobile/201411/sub02_02.html
   <br>> [3]https://www.korea.kr/news/healthView.do?newsId=148896724 (심장질환)
   <br>> [4]https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease (데이터셋)
   <br>> [5]https://computer-science-student.tistory.com/113 (Titanic 생존자 예측)
   <br>> [6]https://www.kaggle.com/code/mrisdal/exploring-survival-on-the-titanic (Exploring Survival on the Titanic)
   <br>> [7]https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks (Github Readme code block)
   <br>> [8]https://partrita.github.io/posts/random-forest-python/ (python random forest analysis)
   <br>> [9]https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a (Logistric Regression)
   <br>> [10]https://www.ibm.com/topics/logistic-regression#anchor--1020983554 (Logistic Regression)
   <br>> [11]https://ko.wikipedia.org/wiki/%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1_%ED%9A%8C%EA%B7%80 (Logistic Regression)
   <br>> [12]https://zephyrus1111.tistory.com/249
   <br>> [13]https://zephyrus1111.tistory.com/245
   <br>> [14]https://ratsgo.github.io/machine%20learning/2017/03/26/tree/
# **Ⅵ. Conclusion: Discussion**
  <br> 이번 프로젝트에서는 머신러닝 기법을 활용한 심장 질환 예측에 대해 selected feature만을 사용하는 경우와 모든 feature를 사용하는 경우의 정확도 비교를 진행하였다. 이때 심장질환 발병(HeartDisease)과 17개의 feature들 사이의 피어슨 상관 계수를 계산하여 절댓값이 0.1 이상인 9개의 feature(PhysicalHealth, Smoking, Stroke, DiffWalking, PhysicalActivity, KidneyDisease, Diabetic, AgeCategory, GenHealth)를 selected feature로 사용하였다. selected feature만을 사용하는 경우, random forest 모델의 정확도는 91.08%, logistic regression 모델의 정확도는 91.64%로 두 모델의 평균 정확도를 계산하면 91.36%이다. 모든 feature를 사용하는 경우, random forest 모델의 정확도는 90.66%, logistic regression 모델의 정확도는 91.53%로 두 모델의 평균 정확도는 91.10%이다. 각각의 평균 정확도를 비교하면 selected feature를 사용한 경우의 정확도가 모든 feature를 사용하는 경우에 비해 0.26% 소폭 높은 것을 확인할 수 있으며 이를 통해 모델 학습 시 선형 관계가 강한 feature들을 선별적으로 사용하는 경우가 모든 feature를 사용하는 경우에 비해 좋은 예측 결과를 나타낼 수 있다는 것을 확인할 수 있다.
  <br> 또한 모델 간의 정확도를 비교하면 selected feature를 사용하는 경우, 모든 feature를 사용하는 경우 각각에 대해 logistic regression 모델의 정확도가 각각 0.56%, 0.87% 소폭 높은 것을 확인할 수 있다. 모델의 정확도는 어떤 데이터를 사용하느냐에 따라 달라질 수 있는데, 심장 질환 데이터의 경우 logistic regression 모델이 random forest 모델에 비해 소폭 좋은 정확도를 가지는 것을 확인할 수 있다.

<br> 도은찬: Dataset preprocessing, Model training, Code implementation, YouTube recording
<br> 정다연: Feature Analysis, Graph analysis, Code implementation, YouTube recording


