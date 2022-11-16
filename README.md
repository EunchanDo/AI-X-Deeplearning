# AI-X-Deeplearning

**Title**: 다양한 머신러닝 기법을 활용한 심장병 발병 예측

**Members**: <br>도은찬, 서울 기계공학부, dec1995@hanyang.ac.kr  <br>정다연, 서울 기계공학부, jdd2635@hanyang.ac.kr
         
**Ⅰ. Proposal(Option A)**

  - Motivation:
  <br> 심장질환은 국내 사망원인 1위 질환인 암 다음으로 우리의 목숨을 위협하는 가장 무서운 질병이다. 이는 돌연사의 원인이 되기도 하며 고령 환자의 주요 사망 원인이다. 국내에서는 매년 20만명이 넘는 환자들이 심부전으로 병원을 찾는다. 우리나라 뿐만 아니라 다른 여러 국가에서도 사망 원인이 되는 심각한 질환이고 특히 미국 내에서는 전 인종이 영향을 받은 주된 사망원인 중 하나이다. 미국 CDC의 통계자료에 따르면 미국인들의 절반이 심장병의 세 가지 주요 원인인 고혈압, 고 콜레스테롤, 흡연 중 적어도 한가지를 가지고 있다고 한다. 이 외에도 심장질환을 발병시킬 수 있는 요인에는 육체 및 정신의 건강한 정도, 뇌졸중 경험 여부 그리고 음주 등이 있다. 따라서 이러한 원인들을 분석하여 심장병 발생 여부를 예측하는 것은 인류의 중대한 과업이라고 할 수 있다. 그러한 과업에 동참하기위해 우리 팀은 미국 CDC의 방대한 심장 질환 지표 데이터를 활용하여 그것을 예측할 수 있는 모델을 만들고 어떤 모델이 최적의 결과를 제시할 수 있는지 분석할 것이다. 
  
  - What do you want to see at the end?
  <br> 심장 질환은 예방 가능성이 높은 질환이다. 위에서 언급한 것처럼 주로 흡연, 운동 부족, 비만 등이 주요 원인이며, 이를 약물치료나 생활 습관의 교정함으로서 심장질환 발병을 예방할 수 있다. 따라서 환자의 성별, 나이, BMI 지수, 흡연 여부 등을 특징인자로 하는 머신러닝 기법들을 적용하여 심장질환 발생 가능성을 예측하고 사망률이 높은 질병 중 하나인 심장질환을 예측하고 이러한 환자들에게 약물치료, 생활 습관 교정 등의 예방법을 제시하여 심장질환으로 인한 사망률을 낮출 수 있는 모델을 제시하고자 한다.
  
**Ⅱ. Datasets**

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
  
**Ⅲ. Methodology**

  - Explaining your choice of algorithms
  <br> SVM, logistic regression, random forest ...
  - Explaining features
  <br> <script src="https://gist.github.com/jdd2635/b5d0088043a497ed0bbe42f1b4864800.js"></script>
  
**Ⅳ. Evaluation & Analysis**

  - Graphs, tables, any statistics
  
**Ⅴ. Related Work**
   <br>> http://www.samsunghospital.com/dept/main/index.do?DP_CODE=XB301&MENU_ID=001002 (심장질환 예방)
   <br>> https://www.nhis.or.kr/magazin/mobile/201411/sub02_02.html
   <br>> https://www.korea.kr/news/healthView.do?newsId=148896724
   <br>> https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease (데이터셋)
   <br>> https://computer-science-student.tistory.com/113
   <br>> https://www.kaggle.com/code/mrisdal/exploring-survival-on-the-titanic
   
**Ⅵ. Conclusion: Discussion**
