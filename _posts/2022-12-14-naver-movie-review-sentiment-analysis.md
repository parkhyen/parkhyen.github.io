---
layout: post
title: KoNLPY를 이용한 네이버 영화 리뷰 감성 분류 모델 만들어보기
tags: [NLP, python, M/L]
---

# 목차

1. 데이터 수집
2. 데이터 정제
3. 피처 엔지니어링
4. 모델 학습
5. 모델 평가
<br>
{% highlight python linenos %}
{% endhighlight %}

<br>
### References
유원준·안상준.  ⌜딥 러닝을 이용한 자연어 처리 입문⌟.   
URL : https://wikidocs.net/book/2155   
dataset : https://github.com/e9t/nsmc/   
<br>

# 데이터 수집
해당 게시글에서 사용할 데이터는 상단 레퍼런스를 참조합니다.   

{% highlight python linenos %}
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')
{% endhighlight %}

샘플 데이터는 훈련 데이터 150,000개와  테스트 데이터 50,000개로 구성되어 총 200,000개의 데이터가 존재합니다.
pandas를 사용하여 데이터셋을 가져옵니다.   

데이터의 구성을 확인하기 위해서 상위 5개의 샘플을 출력해봅니다.   

{% highlight python linenos %}
train_data.head(5)
test_data.head(5)
{% endhighlight %}

>train_data : 
>
>| id | document | label |
>| :--- | :--- | :--- | :--- |
>| 0 | 9976970 | 아 더빙.. 진짜 짜증나네요 목소리 | 0 |
>| 1 | 3819312 | 흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나 | 1 |
>| 2 | 10265843 | 너무재밓었다그래서보는것을추천한다 | 0 |
>| 3 | 9045019 | 교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정 | 0 |
>| 4 | 6483659 | 사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 ... | 1 |

>test_data : 
>
>| id | document | label |
>| :--- | :--- | :--- | :--- |
>| 0	| 6270596	| 굳 ㅋ	| 1 |
>| 1	| 9274899	| GDNTOPCLASSINTHECLUB	| 0 |
>| 2	| 8544678	| 뭐야 이 평점들은.... 나쁘진 않지만 10점 짜리는 더더욱 아니잖아	| 0 |
>| 3	| 6825595	| 지루하지는 않은데 완전 막장임... 돈주고 보기에는....	| 0 |
>| 4	| 6723715	| 3D만 아니었어도 별 다섯 개 줬을텐데.. 왜 3D로 나와서 제 심기를 불편하게 하죠??	| 0 |

데이터는 id, document, label 3열로 구성되어 있는 것을 확인할 수 있습니다.   
인덱스와 id는 모델 학습에 의미가 있는 데이터는 아니므로 리뷰 내용(document)과 긍정(1)·부정(0)을 나타내는 열(label)만 사용합니다.   
<br>

# 데이터 정제
### 데이터 분포 시각화
모델 학습 과정에서 데이터의 분포도 모델이 학습을 하기 때문에, 데이터의 불균형은 문제가 될 수 있습니다.   
데이터의 분포를 그래프로 그려 확인해봅니다.   

{% highlight python linenos %}
train_data['label'].value_counts().plot(kind = 'bar')
{% endhighlight %}

![graph](/assets/img/posts/img_01.png)

그래프를 보면 긍정과 부정의 분포가 균일한 것을 확인할 수 있습니다.   
<br>

### 중복값, null값 제거
중복값과 null값을 확인 후 제거해줍니다.   

{% highlight python linenos %}
train_data.drop_duplicates(subset=['document'], inplace=True)
train_data.drop_duplicates(subset=['document'], inplace=True)
{% endhighlight %}
<br>

### 데이터 전처리를 수행하기 위해 정규표현식 사용
한글[ㄱ-ㅎㅏ-ㅣ가-힣]과 공백을 제외하고 모두 제거해줍니다.   

```python
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
```
결과를 확인해보면 위의 train_data와 차이를 확인할 수 있습니다.   

>Out : 
>
>| id | document | label |
>| :--- | :--- | :--- | :--- |
>| 0 | 9976970 | 아 더빙 진짜 짜증나네요 목소리 | 0 |
>| 1 | 3819312 | 흠포스터보고 초딩영화줄오버연기조차 가볍지 않구나 | 1 |
>| 2 | 10265843 | 너무재밓었다그래서보는것을추천한다 | 0 |
>| 3 | 9045019 | 교도소 이야기구먼 솔직히 재미는 없다평점 조정 | 0 |
>| 4 | 6483659 | 사이몬페그의 익살스런 연기가 돋보였던 영화스파이더맨에서 늙어보이기만 했던 커스틴 던... | 1 |

<br>

### 공백만 있거나 빈 값을 가진 행이 있다면 null 값으로 변경
한글 없이 영어, 숫자, 특수문자로만 구성된 데이터의 경우 전처리 수행 과정에서 공백(white space) 혹은 빈 값(empty)이 되었을 것입니다.   
또, 전처리 과정에서 영어, 숫자, 특수문자가 없어지고 한글과 공백만 남게 되면서 중복값이 생겼을 수도 있습니다.   
공백(white space) 혹은 빈 값(empty)을 가진 행은 null 값으로 변경하고, 중복값과 null값을 다시 한 번 제거해줍니다.   

{% highlight python linenos %}
train_data['document'] = train_data['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)
train_data.drop_duplicates(subset=['document'], inplace=True)
train_data = train_data.dropna(axis=0)
print(len(train_data))
{% endhighlight %}

{: .box-note}
143620   


# 피처 엔지니어링

<br>

# 모델 학습

<br>

# 모델 평가

<br>

