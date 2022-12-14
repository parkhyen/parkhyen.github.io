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

### References
유원준·안상준.  ⌜딥 러닝을 이용한 자연어 처리 입문⌟.   
URL : https://wikidocs.net/book/2155   
dataset : https://github.com/e9t/nsmc/   
<br>

# 데이터 수집

수집한 데이터를 다루기 위하여 pandas를 사용하여 데이터셋을 가져옴.   

해당 게시글에서 사용할 데이터는 상단 레퍼런스를 참조하며, 훈련 데이터 150,000개와  테스트 데이터 50,000개가 존재함.

{% highlight python linenos %}
# 데이터 가져오기
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

# 데이터 샘플 출력
train_data.head(5)
test_data.head(5)
{% endhighlight %}

<br>

# 데이터 정제

<br>

# 피처 엔지니어링

<br>

# 모델 학습

<br>

# 모델 평가

<br>

