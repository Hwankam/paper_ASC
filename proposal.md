
# Online Covariance Matrix Estimation in Stochastic Gradient Descent

### Team Member
* 류환감
* 이지후
* 홍승기

<br>

### Goal of the Project
* Stochastic Optimization 분야에서 나온 최신 Paper를 읽고, **Julia**를 사용해서 이를 직접 구현해본다
* 다른 연구에서 사용된 알고리즘과 본 논문의 방법을 비교 및 분석한다.

<br>


### Simple review of the Paper

리뷰하려고 하는 paper인 *Online Covariance Matrix Estimation in Stochastic Gradien Descent (2021)* 의 주제는 Averaged Stochastic Gradient Descent 알고리즘의 covariance matrix를 fully online setting에서 estimate하는 방법을 제시하고, 해당 방법의 estimator consistency와 convergence rate이 offline counterpart와도 유사함을 보이고 있다. 

해당 방법을 이용하면 새로운 observation이 관측되는 상황에서도 covariance matrix estimator와 그 confidence interval의 빠른 update가 가능하다는 장점이 있고, 더불어 SGD의 장점인 computational / memory efficiency를 취할 수 있다는 장점 또한 가지고 있다.

먼저 SGD에 대한 설명이 필요한데, 이를 위해 기본적인 objective function optimization을 통한 model parameter estimation을 수식으로 나타내면 다음과 같다.

$$
x^*=\underset{x\in\mathbb{R}^d}{argmin}F(x)
$$

해당 문제를 online learning setting에서 풀기 위해 기본적인 deterministic optimization 방법들을 사용하는 경우, 계산 과정에서 계속 모든 data를 저장해야하기 때문에 memory cost와 computational efficiency 측면에서 좋지 않다고 할 수 있다. 이를 위해 도입된 방법 중 하나가 Stochastic Gradient Descent (SGD)이다. 해당 방법의 과정은 다음과 같다.


$$
x_i=x_{i-1}-\eta_i\Delta f(x_{i-1},\xi_i),\ i\ge 1
$$


이 방법은 update가 각 시점에서 이루어지므로 이전 iteration에서의 outcome을 저장하고 있을 필요가 없어 online setting에서 memory cost와 computational efficiency를 모두 가지고 있다 할 수 있다.

그러나, SGD는 update가 굉장히 빈번하게 일어나고 각각의 update에서의 변동성이 클 수 있어 최종 outcome이 큰 폭에서 변할 수 있다는 단점이 있다. 따라서 해당 paper에서는 online setting에서의 uncertainty quantification problem을 직접 다루었다. 특별히, 여기서 제시한 fully online 방법은 covariance matrix의 SGD-based estimate이 SGD iteration의 결과만을 사용하고 있다.

SGD에 대한 연구들은 많이 진행되었고, 대표적인 결과들은 asymptotic convergence, asymptotic distribution에 대한 연구, Averaged SGD (ASGD)의 정의와 modification, 그리고 ASGD estimate의 적당한 regularity condition하에섣의 optimal central limit theorem rate 달성 등이 있다.

<br>

### Problem Formulation

ASGD iterate은 다음과 같이 정의된다.


$$
\bar{x}_n=n^{-1}\sum_{i=1}^n x_i
$$


Polyak and Juditsky (1992)에 의하면

$$
A=\nabla^2 F(x^*),\ S=\mathbb{E}([\nabla f(x^*,\xi)][\nabla f(x^*,\xi)]^T),\ \Sigma=A^{-1}SA^{-1}
$$


와 같이 정의했을 때, 적당한 condition 아래에서

$$
\sqrt{n}(\bar{x}_n-x^*)\Rightarrow N(0,\Sigma)
$$

의 asymptotic normality가 성립한다.

그러나 이 covariance matrix에 대한 Hessian-based estimation이 항상 잘 이루어지는 것은 아니고, 또한 이 plug-in estimator가 computationally expensive estimator인 경우도 존재한다.

따라서, 해당 paper에서는 $\sqrt{n}\bar{x}_n$의 covariance matrix의 online estimate를 오직 SGD iterate인 $\left \{x_1,x_2,\dots,x_n \right \}$ 만으로 구성해 문제를 해결하려고 한다.

제시한 estimator의 경우, update step에서의 computational and memory complexity가 $O(d^2)$ 이라고 한다. 이는 $d\times d$ covariance matrix에 대해 필요한 least scale computation이므로 의미가 있다. 또한, 총 computational cost가 $n$-linear scale이다.

<br>

### Estimators

paper에서 제시한 estimator는 총 2개의 version으로, 각각 full overlapping version과 nonoverlapping version이고 그 구조를 간략히 표현하면 다음과 같다.


$$
\bar{\Sigma}_n=\frac{\sum_{i=1}^n(\sum_{k=t_i}^ix_k-l_i\bar{x}_n)(\sum_{k=t_i}^ix_k-l_i\bar{x}_n)^T}{\sum_{i=1}^nl_i}

$$


where the blocks are $\left\{B_i\right\}_{i\in\mathbb{N}} $ and $l_i$ denotes the length of $B_i$ 

$$
\bar{\Sigma}_{n,NOL}=\frac{1}{n}\sum_{i\in S_n}\big(\sum_{k=t_i}^ix_k-l_i\bar{x}_n\big)\big(\sum_{k=t_i}^ix_k-l_i\bar{x}_n\big)^T
$$


where $S_n=\left\{n\right\}\cup\left\{a_i-1:i>1,a_i\le n\right\}$


<br>

### Simulation
1. Online covariance estatimation에 대한 수렴성을 확인하기 위해, 추정에 사용되는 batch size를 바꾸면서 step에 따른 $|| \hat \Sigma_n - \Sigma ||_2$ 값의 감소 양상을 비교해볼 것이다. 

2. full overlapping version과 non-overlapping version의 MSE 차이를 비교하기 위해 각 step 에 따른 relative efficiency 변화도 확인해볼 것이다. 

3. 시뮬레이션 결과로 online covariance estimator의 수렴성을 확인한다면 asymptotic normality 하에서 averaged coefficient $\mu = 1^T x^*$  에 대한 신뢰구간을 구할 수 있을 것이다.
$$
[1^T \bar x_n - z_{1-q/2}\sqrt{1^T \hat \Sigma_n 1 /n} , 1^T \bar x_n + z_{1-q/2}\sqrt{1^T \hat \Sigma_n 1 /n} ]
$$ 


<br>

### Reference
[1] *Chen, X., J. D. Lee, X. T. Tong, and Y. Zhang (2020). “Statistical inference for model parameters in stochastic gradient descent,” Annals of Statistics, 48(1), 251–273.*

