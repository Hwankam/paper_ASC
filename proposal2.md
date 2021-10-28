
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

리뷰하려고 하는 paper인 *Online Covariance Matrix Estimation in Stochastic Gradient Descent (2021)* 의 주제는 Averaged Stochastic Gradient Descent 알고리즘의 covariance matrix를 fully online setting에서 estimate하는 방법을 제시하고, 해당 방법의 estimator consistency와 convergence rate이 offline counterpart와도 유사함을 보이고 있다. 
해당 방법을 이용하면 새로운 observation이 관측되는 상황에서도 covariance matrix estimator와 그 confidence interval의 빠른 update가 가능하다는 장점이 있고, 더불어 SGD의 장점인 computational / memory efficiency를 취할 수 있다는 장점 또한 가지고 있다.

먼저 SGD에 대한 설명이 필요한데, 이를 위해 기본적인 objective function optimization을 통한 model parameter estimation을 수식으로 나타내면 다음과 같다.

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=x^*=\underset{x\in\mathbb{R}^d}{argmin}F(x)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x^*=\underset{x\in\mathbb{R}^d}{argmin}F(x)" title="x^*=\underset{x\in\mathbb{R}^d}{argmin}F(x)" /></a>
</p>

해당 문제를 online learning setting에서 풀기 위해 기본적인 deterministic optimization 방법들을 사용하는 경우, 계산 과정에서 계속 모든 data를 저장해야하기 때문에 memory cost와 computational efficiency 측면에서 좋지 않다고 할 수 있다. 이를 위해 도입된 방법 중 하나가 Stochastic Gradient Descent (SGD)이다. 해당 방법의 과정은 다음과 같다.

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=x_i=x_{i-1}-\eta_i\Delta&space;f(x_{i-1},\xi_i),\&space;i\ge&space;1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_i=x_{i-1}-\eta_i\Delta&space;f(x_{i-1},\xi_i),\&space;i\ge&space;1" title="x_i=x_{i-1}-\eta_i\Delta f(x_{i-1},\xi_i),\ i\ge 1" /></a>
</p>

이 방법은 update가 각 시점에서 이루어지므로 이전 iteration에서의 outcome을 저장하고 있을 필요가 없어 online setting에서 memory cost와 computational efficiency를 모두 가지고 있다 할 수 있다.  

그러나, SGD는 update가 굉장히 빈번하게 일어나고 각각의 update에서의 변동성이 클 수 있어 최종 outcome이 큰 폭에서 변할 수 있다는 단점이 있다. 따라서 해당 paper에서는 online setting에서의 uncertainty quantification problem을 직접 다루었다. 특별히, 여기서 제시한 fully online 방법은 covariance matrix의 SGD-based estimate이 SGD iteration의 결과만을 사용하고 있다.

SGD에 대한 연구들은 많이 진행되었고, 대표적인 결과들은 asymptotic convergence, asymptotic distribution에 대한 연구, Averaged SGD (ASGD)의 정의와 modification, 그리고 ASGD estimate의 적당한 regularity condition하에서의 optimal central limit theorem rate 달성 등이 있다.


<br>


### Problem Formulation

ASGD iterate은 다음과 같이 정의된다.

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\bar{x}_n=n^{-1}\sum_{i=1}^n&space;x_i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bar{x}_n=n^{-1}\sum_{i=1}^n&space;x_i" title="\bar{x}_n=n^{-1}\sum_{i=1}^n x_i" /></a>
</p>

Polyak and Juditsky (1992)에 의하면
<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=A=\nabla^2&space;F(x^*),\&space;S=\mathbb{E}([\nabla&space;f(x^*,\xi)][\nabla&space;f(x^*,\xi)]^T),\&space;\Sigma=A^{-1}SA^{-1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?A=\nabla^2&space;F(x^*),\&space;S=\mathbb{E}([\nabla&space;f(x^*,\xi)][\nabla&space;f(x^*,\xi)]^T),\&space;\Sigma=A^{-1}SA^{-1}" title="A=\nabla^2 F(x^*),\ S=\mathbb{E}([\nabla f(x^*,\xi)][\nabla f(x^*,\xi)]^T),\ \Sigma=A^{-1}SA^{-1}" /></a>
</p>
와 같이 정의했을 때, 적당한 condition 아래에서
<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{n}(\bar{x}_n-x^*)\Rightarrow&space;N(0,\Sigma)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\sqrt{n}(\bar{x}_n-x^*)\Rightarrow&space;N(0,\Sigma)" title="\sqrt{n}(\bar{x}_n-x^*)\Rightarrow N(0,\Sigma)" /></a>
</p>
의 asymptotic normality가 성립한다. 그러나 이 covariance matrix에 대한 Hessian-based estimation이 항상 잘 이루어지는 것은 아니고, 또한 이 plug-in estimator가 computationally expensive estimator인 경우도 존재한다. 따라서, 해당 paper에서는 <a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{n}\bar{x}_n" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\sqrt{n}\bar{x}_n" title="\sqrt{n}\bar{x}_n" /></a>의 covariance matrix의 online estimate를 오직 SGD iterate인 <a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;\{x_1,x_2,\dots,x_n&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\left&space;\{x_1,x_2,\dots,x_n&space;\right&space;\}" title="\left \{x_1,x_2,\dots,x_n \right \}" /></a> 만으로 구성해 문제를 해결하려고 한다.

제시한 estimator의 경우, update step에서의 computational and memory complexity가 <a href="https://www.codecogs.com/eqnedit.php?latex=O(d^2)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?O(d^2)" title="O(d^2)" /></a>이라고 한다. 이는 <a href="https://www.codecogs.com/eqnedit.php?latex=d\times&space;d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d\times&space;d" title="d\times d" /></a> covariance matrix에 대해 필요한 least scale computation이므로 의미가 있다. 또한, 총 computational cost가 <a href="https://www.codecogs.com/eqnedit.php?latex=n" target="_blank"><img src="https://latex.codecogs.com/svg.latex?n" title="n" /></a>-linear scale이다.


<br>


### Estimators

paper에서 제시한 estimator는 총 2개의 version으로, 각각 full overlapping version과 nonoverlapping version이고 그 구조를 간략히 표현하면 다음과 같다.  

###### *Full overlapping version*

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{\Sigma}_n=\frac{\sum_{i=1}^n(\sum_{k=t_i}^ix_k-l_i\bar{x}_n)(\sum_{k=t_i}^ix_k-l_i\bar{x}_n)^T}{\sum_{i=1}^nl_i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\hat{\Sigma}_n=\frac{\sum_{i=1}^n(\sum_{k=t_i}^ix_k-l_i\bar{x}_n)(\sum_{k=t_i}^ix_k-l_i\bar{x}_n)^T}{\sum_{i=1}^nl_i}" title="\bar{\Sigma}_n=\frac{\sum_{i=1}^n(\sum_{k=t_i}^ix_k-l_i\bar{x}_n)(\sum_{k=t_i}^ix_k-l_i\bar{x}_n)^T}{\sum_{i=1}^nl_i}" /></a>
</p>
where the blocks are <a href="https://www.codecogs.com/eqnedit.php?latex=\left\{B_i\right\}_{i\in\mathbb{N}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\left\{B_i\right\}_{i\in\mathbb{N}}" title="\left\{B_i\right\}_{i\in\mathbb{N}}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=l_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_i" title="l_i" /></a> denotes the length of <a href="https://www.codecogs.com/eqnedit.php?latex=B_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?B_i" title="B_i" /></a>.

<br>  

###### *Non overlapping version*

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{\Sigma}_{n,NOL}=\frac{1}{n}\sum_{i\in&space;S_n}\big(\sum_{k=t_i}^ix_k-l_i\bar{x}_n\big)\big(\sum_{k=t_i}^ix_k-l_i\bar{x}_n\big)^T" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\hat{\Sigma}_{n,NOL}=\frac{1}{n}\sum_{i\in&space;S_n}\big(\sum_{k=t_i}^ix_k-l_i\bar{x}_n\big)\big(\sum_{k=t_i}^ix_k-l_i\bar{x}_n\big)^T" title="\bar{\Sigma}_{n,NOL}=\frac{1}{n}\sum_{i\in S_n}\big(\sum_{k=t_i}^ix_k-l_i\bar{x}_n\big)\big(\sum_{k=t_i}^ix_k-l_i\bar{x}_n\big)^T" /></a>
</p>
where <a href="https://www.codecogs.com/eqnedit.php?latex=S_n=\left\{n\right\}\cup\left\{a_i-1:i>1,a_i\le&space;n\right\}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_n=\left\{n\right\}\cup\left\{a_i-1:i>1,a_i\le&space;n\right\}" title="S_n=\left\{n\right\}\cup\left\{a_i-1:i>1,a_i\le n\right\}" /></a>.


<br>



### Simulation Studies

#### Model Settings

이 paper는 선형회귀모형, 로지스틱 회귀모형 두 경우에 대해 시뮬레이션을 진행한다. 구체적인 세팅은 다음과 같다: 우선 <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\{\xi_1\equiv&space;(a_i,b_i)\}_{i=1,2,\cdots}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\{\xi_1\equiv&space;(a_i,b_i)\}_{i=1,2,\cdots}" title="\{\xi_1\equiv (a_i,b_i)\}_{i=1,2,\cdots}" /></a> 를 i.i.d. 표본이라 하고  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x^*" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;x^*" title="x^*" /></a> 를 모형의 true parameter 라 하자. 그리고 선형회귀모형과 로지스틱 회귀모형 모두에서 <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a_i\sim&space;N(0,\mathbf{I}_d)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;a_i\sim&space;N(0,\mathbf{I}_d)" title="a_i\sim N(0,\mathbf{I}_d)" /></a> 라 하자. 선형회귀모형의 경우, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;b_i&space;=&space;a_i^Tx^*&plus;\epsilon_i,&space;\epsilon_i\sim&space;N(0,1)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;b_i&space;=&space;a_i^Tx^*&plus;\epsilon_i,&space;\epsilon_i\sim&space;N(0,1)" title="b_i = a_i^Tx^*+\epsilon_i, \epsilon_i\sim N(0,1)" /></a> 라 하고, 로지스틱 회귀모형의 경우 <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;b_i|a_i\sim&space;Bernoulli((1&plus;\exp(-a_i^Tx^*))^{-1})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;b_i|a_i\sim&space;Bernoulli((1&plus;\exp(-a_i^Tx^*))^{-1})" title="b_i|a_i\sim Bernoulli((1+\exp(-a_i^Tx^*))^{-1})" /></a> 라 하자. 이 때 손실함수 <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;f(\cdot)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;f(\cdot)" title="f(\cdot)" /></a> 는 negative log likelihood function 로 정의하며, 구체적으로 다음과 같다:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;f(x,a_i,b_i)&space;=&space;\begin{cases}\frac{1}{2}(a_i^Tx-b_i)^2&space;\quad&\text{(linear&space;regression)}\\&space;(1-b_i)a_i^Tx&space;&plus;&space;\log(1&plus;\exp(-a_i^Tx))\quad&\text{(logistic&space;regression)}.&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;f(x,a_i,b_i)&space;=&space;\begin{cases}\frac{1}{2}(a_i^Tx-b_i)^2&space;\quad&\text{(linear&space;regression)}\\&space;(1-b_i)a_i^Tx&space;&plus;&space;\log(1&plus;\exp(-a_i^Tx))\quad&\text{(logistic&space;regression)}.&space;\end{cases}" title="f(x,a_i,b_i) = \begin{cases}\frac{1}{2}(a_i^Tx-b_i)^2 \quad&\text{(linear regression)}\\ (1-b_i)a_i^Tx + \log(1+\exp(-a_i^Tx))\quad&\text{(logistic regression)}. \end{cases}" /></a>

이 paper에서는 true limiting covariance matrix의 계산을 쉽게 하기 위해서 기본적으로 선형회귀모형을 가정한다.

<br>

#### Figure 1


<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;A&space;=&space;E[\nabla^2f(x^*)]&space;=&space;E(aa^T)&space;=&space;\mathbf{I}_d" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;A&space;=&space;E[\nabla^2f(x^*)]&space;=&space;E(aa^T)&space;=&space;\mathbf{I}_d" title="A = E[\nabla^2f(x^*)] = E(aa^T) = \mathbf{I}_d" /></a>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;S&space;=&space;E([\nabla&space;f(x^*,&space;\xi)][\nabla&space;f(x^*,&space;\xi)]^T)=E(\epsilon^2)E(aa^T)&space;=&space;\mathbf&space;I_d" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;S&space;=&space;E([\nabla&space;f(x^*,&space;\xi)][\nabla&space;f(x^*,&space;\xi)]^T)=E(\epsilon^2)E(aa^T)&space;=&space;\mathbf&space;I_d" title="S = E([\nabla f(x^*, \xi)][\nabla f(x^*, \xi)]^T)=E(\epsilon^2)E(aa^T) = \mathbf I_d" /></a>

이므로 limiting covariance matrix는  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\Sigma&space;=&space;A^{-1}SA^{-1}&space;=&space;\mathbf{I}_d" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\Sigma&space;=&space;A^{-1}SA^{-1}&space;=&space;\mathbf{I}_d" title="\Sigma = A^{-1}SA^{-1} = \mathbf{I}_d" /></a>이다.

online estimator의 convergence를 확인하기 위해서는 추정한 분산행렬과 limiting covariance matrix 간의 차에 대한 operator norm의 Loss를 살펴본다. 

Figure 1 은 이를 나타낸 것이다. 

<img src="./Figure1.png" alt="figure 1" style="zoom:40%;" />

Online covariance estatimation에 대한 수렴성을 확인하기 위해, 추정에 사용되는 batch size를 바꾸면서 step에 따른 <img src="https://latex.codecogs.com/svg.latex?\;||\hat\Sigma_n-\Sigma||_2"/>  값 변화를 확인해보았다. Full overlapping version와 Non-overlapping version 모두 동일한 수렴 속도<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;O(n^{-1/8})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;O(n^{-1/8})" title="O(n^{-1/8})" /></a>로 limiting covariance matrix에 수렴하는 것을 알 수 있다. 프로젝트에서는 이를 실제로 구현해보겠다.

<br>

#### Figure 2

<img src="./Figure2.png" alt="figure 2" style="zoom:40%;" />

Full overlapping version과 non-overlapping version의 MSE 차이를 비교하기 위해 각 step 에 따른 relative efficiency 변화도 확인해볼 것이다. Figure 2에서 알 수 있는 바는 시행횟수가 커질수록 reltive efficiency 값이 batch size와 큰 관련이 없다는 것인데 이를 시뮬레이션을 통해 검증해보고, 이후 연구에서는 batch size를 고정한 채로 진행할 것이다.

<br>

#### Figure 3

선형회귀모형에서 (특히 차원 <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;d&space;=&space;5" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;d&space;=&space;5" title="d = 5" /></a> 인 경우), 점근정규성과 신뢰구간(CI) 범위에 대해 수치적으로 체크해볼 것이다. 이때 추정된 공분산행렬을 이용하여, 다음과 같은 <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;95\%" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;95\%" title="95\%" /></a> CI를 구성한다:  averaged coefficient <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mu&space;=&space;1^Tx^*" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\mu&space;=&space;1^Tx^*" title="\mu = 1^Tx^*" /></a>에 대해,

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left[1^T\bar{x}&space;-&space;z_{1-q/2}\sqrt{1^T\hat{\Sigma}_n1/n},&space;1^T\bar{x}&space;&plus;&space;z_{1-q/2}\sqrt{1^T\hat{\Sigma}_n1/n}\right]." target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\left[1^T\bar{x}&space;-&space;z_{1-q/2}\sqrt{1^T\hat{\Sigma}_n1/n},&space;1^T\bar{x}&space;&plus;&space;z_{1-q/2}\sqrt{1^T\hat{\Sigma}_n1/n}\right]." title="\left[1^T\bar{x} - z_{1-q/2}\sqrt{1^T\hat{\Sigma}_n1/n}, 1^T\bar{x} + z_{1-q/2}\sqrt{1^T\hat{\Sigma}_n1/n}\right]." /></a>

그리고 위의 CI를 true limiting covariance matrix를 기반으로한 oracle <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;95\%" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;95\%" title="95\%" /></a> CI와 비교해볼 것이다. 특히 이 paper의 Figure 3 에선 overlapping(Full, 그림에서 파란색 선으로 표기)과 non-overlapping 버전(NOL, 그림에서 녹색으로 표기)를 비교하였으며, 수치적으로 위의 CI가 <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;95\%" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;95\%" title="95\%" /></a>의 범위로 수렴함을 확인할 수 있다(Figure 3 - (a), empirical coverage rate과 step 수를 비교).  
한 편, standardized error <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\sqrt{n}1^T(\hat{x}-x^*)/\sqrt{1^T\hat{\Sigma}_n1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\sqrt{n}1^T(\hat{x}-x^*)/\sqrt{1^T\hat{\Sigma}_n1}" title="\sqrt{n}1^T(\hat{x}-x^*)/\sqrt{1^T\hat{\Sigma}_n1}" /></a> (Figure 3 - (c), standardized error의 density plot을 나타냄)가 근사적으로 정규분포를 따른다는 것을 확인할 수 있다. 또한 추정 CI의 길이가 oracle CI의 길이로 수렴하는 것 또한 Figure 3에서 확인할 수 있다. (Figure 3 - (b), CI의 길이를 나타냄). 우리는 Figure 3를 재현해봄으로써 이 paper에서 제시된 방법론을 통한standardized error의 점근정규성, CI의 길이 및 coverage rate의 수렴성에 대해 수치적으로 살펴볼 것이며, 결과적으로 이 paper의 fully overlapping 또는 non-overlapping method 가 효과적으로 true model에 수렴함을 확인해볼 것이다.

<img src="./Figure3.png" alt="figure 3" style="zoom:40%;" />

<br>

#### Figure 4

한 편, 이 paper에선 기존의 전통적인 추정법, 예를 들면 plug-in method 와 비교하여 새로 제시된 방법론이 더 효율적임을 주장한다. 특히 선형/로지스틱 회귀모형 (차원이 <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;d=5" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;d=5" title="d=5" /></a> 그리고 <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;d=5" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;d=20" title="d=20" /></a> 인 경우) 모두에서 step의 수에 따른 empirical coverage rate를 fully overlapping online-BM, non-overlapping online-BM, 그리고 plug-in method를 기준으로 비교함으로써 주장을 뒷받침한다. Figure 4에선 online-BM 방법론들이 반복횟수가 커지면 커질수록 기존 방법론인 plug-in method와 사실상 유사한 성능을 보인다는 것(Figure 4의 첫 번째와 두 번째 행의 그림들에서 확인할 수 있음)을 수치적으로 확인할 수 있다. 또한 연산 시간을 비교해봤을 때, online-BM 방법론들이 plug-in method에 비해 현저히 빠른 속도로 계산됨을 Figure 4의 마지막 행을 통해 확인할 수 있다. 우리는 이전 Figure들과 마찬가지로 Figure 4 또한 재현해봄으로써 새로 제시된 방법론이 반복횟수가 커질수록 기존 방법론에 견줄만한 성능을 보이는지, 그리고 연산 시간이 현저히 줄어드는지 다시 확인해볼 것이다. 즉, Figure 4를 재현함으로써 새로 제시된 방법론의 준수한 성능과 효율적 계산능력을 확인해볼 것이다.

<img src="./Figure4.png" alt="figure 4" style="zoom:40%;" />


<br>

### Reference
[1] *Chen, X., J. D. Lee, X. T. Tong, and Y. Zhang (2020). “Statistical inference for model parameters in stochastic gradient descent,” Annals of Statistics, 48(1), 251–273.*