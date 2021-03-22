<img src="https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/pf2-small.png"/>

[![pipeline](https://gitlab.com/noahhsmith/starid/badges/master/pipeline.svg)](https://gitlab.com/noahhsmith/statespace/pipelines)
[![pypi](https://img.shields.io/badge/pypi-latest-brightgreen.svg)](https://pypi.org/project/statespace/)
[![docs](https://readthedocs.org/projects/statespace/badge/?version=latest)](https://statespace.readthedocs.io/en/latest/?badge=latest)

processors and models from 

[Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods, James V. Candy](http://a.co/gp4upXd)

[Kalman Filtering: Theory and Practice, Mohinder S. Grewal, Angus P. Andrews](http://a.co/6hAa35c)

210321 settling on a new model structure that separates the parts associated with baseline kalman filter, sigma-point filter, and particle filter. this can eventually gorw upward into a high level base model. concept is that a model needs to know something about the processors that are going to use it. a particle processor has different needs than a kalman processor. this gives a natural shape and structure to the models.

210314 maybe what we need is to bring in a new reference model - to kind of shake things up and help brainstorming around the concept of a high level base model. a strong candidate is the bearings only ship tracking problem - we have working examples in the orgeon library, and it's in the textbooks.

the bearings only problem has some interesting history - it's basically about being on a ww2 era sub. your sub is travelling along, and you begin hearing the sound of a ship at some bearing. over time, as the bearing changes, you can estimate the position and velocity of the ship. keep in mind that both your sub and the ship are moving the entire time. at some point you make a course change for your sub to pursue the ship.

210308 looking more closely at the oregon library documentation, they have an interesting discussion of objectives - something to think about going forward. is a higher-level statespace model something to consider? putting it this way - adapt/remap a higher-level statespace model into specific lower-level statespace models - we're talking about a translator / converter / adaptor... and we already have an extremely primitive form of that - the rccircuit, jazwinski1, jazwinski2 models are fed into classical, modern, particle. we can think about a higher-level model that can express rccircuit, jazwinski1, and jazwinski2.

designed to allow reuse of a state space definition for state, parameter and joint estimation, using a variety of different inference algorithms. you define your system once in a higher-level state space framework, and then the inference framework generator *geninfds* together with the inference system noise source generator *gensysnoiseds* will adapt/remap that model into a lower-level state space framework needed for whatever type of estimation you want to do. This allows you to focus on defining the model, doing data IO, etc. without having to get bogged down into casting the problem into a different framework each time you want to use a different estimator or want to change the type of inference your doing. I.e. the internal inference implementation is hidden or as transparent as possible with respect to the problem definition by the user. [more](https://gitlab.com/noahhsmith/statespace/-/tree/master/docs/liboregon)

210221 brought the documentation via readthedocs up to a minimal level. cleaned up the project and brought some focus to what's going on here. as the docs now make clear - this project focuses on reference problems from Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods, Kalman Filtering: Theory and Practice, and Stochastic Processes and Filtering Theory - in particular, using numpy for matrix and vector manipulation.

190418 brief [lit-review](https://www.linkedin.com/pulse/google-state-space-noah-smith/) posted on linkedin.

190331 concise [motivation piece](https://www.linkedin.com/pulse/shape-uncertainty-noah-smith/) posted on linkedin.

190310 decision-function-based detector is go. simplest possible case - linear rc-circuit system-model and linear kalman-filter tracker. log-likelihood decision function for detection, ensembles of 100 runs each for signal case and noise case. output curves shown in the first plot - green signal, blue noise-only. roc curves in the second plot. 

![](https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/rccircdecfuncs.png)
 
![](https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/rccircroc.png)

<a name="190223">190223</a>

kl-divergence for evaluating sequential monte-carlo - demonstrated below by three pf's in action during the first second of the jazwinksi problem - start-up and convergence. these are 100 hz dist-curves - each dist-curve is a kernel-density-estimate combining hundreds of monte-carlo samples, the fundamental-particles - green dist-curves for truth, blue dist-curves for pf. state-estimates are two red curves on the x,t-plane beneath the dist-curves.

![pf1](https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/pf1.png)

![pf2](https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/pf2.png)

![pf3](https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/pf3.png)

190105 ukf adaptive jazwinksi switched to square-root filtering, qr-factorization, cholesky-factor update and downdate. improved numerical stability and scaled sampling is clear. still a question around scalar-obs and the obs cholesky-factor and gain. with an adhoc stabilizer on the obs cholesky-factor it's working well overall.

181230 pf adaptive jazwinksi. parameter-roughening.

181226 ukf adaptive jazwinski. sample-and-propagate tuning.

180910 ekf adaptive jazwinski. ud-factorized square-root filtering required for numerical stability.

    