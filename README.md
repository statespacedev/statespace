<img src="https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/pf2-small.png"/>

[![pipeline](https://gitlab.com/noahhsmith/starid/badges/master/pipeline.svg)](https://gitlab.com/noahhsmith/statespace/pipelines)
[![pypi](https://img.shields.io/badge/pypi-latest-brightgreen.svg)](https://pypi.org/project/statespace/)
[![docs](https://readthedocs.org/projects/statespace/badge/?version=latest)](https://statespace.readthedocs.io/en/latest/?badge=latest)

implements reference problems from [Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods, James V. Candy](http://a.co/gp4upXd), [Kalman Filtering: Theory and Practice, Mohinder S. Grewal, Angus P. Andrews](http://a.co/6hAa35c), [Stochastic Processes and Filtering Theory, Jazwinski](https://amzn.to/2NLXfVK).

build-test-deploy to [pypi](https://pypi.org/project/statespace) is mostly a placeholder, ubuntu clone-install-develop of gitlab [repo](https://gitlab.com/noahhsmith/statespace) is assumed for now.

    sudo apt-get -qq update -qy
    sudo apt-get -qq install -y python3.6 python3-venv python3-pip
    git clone git@gitlab.com:noahhsmith/statespace.git statespace
    cd statespace
    python3 -m venv venv
    . venv/bin/activate
    python3 setup.py develop
    pytest
    python3 statespace --demo

190418

brief [lit-review](https://www.linkedin.com/pulse/google-state-space-noah-smith/) posted on linkedin.

190331

concise [motivation piece](https://www.linkedin.com/pulse/shape-uncertainty-noah-smith/) posted on linkedin.

190310

decision-function-based detector is go. simplest possible case - linear rc-circuit system-model and linear kalman-filter tracker. log-likelihood decision function for detection, ensembles of 100 runs each for signal case and noise case. output curves shown in the first plot - green signal, blue noise-only. roc curves in the second plot. 

![](https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/rccircdecfuncs.png)
 
![](https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/rccircroc.png)

<a name="190223">190223</a>

kl-divergence for evaluating sequential monte-carlo - demonstrated below by three pf's in action during the first second of the jazwinksi problem - start-up and convergence. these are 100 hz dist-curves - each dist-curve is a kernel-density-estimate combining hundreds of monte-carlo samples, the fundamental-particles - green dist-curves for truth, blue dist-curves for pf. state-estimates are two red curves on the x,t-plane beneath the dist-curves.

![pf1](https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/pf1.png)

![pf2](https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/pf2.png)

![pf3](https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/pf3.png)

190105

ukf adaptive jazwinksi switched to square-root filtering, qr-factorization, cholesky-factor update and downdate. improved numerical stability and scaled sampling is clear. still a question around scalar-obs and the obs cholesky-factor and gain. with an adhoc stabilizer on the obs cholesky-factor it's working well overall.

181230

pf adaptive jazwinksi. parameter-roughening.

181226

ukf adaptive jazwinski. sample-and-propagate tuning.

180910

ekf adaptive jazwinski. ud-factorized square-root filtering required for numerical stability.

    