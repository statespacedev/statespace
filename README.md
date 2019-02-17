<img src="https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/ekf.png" align="center" width="480" height="360"/>

[![pipeline](https://gitlab.com/noahhsmith/starid/badges/master/pipeline.svg)](https://gitlab.com/noahhsmith/statespace/pipelines)
[![pypi](https://img.shields.io/badge/pypi-latest-brightgreen.svg)](https://pypi.org/project/statespace/)
[![blog](https://img.shields.io/badge/blog-latest-brightgreen.svg)](https://gitlab.com/noahhsmith/statespace/blob/master/docs/readme.md)
[![references](https://img.shields.io/badge/references-latest-brightgreen.svg)](https://gitlab.com/noahhsmith/statespace/blob/master/docs/references.md)

uncertainty and confidence, distributions, their evolution with time, noise, and observations, tracking and detection, decisions, risk and the cost of errors, sample-and-propagate, model-based systems, sequential monte carlo, markov-chain monte carlo

[The Flaw of Averages: Why We Underestimate Risk in the Face of Uncertainty, Sam L. Savage](http://a.co/cDDBO9p)

[Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods, James V. Candy](http://a.co/gp4upXd)

[Kalman Filtering: Theory and Practice, Mohinder S. Grewal, Angus P. Andrews](http://a.co/6hAa35c)

google-cloud-platform automated build-test-deploy to kubernetes-engine is the focus at the moment, using cloud-source [repo](https://source.cloud.google.com/statespace-229611/statespace) and cloud-build along the way.

automated build-test-deploy to [pypi](https://pypi.org/project/statespace) is mostly a placeholder, ubuntu clone-install-develop of gitlab [repo](https://gitlab.com/noahhsmith/statespace) is assumed for now.

    sudo apt-get -qq update -qy
    sudo apt-get -qq install -y python3.6 python3-venv python3-pip
    git clone git@gitlab.com:noahhsmith/statespace.git statespace
    cd statespace
    python3 -m venv venv
    . venv/bin/activate
    python3 setup.py develop
    pytest
    python3 statespace --demo
