<img src="https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/ekf.png" align="center" width="480" height="360"/>

[notes](https://gitlab.com/noahhsmith/statespace/blob/master/docs/readme.md)

uncertainty and confidence, their evolution with time, noise, and observations, risk and the cost of errors

[The Flaw of Averages: Why We Underestimate Risk in the Face of Uncertainty, Sam L. Savage](http://a.co/cDDBO9p)

[Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods, James V. Candy](http://a.co/gp4upXd)

[Kalman Filtering: Theory and Practice, Mohinder S. Grewal, Angus P. Andrews](http://a.co/6hAa35c)

[Stochastic Processes and Filtering Theory, Andrew H. Jazwinski](http://a.co/cm5zfQu) 

ubuntu dependencies

    sudo apt-get -qq update -qy
    sudo apt-get -qq install -y python3.6 python3-venv python3-pip

clone, virtual environment, install and test

    git clone git@gitlab.com:noahhsmith/statespace.git statespace
    cd statespace
    python3 -m venv venv
    . venv/bin/activate
    pip install -e .
    pytest

