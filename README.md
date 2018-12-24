<img src="https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/ekf.png" align="center" width="300" height="300"/>

[notes](https://gitlab.com/noahhsmith/statespace/blob/master/docs/readme.md)

uncertainty, confidence, knowledge

their evolution with time and various processes, including noise and observations

risk and the cost of errors

[The Flaw of Averages: Why We Underestimate Risk in the Face of Uncertainty, Sam L. Savage](http://a.co/cDDBO9p)

[Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods, James V. Candy](http://a.co/gp4upXd)

[Stochastic Processes and Filtering Theory, Andrew H. Jazwinski](http://a.co/cm5zfQu) 

[Kalman Filtering: Theory and Practice, Mohinder S. Grewal, Angus P. Andrews](http://a.co/6hAa35c)

[Capital Ideas: The Improbable Origins of Modern Wall Street, Peter L. Bernstein](http://a.co/1Y1DR9p)

[A Demon of Our Own Design: Markets, Hedge Funds, and the Perils of Financial Innovation, Richard Bookstaber](http://a.co/4FvnyfB)

in ubuntu, install or upgrade os-level dependencies

    sudo apt-get -qq update -qy
    sudo apt-get -qq install -y python3.6 python3-venv python3-pip

clone the git project, start a venv virtual environment, install the package

    git clone git@gitlab.com:noahhsmith/statespace.git statespace
    cd statespace
    python3 -m venv venv
    . venv/bin/activate
    python3 setup.py install

