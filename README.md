<img src="https://gitlab.com/noahhsmith/statespace/raw/master/docs/images/udfactoring.png" align="center" width="300" height="300"/>

uncertainty, confidence, knowledge

their evolution with time and various processes, including noise and observations

risk and the cost of errors

[The Flaw of Averages: Why We Underestimate Risk in the Face of Uncertainty, Sam L. Savage](http://a.co/cDDBO9p)

[Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods, James V. Candy](http://a.co/gp4upXd)

[Capital Ideas: The Improbable Origins of Modern Wall Street, Peter L. Bernstein](http://a.co/1Y1DR9p)

[Stochastic Processes and Filtering Theory, Andrew H. Jazwinski](http://a.co/cm5zfQu) 

[A Demon of Our Own Design: Markets, Hedge Funds, and the Perils of Financial Innovation, Richard Bookstaber](http://a.co/4FvnyfB)

[Kalman Filtering: Theory and Practice, Mohinder S. Grewal, Angus P. Andrews](http://a.co/6hAa35c)

in ubuntu, install or upgrade os-level dependencies

    sudo apt-get -qq update -qy
    sudo apt-get -qq install -y python3.6 python3-venv python3-pip

clone the git project, start a venv virtual environment, install the package, and test

    git clone git@gitlab.com:noahhsmith/statespace.git statespace
    cd statespace
    python3 -m venv venv
    . venv/bin/activate (same as source venv/bin/activate)
    python3 setup.py install
    python3 statespace --xbp
    
usage hints

    ~/statespace$ venv/bin/python -m statespace -h
    usage: statespace [-h] [--lzbp] [--xbp] [--spbp] [--sspf] [--abp]
    
    optional arguments:
      -h, --help  show this help message and exit
      --lzbp      linearized bayesian processor, linearized kalman filter
      --xbp       extended bayesian processor, extended kalman filter
      --spbp      sigma-point bayesian processor, unscented kalman filter
      --sspf      state space particle filter, sequential monte carlo processor
      --abp       adaptive bayesian processors, joint bayesian state/parameteric
                  processors

