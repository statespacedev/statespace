
## distributions and decisions

uncertainty, confidence, knowledge

their evolution with various processes

risk and the cost of errors

## processors  

lzkf, linearized processor, linearized kalman filter

ekf, extended processor, extended kalman filter

spkf, sigma-point sampling processor, unscented kalman filter

pf, sequential monte carlo sampling processor, particle filter

sd, sequential detector

## background

[The Flaw of Averages: Why We Underestimate Risk in the Face of Uncertainty, Sam L. Savage](http://a.co/cDDBO9p)

[Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods, James V. Candy](http://a.co/gp4upXd)

## notes

    ~/statespace$ venv/bin/python -m statespace -h
    usage: statespace [-h] [-t] [-lzkf] [-ekf] [-ukf] [-pf]
    
    optional arguments:
      -h, --help  show this help message and exit
      -t          test the package
      -lzkf       linearized processor
      -ekf        extended processor
      -ukf        sigma-point sampling processor
      -pf         sequential monte carlo processor

*13.08.18*

testing [pypi packaging](https://test.pypi.org/project/statespace/) 

