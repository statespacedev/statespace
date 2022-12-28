"""what's it all about? just some of the core applications of computing in the 40s, 50s, and 60s that really drove
those early generations of hardware and what eventually became known as software - what many of those machines in the
old film clips were actually doing, and why fortran was such a big deal. tons of large scale linear algebra - so
matrix and vector operations, in a language that didn't need or want character strings. a fundamental dividing line
between numerical computing and everything else - if you cared about strings, you became a cobol person. that divide
is still with us today in fortran/c/cpp/python and cobol/lisp/java/everything-else. if your stuff had anything to do
with high-performance-computing/super-computing/embedded-real-time, you were in the first camp - otherwise, not.

the beginnings were dantzig and von neumann linear programming - all woven through operations research around the
time of the transition from mechanical to digital computing - let's say, early cold war. one of the things kalman
brought in on top of this was nonlinear elements capable of representing uncertainty - the classic covariance matrix
quadratic form ATAx. this was happening across fields, as the same appeared in finance via markowitz portfolio theory
around the same time - what was driving all this organic evolution was the increasing floating point computing power
- flops - and the ability to harness it via fortran. and the impact was immediate - let's say high cold war - late
50s early 60s - with the early integrated circuitry of the apollo guidance computer and minuteman missile guidance
system. how do you get your life-or-death guidance systems to make better decisions? the birth of machine learning.

here's a fun story illustrating all this evolution. high cold war - increasing flops and fortran immediately drive
the appearance of covariance matrices in what was purely linear optimization - but we're definitely on the edge here,
because those quadtratic forms immediately blow up our numerical ranges - there's a crescendo of cries - numerical
instability! we don't have enough bits in our floating point representations - plain and simple. so what immediately
happens is a new industry within applied mathematics and electrical engineering - optimize the fortran and floating
point units, and factorize the matrices! what does factorize the matrices mean? in a nutshell, only represent
something very much like their square roots - at least internally within our computations. all of this dominates the
60s, 70s, and 80s - let's say high cold war verging into baroque/neo-classical cold war - the era when the name
'cray' could inspire awe. one of our objectives here in project statespace is a straightforward minimalistic
representation of the real-world factorized forms, alongside the classic textbook forms. names like thornton and
bierman aren't widely remembered today - but without the factorized form square root ekf, the apollo guidance
computer wasn't possible at that time - and they were crucial for the transit/gps satnav systems as well. if you know
about cholesky decomposition, ud decomposition, singular value decomposition, etc - then you know."""
import sys
import argparse


def main():
    """configure and call process_model(). -h and --help are built in, so 'python statespace -h' displays an options
    text. """
    cli = argparse.ArgumentParser()
    cli.add_argument('-p', '--processor', choices=['ekf', 'spkf', 'pf'], default='ekf', type=str)
    cli.add_argument('-m', '--model', choices=['one', 'three', 'bearings'], default='three', type=str)
    cli.add_argument('-f', '--factorized', dest='factorized', action='store_true', default=False)
    conf = cli.parse_args()
    if not len(sys.argv) > 1:
        conf.processor = 'pf'
        conf.model = 'three'
        conf.factorized = True
    process_model(conf)


def process_model(conf):
    """bring in a processor and a model, process the model, evaluate the results. """
    processor, model = None, None
    match conf.processor:
        case 'ekf':
            from statespace.processors.extended_kalman_filter import Kalman
            processor = Kalman(conf)
        case 'spkf':
            from statespace.processors.sigma_point_kalman_filter import SigmaPoint
            processor = SigmaPoint(conf)
        case 'pf':
            from statespace.processors.particle_filter import Particle
            processor = Particle(conf)
    match conf.model:
        case 'one':
            from statespace.models.one_state import Onestate
            model = Onestate(conf)
        case 'three':
            from statespace.models.three_state import Threestate
            model = Threestate(conf)
        case 'bearings':
            from statespace.models.bearings_only import BearingsOnly
            model = BearingsOnly(conf)
    processor.run(model)
    model.eval.estimate(processor.log)
    # model.eval.autocorr.run(processor.log)
    model.eval.show()


if __name__ == "__main__":
    main()
