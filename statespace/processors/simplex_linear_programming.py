"""linear programming using matrix and vector representation, as close to the literature as possible. besides being
valuable at work, it's also lots of fun historically - and from 'aesthetic history of ideas' perspective. those
latter two - history and aesthetics - are what we're really looking at here.

with the first generations of hardware in the 50s, what were some of the commercial motivations? well, they could
simulate, and played a critical role there - see enewetak atoll. they could help with command and control,
as in the sage system. but what else could justify the big price tags? in a nutshell - resource allocation. maybe
sounds boring - but more than enough motivation to help drive speedy commercialization across industries. there's a
reflection of all this in the growth of the parallel cultural concept of 'systems analysis'. possibly more intriguing
than 'operations research' - even a bit disreputable and adventurous?

one thing we can say is - the main thing simplex requires is matrix inversion. factorizations such as cholesky can
also enter the picture for diagnostics - but overall, simplex helped drive the early waves of work on matrix
inversion. would love to explore this further, if good records are out there. not surprisingly, dantzig and von
neumann did publish specifically on this topic, around 1950 - likely based on work underway on the eniac from 1946
forward."""
import numpy as np


def main():
    A = np.array([[-2, 1, 1, 0, 0], [-1, 2, 0, 1, 0], [1, 0, 0, 0, 1]])
    b = np.array([2, 7, 3])
    c = np.array([-1, -2, 0, 0, 0])
    x = simplex(A, b, c)
    return x


def simplex(A, b, c):
    """revised simplex algorithm. """
    m, n, o = A.shape[0], A.shape[1], A.shape[1] - A.shape[0]
    ns, v1, v2 = [i for i in range(n)], np.array(c[o:]), np.array(c[:o])
    Binv = np.linalg.inv(A[:, ns[o:]])
    while not np.min(v2 - v1 @ Binv @ A[:, ns[:o]]) > 0:
        n1 = np.argmin(v2 - v1 @ Binv @ A[:, ns[:o]])
        t1, t2 = Binv @ b, Binv @ A[:, ns[n1]]
        n2 = np.argmin([t1[i] / t2[i] if t2[i] > 0 else np.inf for i in range(m)])
        ns[n1], ns[n2 + o], v1[n2], v2[n1] = ns[n2 + o], ns[n1], v2[n1], v1[n2]
        Binv = np.linalg.inv(A[:, ns[o:]])
    return Binv @ b


if __name__ == "__main__":
    main()
