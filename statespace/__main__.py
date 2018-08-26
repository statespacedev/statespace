import argparse, sys
sys.path.append('statespace')

def main():
    parser = argparse.ArgumentParser('statespace')
    parser.add_argument('--lzbp', help='linearized bayesian processor, linearized kalman filter', action='store_true')
    parser.add_argument('--xbp', help='extended bayesian processor, extended kalman filter', action='store_true')
    parser.add_argument('--spbp', help='sigma-point bayesian processor, unscented kalman filter', action='store_true')
    parser.add_argument('--sspf', help='state space particle filter', action='store_true')
    parser.add_argument('--jbp', help='joint bayesian state/parameteric processors', action='store_true')
    args = parser.parse_args()

    if args.lzbp:
        import bsp1_lzbp
        bsp1_lzbp.main()

    if args.xbp:
        import bsp1_xbp
        bsp1_xbp.main()

    if args.spbp:
        import bsp1_spbp
        bsp1_spbp.main()

    if args.sspf:
        import bsp1_sspf
        bsp1_sspf.main()

    if args.jbp:
        import bsp1_jbp
        bsp1_jbp.main()

if __name__ == "__main__":
    main()