import argparse, sys
sys.path.append('statespace')

def main():
    parser = argparse.ArgumentParser('statespace')
    parser.add_argument('-t', '--test', help='test the project', action='store_true')
    parser.add_argument('-l', '--lzkf', help='linearized processor, linearized kalman filter', action='store_true')
    parser.add_argument('-e', '--ekf', help='extended processor, extended kalman filter', action='store_true')
    parser.add_argument('-u', '--ukf', help='sigma-point sampling processor, unscented kalman filter', action='store_true')
    parser.add_argument('-p', '--pf', help='sequential monte carlo processor, particle filter', action='store_true')
    args = parser.parse_args()

    if args.test:
        print('test successful')

    if args.lzkf:
        import lzkf
        lzkf.main()

    if args.ekf:
        import ekf
        ekf.main()

    if args.ukf:
        import ukf
        ukf.main()

    if args.pf:
        import pf
        pf.main()

if __name__ == "__main__":
    main()