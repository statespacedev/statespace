import argparse, sys
sys.path.append('statespace')
import class_statespace

def main():
    parser = argparse.ArgumentParser('statespace')
    parser.add_argument('-t', help='test the package', action='store_true')
    parser.add_argument('-lzkf', help='linearized processor', action='store_true')
    parser.add_argument('-ekf', help='extended processor', action='store_true')
    parser.add_argument('-ukf', help='sigma-point sampling processor', action='store_true')
    parser.add_argument('-pf', help='sequential monte carlo processor', action='store_true')
    args = parser.parse_args()

    ss = class_statespace.Statespace()
    if args.t:
        print('test successful')

    if args.lzkf:
        ss.lzkf()
    if args.ekf:
        ss.ekf()
    if args.ukf:
        ss.ukf()
    if args.pf:
        ss.pf()

if __name__ == "__main__":
    main()