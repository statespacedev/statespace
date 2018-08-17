import argparse

def main():
    parser = argparse.ArgumentParser('starid')
    parser.add_argument('-t', help='test the package', action='store_true')
    args = parser.parse_args()
    if args.t:
        print('test successful')

if __name__ == "__main__":
    main()