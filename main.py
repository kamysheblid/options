import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--price-range', '-p',nargs=2, default=[3000,5000], type=float, help='Price Range. def 3000 5000')
parser.add_argument('--strike', '-s', type=float, default=4500, help='Strike Price. Def 4500')
parser.add_argument('--time', '-t', type=int, default=90, help='Days to Expiry. Def 90')
parser.add_argument('--volatility', '-V', type=float, default=75, help='Volatility as percentage. Def 75')
parser.add_argument('--rate', '-r', type=float, default=5, help='Risk Free Rate as percentage. Def 5')
parser.add_argument('--dividend', '-d', type=float, default=0.1, help='Dividend as percentage. Def 0.1')
parser.add_argument('--option-type', '-o', type=str, default='c', help='Option Type. Def c')
parser.add_argument('--debug-mode','-D', action='store_true', help='Enable DEBUG mode so that program hot reloads when code is changed. Def Off')
parser.add_argument('--server-port', '-P', type=int, default=8050, help='Set Port number for web host. Def 8050')
parser.add_argument('--proxy', type=str, default=None, help='Proxy. Works by setting ALL_PROXY environment variable in the python program')

args = parser.parse_args()

if args.proxy:
    import os
    os.environ['ALL_PROXY'] = args.proxy

if __name__ == '__main__':
    import plot
    plot.main(args)
