from utils.config import parse_args

def main(args):
    print(args.dataset)

if __name__ == '__main__':
    args = parse_args()
    main(args)