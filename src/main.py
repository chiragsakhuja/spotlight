import search
import options
import interface
import sys
import os

def log(out_file, format, *args):
    if out_file:
        out_file.write(format.format(*args) + '\n')
    else:
        print(format.format(*args))

def main():
    args = options.get_args()

    print(' '.join(sys.argv[1:]))

    if args.output_to_file:
        out_file = open(os.path.join(args.output_dir, args.output_filename), 'w')
    else:
        out_file = None

    for i in range(args.trials):
        log(out_file, '---------- trial {} ----------\n', i)
        search.run_search(args, out_file)
        failure_stats = sorted([(v, k) for k, v in interface.failure_stats.items()], reverse=True)
        log(out_file, '{}', '\n'.join([str(x) for x in failure_stats]))
        log(out_file, '{}', str(sum([x[0] for x in failure_stats])))
    if out_file:
        out_file.close()

if __name__ == '__main__':
    main()