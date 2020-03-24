import pkg_resources
import argparse
import os


def truncate_wikibio_files(orig, dest, setname, length=10):
    assert setname in ['train', 'valid', 'test']
    assert int(length) == length and length > 0  # check positiv int
    
    # files with 1 example per line
    for suffix in ['input.txt', 'output.txt', 'tables.jl']:
        origpath = os.path.join(orig, f'{setname}_{suffix}')
        destpath = os.path.join(dest, f'{setname}_{suffix}')
        with open(origpath, mode="r", encoding='utf8') as f, \
                open(destpath, mode="w", encoding='utf8') as g:
            for idx, line in enumerate(f):
                if idx >= length:
                    break
                g.write(line)
                
    # files with one token per line
    for suffix in ['pos.txt']:
        origpath = os.path.join(orig, f'{setname}_{suffix}')
        destpath = os.path.join(dest, f'{setname}_{suffix}')
        with open(origpath, mode="r", encoding='utf8') as f, \
                open(destpath, mode="w", encoding='utf8') as g:
            nline = 0
            for tokens in f:
                if tokens.strip():
                    g.write(tokens.strip() + '\n')
                else:
                    g.write('\n')
                    nline += 1
                    if nline >= length: break


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', dest='folder',
                        help='Name of the folder to put the truncated files'
                        ' (will be inside wikibio/)')
    parser.add_argument('--max_size', dest='max_size', type=float, default=1e3,
                        help='How many lines to keep')
    parser.add_argument('--setnames', dest='setnames', nargs='+',
                        help='Specify any combinason of [train, valid, test]')
    
    args = parser.parse_args()

    orig = pkg_resources.resource_filename(__name__, 'wikibio')
    dest = os.path.join(orig, args.folder)
    if not os.path.exists(dest): os.mkdir(dest)

    for setname in args.setnames:
        truncate_wikibio_files(orig, dest, setname, args.max_size)
