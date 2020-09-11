from onmt.bin.translate import main as translate

import pkg_resources
import argparse
import os, re


def posint(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive int")
    return ivalue


def strposint(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a strictly positive int")
    return ivalue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', dest='dataset', default='wikibio')#,
                        #choices=['wikibio', 'webnlg'])
    parser.add_argument('--setname', dest='setname', default='test',
                        choices=['test', 'valid'])
    parser.add_argument('--experiment', '-e', dest='experiment')
    parser.add_argument('--start-step', dest='start_step', default=0, type=posint)
    parser.add_argument('--step-size', dest='step_size', default=1, type=strposint)
    parser.add_argument('--bms', dest='bms', default=10, type=strposint,
                        help="beam size")
    parser.add_argument('--bsz', dest='bsz', default=64, type=strposint,
                        help="batch size")
    parser.add_argument('--blk', dest='blk', default=0, type=posint, 
                        help="block ngram repeats")
    parser.add_argument('--gpu', dest='gpu', default=0, type=posint)
    parser.add_argument('--small', dest='small', action='store_true')
    
    parser.add_argument('--weights', dest="weights", nargs="+", type=float)
    
    args = parser.parse_args()
    
    print(f"Batch translating models from experiment {args.experiment}")
    
    exp_dir = pkg_resources.resource_filename(__name__, 'experiments')
    exp_dir = os.path.join(exp_dir, args.dataset, args.experiment)
    mdl_dir = os.path.join(exp_dir, 'models')
    gns_dir = os.path.join(exp_dir, 'gens', args.setname)
    
    def get_step(fname):
        return int("".join(re.findall("([0-9]+?)[.]pt", fname)))
    
    models = [fname for fname in os.listdir(mdl_dir) if fname.endswith('.pt')]
    models = sorted(models, key=get_step, reverse=False)

    datadir = pkg_resources.resource_filename(__name__, 'data')
    src = os.path.join(datadir, args.dataset, 
                       'small' if args.small else '', 
                       f'{args.setname}_input.txt')
    tgt = os.path.join(datadir, args.dataset,
                       'small' if args.small else '',  
                       f'{args.setname}_output.txt')
    
    n_processed = -1
    for idx, fname in enumerate(models):
        step = get_step(fname)
        
        n_processed += 1
        if n_processed % args.step_size or step < args.start_step:
            print(f"Skipping step {step}")
            continue
            
        print(idx, "translating", fname)
        
        model = os.path.join(mdl_dir, fname)
        output_pfx = f'bms{args.bms}.blk{args.blk}.bsz{args.bsz}'
        if args.weights:
            output_pfx += f'.wgt{args.weights}'.replace(' ', '')
        output_pfx += f'.{"small" if args.small else "full"}'
        output = os.path.join(gns_dir, f'{output_pfx}-step_{step}.txt')
        log_file = os.path.join(exp_dir, 'translate-log.txt')
        
        cmd_args = [
            f'-model {model}',
            f'-src {src}',
            f'-tgt {tgt}',
            f'-output {output}',
            f'-beam_size {args.bms}',
            f'-block_ngram_repeat {args.blk}',
            f'-batch_size {args.bsz}',
            f'-gpu {args.gpu}',
            f'-log_file {log_file}'
        ]
        
        if args.weights:
            weights = ' '.join([str(w) for w in args.weights])
            cmd_args.append(f'-rnn_weights {weights}')
        
        translate(f'--config translate.cfg {" ".join(cmd_args)}')
