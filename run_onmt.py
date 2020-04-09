from onmt.bin.preprocess import main as preprocess
from onmt.bin.translate import main as translate
from onmt.utils.parse import ArgumentParser
from onmt.bin.train import main as train


if __name__ == '__main__':
    parser = ArgumentParser()
    
    # Simply add an argument for preprocess, train, translate
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preprocess", dest='preprocess', action='store_true',
                      help="Activate to preprocess with OpenNMT")
    mode.add_argument("--train", dest='train', action='store_true',
                      help="Activate to train with OpenNMT")
    mode.add_argument("--translate", dest='translate', action='store_true',
                      help="Activate to translate with OpenNMT")
    
    mode, remaining_args = parser.parse_known_args()
    
    if mode.preprocess:
        preprocess(remaining_args)
    elif mode.train:
        train(remaining_args)
    elif mode.translate:
        args = translate(remaining_args)
        
        # TODO compute scores directly after the translation is done
    