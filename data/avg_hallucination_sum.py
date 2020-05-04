import argparse
from utils import TaggedFileIterable
from statistics import mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='<model>_h.txt file, containing the generated sentences, one token per line, '
                                     'each one along with its hallucination score. '
                                     'Sentences are separated by an empty line.')
    args = parser.parse_args()
    
    tagged_hypotheses = TaggedFileIterable.from_filename(args.file, func=lambda x,y: float(y))
    avg_score = mean([sum(h) for h in tagged_hypotheses])
    print(f'{args.file}:\t{avg_score:.4f}')

    # scores = []
    # accumulator = 0
    # with open(args.file) as f:
    #     for line in f:
    #         if line.strip():
    #             accumulator += float(line.strip().split()[1])
    #         else:
    #             scores.append(accumulator)
    #             accumulator = 0

    # avg_score = mean(scores)
    # print(f'{args.file}:\t{avg_score:.4f}')
