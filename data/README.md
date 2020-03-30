# Dataset creation

Instructions to download and format datasets for this projects.
WikiBIO formatting script comes from [this great repo](https://github.com/tyliupku/wiki2bio/blob/master/preprocess.py)

Scripts assume you are in the `data/` repository, with the following file:

```
.
├── co_occurrence.py
├── count_co_occurrences.py
├── dep_parsing_spacy.py
├── format_weights.py
├── format_wikibio.py
├── pos_tagging.py
├── remove_long_sentences.py
├── run_ner.py
├── utils_ner.py        
```

## WikiBIO download and basic formatting

Download and unpack the dataset:

```bash
git clone https://github.com/DavidGrangier/wikipedia-biography-dataset.git
cd wikipedia-biography-dataset
cat wikipedia-biography-dataset.z?? > tmp.zip
unzip tmp.zip
rm tmp.zip
mkdir ../wikibio
mv  wikipedia-biography-dataset/ ../wikibio/raw
cd ..
rm -rf wikipedia-biography-dataset
```

create the dataset:

```bash
python3 format_wikibio.py
```

The whole OpenNMT-ready dataset can now be found in `wikibio/full`:
```
.
└── wikibio/
│   ├── raw/
│   ├── process_data/
│   └── full/
├── co_occurrence.py
├── count_co_occurrences.py
├── dep_parsing_spacy.py
├── format_weights.py
├── format_wikibio.py
├── pos_tagging.py
├── remove_long_sentences.py
├── run_ner.py
├── utils_ner.py        
```

However, this work relies on Part-of-Speech tagging and Dependency parsing. The former is done using a BERT model
fine-tuned using huggingface/transformers [ner script](https://github.com/huggingface/transformers/tree/master/examples/ner),
which requires a max length setting. Manual exploration showed that the minimum size which will keep all testset
examples is 256.

You can trim the few (approx 13 examples in train+valid) remaining sentences using (for now, only BERT-based model are
supported):

```bash
python3 remove_long_sentences.py --max_size 256 --model bert-base-uncased
```


## WikiBIO pre-processing
In order to train our multi-branch decoder, every training token must be associated to an "hallucination score", which,
in turn, is computed according to the sentence's Part-of-Speech tags and Dependency parsing. 

### Part-of-Speech tagging

We use huggingface/transformers to train a PoS-tagger on [UniversalDependencies english treebank](https://github.com/UniversalDependencies/UD_English-ParTUT).

In order to do this, you will need the following libraries: `pip install transformers pyconll seqeval tensorboardX`

To train the model and tag WikiBIO training set (training is quick (few minutes) however tagging the ~700K
WikiBIO examples can take one or two hours):

```bash
python3 pos_tagging.py --do_train --do_tagging train --gpus 0 1
```

You should now find the PoS file in `wikibio/train_pos.txt`

### Dependency parsing

We use [SpaCy](http://www.spacy.org) to parse WikiBIO training set:

```bash
python3 dep_parsing_spacy.py --orig wikibio/train_output.txt --dest wikibio/train_deprel.txt --format sent
```

You should now find the dependency relations file in `wikibio/train_deprel.txt`

### Sentence scoring

Finally, we compute the tokens' scores basing on the PoS tags, the Dependency relations and the input-output
co-occurrences:

```bash
python3 co_occurrence.py --freq-input wikibio/train_tables.jl --freq-pos wikibio/train_pos.txt --frequencies wikibio/train_freq.pickle \
                         --input wikibio/train_tables.jl --pos wikibio/train_pos.txt --deprel wikibio/train_deprel.txt \
                         --scores wikibio/train_h.txt
```

The script will create the scores file `wikibio/train_h.txt` (this takes 6 to 8 hours, be patient!), which will be used
to train our multi-branch decoder.
