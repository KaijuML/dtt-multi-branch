# Dataset creation

Instructions to download and format datasets for this projects.
WikiBIO formating script comes from [this great repo](https://github.com/tyliupku/wiki2bio/blob/master/preprocess.py)

Scripts assume you are in the `data/` repository, with the following file:

```
.
├── format_wikibio.py
├── pos_tagging.py        
├── remove_long_sentences.py        
├── run_ner.py        
├── utils_ner.py        
```

## WikiBIO download and basic formating

Download and unpack the dataset:

```
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

```
python format_wikibio.py
```

The whole OpenNMT-ready dataset can now be found in `wikibio/full`:
```
.
└── wikibio/
│   ├── raw/
│   ├── process_data/
│   └── full/
├── format_wikibio.py
├── pos_tagging.py        
├── remove_long_sentences.py        
├── run_ner.py        
├── utils_ner.py        
```

However, in this work with rely on Part-of-Speech tagging, using a BERT model fine-tuned using hugginface/transformers [ner script](https://github.com/huggingface/transformers/tree/master/examples/ner), which requires a max length setting. Manual exploration showed that the minimum size which will keep all testset examples is 256.

You can trim the few (approx 13 examples in train+valid) remaining sentences using (for now, only BERT-based model are supported):

```
python remove_long_sentences --max_size 256 --model bert-base-uncased
```


## WikiBIO Part-of-Speech tagging

We use huggingface/transformers to train a PoS-tagger on [UniversalDependencies english treebank](https://github.com/UniversalDependencies/UD_English-ParTUT)

In order to do this, you will need the following libraries: `pip install transformers pyconll seqeval tensorboardX`

To train the model and tag wikibio train/valid/test sets (training is quick (few minutes) however tagging the ~700K wikibio examples can take one or two hours):

```
python pos_tagging.py --do_train --do_tagging train valid test --gpus 0 1
```