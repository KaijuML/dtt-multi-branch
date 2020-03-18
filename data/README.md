# Dataset creation

Instructions to download and format datasets for this projects.
WikiBIO formating script comes from [this great repo](https://github.com/tyliupku/wiki2bio/blob/master/preprocess.py)

Scripts assume you are in the `data/` repository, with the following file:

```
.
├── format_wikibio.py
├── pos_tagging.py        
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

## WikiBIO Part-of-Speech tagging

We use huggingface/transformers to train a PoS-tagger on [UniversalDependencies english treebank](https://github.com/UniversalDependencies/UD_English-ParTUT)

In order to do this, you will need the following libraries: `pip install transformers pyconll`

To train the model and tag wikibio train/valid/test sets:

```
python pos_tagging.py --do_train --do-tagging train valid test
```