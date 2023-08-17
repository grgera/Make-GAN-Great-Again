# Paraphrase Generation with SeqGAN

Experimental adoptation neural network model based on SeqGAN that generates paraphrases for a given sentence. For this model [Quora](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) dataset is used.

# Usage

## Preparation

###### 1: Install dependencies

`pip3 install -r requirements.txt`

###### 2: Activate nltk

`nltk.word_tokenize()` method needs `punkt` package.

```bash
$ python3
>>> import nltk
>>> nltk.download('punkt')
```

###### 3: Install nlg-eval (for evaluation only)

From instructions here: https://github.com/Maluuba/nlg-eval

```bash
$ git clone https://github.com/Maluuba/nlg-eval.git
$ cd nlg-eval
$ pip3 install -e .
$ nlg-eval --setup
```

###### 4: Download pretrained word embeddings

- Download pretrained word embeddings
    - [GloVe](https://nlp.stanford.edu/projects/glove/)
    - [word2vec](https://code.google.com/archive/p/word2vec/)
    - [fastText](https://github.com/icoxfog417/fastTextJapaneseTutorial)
    - [ELMo](https://allennlp.org/elmo)
- Extract into plain text files and put under `dataset/pretrained_word_embeddings/original/`

###### 5: Parse pretrained word embeddings
    
```bash
$ python3 tools/parse_emb.py <word-embedding-file> <output-vector-file> <output-info-file>
```

## Train Model

```bash
$ python3 -m /src/train.py
```

## Evaluate Model

Paraphrases will be generated, and the BLEU-2 and METEOR evaluation metrics will be calculated. The model path below should be the directory path of the pretrained model and end in slash. Pathbuilder tool will parse everything with this.

```bash
$ python3 -m tools.evaluate model/<model-params>/pretrain/<pretrained-model-params>/
```

## Sample Results

Original: what are the safety precautions on handling shotguns proposed by the nra in north carolina ? \
Reference: what are the safety precautions on handling shotguns proposed by the nra in vermont ? \
**Paraphrase: what are the safety precautions proposed by the nra precautions on handling shotguns proposed**

Original: how do i improve professional email writing skills ? \
Reference: how do i improve my email writing skills ? \
**Paraphrase: how can i improve my writing skills ?**

Original: how do i get rid of my belly fat ? \
Reference: what should i do for belly fat ? \
**Paraphrase: how can i i lose fat ? how do i get rid of weight ?**

Original: what is quickbooks tech support number in arizona ? \
Reference: what is the quickbooks customer support phone number usa ? \
**Paraphrase: what is the quickbooks softwares support phone number ?**


# References to original SeqGAN implementation

- [LantaoYu/SeqGAN](https://github.com/LantaoYu/SeqGAN)
- [suragnair/seqGAN](https://github.com/suragnair/seqGAN)