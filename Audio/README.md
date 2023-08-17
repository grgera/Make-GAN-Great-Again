# Fre-GAN: Adversarial Frequency-consistent Audio Synthesis

The repository contains my unofficial implementation of vocoder model [Fre-GAN](https://arxiv.org/pdf/2106.02297.pdf) and some examples of its work.

# Running the model
+ To run code in Google Colab, open `fregan_experiments.ipynb`, run the first cell to clone the repository and then follow the instructions to get data.
+ As steps, you will need, first preprocess
```
python preprocess.py
```
+ Then you can easily train
```
python train.py -n name --training_epochs 3000
```
+ And of course, evaluation step, before that you need to put your .wav file into `/test` directory
```
python inference.py -p ./fg_model/test/fg-3000.pt
```
___
## Google Colab:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grgera/Make-GAN-Great-Again/Audio/blob/main/fregan_experiments.ipynb)
___
## References
In my implementation there may be constructions from implementations:
+ [HiFi-GAN](https://github.com/jik876/hifi-gan)
+ [Fre-GAN](https://github.com/rishikksh20/Fre-GAN-pytorch)

