# FPENN
Implementation of "FPENN Forest path encoding for neural networks".

This paper accepted to Information Fusion.

If you find this code useful in your research, please cite the paper: (TODO: Add citation) 

# Environment Setting
  *Keras
  *Sickit learn
  *Gensim


# Brief Introduction and quick start
This repository is built for the experimental part of our paper. We add all the models and its different variations.
For now, our models only supourts binary clasification task with non-categorical features - i.e. - in order to start experiments you have to make sure:
1. It is a .csv file
2. Binary classification task.
3. Categorical features must be transformed into numerical ones.
4. The "class" column must be the last one in the .csv file.
5. All data .csv files must be insied \Data and in the same level of the script.
6. A \Results folder must be in the same level of the script.

Once you have all the prequisits ready you can run the script (for example - only_rf.py).

