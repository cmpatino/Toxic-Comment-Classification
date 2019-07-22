# Toxic Comment Classification Challenge

This repository has the solutions we developed for he [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) on Kaggle. The challenge was to predict different tags for online comments. The possible tags for a comment were:

+ toxic
+ severe_toxic
+ obscene
+ threat
+ insult
+ identity_hate

## Solution

Our approach was to implement three different sequence models to compare their performance. The three models were an implementation of BERT, a bidirectional GRU followed by a Capsule layer, and a baseline model with an LSTM layer. The [mean column-wise ROC AUC score](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview/evaluation) on Kaggle for each of the models was:
+ BERT: 0.98437
+ CapsuleNet: 0.9765
+ Baseline: 0.9739
