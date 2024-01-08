# Text Emotion Analysis
> Identifying emotions in text with natural language models and deep learning techniques.

## Project Overview
This project contains training, testing, and inference scripts for a natural language model capable of classifying emotions within a given text. The datasets utilized include Google's GoEmotions dataset as well as DAIR.AI's emotion dataset, both available on HuggingFace. The primary architectures utilized are a basic LSTM and a fine-tuned SqueezeBERT model.

## Model Architecture

### Long Short-Term Memory
This implementation utilizes Long Short-Term Memory modules first introduced in [[1]](#1), which have shown to help the model learn relationships between textual word embeddings across arbitrarily long sequences. The LSTM structure also combats vanishing gradients that are prominent in basic recurrent neural networks by allowing effective gradient flow.

The model first takes in tokenized input and retrieves corresponding GloVe embeddings [[2]](#2). These embeddings are then processed through a 2-layer LSTM and a fully connected network (prediction head) to produce an emotion class prediction.

For reference, the LSTM module is depected below (from Christopher Olah's [blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)):
![LSTM module](references/lstm.png)

Note: This version is an older commit and can be accessed [here](https://github.com/joe-lin-tech/playground/commit/4a88246798ed6361afea63272c74f66402c99ec5).

### SqueezeBERT
The current project version fine tunes a pretrained SqueezeBERT, which is at its base a BERT-based architecture but integrates core Computer Vision techniques to boost model efficiency. SqueezeBERT replaces position-wise fully connected layers in BERT with grouped convolutions, which are much less computationally expensive [[3]](#3). This enables SqueezeBERT to achieve reduced inference times, especially on smaller devices.

For reference, the base BERT architecture and training procedure that SqueezeBERT is built on top of is shown below [[4]](#4):
![BERT architecture](references/bert.jpeg)

## References
<a id="1">[1]</a> 
S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. 1997.

<a id="2">[2]</a>
J. Pennington and R. Socher and Christopher D. Manning. 2014.

<a id="3">[3]</a>
F. Iandola and A. Shaw and R. Krishna and K. Keutzer. SqueezeBERT: What can computer vision teach NLP
about efficient neural networks? 2020.

<a id="4">[4]</a>
J. Devlin and M. Chang and K. Lee and K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding. 2019.