# CS224n Assignments: a Python 3 repo

[CS224n](http://web.stanford.edu/class/cs224n/index.html) is awesome, the original code of assignments is based on Python 2, I'm a fan of Python 3 though, for those who prefer 3 to 2, feel free to clone or fork this repo, have fun with it!

**bug fix(early 2018 version)**

Assignment 3:

  * q1_window.py 
    
    WindowModel(NERModel).create_feed_dict(), the default `dropout` should be 0(i.e. `keep_prob=1` in 
    TensorFlow by default, we would like to disable droupout when predicting). However if you treate `dropout` as `keep_prob`, then there's no trouble. 
 
  * q2_rnn.py

    In `RNNModel(NERModel).preprocess_sequence_data()`, should pass `window_size=self.config.window_size()` calling `featurize_windows()`, or your model will crash if you change `window_size` in `Config`.




Progress

- Assignment 1:
  1. Softmax  ✔️
  2. Neural Network Basics ✔️
  3. word2vec ✔️
  4. Sentiment Analysis(I can not fix encoding error in Python3, this was done by Python2)  ✔️

- Assignment 2:
  1. Tensorflow Softmax✔️
  2. Neural Transition-Based Dependency Parsing ✔️

  |  model   | training loss(debug) | dev UAS(debug) | training loss(full) | dev UAS(full) |
  | :------: | :------------------: | :------------: | ------------------- | ------------- |
  | baseline |        0.1203        |     69.97      | 0.0703              | 86.68         |
  | + L2 reg |        0.2402        |     66.38      | 0.1212              | 85.81         |

  adding L2 regularization hurts the model, for the model is simple(low capacity), regularization actually reduce the capacity of baseline model.


- Assignment 3:

  1. A window into NER ✔️

     * best score(Entity level P/R/F1): 0.85/0.87/0.86

     * Token-level confusion matrix

       | gold\guess | PER  | ORG  | LOC  | MISC | O     |
       | :--------: | :--: | :--: | :--: | :--: | ----- |
       |    PER     | 2968 |  45  |  53  |  18  | 65    |
       |    ORG     |  92  | 1738 |  67  |  88  | 107   |
       |    LOC     |  35  |  87  | 1931 |  16  | 25    |
       |    MISC    |  32  |  53  |  32  | 1056 | 95    |
       |     O      |  40  |  43  |  22  |  37  | 42617 |

     * Token-level scores:

       | label | acc  | prec | rec  | f1   |
       | ----- | ---- | ---- | ---- | ---- |
       | PER   | 0.99 | 0.94 | 0.94 | 0.94 |
       | ORG   | 0.99 | 0.88 | 0.83 | 0.86 |
       | LOC   | 0.99 | 0.92 | 0.92 | 0.92 |
       | MISC  | 0.99 | 0.87 | 0.83 | 0.85 |
       | O     | 0.99 | 0.99 | 1    | 0.99 |
       | micro | 0.99 | 0.98 | 0.98 | 0.98 |
       | macro | 0.99 | 0.92 | 0.9  | 0.91 |
       | not-O | 0.99 | 0.91 | 0.89 | 0.9  |
  
  2. Recurrent neural nets for NER ✔️
  3. Grooving with GRUs (30 points) ✔️
      * best score(Entity level) P/R/F1: 0.87/0.86/0.86
    