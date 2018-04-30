CS224n Assignments: a Python 3 repo

CS22n is awesome, the original code of assignments is based on Python 2, I'm a fan of Python 3 though, for those who prefer 3 to 2, feel free to clone or fork this repo, have fun with it!


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

  ​

  ...
