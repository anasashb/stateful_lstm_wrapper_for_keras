# Wrapper Class for implementing stateful LSTM models in Keras
The wrapper class was designed for personal use and is thus tailored for LSTM models for step-ahead financial time series forecasting. It is aimed at easily implementing stateful LSTMs in Keras and includes functions that replicate fitting, grid search (unfinished/in progress), early stopping, etc. functionalities provided by Keras for stateless models. 

The key method of the StatefulLSTM class is the .stateful_fit() method, which implements a custom training loop where cell and hidden states of the model are reset at the ends of each training epochs. To simplify and streamline implementation of the training loop, the method replicates some of the useful Keras functionalities such as having the option to either include validation data in training or not and call training and validation losses from training history. A built-in early stopping functionality with restoration of best weights from training is also included. 

Once a model is compiled in Keras, the StatefulLSTM class can be initialized with the compiled model as the argument, yielding a StatefulLSTM class object that will have the aforementioned .stateful_fit() method. Besides, .save(), .set_weights(), .get_weights() and .predict() functionalities are also available from the class as wrappers of the Keras methods. 

The project also includes a simple_search function to conduct a grid search for optimal hyperparameters. The function was developed for personal use-case and is still thus largely underdeveloped and does not allow for tuning for learning rates, regularization, optimizers, etc. 

### For further details refer to:
- [Code](stateful_lstm.py)
- [Demo Notebook](stateful_lstm_demo.ipynb)
