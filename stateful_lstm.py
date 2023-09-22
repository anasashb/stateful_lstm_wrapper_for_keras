import time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import regularizers


class StatefulLSTM:
    '''
    Wrapper class for stateful LSTM models from Keras. Intended for use with time series data.
    Custom training function ensures hidden states are reset after an epoch of training. 
    Earlystopping functionality is included.

    # Arguments:
        model: An already compiled Keras LSTM model. LSTM layers should be stateful
    '''
    
    def __init__(self, model):
        
        # Some error handling
        if not isinstance(model, Model):
            raise ValueError('Given model is not supported. Please use a Keras model.')   

        stateful_check = all(layer.stateful for layer in model.layers if isinstance(layer, LSTM))  
        if not stateful_check:
            raise ValueError('Provided model does not contain stateful LSTM layer(s).\nThere is no point in using this class if you don\'t use stateful LSTMs.')

        self.model = model
        self.train_losses = []
        self.val_losses = []
        self.best_weights = None
        self.wait = 0
        self.best_loss = np.inf

    
    def stateful_fit(self, 
            x=None,
            y = None, 
            batch_size=None,
            epochs=1, 
            verbose=False,
            validation_data=None,
            early_stopping=False,
            min_delta=1e-3,
            patience=10):
        '''
        Custom fit function intended for stateful LSTM models.

        # Main Arguments
            x: Inputs
            y: Targets
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs for training
        # Optional Arguments
            verbose: Boolean argument on whether to display losses at epoch ends.
            validation_data: Tuple containing inputs and targets to evaluate model at epoch ends.
            early_stopping: Boolean argument to switch on early stopping.
            min_delta: Float determining minimum change in loss to qualifiy as sufficient change.
            patience: Number of epochs without improvement before early stopping is triggered.  
        '''

        # Error Handling to Ensure Input Dimensions and Batch Size
        # Are Compatible with Stateful LSTM Training
        if x.shape[0] % batch_size != 0 or y.shape[0] % batch_size != 0:
            raise ValueError('Training data dimensions and given batch size are not compatible with stateful LSTM training. '
                             'Make sure first dimensions of inputs and targets are divisible by batch size without remainder.')
        if validation_data:
            if validation_data[0].shape[0] % batch_size != 0 or validation_data[1].shape[0] % batch_size != 0:
                raise ValueError('Validation data dimensions and given batch size are not compatible with stateful LSTM training. '
                             'Make sure first dimensions of inputs and targets are divisible by batch size without remainder.')   
        
        for i in range(epochs):
            # Begin Tracking Time
            start_time = time.time()
            # Fits According to Arguments Given
            # Verbose = 0 Because We Have Own Print Statements 
            history = self.model.fit(x,
                                     y,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     verbose = 0,
                                     shuffle=False)
            # Model States Reset After Every Epoch
            self.model.reset_states()
            # Get Loss After Epoch
            train_loss = history.history['loss'][0]
            # Append Loss to Losses
            self.train_losses.append(train_loss)
            # Get Model Weights
            model_weights = self.model.get_weights()
            # Calculate End Time
            end_time = time.time()
            # Calculate Elapsed Time
            elapsed_time = end_time - start_time

            # If Statement to Control for Validation Set
            # If Validation Set Given, Current Loss to Track for Early Stopping Will Be Val Loss
            # Else It Will Be Train Loss
            if validation_data:
                val_metrics = self.model.evaluate(validation_data[0],
                                               validation_data[1],
                                               batch_size=batch_size,
                                               verbose=0)
                # To Handle Multiple Val Losses in Case Model is Compiled with Additional Metrics
                if isinstance(val_metrics, list):
                    val_loss = val_metrics[0]
                else:
                    val_loss = val_metrics
                self.val_losses.append(val_loss)
                loss_on_epoch_end = val_loss
                if verbose:
                    print(f"Epoch {i+1}/{epochs} - {elapsed_time:.2f}s - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
                    print('-' * 100)
            else:
                loss_on_epoch_end = train_loss
                if verbose:
                    print(f"Epoch {i+1}/{epochs} - {elapsed_time:.2f}s - loss: {train_loss:.4f}")
                    print('-' * 100)


            # Early Stopping
            if early_stopping:
                if np.abs(loss_on_epoch_end - self.best_loss) > min_delta:
                    self.best_loss = loss_on_epoch_end
                    self.best_weights = model_weights
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= patience:
                        self.model.set_weights(self.best_weights)
                        print(f'Early Stopping Triggered on Epoch {i+1}.')
                        break

    # Wrappers for Saving Model and Weights, Making Predictions
    def save(self, filepath):
        '''
        Basic wrapper to save Keras model. Refer to Keras documentation for more.
        '''
        self.model.save(filepath)

    def get_weights(self):
        '''
        Basic wrapper to get Keras model's weights. Refer to Keras documentation for more.
        
        # Returns
            A list of arrays with model weights.
        '''
        return self.model.get_weights()
    
    
    def set_weights(self, weights):
        '''
        Basic wrapper to set Keras model's weights. Refer to Keras documentation for more.
        
        # Arguments
            weights: A list of arrays with model weights.
        '''
        self.model.set_weights(weights)


    def save_weights(self, filepath):
        '''
        Basic wrapper to save Keras models weights. Refer to Keras documentation for more
        '''
        self.model.save_weights(filepath) 

    def predict(self, x, batch_size=None):
        '''
        Makes predictions based on input.
        
        # Arguments
            x: Inputs
            batch_size: Number of samples per batch.
        
        # Returns
            Predictions
        '''

        if x.shape[0] % batch_size != 0:
            raise ValueError('Test data dimensions and given batch size are not compatible with stateful LSTM training. '
                             'Make sure the first dimension of inputs is divisible by batch size without remainder.')   
        return self.model.predict(x, batch_size=batch_size, verbose = 1)
    


# Separate Function for Grid Search that uses the StatefulLSTM class
def simple_search(x=None,
                  y=None,
                  batch_size=None,
                  timesteps = None,
                  train_epochs=1,
                  validation_data=None,
                  num_layers=None,
                  num_units=None,
                  regularize=False,
                  verbose=False):
    
    '''
    A simple function for gridsearch to tune for number of layers and number of units using the stateful_fit method of the StatefulLSTM wrapper class.
    Early stopping and verbose arguments of the stateful_fit method are turned off. Keras models are built inside the function, but user needs to build the obtained best model outside the function and initialize the StatefulLSTM class.
    Current form does not allow for searching different amounts of units per different amounts of layers.

    # Arguments
        x: Inputs
        y: Targets
        batch_size: Number of samples per gradient update
        timesteps: Number of lags to predict step ahead.
        train_epochs: Number of epochs for training per trial
        validation_data: Tuple containing inputs and targets to evaluate model at epoch ends.
        num_layers: Tuple containing integers as (start, stop, step size) for number of stateful LSTM layers to search. 
        num_units: Tuple containing integers as (start, stop, step size) for number of units to search per layer. 
        regularize: Boolean argument to include pre-defined L2 penalties and a dropout layer stacked on top of LSTM layer(s).
        verbose: Boolean argument on whether to display trial information.
    '''    

    mandatory_arguments = {
        "Input 'x'": x,
        "Target 'y'": y,
        "Batch size": batch_size,
        "Timesteps": timesteps, 
        'Validation data': validation_data,
        'Number of layers': num_layers,
        'Number of units': num_units,
        }
    
    # Making Sure All Arguments Provided
    for argument_name, argument_value in mandatory_arguments.items():
        if argument_value is None:
            raise ValueError(f"{argument_name} cannot be None.")
    # Error Handling for Validation Data
    if not isinstance(validation_data, tuple) or len(validation_data) != 2: 
        raise TypeError("Validation data must be provided as a tuple containing hold out inputs and targets.")
    # Error Handling for Number of Layers
    if isinstance(num_layers, tuple) and len(num_layers) == 3 and all(isinstance(number, int) for number in num_layers):
        num_layers = range(*num_layers)
    else:
        raise ValueError("num_layers must be a tuple of integers in the form (start, stop, step size).")
    # Error Handling for Number of Units
    if isinstance(num_units, tuple) and len(num_units) == 3 and all(isinstance(number, int) for number in num_units):
        num_units = range(*num_units)
    else:
        raise ValueError("num_units must be a tuple of integers in the form (start, stop, step size).") 

    # If Successful, num_layers and num_units Are Turned into Range Iterators

    # Set Up Best Validation Loss, Best Number of Layers and Best Number of Units
    # Search is Carried Out Looking for Minimum Loss Obtained Within Any Evaluation, Similarly to Keras-Tuner
    best_val_loss = np.inf
    train_loss_for_best_val = np.inf
    best_num_units = None
    best_num_layers = None


    for layer in num_layers:
        for unit in num_units:
            if verbose:
                print(f"Trial with {layer} Layer(s), {unit} Unit(s). Model is Being Fitted and Evaluated...")
                print('-'*120)
            # Construct Model    
            model = Sequential()
            for i in range(layer):
                if regularize:
                    model.add(LSTM(unit,
                                activation='tanh',
                                recurrent_activation='sigmoid',
                                kernel_regularizer=regularizers.L2(1e-3),
                                recurrent_regularizer=regularizers.L2(1e-3),
                                bias_regularizer=regularizers.L2(1e-3),
                                recurrent_dropout=0,
                                unroll=False,
                                use_bias=True,
                                stateful=True,
                                return_sequences=True if i < layer - 1 else False,
                                batch_input_shape=(batch_size, timesteps, 1)))
                    model.add(Dropout(0.1))
                else:
                    model.add(LSTM(unit,
                                activation='tanh',
                                recurrent_activation='sigmoid',
                                recurrent_dropout=0,
                                unroll=False,
                                use_bias=True,
                                stateful=True,
                                return_sequences=True if i < layer - 1 else False,
                                batch_input_shape=(batch_size, timesteps, 1)))
            model.add(Dense(1))
            model.compile(loss = 'mean_squared_error', optimizer = 'adam')
            # Initialize Stateful LSTM Class    
            stateful_model = StatefulLSTM(model)
            # Actual Fitting and Validation
            stateful_model.stateful_fit(
                x=x,
                y=y,
                batch_size=batch_size,
                epochs=train_epochs,
                verbose=False,
                validation_data=validation_data,
                early_stopping=False
            )            
            # Grab Losses
            train_losses = stateful_model.train_losses
            val_losses = stateful_model.val_losses
            # Get the Minimum Validation Loss and Corresponding Train Loss
            min_val_loss_idx = val_losses.index(min(val_losses))
            min_val_loss = min(val_losses)
            relevant_train_loss = train_losses[min_val_loss_idx]
            # Check if Val Loss Beats Best Val Loss So Far
            if min_val_loss < best_val_loss:
                best_val_loss = min_val_loss
                best_num_layers = layer
                best_num_units = unit
                train_loss_for_best_val = relevant_train_loss
                if verbose:
                    print(f"Trial Finished. New Best Parameters Found: Layers: {best_num_layers}, Units: {best_num_units}, Val Loss: {best_val_loss:.4f}, Corresponding Train Loss: {train_loss_for_best_val:.4f}.")
            else:
                if verbose:
                    print(f"Trial Finished. No Best Parameters Found at This Stage. Moving On...")
            if verbose:
                print('=' * 120)

    print(f"Search Finished.")
    print(f"Best Number of Layers: {best_num_layers}\nBest Number of Units: {best_num_units}\nBest Validation Loss Achieved: {best_val_loss:.4f}\nCorresponding Train Loss: {train_loss_for_best_val:.4f}.")


    