#!/usr/bin/env python
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

class MLP(object):

    def __init__(self, input_dim =(28, 28), output_size=10, batch_size=None, hidden_layers=[800, 800], drop_input_p=.2, drop_hidden_p=.5 ):
        '''
        input_dim =         N X M size of input matrix
        output_size =       Number of classes to classify
        batch_size =        SGD batch size
        theano_sym_var =    Symbolic var name for theano ex: 'x'
        hidden_layers =     Array of hidden layer sizes [100, 200, 300]  (3 layers, 1st = 100 neurons, etc...)
        drop_input_p =      Overfitting dropout probability for input layer
        drop_hidden_p =     Overfitting dropout probability for hidden layer
        '''
        # --------------------------------------------------------------------------------------------
        # ABOUT LASAGNE:
        # lasagne works by stacking layers on previous layers and specifying attributes of each layer.
        # in this network, we start with an input layer, then add dropout layer, then same for hidden layers and output layer
        # shape of 2 hidden layer network is:   INPUT -> DROPOUT -> H1 -> H1_DROPOUT -> H2 -> H2_DROPOUT -> OUTPUT
        # --------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------
        # BUILD NETWORK
        # --------------------------------------------------------------------------------------------
        print('building network...')
        # nn constants
        self.epochs = 0
        self.batchsize = 0

        # theano inputs var
        self.input_var = T.tensor4('inputs')

        # 1. Input layer
        network = lasagne.layers.InputLayer(shape=(batch_size, 1, input_dim[0], input_dim[1]), input_var=self.input_var)

        # 2. Input dropout layer 
        #    (prevent overfitting by adding dropout layer with P(dropout) = p)
        if drop_input_p:
            network = lasagne.layers.dropout(network, p=drop_input_p)

        # Hidden layers and dropout
        nonlin = lasagne.nonlinearities.rectify
        for layer_width in hidden_layers:

            # 3. Hidden layer N
            network = lasagne.layers.DenseLayer(network, layer_width, nonlinearity=nonlin)

            # 4. Hidden layer N dropout
            #    (prevent overfitting in hidden layer with prob = p by adding dropout layer)
            if drop_hidden_p:
                network = lasagne.layers.dropout(network, p=drop_hidden_p)
        
        # 5. Output layer:
        softmax = lasagne.nonlinearities.softmax
        network = lasagne.layers.DenseLayer(network, output_size, nonlinearity=softmax)
        
        # our fully working network
        self.network = network

        print('network built!')


    def predict(self, X):
        # define prediction function
        prediction = lasagne.layers.get_output(self.network, deterministic=True)
        predict_fn = theano.function([self.input_var], T.argmax(prediction, axis=1))

        # hold prediction results
        predictions = []
        for x in X:

            # make prediction and track
            y_hat = predict_fn(x)
            predictions.append(y_hat)
        
        return predictions


    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs=15, batchsize=500, learning_rate=0.01, momentum=0.9):
        self.batchsize = batchsize
        self.epochs = epochs
        
        # Prepare Theano variables for inputs and targets
        target_var = T.ivector('targets')

        # define loss function
        loss = self.__create_loss_fx(target_var)

        # define update rules
        updates = self.__create_update_rules(learning_rate, momentum, loss)

        # define test loss fx
        test_prediction, test_loss = self.__create_test_loss_fx(target_var)

        # Expression for the classification accuracy
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss
        train_fn = theano.function([self.input_var, target_var], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy
        val_fn = theano.function([self.input_var, target_var], [test_loss, test_acc])

        # train on the data
        self.__train(X_train, y_train, X_val, y_val, train_fn, val_fn)

        # see how accurate it is
        self.__compute_test_error(X_test, y_test, val_fn)


    def __train(self,  X_train, y_train, X_val, y_val, train_fn, val_fn):
        print("Starting training...")

        # iterate each epoch
        for epoch in range(self.epochs):

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.__iterate_minibatches(X_train, y_train, self.batchsize, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.__iterate_minibatches(X_val, y_val, self.batchsize, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))


    def __compute_test_error(self, X_test, y_test, val_fn):
        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0

        for batch in self.__iterate_minibatches(X_test, y_test, self.batchsize, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1

        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))


    def __create_loss_fx(self, target_var):
        '''
        Define loss function and use cross entropy loss
        '''
        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        return loss


    def __create_test_loss_fx(self, target_var):
        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        return test_prediction, test_loss


    def __create_update_rules(self, learning_rate, momentum, loss):
        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate, momentum)

        return updates


    def __save_network(self, model_name='model.npz'):
        np.savez(model_name, *lasagne.layers.get_all_param_values(self.network))


    def __load_network(self, model_name='model.npz'):
        with np.load(model_name) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.network, param_values)


    # ############################# Batch iterator ###############################
    # This is just a simple helper function iterating over training data in
    # mini-batches of a particular size, optionally in random order. It assumes
    # data is available as numpy arrays. For big datasets, you could load numpy
    # arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
    # own custom data iteration function. For small datasets, you can also copy
    # them to GPU at once for slightly improved performance. This would involve
    # several changes in the main program, though, and is not demonstrated here.
    # Notice that this function returns only mini-batches of size `batchsize`.
    # If the size of the data is not a multiple of `batchsize`, it will not
    # return the last (remaining) mini-batch.

    def __iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)

        # shuffle if requested
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)


        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)

            # generator function for next batch
            yield inputs[excerpt], targets[excerpt]