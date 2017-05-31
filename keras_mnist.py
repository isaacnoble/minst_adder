from __future__ import print_function

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Embedding, Conv2D, MaxPooling2D, concatenate, Flatten, Reshape, GRU, Dropout
from keras import backend as K

# set learning phase to 0
K.set_learning_phase(0)

length = 15
num_samples = 200000
epochs = 10
test_cases = 15

image_cols = 28
image_rows = 28

class DataBuilder:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def Sample(self, length):
        images = np.zeros([length, 1, image_rows, image_cols])
        symbols = np.zeros([length, 1])
        total = 0
        label = np.zeros([10*length])
        for i in range(0,length):
            index = np.random.randint(0,self.labels.size)
            symbols[i][0] = self.labels[index]
            #images[i][0] = self.images[index]
            total += self.labels[index]
        label[total] = 1
        return images, symbols, label

    def DataSet(self, num_samples, length):
        image_samples = np.zeros([num_samples, length, 1, image_rows, image_cols])
        symbol_samples = np.zeros([num_samples, length, 1])
        labels = np.zeros([num_samples, 10*length])
        for i in range(0, num_samples):
            image_samples[i], symbol_samples[i], labels[i] = self.Sample(length)
        return image_samples, symbol_samples, labels
            

image_input = Input(shape=(1, image_rows, image_cols), name='image_input')
conved = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_rows, image_cols), data_format='channels_first')(image_input)
conved = Conv2D(64, (3, 3), activation='relu', data_format='channels_first')(conved)
conved = MaxPooling2D(pool_size=(2, 2))(conved)
conved = Dropout(0.25)(conved)
conved = Flatten()(conved)
conved = Dense(128, activation='relu')(conved)
conved = Dropout(0.5)(conved)
conved_out = Dense(10, activation='relu')(conved)

conv_model = Model(inputs=image_input, outputs=conved_out)

image_seq_input = Input(shape=(length, 1, image_rows, image_cols), name='image_seq_input')
symbol_seq_input = Input(shape=(length, 1), name='symbol_seq_input')

conved_images = TimeDistributed(conv_model)(image_seq_input)
embedded_symbols = Embedding(10, 4, input_length=length)(symbol_seq_input)
embedded_symbols = Reshape([length, 4])(embedded_symbols)

#sequence_data = concatenate([conved_images, embedded_symbols], axis=2)

#x = GRU(16, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=True, input_shape=(length, 14))(squence_data)
#x = GRU(16, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=True, input_shape=(length, 4))(embedded_symbols)
x = GRU(16, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=True, input_shape=(length, 10))(conved_images)
x = GRU(16, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False)(x)
x = Dense(10, activation='relu')(x)

prediction = Dense(10*length, activation='softmax', name='main_output')(x)

#model = Model(inputs=[image_seq_input, symbol_seq_input], outputs=prediction)
#model = Model(inputs=symbol_seq_input, outputs=prediction)
model = Model(inputs=image_seq_input, outputs=prediction)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()    

# The data, shuffled and split between train and test sets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

training_builder = DataBuilder(x_train, y_train)
test_builder = DataBuilder(x_test, y_test)

def test_model(model, cases):
    known_images, known_symbols, known_labels = test_builder.DataSet(cases, length)
    #guesses = model.predict([known_images, known_symbols])
    #guesses = model.predict(known_symbols)
    guesses = model.predict(known_images)
    for x in range(0, cases):
        guess = np.argmax(guesses[x])
        label = np.argmax(known_labels[x])
        print("guess = " + repr(guess) + " label = " + repr(label))

test_model(model, test_cases)

for e in range(0, epochs):
    images, symbols, labels = training_builder.DataSet(num_samples, length)
    #model.fit([images, symbols], labels, epochs=2, batch_size=100)
    #model.fit(symbols, labels, epochs=2, batch_size=100)
    model.fit(images, labels, epochs=2, batch_size=100)

    test_model(model, test_cases)
    test_images, test_symbols, test_labels = test_builder.DataSet(num_samples/10, length)
    #score = model.evaluate([test_images, test_symbols], test_labels, batch_size=100)
    #score = model.evaluate(test_symbols, test_labels, batch_size=100)
    score = model.evaluate(test_images, test_labels, batch_size=100)
    print (score)
