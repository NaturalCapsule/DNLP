# Importing libraries.
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd

# Loading the datasets.
train_df = pd.read_csv('training.csv')
test_df = pd.read_csv('test.csv')
val_df = pd.read_csv('validation.csv')

# Defining (x_train, y_train, x_test, y_test, x_val, y_val).
x_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

x_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

x_val = val_df.iloc[:, :-1].values
y_val = val_df.iloc[:, -1].values

# Creating TextVectorization layer.
max_tokens = 5000
max_len = 15
vec_layer = layers.experimental.preprocessing.TextVectorization(max_tokens = max_tokens,
                                                                output_sequence_length = max_len)
# Adapting the TextVectorization layer to the (x_train).
vec_layer.adapt(x_train)
vec_layer.adapt(x_test)
vec_layer.adapt(x_val)

# Creating Embedding layer.
embedding_layer = layers.Embedding(input_dim = max_tokens,
                     output_dim = 128,
                     input_length = max_len,
                     )
# Building the functional API.
def functional_model():
    inputs = layers.Input(shape = (1, ), dtype = tf.string)
    x = vec_layer(inputs)
    x = embedding_layer(x)

    x = layers.GRU(256, dropout=0.2, return_sequences = True)(x)
    x = layers.GRU(128, dropout=0.3, return_sequences = True)(x)
    x = layers.GRU(64, dropout=0.4, return_sequences = True)(x)
    x = layers.GRU(32, dropout=0.2, return_sequences = True)(x)

    x = layers.Dense(256, activation = 'relu')(x)
    x = layers.Dense(128, activation = 'relu')(x)
    x = layers.Dense(64, activation = 'relu')(x)
    x = layers.Dense(32, activation = 'relu')(x)

    x = tf.keras.layers.MaxPool1D()(x)
    x = tf.keras.layers.Flatten()(x)

    outputs = layers.Dense(units = 6, activation = 'softmax')(x)

    nlp_model = tf.keras.Model(inputs, outputs)

    return nlp_model

model = functional_model()

#Compiling the model.
model.compile(optimizer = tf.keras.optimizers.Adam(3e-4), loss = tf.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])

# Creating EarlyStopping callback to prevent overfitting (stops the training when it overfits).
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 5, verbose = 2)

# Training the model.
model.fit(x = x_train, y = y_train, validation_data = (x_val, y_val), epochs = 50, callbacks = [early_stopping])

#Evaluating the model on the test sets.
model.evaluate(x_test, y_test, batch_size = 32)
#### AS ALWAYS YOU CAN LOOK AT THE DOCUMENTATIONS IF YOU DON'T UNDERSTAND WHAT IS GOING ON.