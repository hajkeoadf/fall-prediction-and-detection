from tensorflow.keras import Sequential,layers

lstmmodel = Sequential([
    layers.LSTM(units=256,input_shape=(5,34),return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(units=256,return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(units=128,return_sequences=True),
    layers.LSTM(units=32),
    layers.Dense(1)
])
lstmmodel.load_weights('best_model_drop.weights.h5')