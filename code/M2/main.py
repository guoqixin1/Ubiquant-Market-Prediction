import csv
import tensorflow as tf
import numpy as np
from model import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from utils import get_data, My_input, Time_embedding, Input_embedding, evalulate


def build_model():
    D_MODEL = 300
    SEQ_LEN = 3773
    lr = 1e-5
    inp = My_input(shape=(SEQ_LEN, 3773))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    x = Input_embedding(inp[1:])
    x += Time_embedding(inp[0])

    x, self_attn = EncoderLayer(d_model=D_MODEL, d_inner_hid=512, n_head=4, d_k=64, d_v=64, dropout=0.2)(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(128, activation="relu")(conc)
    x = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=x)
    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer)

    return model


if __name__ == '__main__':
    EPOCHS = 10
    BATCH_SIZE = 128

    multi_head = build_model()
    # multi_head.summary()
    trains, valids, tests = get_data()
    X_train, Y_train = trains
    X_valid, Y_valid = valids
    X_test, Y_test = tests
    callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = multi_head.fit(x=X_train, y=Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                             validation_data=(X_valid, Y_valid), callbacks=[callback])
    predicted_stock_price_multi_head = multi_head.predict(X_test)
    predicted_stock_price = np.vstack((np.full((60, 1), np.nan), predicted_stock_price_multi_head))
    evalulate(predicted_stock_price, Y_test)
    with open('../../data/final_res.csv', 'w')as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(predicted_stock_price)
