import csv

import numpy as np
from model import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from utils import get_data


def build_model():
    inp = Input(shape=(SEQ_LEN, 9))

    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # for i in range(2):
    x, self_attn = EncoderLayer(
        d_model=D_MODEL,
        d_inner_hid=512,
        n_head=4,
        d_k=64,
        d_v=64,
        dropout=0.2)(x)

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
    multi_head = build_model()
    # multi_head.summary()
    X_train, y_train = get_data()
    callback = EarlyStopping(monitor='val_loss',
                             patience=3,
                             restore_best_weights=True)
    history = multi_head.fit(x=X_train,
                             y=y_train,
                             batch_size=BATCH_SIZE,
                             epochs=EPOCHS,
                             validation_data=(X_valid, y_valid),
                             callbacks=[callback])
    predicted_stock_price = np.vstack((np.full((60, 1), np.nan), predicted_stock_price_multi_head))
    with open('../../data/final_res.csv', 'w')as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(predicted_stock_price)
