# !/usr/bin/env python
# -*-coding:utf-8 -*-
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from hotp.entity.dataloader import get_dataloader
from hotp.model.my_model import *
from hotp.utils import Log
from hotp.model.callbacks import EpochEndCheckpoint
import joblib
from hotp.utils.common import seed_everything

model_type = "cnn"  # cnn   cnnlstm

seed_everything(100)


def main():
    epochs = 20
    batch_size = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log = Log("log.log", "./log/")
    df = pd.read_csv('data/bufen.csv')
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df['用户ID'] = user_encoder.fit_transform(df['用户ID'])
    df['商品ID'] = item_encoder.fit_transform(df['商品ID']) + 1

    joblib.dump(user_encoder, "./save_model/user_encoder")
    joblib.dump(item_encoder, "./save_model/item_encoder")

    dtr, dte = get_dataloader(df, 0.8)
    num_users = max(df['用户ID']) + 1
    num_items = max(df['商品ID']) + 2
    embed_dim = 64
    model = ECommerceModel(num_users, num_items, embed_dim, batch_size=batch_size, device=device, log=log).to(device)

    model.compile("adam", "binary_crossentropy",
                  metrics=["auc", 'acc'],
                  lr=5e-4) # 0.0001
    callback = EpochEndCheckpoint(out=log, filepath="./save_model/model", mode='max', monitor='val_auc')

    log_list = model.fit(train=dtr, epochs=epochs, validation_data=dte, callback=callback)



if __name__ == '__main__':
    main()
