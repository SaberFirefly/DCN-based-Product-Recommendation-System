from copy import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import defaultdict

'''
然后，我们定义一个数据处理函数，该函数将原始数据转换为模型可以接受的格式：
'''

click_count = 10
buy_count = 3
fav_count = 5

import random

random.seed(10)
random.randrange(10)


def process_data(data, max_len=10):
    # 将数据按照时间戳排序
    data = data.sort_values(by='时间戳')

    # 创建字典来存储用户的历史行为
    user_history = defaultdict(lambda: {'pv': [], 'buy': [], 'fav': []})

    # 遍历数据，填充用户历史行为字典
    for _, row in data.iterrows():
        user_id = row['用户ID']
        item_id = row['商品ID']
        action = row['行为类型']
        if action in user_history[user_id]:
            user_history[user_id][action].append(item_id)

    # 处理数据，使得每个用户的历史行为列表长度相同
    # for user_id in user_history:
    #     for action in user_history[user_id]:
    #         history = user_history[user_id][action]
    #         if len(history) < max_len:
    #             history.extend(['0'] * (max_len - len(history)))
    #         else:
    #             history = history[:max_len]
    #         user_history[user_id][action] = history

    # 创建最终的数据集
    def tran_and_fill(l, count):
        if len(l) < count:
            l.extend([0] * (count - len(l)))
        else:
            l = l[:count]
        return l

    processed_data = []
    user_history_index = defaultdict(lambda: {'pv': 0, 'buy': 0, 'fav': 0})
    for _, row in data.iterrows():
        user_id = row['用户ID']
        item_id = row['商品ID']
        label = 1 if row['行为类型'] == 'buy' else 0
        if row['行为类型'] in user_history[user_id]:
            user_history_index[user_id][row['行为类型']] += 1
            click_list = copy(user_history[user_id]['pv'][:user_history_index[user_id]['pv']])
            buy_list = copy(user_history[user_id]['buy'][:user_history_index[user_id]['buy']])
            fav_list = copy(user_history[user_id]['fav'][:user_history_index[user_id]['fav']])

            click_list = tran_and_fill(click_list, click_count)
            buy_list = tran_and_fill(buy_list, buy_count)
            fav_list = tran_and_fill(fav_list, fav_count)

            item = {
                '用户ID': user_id,
                '商品ID': item_id,
                '用户历史点击列表': click_list,
                '用户历史购买列表': buy_list,
                '用户历史喜欢列表': fav_list,
                '用户是否购买商品': label
            }
            if label == 1:
                processed_data.append(item)
            if label == 0 and (random.randrange(10) == 0):  # 抽1/10出来作为负样本
                processed_data.append(item)

    return pd.DataFrame(processed_data)


'''
接下来，我们定义一个PyTorch的Dataset类来加载处理后的数据：
'''


class ECommerceDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        user_id = torch.tensor(row['用户ID'], dtype=torch.long)
        item_id = torch.tensor(row['商品ID'], dtype=torch.long)
        pv_history = torch.tensor([int(x) for x in row['用户历史点击列表']], dtype=torch.long)
        buy_history = torch.tensor([int(x) for x in row['用户历史购买列表']], dtype=torch.long)
        fav_history = torch.tensor([int(x) for x in row['用户历史喜欢列表']], dtype=torch.long)
        label = torch.tensor(row['用户是否购买商品'], dtype=torch.float)
        return user_id, item_id, pv_history, buy_history, fav_history, label


'''
现在我们定义一个简单的神经网络模型来处理这些数据
'''


class ECommerceModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(ECommerceModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, item_id, pv_history, buy_history, fav_history):
        user_embedded = self.user_embedding(user_id)
        item_embedded = self.item_embedding(item_id)
        pv_embedded = self.item_embedding(pv_history).mean(1)
        buy_embedded = self.item_embedding(buy_history).mean(1)
        fav_embedded = self.item_embedding(fav_history).mean(1)
        x = torch.cat((user_embedded, item_embedded, pv_embedded, buy_embedded, fav_embedded), 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze()


'''
最后，我们可以使用以下代码来训练模型：
'''
# 假设我们已经有了一个DataFrame 'df' 包含了原始数据
# 你需要替换这里的代码来加载你的数据
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('data/bufen.csv')[:1000]
df['用户ID'] = LabelEncoder().fit_transform(df['用户ID'])
df['商品ID'] = LabelEncoder().fit_transform(df['商品ID']) + 1
# 数据处理
processed_df = process_data(df)

# 创建数据集

dataset = ECommerceDataset(processed_df)

# 创建数据加载器
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型参数
num_users = max(df['用户ID']) + 1
num_items = max(df['商品ID']) + 2
embedding_dim = 128

# 初始化模型
model = ECommerceModel(num_users, num_items, embedding_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for user_id, item_id, pv_history, buy_history, fav_history, label in data_loader:
        # 清除梯度
        optimizer.zero_grad()

        # 前向传播
        predictions = model(user_id, item_id, pv_history, buy_history, fav_history)

        # 计算损失
        loss = criterion(predictions, label)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
