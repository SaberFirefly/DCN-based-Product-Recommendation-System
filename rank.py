import joblib
import pandas as pd


def most_popular_item():
    data = pd.read_csv("./data/bufen.csv")

    pupular_item = data.groupby("商品ID").count()

    return pupular_item['用户ID'].sort_values(ascending=False)[:1000].index


recommend_user = 884509  # 输入需要推荐的用户列表，用户可以在 ./data/bufen.csv中取看

if __name__ == '__main__':
    pupular_item_index = most_popular_item()

    user_encoder = joblib.load("./save_model/user_encoder")
    item_encoder = joblib.load("./save_model/item_encoder")
    user_portrait = joblib.load("./data/user_portrait")

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()

    pupular_item_index = item_encoder.transform(pupular_item_index) + 1

    user_index = user_encoder.transform([recommend_user])[0]
    '''
     '用户历史点击列表': click_list,
                '用户历史购买列表': buy_list,
                '用户历史喜欢列表': fav_list,
    '''

    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load("./save_model/model").to(device).eval()
    with torch.no_grad():
        user = torch.tensor([user_index] * len(pupular_item_index)).to(device)
        item = torch.tensor(pupular_item_index).to(device)
        click = torch.tensor(user_portrait[user_index]['用户历史购买列表'] * len(pupular_item_index)).view(
            (len(pupular_item_index), -1)).to(device)
        buy_list = torch.tensor(user_portrait[user_index]['用户历史点击列表'] * len(pupular_item_index)).view(
            (len(pupular_item_index), -1)).to(device)
        fav_list = torch.tensor(user_portrait[user_index]['用户历史喜欢列表'] * len(pupular_item_index)).view(
            (len(pupular_item_index), -1)).to(device)

        score = model(user, item, click, buy_list, fav_list)

        df = pd.DataFrame()
        df['item_id'] = item.cpu().numpy()
        df['score'] = score.cpu().numpy()
        df.sort_values(by=['score'], ascending=False, inplace=True)
        df['item_id'] = item_encoder.inverse_transform(df['item_id'] - 1)
        print(f"用户{recommend_user}的推荐列表为：")
        print(df['item_id'].values)
