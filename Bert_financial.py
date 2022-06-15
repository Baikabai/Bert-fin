#coding=utf-8
import pandas as pd
import numpy as np
import torch
import transformers
from numpy import dot
from numpy.linalg import norm
from transformers import BertJapaneseTokenizer


class BertSequenceVectorizer:
    def __init__(self):
        """
        The function takes in a sentence, tokenizes it, and then returns the embedding of the sentence
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'izumi-lab/bert-base-japanese-fin-additional'
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128
            

    def vectorize(self, sentence):
        """
        It takes a sentence, tokenizes it, pads it, and then passes it through the BERT model. 
        
        The output is a tensor of shape (1, 768) which is the output of the last layer of the BERT model. 
        
        The output of the last layer is a vector of 768 numbers. 
        
        This vector is a representation of the sentence. 
        
        This is the vector that we will use to train our classifier.
        
        :param sentence: the sentence you want to vectorize
        """
        input_data = self.tokenizer.encode(sentence)
        len_input_data = len(input_data)

        if len_input_data >= self.max_len:
            input_datauts = input_data[:self.max_len]
            masks = [1] * self.max_len
        else:
            input_datauts = input_data + [0] * (self.max_len - len_input_data)
            masks = [1] * len_input_data + [0] * (self.max_len - len_input_data)

        input_datauts_tensor = torch.tensor([input_datauts], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)
        
        seq_out = self.bert_model(input_datauts_tensor, masks_tensor)[0]
        pooled_out = self.bert_model(input_datauts_tensor, masks_tensor)[1]
        # 768 dementional vector
        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy() 
        else:
            return seq_out[0][0].detach().numpy()


def cosine_similarity(matrix):

# Calculating the cosine similarity between the first 13 rows and the last 4 rows.
    d = matrix @ matrix.T 

    norm = (matrix * matrix).sum(axis=1, keepdims=True) ** .5

    return d / norm / norm.T


if __name__ == '__main__':

# Creating a dataframe with the given sentences.
    df = pd.DataFrame(['当第３四半期連結累計期間において、新たな事業等のリスクの発生又は前事業年度の有価証券報告書に記載した事業等のリスクについての重要な変更はありません。','文中の将来に関する事項は、当四半期連結会計期間の末日現在において当社グループが判断したものであります。','当第３四半期連結累計期間における経営成績は、売上高は2,439,110百万円、営業利益は76,497百万円、経常利益は91,478百万円、親会社株主に帰属する四半期純利益は51,623百万円となりました。','セグメントの経営成績は、前年同四半期連結累計期間対比で次のとおりであります。','新型コロナウイルス感染症の影響により主として前連結会計年度の第１四半期連結会計期間に需要が大きく落ち込んだワイヤーハーネスや自動車電装部品、防振ゴム・ホースの需要が回復したことにより、売上高は1,274,106百万円と133,792百万円の増収となりました。','しかしながら、世界的な半導体供給不足の影響等による自動車生産の減産の動きが当連結会計年度の第２四半期連結会計期間以降に強まったほか、原材料の価格高騰、コンテナ不足や港湾混雑による物流コストの増加もあり、営業損失は2,972百万円と16,524百万円の悪化となりました。','データセンター用の光配線機器やアクセス系ネットワーク機器などで拡販を進め、売上高は175,562百万円と9,583百万円の増収となりました。','営業利益は、光・電子デバイスの品種構成の変化に伴う収益性の悪化により、17,313百万円と2,159百万円の減益となりました。','電子ワイヤー製品などで需要の捕捉を進めたことに加え、㈱テクノアソシエにおける自動車関連製品の需要増加などもあり、売上高は217,644百万円と34,592百万円の増収となりました。','営業利益は、売上増加に加え、FPC（フレキシブルプリント回路）の新製品拡販や生産性改善による収益力向上もあり、15,593百万円と9,151百万円の増益となりました。','電力ケーブルや巻線などの拡販を進めたほか、銅価格上昇の影響もあり、売上高は588,493百万円と162,273百万円の増収となり、営業利益は30,149百万円と19,508百万円の増益となりました。','超硬工具やダイヤ・CBN工具、ばね用鋼線、スチールコードなどの需要が増加し、売上高は241,538百万円と25,692百万円の増収となりました。営業利益は、売上増加に加え、工場の稼働率上昇に伴う収益性の改善もあり、16,635百万円と15,707百万円の増益となりました。','なお、各セグメントの営業利益又は営業損失は、四半期連結損益計算書の営業利益又は営業損失に対応しております。','電気自動車','医療機器','eスポーツ','自動車'], columns=['text'])


    Bert_model = BertSequenceVectorizer()
    df['vector'] = df['text'].progress_apply(lambda x: Bert_model.vectorize(x))
# Calculating the cosine similarity between the first 13 rows and the last 4 rows.
    a = np.sum(df[0:13]['vector'])
    print(cosine_similarity(np.stack(df.vector)))
    cos_sim1 = dot(a, df['vector'][13])/(norm(a)*norm(df['vector'][13]))
    print(cos_sim1)
    cos_sim2 = dot(a, df['vector'][14])/(norm(a)*norm(df['vector'][14]))
    print(cos_sim2)
    cos_sim3 = dot(a, df['vector'][15])/(norm(a)*norm(df['vector'][15]))
    print(cos_sim3)
    cos_sim4 = dot(a, df['vector'][16])/(norm(a)*norm(df['vector'][16]))
    print(cos_sim4)
    
