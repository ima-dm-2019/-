import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def data_cut(content):
    """利用Jieba进行分词"""
    content_cut = jieba.cut(content)
    jieba.suggest_freq('拉克兰', True)
    jieba.suggest_freq('副主席', True)
    return content_cut


def stp_wrd(words_data):
    """停用词处理"""
    with open('stop_words.txt', 'r', encoding='utf-8') as stp:
        stop_words = [line.strip() for line in stp.readlines()]
        stop_words.append(' ')
        words_data = [word for word in words_data if word not in stop_words]
        stp.close()
        return words_data


def feature_extraction(content):
    """特征工程"""

    vectorizer = CountVectorizer()
    # 输出：词频矩阵，(文本序号，词序号) 词频
    word_matrix = vectorizer.fit_transform(content)
    # print(matrix_word)

    # 词袋模型中词的名称
    word_name = vectorizer.get_feature_names()
    # print(word_name)

    transformer = TfidfTransformer()
    # 计算tf-idf
    tfidf = transformer.fit_transform(word_matrix)

    # tf-idf矩阵
    weight = tfidf.toarray()
    # print(vectorizer.fit_transform(data).toarray())

    # 每类文本的tf - idf词语权重
    for i in range(len(weight)):
        # print(list(zip(word_name, weight[i])))
        # print("-------这里输出第", i, u"类文本的词语tf-idf权重------")
        # print(list(zip(word, weight[i])))
        for j in range(len(word_name)):
            X = word_name[j], weight[i][j]
            # print(X)
    # 返回tf-idf矩阵
    return weight


def pre_main():
    # 去除<br>标签和\n
    # with open('data_ex - 副本.csv', encoding='utf-8') as fin:
    #     lines = fin.read().replace("<br>", "").replace("\n", "")
    # with open('data_ex - 副本.csv', "w", encoding='utf-8', newline='') as f1:
    #     f1.write(lines)
    #     f1.close()

    #
    # data = pd.read_csv("data_ex.csv", encoding='utf-8')
    # data = data.fillna('?')
    data = pd.read_excel("news_2020_02_19_kr.xlsx", usecols=[2], names=None)
    df_li = data.values.tolist()
    content_ = []
    # row, column = data.shape
    # for i in range(row):
    for s_li in df_li:
        # n_content = data.loc[i, :][0] + data.loc[i, :][1] + data.loc[i, :][2]
        n_content = s_li[0]
        n_words = data_cut(n_content)
        true_words = stp_wrd(n_words)
        content_.append(" ".join(true_words))
    tfidf_weight = feature_extraction(content_)
    return content_, tfidf_weight

