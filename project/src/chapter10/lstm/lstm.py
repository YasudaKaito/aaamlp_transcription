import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedding_matrix) -> None:
        """
        :param embedding_matrix: 全ての語彙に対する埋め込み表現
        """
        super().__init__()
        # 単語数
        num_words = embedding_matrix.shape[0]
        # 埋め込み次元数
        embed_dim = embedding_matrix.shape[1]
        # 埋め込み層
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=embed_dim)
        # 埋め込み表現を、埋め込み層の重みとして利用
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        # 事前学習済の埋め込み表現は学習しない
        self.embedding.weight.requires_grad = False

        # 簡単な双方向 LSTM 層
        # 隠れ層のサイズは 128
        self.lstm = nn.LSTM(embed_dim, 128, bidirectional=True, batch_first=True)
