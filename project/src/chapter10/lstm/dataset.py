import torch


class IMDBDataset:
    def __init__(self, reviews, targets) -> None:
        """
        :param reviews: numpy 配列
        :param targets: numpy 配列
        """
        self.reviews = reviews
        self.target = targets

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item: int):
        # インデックスに対して、該当する reviews と target テンソルを返す
        review = self.reviews[item, :]
        target = self.target[item]
        return {
            "review": torch.tensor(review, dtype=torch.long),
            "target": torch.tensor(review, dtype=torch.float),
        }
