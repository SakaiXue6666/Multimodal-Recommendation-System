import pandas as pd
import torch
import torch.utils.data as data
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class EmbDataset(data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_parquet(data_path)

        # ====== 文本 / 图像 ======
        self.text_emb = np.stack(self.df["text_emb"].values).astype(np.float32)
        self.image_emb = np.stack(self.df["image_emb"].values).astype(np.float32)

        # ====== 数值模态 ======
        self.price = np.stack(self.df["price_norm"].values).astype(np.float32)

        # ====== 类别模态 ======
        self.brand = np.stack(self.df["brand_idx"].values).astype(np.int64)
        self.categories = self.df["categories_idx"].tolist()  # list[list[int]]

        # ====== 维度统计 ======
        self.text_dim = self.text_emb.shape[-1]
        self.image_dim = self.image_emb.shape[-1]
        self.dim = self.text_dim ###

        self.num_dim = {
            "price": self.price.shape[-1]
        }

        # 类别数量（最大索引 + 1）
        num_brands = int(np.max(self.brand)) + 1
        num_categories = int(max([max(c) if len(c) > 0 else 0 for c in self.categories])) + 1
        self.cls_dim = {
            "brand": num_brands,
            "categories": num_categories
        }

        print(f"✅ Loaded {len(self.df)} items from {data_path}")
        print(f" text_emb: {self.text_dim}, image_emb: {self.image_dim}, num: {self.num_dim}")
        print(f" brand classes: {num_brands}, category classes: {num_categories}")

    def __getitem__(self, index):
        # --- 文本 & 图像 ---
        text = torch.tensor(self.text_emb[index], dtype=torch.float32)
        image = torch.tensor(self.image_emb[index], dtype=torch.float32)

        # --- 数值模态 ---
        num = {
            "price": torch.tensor(self.price[index], dtype=torch.float32)
        }

        # --- 分类模态 ---
        brand = torch.tensor(self.brand[index], dtype=torch.long)
        categories = torch.tensor(self.categories[index], dtype=torch.long)
        cls = {
            "brand": brand,
            "categories": categories
        }

        return {
            "text_emb": text,
            "image_emb": image,
            "num": num,
            "cls": cls
        }

    def __len__(self):
        return len(self.df)


    def collate_fn(self, batch):
        text = torch.stack([b["text_emb"] for b in batch])
        image = torch.stack([b["image_emb"] for b in batch])
        price = torch.stack([b["num"]["price"] for b in batch])

        brand = torch.stack([b["cls"]["brand"] for b in batch])

        # padding categories
        categories = [b["cls"]["categories"] for b in batch]
        categories_padded = pad_sequence(categories, batch_first=True, padding_value=0)

        return {
            "text_emb": text,
            "image_emb": image,
            "num": {"price": price},
            "cls": {
                "brand": brand,
                "categories": categories_padded
            }
        }