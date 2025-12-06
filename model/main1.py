import torch
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
import os
import random
import pandas as pd
from tqdm import tqdm
import logging
from dataset import GenRecDataset
from dataloader import GenRecDataLoader
from datetime import datetime

# class TIGER(nn.Module):
#     def __init__(self, config: Dict[str, Any]):
#         super(TIGER, self).__init__()
#         t5config = T5Config(
#         num_layers=config['num_layers'],
#         num_decoder_layers=config['num_decoder_layers'],
#         d_model=config['d_model'],
#         d_ff=config['d_ff'],
#         num_heads=config['num_heads'],
#         d_kv=config['d_kv'],
#         dropout_rate=config['dropout_rate'],
#         vocab_size=config['vocab_size'],
#         pad_token_id=config['pad_token_id'],
#         eos_token_id=config['eos_token_id'],
#         decoder_start_token_id=config['pad_token_id'],
#         feed_forward_proj=config['feed_forward_proj'],
#     )
#         # Initialize T5 model with the specified configuration
#         self.model = T5ForConditionalGeneration(t5config)

class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
        num_layers=config['num_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        d_kv=config['d_kv'],
        dropout_rate=config['dropout_rate'],
        vocab_size=config['vocab_size'],
        # pad_token_id=config['pad_token_id'],
        # eos_token_id=config['eos_token_id'],
        # decoder_start_token_id=config['pad_token_id'],
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=1,  # 👈 用 <eos> 或 <bos> ID
        feed_forward_proj=config['feed_forward_proj'],
    )
        # Initialize T5 model with the specified configuration
        self.model = T5ForConditionalGeneration(t5config)
    
    @property
    def n_parameters(self):
      """Calculates the number of trainable parameters in the model.

      Returns:
          str: A string containing the number of embedding parameters,
          non-embedding parameters, and total trainable parameters.
      """
      num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
      total_params = num_params(self.parameters())
      emb_params = num_params(self.model.get_input_embeddings().parameters())
      return (
          f'#Embedding parameters: {emb_params}\n'
          f'#Non-embedding parameters: {total_params - emb_params}\n'
          f'#Total trainable parameters: {total_params}\n'
      )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
      """Forward pass of the model. Returns the output logits and the loss value.

      Args:
          batch (dict): A dictionary containing the input data for the model.

      Returns:
          outputs (ModelOutput):
              The output of the model, which includes:
              - loss (torch.Tensor)
              - logits (torch.Tensor)
      """
      outputs = self.model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          labels=labels
      )
      return outputs.loss, outputs.logits
    
    # def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,  num_beams: int = 20, **kwargs):
    #     """Generate recommendations using the model.
    #
    #     Args:
    #         input_ids (torch.Tensor): Input tensor for the model.
    #         attention_mask (Optional[torch.Tensor]): Attention mask for the input.
    #         max_length (int): Maximum length of the generated sequence.
    #         num_beams (int): Number of beams for beam search.
    #
    #     Returns:
    #         torch.Tensor: Generated output tensor.
    #     """
    #     return self.model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         max_length=5,
    #         num_beams=num_beams,
    #         num_return_sequences=num_beams,
    #         **kwargs
    #     )
    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            num_beams: int = 10,  # 默认与 topk 对齐
            num_return_sequences: Optional[int] = None,
            max_length: int = 5,
            early_stopping: bool = True,
            use_cache: bool = True,
            do_sample: bool = False,
            **kwargs
    ):
        """Generate recommendations efficiently using beam search.

        Args:
            input_ids (torch.Tensor): Input tensor for the model.
            attention_mask (Optional[torch.Tensor]): Attention mask.
            num_beams (int): Number of beams for beam search (default=10).
            num_return_sequences (Optional[int]): Number of returned sequences (default=num_beams).
            max_length (int): Maximum length of the generated sequence (default=5).
            early_stopping (bool): Whether to stop generation early when <eos> is reached.
            use_cache (bool): Whether to cache past key/value states for faster decoding.
            do_sample (bool): Whether to use sampling instead of beam search.
        Returns:
            torch.Tensor: Generated sequences tensor.
        """
        if num_return_sequences is None:
            num_return_sequences = num_beams  # 默认为与beam数量一致

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=early_stopping,
            use_cache=use_cache,
            do_sample=do_sample,
            **kwargs
        )


def calculate_pos_index(preds, labels, maxk=20):
    """Calculate the position index of the ground truth items.

    Args:
      preds: The predicted token sequences, of shape
        (batch_size, maxk, seq_len).
      labels: The ground truth token sequences, of shape (batch_size, seq_len).

    Returns:
      A boolean tensor of shape (batch_size, maxk) indicating whether the
      prediction at each position is correct.
    """
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert (
        preds.shape[1] == maxk
    ), f'preds.shape[1] = {preds.shape[1]} != {maxk}'

    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
      cur_label = labels[i].tolist()
      for j in range(maxk):
        cur_pred = preds[i, j].tolist()
        if cur_pred == cur_label:
          pos_index[i, j] = True
          break
    return pos_index

def calculate_pos_index_fast(preds: torch.Tensor, labels: torch.Tensor, pad_token_id=None):
    """
    preds: [B, K, L], labels: [B, L]
    返回: [B, K]，每个元素表示该候选是否与 labels 完全一致（可选忽略 pad）
    """
    # [B, 1, L] 与 [B, K, L] 广播比较 -> [B, K, L]
    if pad_token_id is None:
        eq = (preds == labels.unsqueeze(1))               # 逐位相等
    else:
        # 可选：忽略 pad 位置，仅在非 pad 处比较
        valid = (labels != pad_token_id).unsqueeze(1)     # [B, 1, L]
        eq = torch.where(valid, preds == labels.unsqueeze(1), torch.ones_like(preds, dtype=torch.bool))
    # 沿 L 维全为真 -> 整句相等
    pos_index = eq.all(dim=-1)                            # [B, K]
    return pos_index


def recall_at_k(pos_index, k):
  return pos_index[:, :k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index, k):
  # Assume only one ground truth item per example
  ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
  dcg = 1.0 / torch.log2(ranks + 1)
  dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
  return dcg[:, :k].sum(dim=1).cpu().float()

# def train(model, train_loader, optimizer, device):
#     model.train()
#     total_loss = 0.0
#     for batch in tqdm(train_loader, desc="Training"):
#         input_ids = batch['history'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['target'].to(device)
#
#         optimizer.zero_grad()
#         loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     return total_loss / len(train_loader)

def train(model, train_loader, optimizer, device, epoch=None, num_epochs=None):
    model.train()
    total_loss = 0.0

    # ✅ 在进度条描述中显示 epoch
    desc = f"Epoch {epoch}/{num_epochs}" if epoch is not None else "Training"
    progress_bar = tqdm(train_loader, desc=desc, dynamic_ncols=True)

    for batch in progress_bar:
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ✅ 在进度条后端显示当前 loss
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


# def evaluate(model, eval_loader, topk_list, beam_size, device):
#     model.eval()
#     recalls = {'Recall@' + str(k): [] for k in topk_list}
#     ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
#
#     with torch.no_grad():
#         for batch in tqdm(eval_loader, desc="Evaluating"):
#             input_ids = batch['history'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['target'].to(device)
#
#             preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
#             preds = preds[:, 1:]  # Exclude the start token
#             preds = preds.reshape(input_ids.shape[0], beam_size, -1)  # Reshape to (batch_size, beam_size, seq_len)
#             pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
#             # print(f"pos_index shape: {pos_index.shape}, pos_index: {pos_index}")
#             for k in topk_list:
#                 recall = recall_at_k(pos_index, k).mean().item()
#                 ndcg = ndcg_at_k(pos_index, k).mean().item()
#                 recalls['Recall@' + str(k)].append(recall)
#                 ndcgs['NDCG@' + str(k)].append(ndcg)
#     # Calculate average recalls and ndcgs
#     avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
#     avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
#     return avg_recalls, avg_ndcgs

def evaluate(model, eval_loader, topk_list, beam_size, device):
    model.eval()
    recalls = {f'Recall@{k}': [] for k in topk_list}
    ndcgs = {f'NDCG@{k}': [] for k in topk_list}

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=5,
                num_beams=max(topk_list),  # 用最大的topk即可
                num_return_sequences=max(topk_list),
                early_stopping=True,
                use_cache=True,
            )

            preds = preds[:, 1:]  # 去掉起始token
            beam_k = max(topk_list)
            preds = preds.reshape(input_ids.size(0), beam_k, -1)

            pos_index = calculate_pos_index(preds, labels, maxk=beam_k)

            for k in topk_list:
                recalls[f'Recall@{k}'].append(recall_at_k(pos_index, k).mean().item())
                ndcgs[f'NDCG@{k}'].append(ndcg_at_k(pos_index, k).mean().item())

    avg_recalls = {k: sum(v)/len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v)/len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs


# def set_seed(seed):
#     """Set random seed for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
def set_seed(seed):
    import random, numpy as np, torch, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ✅ 保证 cuDNN 算法确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # # ✅ 禁用非确定性行为（PyTorch >= 1.9）
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception as e:
    #     print("Deterministic algorithms not fully supported:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIGER configuration")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--infer_size', type=int, default=96, help='Inference size for generating recommendations')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers in the model')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--d_ff', type=int, default=1024, help='Dimension of the feed-forward layer')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--d_kv', type=int, default=64, help='Dimension of key and value vectors')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--vocab_size', type=int, default=1025, help='Vocabulary size')
    parser.add_argument('--pad_token_id', type=int, default=0, help='Padding token ID')
    parser.add_argument('--eos_token_id', type=int, default=0, help='End of sequence token ID')
    parser.add_argument('--feed_forward_proj', type=str, default='relu', help='Feed forward projection type')
    parser.add_argument('--max_len', type=int, default=20, help='Maximum length for padding or truncation')
    
    # parser.add_argument('--dataset_path', type=str, default='../data/All_Beauty', help='Path to the dataset')
    # parser.add_argument('--code_path', type=str, default='../data/All_Beauty/All_Beauty_t5_rqvae.npy', help='Path to the item-to-code mapping file')
    parser.add_argument('--dataset_path', type=str, default='../data/Beauty', help='Path to the dataset')
    parser.add_argument('--code_path', type=str, default='../data/Beauty/Beauty_clip_rqvae.npy', help='Path to the item-to-code mapping file')

    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluation'], help='Mode of operation')
    parser.add_argument('--log_path', type=str, default='./logs/tiger_20251109.log', help='Path to the log file')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./ckpt_20251109/tiger.pth', help='Path to save the trained model')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--topk_list', type=list, default=[5,10], help='List of top-k values for evaluation metrics')
    parser.add_argument('--beam_size', type=int, default=10, help='Beam size for generation')
    config = vars(parser.parse_args())
    # Set up logging
    logging.basicConfig(
        filename=config['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration: {config}")
    
    # Initialize model
    model = TIGER(config)
    print(model.n_parameters)
    logging.info(model.n_parameters)

    # Set random seed for reproducibility
    set_seed(config['seed'])
    # Check if the device is available
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    train_dataset = GenRecDataset(
        dataset_path=config['dataset_path']+ '/train.parquet',
        code_path=config['code_path'],
        mode='train',
        max_len=config['max_len']
    )
    validation_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/valid.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len']
    )
    test_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/test.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len']
    )

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=config['infer_size'], shuffle=False)
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Train the model
    model.to(device)
    best_ndcg = 0.0


    # 创建CSV日志文件（带时间戳）
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # csv_path = f"./logs/tiger_metrics_{timestamp}.csv"
    # ✅ 创建独立实验文件夹（包含模型与日志）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"./runs/tiger_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    # ✅ 设置模型与日志保存路径
    csv_path = os.path.join(exp_dir, "metrics.csv")
    config['save_dir'] = exp_dir  # 保存路径根目录

    # 写入CSV表头
    columns = ["epoch", "train_loss"] + \
              [f"Recall@{k}" for k in config['topk_list']] + \
              [f"NDCG@{k}" for k in config['topk_list']]
    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

    best_metric = 0.0
    best_epoch = -1
    early_stop_counter = 0
    topk_metric = f"NDCG@{max(config['topk_list'])}"

    val_interval = 10  # ✅ 每隔多少个 epoch 验证一次

    for epoch in range(config['num_epochs']):
        logging.info(f"\n===== Epoch {epoch + 1}/{config['num_epochs']} =====")

        # ---- 1️⃣ 训练阶段 ----
        # train_loss = train(model, train_dataloader, optimizer, device)
        train_loss = train(model, train_dataloader, optimizer, device,
                           epoch=epoch + 1, num_epochs=config['num_epochs'])

        logging.info(f"Training loss: {train_loss:.4f}")

        # ---- 2️⃣ 验证阶段（每隔 val_interval 执行一次） ----
        if epoch + 1 >= 1 and (epoch + 1) % val_interval == 0:
            topk_list_val = [5]  # ✅ 验证阶段只评估 Top-5
            avg_recalls, avg_ndcgs = evaluate(
                model, validation_dataloader, topk_list_val, beam_size=5, device=device
            )
            val_metric = avg_ndcgs["NDCG@5"]  # ✅ early stopping 基于 NDCG@5

            # ---- 打印与记录 ----
            print(f"\n📊 Validation Results (Epoch {epoch + 1})")
            print(f"  Train Loss: {train_loss:.4f}")
            for k in topk_list_val:
                print(f"  Recall@{k}: {avg_recalls[f'Recall@{k}']:.4f} | NDCG@{k}: {avg_ndcgs[f'NDCG@{k}']:.4f}")

            record = {"epoch": epoch + 1, "train_loss": train_loss}
            record.update(avg_recalls)
            record.update(avg_ndcgs)
            df_row = pd.DataFrame([record]).reindex(columns=columns)
            df_row.to_csv(csv_path, mode='a', index=False, header=False, na_rep="")

            # ---- 4️⃣ early stopping + 保存 ----
            if val_metric > best_metric:
                best_metric = val_metric
                best_epoch = epoch + 1
                early_stop_counter = 0

                # base, ext = os.path.splitext(config['save_path'])
                # save_path = f"{base}_epoch{best_epoch}{ext}"
                # ✅ 每次保存到当前实验文件夹内
                save_path = os.path.join(config['save_dir'], f"tiger_epoch{best_epoch}.pth")
                torch.save(model.state_dict(), save_path)
                # torch.save(model.state_dict(), save_path)
                logging.info(f"✅ Epoch {best_epoch}: Improved NDCG@5={best_metric:.4f}, model saved to {save_path}")
                print(f"✅ Saved best model (Epoch {best_epoch}) → {save_path}")
            else:
                early_stop_counter += 1
                logging.info(f"No improvement ({early_stop_counter}/{config['early_stop']})")
                if early_stop_counter >= config['early_stop']:
                    logging.info("🛑 Early stopping triggered.")
                    break

        else:
            # ---- 非验证轮次 ----
            logging.info(f"⏩ Skipping validation at epoch {epoch + 1} (only every {val_interval} epochs).")
            # 只写入训练损失到CSV
            record = {"epoch": epoch + 1, "train_loss": train_loss}
            df_row = pd.DataFrame([record]).reindex(columns=columns)
            df_row.to_csv(csv_path, mode='a', index=False, header=False, na_rep="")

    # ---- 5️⃣ 测试阶段（加载最佳模型） ----
    print("\n🚀 Evaluating best model on test set...")
    # model.load_state_dict(torch.load(config['save_path']))
    # best_model_path = f"{base}_epoch{best_epoch}{ext}"
    # model.load_state_dict(torch.load(best_model_path))
    best_model_path = os.path.join(config['save_dir'], f"tiger_epoch{best_epoch}.pth")
    model.load_state_dict(torch.load(best_model_path))
    test_recalls, test_ndcgs = evaluate(
        model, test_dataloader, config['topk_list'], config['beam_size'], device
    )

    # 打印测试结果
    print("\n🏁 Final Test Results")
    for k in config['topk_list']:
        print(f"  Recall@{k}: {test_recalls[f'Recall@{k}']:.4f} | NDCG@{k}: {test_ndcgs[f'NDCG@{k}']:.4f}")

    # 追加写入CSV
    test_record = {
        "epoch": "test",
        "train_loss": "",  # 测试集不需要这个值，用空字符串占位
    }
    test_record.update(test_recalls)
    test_record.update(test_ndcgs)
    pd.DataFrame([test_record]).to_csv(csv_path, mode='a', index=False, header=False)

    print(f"\n📂 Metrics saved continuously to: {csv_path}")
