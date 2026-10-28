import torch
import torch.nn as nn
import json
import gzip
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 設定random seeds，確保每次分割的資料都一樣
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 定義數值projection layer
class FlowEmbedding(nn.Module):
    def __init__(self, input_dim=2, hidden_size=768):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_size)
    def forward(self, x):
        return self.projection(x)

class FlowDataset(Dataset):
    def __init__(self, data, scaler):
        # 提取feature並標準化
        raw_features = np.array([item['feature'] for item in data], dtype=np.float32)
        N, S, F = raw_features.shape
        # 使用傳入的 scaler 進行轉換
        scaled = scaler.transform(raw_features.reshape(-1, F))
        self.features = torch.tensor(scaled.reshape(N, S, F), dtype=torch.float32)
        self.labels = torch.tensor([item['app_label'] for item in data], dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {'inputs_embeds': self.features[idx], 'labels': self.labels[idx]}

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./processed_data" 
    
    # 1. 載入資料並過濾 None label
    all_data = []
    for f in Path(data_dir).glob("*.json.gz"):
        with gzip.open(f, 'rt') as f_in:
            for line in f_in:
                item = json.loads(line)
                if item.get('app_label') is not None:
                    all_data.append(item)
    
    if not all_data:
        print("錯誤：找不到有效資料，請檢查預處理結果。")
        return

    # 2. 準備 Scaler (必須使用所有資料來 fit，或僅使用訓練集)
    train_raw = np.array([x['feature'] for x in all_data]).reshape(-1, 2)
    scaler = StandardScaler().fit(train_raw)
    
    # 3. 分割資料 (固定 random_state)
    train_data, val_data = train_test_split(
        all_data, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"train: {len(train_data)}, test: {len(val_data)}")

    train_loader = DataLoader(
        FlowDataset(train_data, scaler),
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        FlowDataset(val_data, scaler),
        batch_size=4,
        num_workers=0
    )

    # 4. initalize
    config = BertConfig(
        vocab_size=1, 
        hidden_size=768, 
        num_hidden_layers=6, 
        num_labels=16 # 根據你的 utils.py 類別數量調整
    )
    model = BertForSequenceClassification(config).to(device)
    embedding_proj = FlowEmbedding(input_dim=2).to(device)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(embedding_proj.parameters()), 
        lr=5e-5
    )

    # 5. training
    num_epochs = 10 # 增加 epoch 提高準確率
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            embeds = embedding_proj(batch['inputs_embeds'].to(device))
            outputs = model(inputs_embeds=embeds, labels=batch['labels'].to(device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # 6. 儲存模型與所有必要參數
    save_path = "bert_flow_vpn.pth"
    torch.save({
        'model': model.state_dict(),
        'proj': embedding_proj.state_dict(),
        'scaler': scaler,
        'config': config,
        'seed': RANDOM_SEED
    }, save_path)
    print(f"model saved to {save_path}")

if __name__ == "__main__":
    train()
