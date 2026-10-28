import torch
import gzip
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

# 引用訓練腳本中的類別
from train_flow_bert import (
    FlowDataset,
    FlowEmbedding,
    BertConfig,
    BertForSequenceClassification
)

def evaluate():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 1. 載入模型
    save_path = "bert_flow.pth"

    if not Path(save_path).exists():
        print(f"錯誤：找不到模型檔案 {save_path}")
        return

    checkpoint = torch.load(
        save_path,
        weights_only=False
    )

    scaler = checkpoint['scaler']
    config = checkpoint['config']
    seed = checkpoint.get('seed', 42)

    # 2. 建立模型
    model = BertForSequenceClassification(config).to(device)

    model.load_state_dict(
        checkpoint['model']
    )

    embedding_proj = FlowEmbedding(
        input_dim=2
    ).to(device)

    embedding_proj.load_state_dict(
        checkpoint['proj']
    )

    # 3. 載入資料
    all_data = []

    data_dir = "./processed_data"

    for f in Path(data_dir).glob("*.json.gz"):

        with gzip.open(f, 'rt') as f_in:

            for line in f_in:

                item = json.loads(line)

                if item.get('app_label') is not None:
                    all_data.append(item)

    # 4. 與訓練相同 split
    _, val_data = train_test_split(
        all_data,
        test_size=0.2,
        random_state=seed
    )

    test_loader = DataLoader(
        FlowDataset(val_data, scaler),
        batch_size=32
    )

    # 5. Evaluation
    model.eval()
    embedding_proj.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in test_loader:

            embeds = embedding_proj(
                batch['inputs_embeds'].to(device)
            )

            outputs = model(
                inputs_embeds=embeds
            )

            preds = torch.argmax(
                outputs.logits,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                batch['labels'].numpy()
            )

    # 6. Report
    print("\n" + "=" * 50)
    print("        BERT-Flow 最終測試報告")
    print("=" * 50)

    print(
        classification_report(
            all_labels,
            all_preds,
            digits=4
        )
    )

    # 7. Confusion Matrix
    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    print("\n--- 混淆矩陣已生成 ---")

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    evaluate()