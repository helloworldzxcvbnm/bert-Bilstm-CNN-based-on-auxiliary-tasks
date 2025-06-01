import os
import re
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import Counter, defaultdict

# ===========================
# 1. 中文类别 → 整数 ID（0~9） 映射
# ===========================
label2id = {
    "体育": 0,
    "娱乐": 1,
    "家居": 2,
    "时尚": 3,
    "时政": 4,
    "游戏": 5,
    "科技": 6,
    "财经": 7,
    "房产": 8,
    "教育": 9
}


# ===========================
# 2. 定义“清洗文本”函数：去掉非中文/英文字母/数字/常见中文标点等
# ===========================
def clean_text(text: str) -> str:
    """
    将 text 中所有“乱码”字符（不属于以下字符集）都去掉：
      - 中文（\u4e00-\u9fff）
      - ASCII 字母与数字（A-Za-z0-9）
      - 常见中文标点：。，！？、：；“”‘’《》（）…—- 以及空格
    """
    allowed_pattern = r"[^\u4e00-\u9fffA-Za-z0-9\s，。！？、：；“”‘’《》（）…—\-]"
    return re.sub(allowed_pattern, "", text)


# ===========================
# 3. 加载并清洗 cnews 数据的函数，同时统计总数与各类数量
# ===========================
def load_and_summarize_cnews(
        file_path: str,
        label2id_map: dict,
        clean_func,
        skip_invalid: bool = True,
        min_chinese_ratio: float = 0.3
):
    """
    读取 cnews 格式数据文件：每行 "<标签>\t<标题+内容>"，
    清洗正文中的“乱码”字符，统计并返回：
      - samples: List[(label_id: int, cleaned_text: str)]
      - total_count: 样本总数
      - class_counts: Counter，每个类别的样本数

    参数：
      - file_path:       数据文件路径，UTF-8 编码，每行形如 "体育\t标题 内容..."
      - label2id_map:    标签到整数的映射字典，例如 {"体育":0, ...}
      - clean_func:      文本清洗函数，接收原始字符串，返回去掉乱码后的字符串
      - skip_invalid:    是否跳过“清洗后文本太短或中文比例太低”的行
      - min_chinese_ratio: 若跳过无效行，则要求“清洗后文本中中文字符数 / 文本长度 >= min_chinese_ratio”才保留

    返回：
      - samples: List[tuple], 每个是 (label_id: int, cleaned_text: str)
      - total_count: int, 有效样本总数
      - class_counts: Counter, 每个类别对应的样本数
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件：{file_path}")

    samples = []
    class_counts = Counter()
    chinese_regex = re.compile(r"[\u4e00-\u9fff]")

    with open(file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # 1) 按制表符切分：label_str, title_and_content
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue

            label_str = parts[0].strip()
            raw_text = parts[1].strip()  # 包含标题+正文

            # 2) 标签映射
            if label_str not in label2id_map:
                continue
            label_id = label2id_map[label_str]

            # 3) 清洗“乱码”字符
            cleaned_text = clean_func(raw_text)
            if not cleaned_text:
                continue

            # 5) 统计并加入样本列表
            samples.append((label_id, cleaned_text))
            class_counts[label_id] += 1

    total_count = len(samples)
    return samples, total_count, class_counts


# ===========================
# 4. 定义 PyTorch Dataset：接收上面返回的 samples
# ===========================
class CNewsDataset(Dataset):
    def __init__(self, raw_samples, tokenizer, max_len, num_classes=10):
        """
        raw_samples: List[(label_id: int, text: str)]
        tokenizer:    BertTokenizer
        max_len:      最大序列长度
        num_classes:  一共多少类别（这里默认为 10）
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes

        for label_id, text in tqdm(raw_samples, desc="Tokenizing", ncols=80):
            # Tokenizer 编码
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].squeeze(0)  # [max_len]
            attention_mask = encoding["attention_mask"].squeeze(0)  # [max_len]

            # 构造 one-hot 向量
            label_vec = torch.zeros(self.num_classes, dtype=torch.float)
            label_vec[label_id] = 1.0

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": label_vec
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["input_ids"], s["attention_mask"], s["label"]


# ===========================
# 5. 定义模型：BERT + BiLSTM + CNN + Multi-Sigmoid Head
# ===========================
class BertBiLstmCnnMultiHead(nn.Module):
    def __init__(
            self,
            bert_path,
            lstm_hidden_size=256,
            lstm_layers=1,
            cnn_kernel_sizes=(2, 3, 4),
            cnn_out_channels=128,
            num_classes=10
    ):
        super(BertBiLstmCnnMultiHead, self).__init__()
        # 1) 加载本地 BERT
        self.bert = BertModel.from_pretrained(bert_path)
        bert_hidden_size = self.bert.config.hidden_size  # 通常是 768

        # 2) BiLSTM
        self.bilstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0
        )

        # 3) CNN：用多种 kernel size 提取 n-gram 特征
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=lstm_hidden_size * 2,  # BiLSTM 双向 => 隐层输出 hidden_size*2
                out_channels=cnn_out_channels,
                kernel_size=k
            ) for k in cnn_kernel_sizes
        ])

        self.dropout = nn.Dropout(p=0.1)

        # 4) 最终拼接后特征维度 = len(cnn_kernel_sizes) * cnn_out_channels
        feat_dim = len(cnn_kernel_sizes) * cnn_out_channels

        # 5) 构建 num_classes 个 二分类 Sigmoid 头
        self.classifiers = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(num_classes)])
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask):
        # 1) BERT 编码，输出 last_hidden_state: [batch, seq_len, hidden_size]
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_outputs.last_hidden_state  # [B, L, D_bert]

        # 2) BiLSTM
        lstm_out, _ = self.bilstm(hidden_states)  # [B, L, hidden_size*2]

        # 3) CNN：需要把维度转成 [B, hidden_size*2, L]
        cnn_input = lstm_out.permute(0, 2, 1)  # [B, hidden*2, L]

        conv_results = []
        for conv in self.convs:
            x = conv(cnn_input)  # [B, out_channels, L - k + 1]
            x = nn.functional.relu(x)
            x = nn.functional.max_pool1d(x, kernel_size=x.shape[2])  # [B, out_channels, 1]
            conv_results.append(x.squeeze(2))  # [B, out_channels]

        feat = torch.cat(conv_results, dim=1)  # [B, feat_dim]
        feat = self.dropout(feat)

        logits = []
        for classifier in self.classifiers:
            logits.append(classifier(feat))  # 每个输出 [B, 1]
        logits = torch.cat(logits, dim=1)  # [B, num_classes]

        probs = torch.sigmoid(logits)  # [B, num_classes]
        return probs


# ===========================
# 6. 训练/验证函数
# ===========================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Training"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)  # [B, num_classes]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)  # [B, num_classes]
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            true_labels = torch.argmax(labels, dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(true_labels)

    report = classification_report(all_labels, all_preds, digits=4)
    return report


# ===========================
# 7. main 函数：整合所有步骤
# ===========================
def main():
    # --------- 7.1 配置路径 & 参数 ---------
    BERT_PATH = r"/path/to/bert-base-chinese"    # 替换为你的bert模型路径
    TRAIN_FILE = r"/path/to/cnews/cnews.train.txt"   # 替换训练集路径
    VAL_FILE = r"/path/to/cnews/cnews.val.txt"       # 替换验证集路径
    TEST_FILE = r"/path/to/cnews/cnews.test.txt"      # 替换测试集路径

    MAX_SEQ_LEN = 256
    BATCH_SIZE = 256
    NUM_CLASSES = 10
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- 7.2 先加载并统计原始训练数据 ---------
    print("┌──────────────────────────────────────────────────┐")
    raw_train_samples, train_total, train_counts = load_and_summarize_cnews(
        TRAIN_FILE, label2id, clean_text, skip_invalid=True, min_chinese_ratio=0.3
    )
    print(f"Train 原始总样本数: {train_total}")
    for lbl, cnt in sorted(train_counts.items()):
        class_name = [k for k, v in label2id.items() if v == lbl][0]
        print(f"  类别 “{class_name}” (ID={lbl}) 原始样本数: {cnt}")
    print("└──────────────────────────────────────────────────┘\n")

    # --------- 7.2.1 对训练集进行按类别逐级递减的子采样 ---------
    # 每个类别按 ID 计算减少比例：类别 ID=i 时，保留比例 = 1 - 0.05*i
    grouped = defaultdict(list)
    for label_id, text in raw_train_samples:
        grouped[label_id].append((label_id, text))

    subsampled_train_samples = []
    subsampled_counts = {}
    random.seed(42)  # 确保可复现
    for label_id in range(NUM_CLASSES):
        original_list = grouped[label_id]
        original_count = len(original_list)
        retain_ratio = 1.0 - 0.05 * label_id
        retain_count = int(original_count * retain_ratio)

        # 随机抽样
        if retain_count >= original_count:
            selected = original_list.copy()
        else:
            selected = random.sample(original_list, retain_count)

        subsampled_train_samples.extend(selected)
        subsampled_counts[label_id] = len(selected)

    print("┌──────────────────────────────────────────────────┐")
    print(f"Train 子采样后总样本数: {len(subsampled_train_samples)}")
    for lbl, cnt in sorted(subsampled_counts.items()):
        class_name = [k for k, v in label2id.items() if v == lbl][0]
        print(f"  类别 “{class_name}” (ID={lbl}) 子采样后样本数: {cnt}")
    print("└──────────────────────────────────────────────────┘\n")

    # --------- 7.3 加载验证集与测试集 ---------
    print("┌──────────────────────────────────────────────────┐")
    val_samples, val_total, val_counts = load_and_summarize_cnews(
        VAL_FILE, label2id, clean_text, skip_invalid=True, min_chinese_ratio=0.3
    )
    print(f"Val   总样本数: {val_total}")
    for lbl, cnt in sorted(val_counts.items()):
        class_name = [k for k, v in label2id.items() if v == lbl][0]
        print(f"  类别 “{class_name}” (ID={lbl}) 样本数: {cnt}")
    print("└──────────────────────────────────────────────────┘\n")

    print("┌──────────────────────────────────────────────────┐")
    test_samples, test_total, test_counts = load_and_summarize_cnews(
        TEST_FILE, label2id, clean_text, skip_invalid=True, min_chinese_ratio=0.3
    )
    print(f"Test  总样本数: {test_total}")
    for lbl, cnt in sorted(test_counts.items()):
        class_name = [k for k, v in label2id.items() if v == lbl][0]
        print(f"  类别 “{class_name}” (ID={lbl}) 样本数: {cnt}")
    print("└──────────────────────────────────────────────────┘\n")

    # --------- 7.4 构建 Dataset 与 DataLoader ---------
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    print("正在构建 train 数据集...\n")
    train_dataset = CNewsDataset(subsampled_train_samples, tokenizer, MAX_SEQ_LEN, num_classes=NUM_CLASSES)
    print("正在构建 val 数据集...\n")
    val_dataset = CNewsDataset(val_samples, tokenizer, MAX_SEQ_LEN, num_classes=NUM_CLASSES)
    print("正在构建 test 数据集...\n")
    test_dataset = CNewsDataset(test_samples, tokenizer, MAX_SEQ_LEN, num_classes=NUM_CLASSES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"TrainLoader 样本数: {len(train_dataset)}")
    print(f"ValLoader   样本数: {len(val_dataset)}")
    print(f"TestLoader  样本数: {len(test_dataset)}\n")

    # --------- 7.5 初始化模型、损失函数 & 优化器 ---------
    model = BertBiLstmCnnMultiHead(
        bert_path=BERT_PATH,
        lstm_hidden_size=256,
        lstm_layers=1,
        cnn_kernel_sizes=(2, 3, 4),
        cnn_out_channels=128,
        num_classes=NUM_CLASSES
    )
    model = nn.DataParallel(model)
    model.to(DEVICE)

    criterion = nn.BCELoss()  # 多个 Sigmoid 头，用 BCE
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --------- 7.6 训练循环：每个 epoch 训练 + 验证 ---------
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"========== Epoch {epoch}/{NUM_EPOCHS} ==========")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"  Training Loss: {train_loss:.4f}")

        print("  ---> Validating on validation set:")
        val_report = evaluate(model, val_loader, DEVICE)
        print(val_report)

    # --------- 7.7 最后在测试集上评测 ---------
    print("========== Final evaluation on Test Set ==========")
    test_report = evaluate(model, test_loader, DEVICE)
    print(test_report)


if __name__ == "__main__":
    main()
