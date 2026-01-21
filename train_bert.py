"""
BERT asosida Huquqiy Savol-Javob Modeli
=======================================
Pre-trained BERT modelini fine-tuning qilish
"""

import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertModel,
    BertConfig,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm


# ============================================
# KONFIGURATSIYA
# ============================================

class Config:
    # Ma'lumotlar
    DATA_PATH = "data.jsonl"
    TRAIN_SPLIT = 0.8

    # Model
    BERT_MODEL = "bert-base-multilingual-cased"  # Ko'p tilli BERT
    MAX_INPUT_LEN = 256
    MAX_OUTPUT_LEN = 64

    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 5
    WARMUP_RATIO = 0.1

    # Qurilma
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# MA'LUMOTLARNI YUKLASH
# ============================================

def load_data(filepath: str):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


# ============================================
# DATASET
# ============================================

class LegalQADataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=256, max_output_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Input: [CLS] context [SEP] question [SEP]
        input_text = f"{item['context']} [SEP] {item['question']}"
        target_text = item['answer']

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_output_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'labels_mask': target_encoding['attention_mask'].squeeze()
        }


# ============================================
# BERT SEQ2SEQ MODEL
# ============================================

class BertEncoder(nn.Module):
    """BERT Encoder"""

    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, outputs.pooler_output


class BertDecoder(nn.Module):
    """Transformer Decoder for BERT"""

    def __init__(self, vocab_size, hidden_size, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Embedding(512, hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        seq_len = tgt.size(1)
        positions = torch.arange(0, seq_len, device=tgt.device).unsqueeze(0)

        tgt_emb = self.dropout(self.embedding(tgt) + self.pos_encoding(positions))

        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        return self.fc_out(output)


class BertSeq2Seq(nn.Module):
    """BERT-based Seq2Seq Model"""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(self.device)

    def forward(self, input_ids, attention_mask, labels=None):
        # Encode
        encoder_output, _ = self.encoder(input_ids, attention_mask)

        # Decode
        if labels is not None:
            tgt_mask = self.generate_square_subsequent_mask(labels.size(1))
            memory_mask = (attention_mask == 0)
            output = self.decoder(labels, encoder_output, tgt_mask=tgt_mask, memory_mask=memory_mask)
            return output

        return encoder_output

    def generate(self, input_ids, attention_mask, tokenizer, max_len=64):
        self.eval()

        with torch.no_grad():
            encoder_output, _ = self.encoder(input_ids, attention_mask)
            memory_mask = (attention_mask == 0)

            # Start with [CLS] token
            generated = torch.tensor([[tokenizer.cls_token_id]], device=self.device)

            for _ in range(max_len):
                tgt_mask = self.generate_square_subsequent_mask(generated.size(1))
                output = self.decoder(generated, encoder_output, tgt_mask=tgt_mask, memory_mask=memory_mask)

                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == tokenizer.sep_token_id:
                    break

            return generated


# ============================================
# TRAINING
# ============================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0

    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward (teacher forcing)
        output = model(input_ids, attention_mask, labels[:, :-1])

        # Loss
        output = output.reshape(-1, output.size(-1))
        target = labels[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids, attention_mask, labels[:, :-1])

            output = output.reshape(-1, output.size(-1))
            target = labels[:, 1:].reshape(-1)

            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("BERT HUQUQIY SAVOL-JAVOB MODELI")
    print("=" * 60)

    print(f"\nQurilma: {Config.DEVICE}")
    print(f"Model: {Config.BERT_MODEL}")

    # 1. Ma'lumotlar
    print("\n[1/5] Ma'lumotlar yuklanmoqda...")
    data = load_data(Config.DATA_PATH)

    random.seed(42)
    random.shuffle(data)

    split_idx = int(len(data) * Config.TRAIN_SPLIT)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # 2. Tokenizer
    print("\n[2/5] Tokenizer yuklanmoqda...")
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL)

    # 3. Dataset
    print("\n[3/5] Dataset yaratilmoqda...")
    train_dataset = LegalQADataset(train_data, tokenizer, Config.MAX_INPUT_LEN, Config.MAX_OUTPUT_LEN)
    val_dataset = LegalQADataset(val_data, tokenizer, Config.MAX_INPUT_LEN, Config.MAX_OUTPUT_LEN)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    # 4. Model
    print("\n[4/5] Model yaratilmoqda...")
    encoder = BertEncoder(Config.BERT_MODEL)
    decoder = BertDecoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=encoder.hidden_size,
        num_layers=4,
        num_heads=8
    )
    model = BertSeq2Seq(encoder, decoder, Config.DEVICE).to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Jami parametrlar: {total_params:,}")
    print(f"O'rganiladigan: {trainable_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * Config.WARMUP_RATIO),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 5. Training
    print("\n[5/5] Training boshlanmoqda...")
    print("-" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, Config.DEVICE)
        val_loss = evaluate(model, val_loader, criterion, Config.DEVICE)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_name': Config.BERT_MODEL,
            }, 'best_bert_model.pt')
            print(">>> Model saqlandi!")

    # Test
    print("\n" + "=" * 60)
    print("TEST")
    print("=" * 60)

    test_samples = [
        ("Konstitutsiya davlat hayotida eng oliy yuridik kuchga ega.", "Konstitutsiya nima?"),
        ("Mehnat huquqi bo'yicha xodim dam olish huquqiga ega.", "Xodim huquqlari qanday?"),
    ]

    for ctx, q in test_samples:
        input_text = f"{ctx} [SEP] {q}"
        encoding = tokenizer(input_text, return_tensors='pt', max_length=Config.MAX_INPUT_LEN, truncation=True)

        input_ids = encoding['input_ids'].to(Config.DEVICE)
        attention_mask = encoding['attention_mask'].to(Config.DEVICE)

        output = model.generate(input_ids, attention_mask, tokenizer, max_len=Config.MAX_OUTPUT_LEN)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"\nSavol: {q}")
        print(f"Javob: {answer}")

    print(f"\nEng yaxshi val loss: {best_val_loss:.4f}")
    print("Yakunlandi!")


if __name__ == "__main__":
    main()
