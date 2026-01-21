"""
GPT asosida Huquqiy Savol-Javob Modeli
======================================
GPT-2 modelini fine-tuning qilish (Decoder-only)
"""

import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
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
    GPT_MODEL = "gpt2"  # gpt2, gpt2-medium, gpt2-large
    MAX_LENGTH = 256

    # Training
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    EPOCHS = 5
    WARMUP_RATIO = 0.1
    GRADIENT_ACCUMULATION = 4

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
    """GPT uchun Dataset - Causal LM format"""

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format: <|context|> context <|question|> question <|answer|> answer <|endoftext|>
        text = f"<|context|>{item['context']}<|question|>{item['question']}<|answer|>{item['answer']}<|endoftext|>"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Labels = input_ids (causal LM)
        labels = input_ids.clone()
        # Padding tokenlarni -100 qilish (loss hisoblanmasin)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# ============================================
# TRAINING
# ============================================

def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    progress = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()
        progress.set_postfix({'loss': f'{outputs.loss.item():.4f}'})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def generate_answer(model, tokenizer, context, question, device, max_new_tokens=64):
    """Javob generatsiya qilish"""
    model.eval()

    prompt = f"<|context|>{context}<|question|>{question}<|answer|>"

    encoding = tokenizer(prompt, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=False)

    # Javobni ajratib olish
    if "<|answer|>" in generated:
        answer = generated.split("<|answer|>")[-1]
        answer = answer.replace("<|endoftext|>", "").strip()
        return answer

    return generated


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("GPT HUQUQIY SAVOL-JAVOB MODELI")
    print("=" * 60)

    print(f"\nQurilma: {Config.DEVICE}")
    print(f"Model: {Config.GPT_MODEL}")

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
    tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT_MODEL)

    # Maxsus tokenlar qo'shish
    special_tokens = {
        'pad_token': '<|pad|>',
        'additional_special_tokens': ['<|context|>', '<|question|>', '<|answer|>']
    }
    tokenizer.add_special_tokens(special_tokens)

    # 3. Dataset
    print("\n[3/5] Dataset yaratilmoqda...")
    train_dataset = LegalQADataset(train_data, tokenizer, Config.MAX_LENGTH)
    val_dataset = LegalQADataset(val_data, tokenizer, Config.MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    # 4. Model
    print("\n[4/5] Model yuklanmoqda...")
    model = GPT2LMHeadModel.from_pretrained(Config.GPT_MODEL)

    # Tokenizer o'zgargani uchun embedding resize
    model.resize_token_embeddings(len(tokenizer))

    model = model.to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Jami parametrlar: {total_params:,}")
    print(f"O'rganiladigan: {trainable_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    total_steps = (len(train_loader) // Config.GRADIENT_ACCUMULATION) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * Config.WARMUP_RATIO),
        num_training_steps=total_steps
    )

    # 5. Training
    print("\n[5/5] Training boshlanmoqda...")
    print("-" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            Config.DEVICE, Config.GRADIENT_ACCUMULATION
        )
        val_loss = evaluate(model, val_loader, Config.DEVICE)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Model va tokenizerni saqlash
            model.save_pretrained('best_gpt_model')
            tokenizer.save_pretrained('best_gpt_model')
            print(">>> Model saqlandi!")

    # Test
    print("\n" + "=" * 60)
    print("TEST")
    print("=" * 60)

    test_samples = [
        ("Konstitutsiya davlat hayotida eng oliy yuridik kuchga ega.", "Konstitutsiya nima?"),
        ("Mehnat huquqi bo'yicha xodim dam olish huquqiga ega.", "Xodim huquqlari qanday?"),
        ("Mulk egasi egalik qilish, foydalanish va tasarruf etish huquqlariga ega.", "Mulk huquqi nima?"),
    ]

    for ctx, q in test_samples:
        answer = generate_answer(model, tokenizer, ctx, q, Config.DEVICE)
        print(f"\nSavol: {q}")
        print(f"Javob: {answer}")

    print(f"\nEng yaxshi val loss: {best_val_loss:.4f}")
    print("Model saqlandi: best_gpt_model/")
    print("Yakunlandi!")


if __name__ == "__main__":
    main()
