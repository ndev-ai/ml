"""
Training script - LSTM modelini o'rgatish
"""

import json
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from main import (
    Config, Tokenizer, Vocabulary, LegalQADataset,
    Encoder, Decoder, Seq2Seq,
    load_data, prepare_input_output, collate_fn,
    train_epoch, evaluate, generate_answer
)


def train():
    print("=" * 60)
    print("HUQUQIY SAVOL-JAVOB LSTM MODELI - TRAINING")
    print("=" * 60)

    print(f"\nQurilma: {Config.DEVICE}")

    # 1. Ma'lumotlarni yuklash
    print("\n[1/5] Ma'lumotlar yuklanmoqda...")
    data = load_data(Config.DATA_PATH)
    print(f"Jami: {len(data)}")

    random.seed(42)
    random.shuffle(data)

    split_idx = int(len(data) * Config.TRAIN_SPLIT)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # 2. Tokenizer va Vocabulary
    print("\n[2/5] Vocabulary yaratilmoqda...")
    tokenizer = Tokenizer()

    all_tokens = []
    for item in data:
        inp, out = prepare_input_output(item, tokenizer)
        all_tokens.append(inp)
        all_tokens.append(out)

    vocab = Vocabulary(min_freq=Config.MIN_FREQ, max_size=Config.MAX_VOCAB_SIZE)
    vocab.build(all_tokens)

    # 3. DataLoader
    print("\n[3/5] DataLoader yaratilmoqda...")
    train_dataset = LegalQADataset(train_data, tokenizer, vocab)
    val_dataset = LegalQADataset(val_data, tokenizer, vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn
    )

    # 4. Model
    print("\n[4/5] Model yaratilmoqda...")
    encoder_hidden = Config.HIDDEN_DIM * 2  # bidirectional

    encoder = Encoder(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        rnn_type="LSTM",
        bidirectional=True,
        pad_idx=vocab.pad_idx
    )

    decoder = Decoder(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        encoder_hidden_dim=encoder_hidden,
        decoder_hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        rnn_type="LSTM",
        use_attention=True,
        pad_idx=vocab.pad_idx
    )

    model = Seq2Seq(encoder, decoder, Config.DEVICE).to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametrlar: {total_params:,}")

    # Optimizer va Loss
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 5. Training
    print("\n[5/5] Training...")
    print("-" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            Config.CLIP_GRAD, Config.DEVICE, Config.TEACHER_FORCING_RATIO
        )

        val_loss = evaluate(model, val_loader, criterion, Config.DEVICE)
        scheduler.step(val_loss)

        ppl = math.exp(val_loss) if val_loss < 10 else float('inf')
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'tokenizer': tokenizer,
            }, 'best_model.pt')
            print(">>> Model saqlandi!")

    # Test
    print("\n" + "=" * 60)
    print("TEST")
    print("=" * 60)

    tests = [
        ("Konstitutsiya davlat hayotida eng oliy yuridik kuchga ega.", "Konstitutsiya nima?"),
        ("Mehnat huquqi bo'yicha xodim dam olish huquqiga ega.", "Xodim huquqlari qanday?"),
    ]

    for ctx, q in tests:
        answer = generate_answer(model, ctx, q, tokenizer, vocab, Config.DEVICE)
        print(f"\nSavol: {q}")
        print(f"Javob: {answer}")

    print(f"\nEng yaxshi val loss: {best_val_loss:.4f}")
    print("Yakunlandi!")


if __name__ == "__main__":
    train()
