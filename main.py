"""
Huquqiy Savol-Javob RNN Modeli (Seq2Seq)
==========================================
Bu model O'zbekiston huquqiy ma'lumotlari asosida
savollarga javob berish uchun o'rgatiladi.

Model arxitekturasi: Encoder-Decoder (LSTM/GRU)
"""

import json
import random
import re
from collections import Counter
from typing import List, Tuple, Dict, Optional
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


# ============================================
# 1. KONFIGURATSIYA
# ============================================

class Config:
    """Model va training konfiguratsiyasi"""
    # Ma'lumotlar
    DATA_PATH = "data.jsonl"
    TRAIN_SPLIT = 0.8

    # Tokenizatsiya
    MIN_FREQ = 2  # Minimal so'z chastotasi
    MAX_VOCAB_SIZE = 50000
    MAX_INPUT_LEN = 256  # Kontekst + savol maksimal uzunligi
    MAX_OUTPUT_LEN = 64  # Javob maksimal uzunligi

    # Model arxitekturasi
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    RNN_TYPE = "LSTM"  # "LSTM" yoki "GRU"
    BIDIRECTIONAL_ENCODER = True
    ATTENTION = True  # Attention mexanizmi

    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 15
    TEACHER_FORCING_RATIO = 0.5
    CLIP_GRAD = 1.0

    # Qurilma
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Maxsus tokenlar
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"


# ============================================
# 2. TOKENIZATSIYA VA VOCABULARY
# ============================================

class Tokenizer:
    """O'zbek tili uchun oddiy tokenizer"""

    def __init__(self):
        self.pattern = re.compile(r"[\w']+|[.,!?;:\-\(\)\[\]\"']")

    def tokenize(self, text: str) -> List[str]:
        """Matnni tokenlarga ajratish"""
        text = text.lower().strip()
        tokens = self.pattern.findall(text)
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """Tokenlarni matnga birlashtirish"""
        return " ".join(tokens)


class Vocabulary:
    """So'z lug'ati"""

    def __init__(self, min_freq: int = 2, max_size: int = 50000):
        self.min_freq = min_freq
        self.max_size = max_size

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()

        # Maxsus tokenlarni qo'shish
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Maxsus tokenlarni qo'shish"""
        special_tokens = [
            Config.PAD_TOKEN,
            Config.SOS_TOKEN,
            Config.EOS_TOKEN,
            Config.UNK_TOKEN
        ]
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    @property
    def pad_idx(self) -> int:
        return self.word2idx[Config.PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.word2idx[Config.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.word2idx[Config.EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[Config.UNK_TOKEN]

    def build(self, texts: List[List[str]]):
        """Lug'atni qurish"""
        # So'z chastotalarini hisoblash
        for tokens in texts:
            self.word_freq.update(tokens)

        # Minimal chastotadan yuqori so'zlarni qo'shish
        current_idx = len(self.word2idx)
        for word, freq in self.word_freq.most_common(self.max_size):
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = current_idx
                self.idx2word[current_idx] = word
                current_idx += 1

        print(f"Lug'at hajmi: {len(self.word2idx)} so'z")

    def encode(self, tokens: List[str]) -> List[int]:
        """Tokenlarni indekslarga o'girish"""
        return [self.word2idx.get(t, self.unk_idx) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Indekslarni tokenlarga o'girish"""
        tokens = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx not in (self.pad_idx, self.sos_idx):
                tokens.append(self.idx2word.get(idx, Config.UNK_TOKEN))
        return tokens

    def __len__(self) -> int:
        return len(self.word2idx)


# ============================================
# 3. MA'LUMOTLARNI YUKLASH VA TAYYORLASH
# ============================================

def load_data(filepath: str) -> List[Dict]:
    """JSONL fayldan ma'lumotlarni yuklash"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


def prepare_input_output(item: Dict, tokenizer: Tokenizer) -> Tuple[List[str], List[str]]:
    """Ma'lumotni input va output formatiga o'girish"""
    # Input: [KONTEKST] [SAVOL]
    context = item['context']
    question = item['question']
    input_text = f"{context} [SAVOL] {question}"

    # Output: javob
    output_text = item['answer']

    input_tokens = tokenizer.tokenize(input_text)
    output_tokens = tokenizer.tokenize(output_text)

    return input_tokens, output_tokens


class LegalQADataset(Dataset):
    """Huquqiy savol-javob dataseti"""

    def __init__(
            self,
            data: List[Dict],
            tokenizer: Tokenizer,
            vocab: Vocabulary,
            max_input_len: int = 256,
            max_output_len: int = 64
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        item = self.data[idx]
        input_tokens, output_tokens = prepare_input_output(item, self.tokenizer)

        # Uzunlikni cheklash
        input_tokens = input_tokens[:self.max_input_len]
        output_tokens = output_tokens[:self.max_output_len - 1]  # EOS uchun joy

        # Indekslarga o'girish
        input_indices = self.vocab.encode(input_tokens)
        output_indices = [self.vocab.sos_idx] + self.vocab.encode(output_tokens) + [self.vocab.eos_idx]

        return (
            torch.tensor(input_indices, dtype=torch.long),
            torch.tensor(output_indices, dtype=torch.long),
            len(input_indices),
            len(output_indices)
        )


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch yaratish uchun collate funksiyasi"""
    inputs, outputs, input_lens, output_lens = zip(*batch)

    # Padding
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    outputs_padded = pad_sequence(outputs, batch_first=True, padding_value=0)

    input_lens = torch.tensor(input_lens, dtype=torch.long)
    output_lens = torch.tensor(output_lens, dtype=torch.long)

    return inputs_padded, outputs_padded, input_lens, output_lens


# ============================================
# 4. SEQ2SEQ MODEL ARXITEKTURASI
# ============================================

class Encoder(nn.Module):
    """RNN Encoder"""

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            rnn_type: str = "LSTM",
            bidirectional: bool = True,
            pad_idx: int = 0
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        rnn_class = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Bidirectional bo'lsa, hidden ni birlashtirish uchun
        if bidirectional:
            self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
            self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
            self,
            src: torch.Tensor,
            src_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            src: [batch_size, src_len]
            src_lens: [batch_size]
        Returns:
            outputs: [batch_size, src_len, hidden_dim * num_directions]
            hidden: tuple of hidden states
        """
        embedded = self.dropout(self.embedding(src))

        # Pack sequence
        packed = pack_padded_sequence(
            embedded,
            src_lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        outputs, hidden = self.rnn(packed)

        # Unpack
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # Bidirectional hidden state ni birlashtirish
        if self.bidirectional:
            if isinstance(hidden, tuple):  # LSTM
                hidden_cat = torch.cat([hidden[0][-2], hidden[0][-1]], dim=1)
                cell_cat = torch.cat([hidden[1][-2], hidden[1][-1]], dim=1)
                hidden = (
                    torch.tanh(self.fc_hidden(hidden_cat)).unsqueeze(0),
                    torch.tanh(self.fc_cell(cell_cat)).unsqueeze(0)
                )
            else:  # GRU
                hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
                hidden = torch.tanh(self.fc_hidden(hidden_cat)).unsqueeze(0)

        return outputs, hidden


class Attention(nn.Module):
    """Bahdanau Attention mexanizmi"""

    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int):
        super().__init__()

        self.attn = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(
            self,
            hidden: torch.Tensor,
            encoder_outputs: torch.Tensor,
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            hidden: [batch_size, decoder_hidden_dim]
            encoder_outputs: [batch_size, src_len, encoder_hidden_dim]
            mask: [batch_size, src_len]
        Returns:
            attention_weights: [batch_size, src_len]
        """
        src_len = encoder_outputs.shape[1]

        # hidden ni takrorlash
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Attention skorlarini hisoblash
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)

        # Mask qo'llash
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    """RNN Decoder with Attention"""

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            encoder_hidden_dim: int,
            decoder_hidden_dim: int,
            num_layers: int,
            dropout: float,
            rnn_type: str = "LSTM",
            use_attention: bool = True,
            pad_idx: int = 0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.use_attention = use_attention
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        rnn_input_dim = embedding_dim + encoder_hidden_dim if use_attention else embedding_dim

        rnn_class = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_class(
            rnn_input_dim,
            decoder_hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        if use_attention:
            self.attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
            self.fc_out = nn.Linear(encoder_hidden_dim + decoder_hidden_dim + embedding_dim, vocab_size)
        else:
            self.fc_out = nn.Linear(decoder_hidden_dim, vocab_size)

    def forward(
            self,
            input: torch.Tensor,
            hidden: Tuple[torch.Tensor, ...],
            encoder_outputs: torch.Tensor,
            mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Args:
            input: [batch_size, 1]
            hidden: decoder hidden state
            encoder_outputs: [batch_size, src_len, encoder_hidden_dim]
            mask: [batch_size, src_len]
        Returns:
            output: [batch_size, vocab_size]
            hidden: updated hidden state
            attention_weights: [batch_size, src_len]
        """
        embedded = self.dropout(self.embedding(input))

        attention_weights = None

        if self.use_attention:
            # Hidden dan oxirgi qatlamni olish
            if isinstance(hidden, tuple):
                attn_hidden = hidden[0][-1]
            else:
                attn_hidden = hidden[-1]

            # Attention
            attention_weights = self.attention(attn_hidden, encoder_outputs, mask)

            # Weighted context
            weighted = attention_weights.unsqueeze(1).bmm(encoder_outputs)

            # RNN inputi
            rnn_input = torch.cat([embedded, weighted], dim=2)
        else:
            rnn_input = embedded

        output, hidden = self.rnn(rnn_input, hidden)

        if self.use_attention:
            output = self.fc_out(torch.cat([output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)], dim=1))
        else:
            output = self.fc_out(output.squeeze(1))

        return output, hidden, attention_weights


class Seq2Seq(nn.Module):
    """Seq2Seq model"""

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """Padding mask yaratish"""
        return (src != pad_idx)

    def forward(
            self,
            src: torch.Tensor,
            src_lens: torch.Tensor,
            trg: torch.Tensor,
            teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Args:
            src: [batch_size, src_len]
            src_lens: [batch_size]
            trg: [batch_size, trg_len]
            teacher_forcing_ratio: teacher forcing ehtimoli
        Returns:
            outputs: [batch_size, trg_len, vocab_size]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.vocab_size

        # Chiqishlar uchun tensor
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)

        # Encoder
        encoder_outputs, hidden = self.encoder(src, src_lens)

        # Mask
        mask = self.create_mask(src, 0)

        # LSTM num_layers uchun hidden state ni moslash
        if isinstance(hidden, tuple):
            hidden = (
                hidden[0].repeat(self.decoder.rnn.num_layers, 1, 1),
                hidden[1].repeat(self.decoder.rnn.num_layers, 1, 1)
            )
        else:
            hidden = hidden.repeat(self.decoder.rnn.num_layers, 1, 1)

        # Decoder boshlang'ich inputi (SOS token)
        input = trg[:, 0:1]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:, t, :] = output

            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = trg[:, t:t + 1] if teacher_force else top1.unsqueeze(1)

        return outputs


# ============================================
# 5. TRAINING VA EVALUATION
# ============================================

def train_epoch(
        model: Seq2Seq,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        clip: float,
        device: torch.device,
        teacher_forcing_ratio: float = 0.5
) -> float:
    """Bir epoch training"""
    model.train()
    total_loss = 0

    for batch_idx, (src, trg, src_lens, trg_lens) in enumerate(dataloader):
        src = src.to(device)
        trg = trg.to(device)
        src_lens = src_lens.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(src, src_lens, trg, teacher_forcing_ratio)

        # Loss hisoblash (SOS tokendan tashqari)
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        total_loss += loss.item()

        # Progress
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(
        model: Seq2Seq,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> float:
    """Model baholash"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, trg, src_lens, trg_lens in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            src_lens = src_lens.to(device)

            # Forward pass (teacher forcing yo'q)
            output = model(src, src_lens, trg, teacher_forcing_ratio=0)

            # Loss
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def generate_answer(
        model: Seq2Seq,
        context: str,
        question: str,
        tokenizer: Tokenizer,
        vocab: Vocabulary,
        device: torch.device,
        max_len: int = 64
) -> str:
    """Model yordamida javob generatsiya qilish"""
    model.eval()

    # Input tayyorlash
    input_text = f"{context} [SAVOL] {question}"
    input_tokens = tokenizer.tokenize(input_text)[:Config.MAX_INPUT_LEN]
    input_indices = vocab.encode(input_tokens)

    src = torch.tensor([input_indices], dtype=torch.long).to(device)
    src_lens = torch.tensor([len(input_indices)], dtype=torch.long).to(device)

    with torch.no_grad():
        # Encode
        encoder_outputs, hidden = model.encoder(src, src_lens)
        mask = model.create_mask(src, vocab.pad_idx)

        # Hidden state ni moslash
        if isinstance(hidden, tuple):
            hidden = (
                hidden[0].repeat(model.decoder.rnn.num_layers, 1, 1),
                hidden[1].repeat(model.decoder.rnn.num_layers, 1, 1)
            )
        else:
            hidden = hidden.repeat(model.decoder.rnn.num_layers, 1, 1)

        # Decode
        input_token = torch.tensor([[vocab.sos_idx]], dtype=torch.long).to(device)
        output_indices = []

        for _ in range(max_len):
            output, hidden, _ = model.decoder(input_token, hidden, encoder_outputs, mask)
            top1 = output.argmax(1).item()

            if top1 == vocab.eos_idx:
                break

            output_indices.append(top1)
            input_token = torch.tensor([[top1]], dtype=torch.long).to(device)

    # Tokenlarni matnga o'girish
    output_tokens = vocab.decode(output_indices)
    return tokenizer.detokenize(output_tokens)


# ============================================
# 6. ASOSIY FUNKSIYA
# ============================================

def main():
    """Asosiy training funksiyasi"""
    print("=" * 60)
    print("HUQUQIY SAVOL-JAVOB RNN MODELI")
    print("=" * 60)

    # Konfiguratsiya
    print(f"\nQurilma: {Config.DEVICE}")
    print(f"RNN turi: {Config.RNN_TYPE}")
    print(f"Attention: {Config.ATTENTION}")

    # ---- 1. Ma'lumotlarni yuklash ----
    print("\n[1/6] Ma'lumotlar yuklanmoqda...")
    data = load_data(Config.DATA_PATH)
    print(f"Jami ma'lumotlar: {len(data)}")

    # Shuffle va split
    random.seed(42)
    random.shuffle(data)

    split_idx = int(len(data) * Config.TRAIN_SPLIT)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Training: {len(train_data)}, Validation: {len(val_data)}")

    # ---- 2. Tokenizer va Vocabulary ----
    print("\n[2/6] Tokenizer va Vocabulary yaratilmoqda...")
    tokenizer = Tokenizer()

    # Barcha matnlarni tokenizatsiya qilish
    all_tokens = []
    for item in data:
        input_tokens, output_tokens = prepare_input_output(item, tokenizer)
        all_tokens.append(input_tokens)
        all_tokens.append(output_tokens)

    vocab = Vocabulary(min_freq=Config.MIN_FREQ, max_size=Config.MAX_VOCAB_SIZE)
    vocab.build(all_tokens)

    # ---- 3. Dataset va DataLoader ----
    print("\n[3/6] Dataset va DataLoader yaratilmoqda...")
    train_dataset = LegalQADataset(
        train_data, tokenizer, vocab,
        max_input_len=Config.MAX_INPUT_LEN,
        max_output_len=Config.MAX_OUTPUT_LEN
    )
    val_dataset = LegalQADataset(
        val_data, tokenizer, vocab,
        max_input_len=Config.MAX_INPUT_LEN,
        max_output_len=Config.MAX_OUTPUT_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # ---- 4. Model yaratish ----
    print("\n[4/6] Model yaratilmoqda...")

    encoder_hidden_dim = Config.HIDDEN_DIM * (2 if Config.BIDIRECTIONAL_ENCODER else 1)

    encoder = Encoder(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        rnn_type=Config.RNN_TYPE,
        bidirectional=Config.BIDIRECTIONAL_ENCODER,
        pad_idx=vocab.pad_idx
    )

    decoder = Decoder(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        rnn_type=Config.RNN_TYPE,
        use_attention=Config.ATTENTION,
        pad_idx=vocab.pad_idx
    )

    model = Seq2Seq(encoder, decoder, Config.DEVICE).to(Config.DEVICE)

    # Model statistikasi
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Jami parametrlar: {total_params:,}")
    print(f"O'rganiladigan parametrlar: {trainable_params:,}")

    # ---- 5. Optimizer va Loss ----
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # ---- 6. Training ----
    print("\n[5/6] Training boshlanmoqda...")
    print("-" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")

        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            Config.CLIP_GRAD, Config.DEVICE, Config.TEACHER_FORCING_RATIO
        )

        # Validation
        val_loss = evaluate(model, val_loader, criterion, Config.DEVICE)

        # Scheduler
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.2f}")

        # Eng yaxshi modelni saqlash
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab,
                'config': Config,
            }, 'best_model.pt')
            print(">>> Eng yaxshi model saqlandi!")

    # ---- 7. Test ----
    print("\n[6/6] Model test qilinmoqda...")
    print("-" * 60)

    # Test namunalari
    test_samples = [
        {
            "context": "Konstitutsiya davlat va jamiyat hayotida eng oliy yuridik kuchga ega.",
            "question": "Konstitutsiya nima?"
        },
        {
            "context": "Mehnat huquqi bo'yicha xodim dam olish va ta'til olish huquqiga ega.",
            "question": "Xodimning asosiy huquqlari qanday?"
        },
        {
            "context": "Mulk egasi egalik qilish, foydalanish va tasarruf etish huquqlariga ega.",
            "question": "Mulk huquqi nimalarni o'z ichiga oladi?"
        }
    ]

    print("\nTest natijalari:")
    for i, sample in enumerate(test_samples, 1):
        answer = generate_answer(
            model, sample["context"], sample["question"],
            tokenizer, vocab, Config.DEVICE
        )
        print(f"\n--- Test #{i} ---")
        print(f"Kontekst: {sample['context'][:100]}...")
        print(f"Savol: {sample['question']}")
        print(f"Javob: {answer}")

    print("\n" + "=" * 60)
    print("TRAINING YAKUNLANDI!")
    print("=" * 60)
    print(f"Eng yaxshi validation loss: {best_val_loss:.4f}")
    print(f"Model saqlandi: best_model.pt")


if __name__ == "__main__":
    main()