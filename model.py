"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import copy
import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gdown
except ImportError:
    gdown = None


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(torch.bool)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    attn_w = F.softmax(scores, dim=-1)
    if mask is not None:
        attn_w = attn_w.masked_fill(mask, 0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    _, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt.device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(1)
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(query))
        K = split_heads(self.W_k(key))
        V = split_heads(self.W_v(value))

        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = self.dropout(attn_output)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :].to(dtype=x.dtype)
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class _SimpleVocab:
    def __init__(self, itos: list[str]) -> None:
        self.itos = list(itos)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi.get("<unk>", 0))

    def lookup_token(self, idx: int) -> str:
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return "<unk>"


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    NOTE FOR AUTOGRADER INFERENCE MODE:
      - Can be instantiated with zero args: Transformer()
      - In that mode, __init__ loads tokenizers, vocabs and trained weights.
    """

    # Replace these with your Google Drive IDs before submission if needed.
    CHECKPOINT_DRIVE_ID = "1yXdDtR0CbEFymVo34n2gqoxHL4dGC9kz"
    VOCAB_DRIVE_ID = "1ide41CB5aPZwjJkm9kZTRC_cilwcdHDl"

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        max_infer_len: int = 100,
    ) -> None:
        super().__init__()

        self.max_infer_len = max_infer_len

        # Defaults for artifact paths; autograder can just call Transformer().
        self.checkpoint_path = checkpoint_path or "checkpoint.pt"
        self.vocab_path = vocab_path or "vocab.pt"

        # In inference mode (Transformer() with no explicit vocab sizes),
        # load everything (tokenizer/vocab/weights) inside __init__.
        auto_infer_mode = src_vocab_size is None or tgt_vocab_size is None
        checkpoint_bundle = None

        if auto_infer_mode:
            self._load_tokenizers()

            # Load checkpoint first (required by assignment), then vocab either
            # from vocab artifact or from checkpoint payload fallback.
            self._ensure_artifact(
                self.checkpoint_path,
                os.getenv("TRANSFORMER_CHECKPOINT_DRIVE_ID", self.CHECKPOINT_DRIVE_ID),
                "checkpoint",
            )
            checkpoint_bundle = torch.load(self.checkpoint_path, map_location="cpu")

            if os.path.exists(self.vocab_path):
                self.src_vocab, self.tgt_vocab = self._load_vocab_pair(self.vocab_path)
            else:
                vocab_drive_id = os.getenv("TRANSFORMER_VOCAB_DRIVE_ID", self.VOCAB_DRIVE_ID)
                if vocab_drive_id and gdown is not None:
                    self._ensure_artifact(self.vocab_path, vocab_drive_id, "vocab")
                    self.src_vocab, self.tgt_vocab = self._load_vocab_pair(self.vocab_path)
                elif isinstance(checkpoint_bundle, dict) and "src_itos" in checkpoint_bundle and "tgt_itos" in checkpoint_bundle:
                    self.src_vocab = _SimpleVocab([str(x) for x in checkpoint_bundle["src_itos"]])
                    self.tgt_vocab = _SimpleVocab([str(x) for x in checkpoint_bundle["tgt_itos"]])
                else:
                    raise FileNotFoundError(
                        "Could not load vocab for infer-mode. Provide vocab.pt (or drive id), "
                        "or save src_itos/tgt_itos in checkpoint payload."
                    )

            src_vocab_size = len(self.src_vocab)
            tgt_vocab_size = len(self.tgt_vocab)

            self.src_pad_idx = self._token_id(self.src_vocab, "<pad>", 1)
            self.tgt_pad_idx = self._token_id(self.tgt_vocab, "<pad>", 1)
            self.src_sos_idx = self._token_id(self.src_vocab, "<sos>", 2)
            self.src_eos_idx = self._token_id(self.src_vocab, "<eos>", 3)
            self.tgt_sos_idx = self._token_id(self.tgt_vocab, "<sos>", 2)
            self.tgt_eos_idx = self._token_id(self.tgt_vocab, "<eos>", 3)

            model_cfg = checkpoint_bundle.get("model_config", {}) if isinstance(checkpoint_bundle, dict) else {}

            d_model = int(model_cfg.get("d_model", d_model))
            N = int(model_cfg.get("N", N))
            num_heads = int(model_cfg.get("num_heads", num_heads))
            d_ff = int(model_cfg.get("d_ff", d_ff))
            dropout = float(model_cfg.get("dropout", dropout))

        self.src_vocab_size = int(src_vocab_size)
        self.tgt_vocab_size = int(tgt_vocab_size)
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout

        self.src_embed = nn.Embedding(self.src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(self.tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout=dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout=dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.output_projection = nn.Linear(d_model, self.tgt_vocab_size)

        self.model_config = {
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "d_model": d_model,
            "N": N,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "dropout": dropout,
        }

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Load model weights in __init__ per assignment announcement.
        if checkpoint_bundle is not None:
            self._load_weights_from_bundle(checkpoint_bundle)
        elif checkpoint_path is not None and os.path.exists(self.checkpoint_path):
            self._load_weights_from_bundle(torch.load(self.checkpoint_path, map_location="cpu"))

    # ── Helper methods used by __init__ / infer ───────────────────────

    def _load_tokenizers(self) -> None:
        import spacy

        try:
            self.src_tokenizer = spacy.load("de_core_news_sm")
        except Exception:
            self.src_tokenizer = spacy.blank("de")

        try:
            self.tgt_tokenizer = spacy.load("en_core_web_sm")
        except Exception:
            self.tgt_tokenizer = spacy.blank("en")

    def _ensure_artifact(self, path: str, drive_id: str, artifact_name: str) -> None:
        if os.path.exists(path):
            return

        if drive_id and gdown is not None:
            gdown.download(id=drive_id, output=path, quiet=False)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{artifact_name} file not found at '{path}'. "
                f"Set {artifact_name} path or provide Google Drive id in "
                f"TRANSFORMER_{artifact_name.upper()}_DRIVE_ID env var / class constant."
            )

    def _extract_itos(self, obj: Any) -> list[str]:
        if isinstance(obj, _SimpleVocab):
            return list(obj.itos)
        if hasattr(obj, "itos"):
            return list(obj.itos)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            if "itos" in obj and isinstance(obj["itos"], list):
                return [str(x) for x in obj["itos"]]
            if "stoi" in obj and isinstance(obj["stoi"], dict):
                stoi = obj["stoi"]
                size = max(stoi.values()) + 1
                itos = ["<unk>"] * size
                for tok, idx in stoi.items():
                    if 0 <= idx < size:
                        itos[idx] = str(tok)
                return itos
            if obj and all(isinstance(k, str) and isinstance(v, int) for k, v in obj.items()):
                size = max(obj.values()) + 1
                itos = ["<unk>"] * size
                for tok, idx in obj.items():
                    if 0 <= idx < size:
                        itos[idx] = str(tok)
                return itos
        raise ValueError("Unsupported vocab serialization format")

    def _load_vocab_pair(self, path: str) -> Tuple[_SimpleVocab, _SimpleVocab]:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".json", ".js"}:
            blob = json.load(open(path, "r", encoding="utf-8"))
        else:
            blob = torch.load(path, map_location="cpu")

        src_candidate = None
        tgt_candidate = None

        if isinstance(blob, dict):
            for key in ("src_vocab", "src", "de_vocab", "src_itos", "de_itos"):
                if key in blob:
                    src_candidate = blob[key]
                    break
            for key in ("tgt_vocab", "tgt", "en_vocab", "tgt_itos", "en_itos"):
                if key in blob:
                    tgt_candidate = blob[key]
                    break

        if src_candidate is None or tgt_candidate is None:
            raise ValueError(
                "Could not find source/target vocab in vocab file. "
                "Expected keys like src_vocab/tgt_vocab or src_itos/tgt_itos."
            )

        src_itos = self._extract_itos(src_candidate)
        tgt_itos = self._extract_itos(tgt_candidate)
        return _SimpleVocab(src_itos), _SimpleVocab(tgt_itos)

    def _token_id(self, vocab: Any, token: str, default: int) -> int:
        if hasattr(vocab, "stoi"):
            return int(vocab.stoi.get(token, vocab.stoi.get("<unk>", default)))
        if isinstance(vocab, dict):
            return int(vocab.get(token, vocab.get("<unk>", default)))
        try:
            return int(vocab[token])
        except Exception:
            return int(default)

    def _id_to_token(self, vocab: Any, idx: int) -> str:
        if hasattr(vocab, "itos"):
            if 0 <= idx < len(vocab.itos):
                return str(vocab.itos[idx])
            return "<unk>"
        if hasattr(vocab, "lookup_token"):
            return str(vocab.lookup_token(idx))
        if isinstance(vocab, dict):
            inv = {v: k for k, v in vocab.items()}
            return str(inv.get(idx, "<unk>"))
        return "<unk>"

    def _load_weights_from_bundle(self, bundle: Any) -> None:
        if isinstance(bundle, dict) and "model_state_dict" in bundle:
            self.load_state_dict(bundle["model_state_dict"])
        elif isinstance(bundle, dict):
            self.load_state_dict(bundle)
        else:
            raise ValueError("Unsupported checkpoint format")

    # ── AUTOGRADER HOOKS ──────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        dec_out = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return self.output_projection(dec_out)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        End-to-end DE->EN inference as required by assignment announcement:
          raw sentence -> tokenize -> ids -> autoregressive decode -> detokenized string
        """
        required = [
            "src_tokenizer",
            "tgt_tokenizer",
            "src_vocab",
            "tgt_vocab",
            "src_pad_idx",
            "tgt_pad_idx",
            "src_sos_idx",
            "src_eos_idx",
            "tgt_sos_idx",
            "tgt_eos_idx",
        ]
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                "infer() requires tokenizer/vocab/assets to be loaded in __init__. "
                f"Missing: {missing}"
            )

        self.eval()
        device = next(self.parameters()).device

        src_tokens = [tok.text.lower() for tok in self.src_tokenizer(src_sentence)]
        src_ids = [
            self.src_sos_idx,
            *[self._token_id(self.src_vocab, tok, 0) for tok in src_tokens],
            self.src_eos_idx,
        ]

        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, pad_idx=self.src_pad_idx)

        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys = torch.tensor([[self.tgt_sos_idx]], dtype=torch.long, device=device)

            for _ in range(self.max_infer_len - 1):
                tgt_mask = make_tgt_mask(ys, pad_idx=self.tgt_pad_idx)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_word = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                ys = torch.cat(
                    [ys, torch.tensor([[next_word]], dtype=torch.long, device=device)],
                    dim=1,
                )
                if next_word == self.tgt_eos_idx:
                    break

        out_tokens = []
        for idx in ys.squeeze(0).tolist():
            if idx in (self.tgt_sos_idx, self.tgt_pad_idx):
                continue
            if idx == self.tgt_eos_idx:
                break
            tok = self._id_to_token(self.tgt_vocab, idx)
            if tok not in {"<unk>", "<pad>", "<sos>", "<eos>"}:
                out_tokens.append(tok)

        text = " ".join(out_tokens).strip()

        # Minimal detokenization cleanup.
        for punct in [".", ",", "!", "?", ":", ";"]:
            text = text.replace(f" {punct}", punct)
        text = text.replace(" n't", "n't")
        text = text.replace(" 'm", "'m").replace(" 's", "'s").replace(" 're", "'re")

        return text
