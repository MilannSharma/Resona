"""
Resona Neural TTS Generator Model
Ultra-lightweight 82M parameter neural speech synthesizer with AdaIN style conditioning.
100% Offline & Self-Contained.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
from transformers import AlbertConfig

from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder


class ResonaModel(nn.Module):
    """
    ResonaModel is a self-contained PyTorch neural vocoder and style-guided prosody generator.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cpu"
    ):
        super().__init__()

        # Find default config if not provided
        if not config_path:
            cand_config = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "config.json"))
            if os.path.isfile(cand_config):
                config_path = cand_config

        if not config_path or not os.path.isfile(config_path):
            raise FileNotFoundError(f"Resona config.json not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.vocab = config["vocab"]
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config["n_token"], **config["plbert"]))
        self.bert_encoder = nn.Linear(self.bert.config.hidden_size, config["hidden_dim"])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config["style_dim"],
            d_hid=config["hidden_dim"],
            nlayers=config["n_layer"],
            max_dur=config["max_dur"],
            dropout=config["dropout"],
        )
        self.text_encoder = TextEncoder(
            channels=config["hidden_dim"],
            kernel_size=config["text_encoder_kernel_size"],
            depth=config["n_layer"],
            n_symbols=config["n_token"],
        )
        self.decoder = Decoder(
            dim_in=config["hidden_dim"],
            style_dim=config["style_dim"],
            dim_out=config["n_mels"],
            disable_complex=False,
            **config["istftnet"],
        )

        # Default model weights selection (Indic fine-tuned preferred)
        if not model_path:
            cand_indic = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "resona-indic-v1.pth"))
            cand_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "resona-v1.pth"))
            if os.path.isfile(cand_indic):
                model_path = cand_indic
            elif os.path.isfile(cand_base):
                model_path = cand_base

        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Resona model weights (.pth) not found at {model_path}")

        # Load weights
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        for key, state_dict in state.items():
            if hasattr(self, key):
                try:
                    getattr(self, key).load_state_dict(state_dict)
                except Exception:
                    clean_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    getattr(self, key).load_state_dict(clean_dict, strict=False)

        self.to(device)
        self.eval()

    @property
    def device(self):
        return next(self.parameters()).device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long,
        )

        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(input_ids.device)

        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration.squeeze()).clamp(min=1).long()

        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    @torch.no_grad()
    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
    ) -> Output:
        tokens = [self.vocab[p] for p in phonemes if p in self.vocab]
        if not tokens:
            return self.Output(audio=torch.zeros(0, device=self.device))
        
        # Add start/end tokens
        tokens = [0] + tokens + [0]
        input_ids = torch.LongTensor([tokens]).to(self.device)
        ref_s = ref_s.to(self.device)

        # Style tensor resolution:
        # If full voice pack [510, 1, 256] or [510, 256] passed, index by phoneme length
        if ref_s.dim() >= 2 and ref_s.shape[0] > 1:
            idx = min(len(phonemes) - 1, ref_s.shape[0] - 1)
            ref_s = ref_s[idx]

        # Ensure shape [1, 256]
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)
        elif ref_s.dim() == 2 and ref_s.shape[0] != 1 and ref_s.shape[1] == 1:
            ref_s = ref_s.squeeze(1).unsqueeze(0)
        elif ref_s.dim() == 3:
            ref_s = ref_s.view(1, -1)

        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        return self.Output(audio=audio, pred_dur=pred_dur)
