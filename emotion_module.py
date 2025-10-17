import warnings
from typing import Optional, Tuple

import numpy as np
import torch

try:
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        AutoModelForAudioClassification,
        Wav2Vec2FeatureExtractor,
    )
except Exception:  # transformers not available
    AutoImageProcessor = None
    AutoModelForImageClassification = None
    AutoModelForAudioClassification = None
    Wav2Vec2FeatureExtractor = None


class EmotionModels:
    """
    Lightweight wrapper that loads pretrained emotion models for
    - visual (image-based) emotion classification
    - audio (wav-based) emotion classification

    It runs with local cache only by default to avoid network.
    """

    def __init__(
        self,
        visual_model_name: str,
        audio_model_name: str,
        device: torch.device,
        local_files_only: bool = True,
    ) -> None:
        self.device = device
        self.ok = False

        if any(m is None for m in [AutoImageProcessor, AutoModelForImageClassification, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor]):
            warnings.warn("transformers is not available; disabling emotion consistency.")
            return

        try:
            self.visual_processor = AutoImageProcessor.from_pretrained(
                visual_model_name, local_files_only=local_files_only
            )
            self.visual_model = AutoModelForImageClassification.from_pretrained(
                visual_model_name, local_files_only=local_files_only
            ).to(device)
            self.visual_model.eval()

            self.audio_feature = Wav2Vec2FeatureExtractor.from_pretrained(
                audio_model_name, local_files_only=local_files_only
            )
            self.audio_model = AutoModelForAudioClassification.from_pretrained(
                audio_model_name, local_files_only=local_files_only
            ).to(device)
            self.audio_model.eval()
            self.ok = True
        except Exception as e:
            warnings.warn(
                f"Failed to load emotion models ({e}); emotion consistency will be disabled."
            )
            self.ok = False

    @torch.no_grad()
    def logits_from_image(self, image_rgb: np.ndarray) -> Optional[torch.Tensor]:
        if not self.ok:
            return None
        # image_rgb: HxWxC in [0,255]
        inputs = self.visual_processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.visual_model(**inputs)
        return out.logits  # [1, C]

    @torch.no_grad()
    def logits_from_audio(self, audio_wave: np.ndarray, sr: int = 16000) -> Optional[torch.Tensor]:
        if not self.ok:
            return None
        inputs = self.audio_feature(audio_wave, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.audio_model(**inputs)
        return out.logits  # [1, C]


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) on last dim; expects unnormalized logits.
    Returns scalar tensor.
    """
    p = torch.softmax(p_logits.float(), dim=-1)
    log_q = torch.log_softmax(q_logits.float(), dim=-1)
    kl = torch.sum(p * (torch.log(torch.clamp(p, min=1e-12)) - log_q), dim=-1)
    return kl.mean()


def compute_emotion_inconsistency(
    emo: EmotionModels,
    frame_rgb: np.ndarray,
    audio_wave: np.ndarray,
    sr: int = 16000,
) -> Optional[float]:
    """
    Compute symmetric KL divergence between visual and audio emotion distributions.
    Returns float or None if models unavailable.
    """
    if emo is None or not emo.ok:
        return None
    v_logits = emo.logits_from_image(frame_rgb)
    a_logits = emo.logits_from_audio(audio_wave, sr=sr)
    if v_logits is None or a_logits is None:
        return None
    # Align to common space if label dims mismatch: project to min dim by slicing
    if v_logits.shape[-1] != a_logits.shape[-1]:
        C = min(int(v_logits.shape[-1]), int(a_logits.shape[-1]))
        v_logits = v_logits[..., :C]
        a_logits = a_logits[..., :C]
    kl_va = kl_divergence(v_logits, a_logits)
    kl_av = kl_divergence(a_logits, v_logits)
    return float(0.5 * (kl_va + kl_av).item())

