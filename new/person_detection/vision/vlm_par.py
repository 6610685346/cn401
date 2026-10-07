"""Inference wrapper for the bundled VLM-PAR pedestrian checkpoint.

The checkpoint contains a SigLIP 2 vision encoder and five trained classifier
heads. It does not contain the paper's described cross-attention parameters, so
this adapter uses the trained image encoder and classifier heads present in
the file.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from ..core.geometry import Box, Frame, crop_box

# The checkpoint has eleven color classes, but it does not include their names.
# This provisional index mapping is inferred from synthetic color probes, not
# verified against the original training labels. In particular, class 2
# responded mostly to muted brown/tan, while class 5 responded mostly to
# brighter coral/terracotta. Class 10 also responds to yellow in the probes.
DEFAULT_COLOR_LABELS = (
    "black",   # class_0
    "blue",    # class_1
    "brown",   # class_2: muted brown/tan shades
    "gray",    # class_3
    "green",   # class_4
    "coral",   # class_5: coral/terracotta shades
    "pink",    # class_6
    "purple",  # class_7
    "red",     # class_8
    "white",   # class_9
    "orange",  # class_10: orange and yellow both triggered this class
)


@dataclass(frozen=True)
class PedestrianAttributes:
    upper_color: str
    lower_color: str
    bag: bool
    hat: bool
    confidence: float

    def __str__(self) -> str:
        extras = ", ".join(
            name for name, present in (("bag", self.bag), ("hat", self.hat)) if present
        )
        clothing = f"top: {self.upper_color}, bottom: {self.lower_color}"
        return f"{clothing}" + (f", {extras}" if extras else "")


class VLMParReader:
    """Predict clothing colors and accessory presence from a person crop."""

    INPUT_SIZE = 224
    MIN_BOX_HEIGHT_PX = 48

    def __init__(self, checkpoint: str | Path, color_labels=DEFAULT_COLOR_LABELS):
        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
        except ImportError as exc:
            raise RuntimeError(
                "VLM-PAR needs Hugging Face Transformers. Install project "
                "dependencies with: pip install -r requirements.txt"
            ) from exc

        self.color_labels = tuple(color_labels)
        if len(self.color_labels) != 11:
            raise ValueError("VLM-PAR requires exactly 11 color labels")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # SigLIP 2 base patch16/224 dimensions, matching the bundled weights.
        config = SiglipVisionConfig(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=224,
            patch_size=16,
        )
        self.vision = SiglipVisionModel(config)
        self.heads = nn.ModuleDict(
            {
                "upper_body_color": nn.Linear(768, 11),
                "lower_body_color": nn.Linear(768, 11),
                "gender": nn.Linear(768, 2),
                "bag": nn.Linear(768, 2),
                "hat": nn.Linear(768, 2),
            }
        )

        # weights_only avoids executing checkpoint pickle objects.
        checkpoint_data = torch.load(
            checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint_data.get("model_state_dict", checkpoint_data)
        vision_state = {
            key.removeprefix("module.vlm_base.vision_model."): value
            for key, value in state.items()
            if key.startswith("module.vlm_base.vision_model.")
        }
        self.vision.load_state_dict(vision_state, strict=True)
        for name, head in self.heads.items():
            prefix = f"module.classifier_heads.{name}."
            head.load_state_dict(
                {
                    key[len(prefix):]: value
                    for key, value in state.items()
                    if key.startswith(prefix)
                },
                strict=True,
            )
        self.vision.to(self.device).eval()
        self.heads.to(self.device).eval()

    @torch.inference_mode()
    def read(self, frame: Frame, box: Box) -> PedestrianAttributes | None:
        x1, y1, x2, y2 = box
        if y2 - y1 < self.MIN_BOX_HEIGHT_PX:
            return None
        crop = crop_box(frame, box)
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_CUBIC
        )
        pixels = (
            torch.from_numpy(np.ascontiguousarray(
                rgb)).permute(2, 0, 1).float() / 255.0
        )
        pixels = (pixels - 0.5) / 0.5
        pixels = pixels.unsqueeze(0).to(self.device)
        # SigLIP2's vision head performs the learned attention pooling.
        embedding = self.vision(pixel_values=pixels).pooler_output
        probabilities = {
            name: head(embedding).softmax(dim=-1)[0]
            for name, head in self.heads.items()
        }
        upper_idx = int(probabilities["upper_body_color"].argmax())
        lower_idx = int(probabilities["lower_body_color"].argmax())
        return PedestrianAttributes(
            upper_color=self.color_labels[upper_idx],
            lower_color=self.color_labels[lower_idx],
            bag=bool(probabilities["bag"].argmax().item() == 1),
            hat=bool(probabilities["hat"].argmax().item() == 1),
            confidence=float(
                torch.stack([p.max()
                            for p in probabilities.values()]).mean().item()
            ),
        )
