"""SAM segmentation for the rig: one-shot text prompt + per-camera mask tracking.

SAM 3.1 turns a text prompt ("sponge") into an initial mask on one frame; the
real-time SAM 2 fork then tracks that mask frame-to-frame at camera rate — one
tracker instance per camera, each with its own memory bank. The policy never
sees a mask: downstream code reduces each view's mask to a centroid, and the
pair of centroids triangulates in the base frame (real.stereo), replacing the
demo script's monocular depth model.

The model checkouts and checkpoints live outside this repo (both envs share
the same editable installs); the paths are pinned here as the single source.
Everything runs CUDA + bfloat16, matching how the checkpoints were validated.
"""
import cv2
import numpy as np
import torch
from PIL import Image

SAM2_DIR = "/home/mikhail/robot_arm/camera_stuff/RGBTrack/segment-anything-2-real-time"
# ~18 ms/frame (tiny) vs ~24 ms (base+) per 720p view on the rig's RTX 4090.
SAM2_MODELS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml",
             f"{SAM2_DIR}/checkpoints/sam2.1_hiera_tiny.pt"),
    "base+": ("configs/sam2.1/sam2.1_hiera_b+.yaml",
              f"{SAM2_DIR}/checkpoints/sam2.1_hiera_base_plus.pt"),
}


def load_sam3():
    """Build the SAM 3.1 image model + processor for one-shot text prompting.

    Import is deferred so that merely importing this module (e.g. the panel
    registry check) doesn't pull the full SAM3 stack.
    """
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    ckpt = download_ckpt_from_hf(version="sam3.1")
    return Sam3Processor(build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False))


def text_to_mask(sam3_processor, frame_bgr, prompt):
    """Highest-scoring mask for `prompt`: (HxW bool at frame resolution, score).

    Raises when nothing matches — bad framing must fail loud at startup, not
    silently track an empty mask.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam3_processor.set_image(Image.fromarray(rgb))
        out = sam3_processor.set_text_prompt(prompt=prompt, state=state)
    masks, scores = out["masks"], out["scores"]
    if len(masks) == 0:
        raise RuntimeError(f"SAM3 found no match for {prompt!r}")
    top = int(scores.argmax())
    return masks[top].squeeze(0).cpu().numpy().astype(bool), float(scores[top])


class MaskTracker:
    """Real-time SAM 2 mask tracker bound to one camera stream."""

    def __init__(self, model):
        from sam2.build_sam import build_sam2_camera_predictor
        cfg, ckpt = SAM2_MODELS[model]
        self._pred = build_sam2_camera_predictor(cfg, ckpt)

    def prime(self, frame_bgr, mask):
        """Seed the tracker's memory with an initial mask (from text_to_mask)."""
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._pred.load_first_frame(frame_bgr)
            self._pred.add_new_mask(frame_idx=0, obj_id=1,
                                    mask=torch.from_numpy(mask))

    def track(self, frame_bgr):
        """Propagate to the next frame: HxW bool mask at frame resolution.

        An all-False mask means the tracker lost the object this frame — the
        caller's held-pose/age convention deals with it, same as a hidden tag.
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, logits = self._pred.track(frame_bgr)
        return logits[0, 0].cpu().numpy() > 0


def mask_centroid(mask):
    """(u, v) pixel centroid of a bool mask, or None for an empty mask."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())
