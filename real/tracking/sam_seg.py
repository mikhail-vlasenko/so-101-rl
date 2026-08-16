"""SAM segmentation for the rig: one-shot text prompt + per-camera mask tracking.

SAM 3 turns a text prompt ("sponge") into an initial mask on one frame; SAM 2
video sessions in *streaming* mode then track that mask frame-to-frame at
camera rate — one session per camera, each with its own memory bank. The
policy never sees a mask: downstream code reduces each view's mask to a
centroid, and the pair of centroids triangulates in the base frame
(real.vision.stereo).

Both models come from `transformers`, weights from the HF hub cache — no
local checkouts or forks; HF's Sam2Video provides the streaming inference the
official Meta repo lacks. SAM 2 runs bfloat16 (~15 ms per 720p frame for tiny
on the rig's RTX 4090 — fp32 is 3.6x slower); SAM 3 stays fp32, it runs once.
`facebook/sam3` is the hub's transformers-format checkpoint (sam3.1 exists
only in the original format); measured mask parity on rig frames: centroid
within 0.2 px of sam3.1.
"""
import cv2
import numpy as np
import torch

SAM3_ID = "facebook/sam3"
SAM2_MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "base+": "facebook/sam2.1-hiera-base-plus",
}

# A streaming session keeps every frame's memory features, on the inference
# device, for as long as it lives — 3600 frames of that exhausted a 24 GB GPU
# while masking the shape dataset. SAM 2 only reads the most recent few plus
# the conditioning frame, so the rest is dead weight. Roll the session at this
# many frames, re-seeding it from the mask just produced so tracking carries
# on unbroken. Bounds a rollout of any length, not just the offline pass.
SESSION_MAX_FRAMES = 200


class SAMPromptNoMatchError(RuntimeError):
    """The requested object was absent from a SAM text-prompt frame."""


def load_sam3():
    """Load the SAM 3 model + processor for one-shot text prompting.

    Imports are deferred so importing this module stays cheap; transformers
    alone costs seconds.
    """
    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained(SAM3_ID).to("cuda").eval()
    processor = Sam3Processor.from_pretrained(SAM3_ID)
    return model, processor


def find_text_mask(sam3, frame_bgr, prompt):
    """Highest-scoring mask for `prompt`: (HxW bool at frame resolution, score),
    or None when the frame contains no match.

    None is a real answer offline, where a recording may run on after the
    object left the view; callers that need the object present should use
    `text_to_mask`.
    """
    model, processor = sam3
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, text=prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist())[0]
    scores = results["scores"]
    if len(scores) == 0:
        return None
    top = int(scores.argmax())
    return results["masks"][top].cpu().numpy().astype(bool), float(scores[top])


def text_to_mask(sam3, frame_bgr, prompt):
    """`find_text_mask`, but refusing to proceed without a match — bad framing
    must fail loud at rollout startup, not silently track an empty mask."""
    found = find_text_mask(sam3, frame_bgr, prompt)
    if found is None:
        raise SAMPromptNoMatchError(f"SAM3 found no match for {prompt!r}")
    return found


class MaskTracker:
    """Streaming SAM 2 mask tracker bound to one camera stream."""

    def __init__(self, model, session_max_frames=SESSION_MAX_FRAMES):
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        repo = SAM2_MODELS[model]
        self._model = Sam2VideoModel.from_pretrained(
            repo, dtype=torch.bfloat16).to("cuda").eval()
        self._processor = Sam2VideoProcessor.from_pretrained(repo)
        self._session = None
        self._session_max_frames = session_max_frames
        self._frames = 0

    def prime(self, frame_bgr, mask):
        """Seed a fresh streaming session with an initial mask (from
        text_to_mask) and condition it on this frame. Also how a session is
        rolled over — cheap, since the model itself is not reloaded."""
        self._session = self._processor.init_video_session(
            inference_device="cuda", dtype=torch.bfloat16)
        self._frames = 0
        inputs = self._preprocess(frame_bgr)
        self._processor.add_inputs_to_inference_session(
            inference_session=self._session, frame_idx=0, obj_ids=1,
            input_masks=mask, original_size=inputs.original_sizes[0])
        self._forward(inputs)

    def track(self, frame_bgr):
        """Propagate to the next frame: HxW bool mask at frame resolution.

        An all-False mask means the tracker lost the object this frame — the
        caller's held-pose/age convention deals with it, same as a hidden tag.
        """
        mask = self._forward(self._preprocess(frame_bgr))
        self._frames += 1
        # Roll only on a good mask: an empty one cannot re-seed a session.
        if self._frames >= self._session_max_frames and mask.any():
            self.prime(frame_bgr, mask)
        return mask

    def _preprocess(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._processor(images=rgb, device="cuda", return_tensors="pt")

    def _forward(self, inputs):
        with torch.inference_mode():
            out = self._model(inference_session=self._session,
                              frame=inputs.pixel_values[0].to(torch.bfloat16))
        masks = self._processor.post_process_masks(
            [out.pred_masks], original_sizes=inputs.original_sizes)[0]
        return masks[0, 0].cpu().numpy().astype(bool)


def mask_centroid(mask):
    """(u, v) pixel centroid of a bool mask, or None for an empty mask."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())
