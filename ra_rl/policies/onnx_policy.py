"""ONNX Runtime-based policy for the commander network (CPU-only, no PyTorch needed)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import onnxruntime as ort

from ra_env import StepResult, pb
from ra_rl.features import CAND_DIM, GLOBAL_DIM, Features, is_tactical_request, tactical_channel_id
from ra_rl.vocab import QUEUE_INDEX


class OnnxPolicy:
    """The commander network evaluated via ONNX Runtime (fast, pure CPU, zero PyTorch)."""

    brain = "external"

    def __init__(
        self,
        item_onnx_path: str | Path,
        army_onnx_path: str | Path,
        ability_onnx_path: str | Path | None = None,
        macro_onnx_path: str | Path | None = None,
    ):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess_item = ort.InferenceSession(str(item_onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sess_army = ort.InferenceSession(str(army_onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

        self.sess_ability = None
        if ability_onnx_path and Path(ability_onnx_path).is_file():
            self.sess_ability = ort.InferenceSession(str(ability_onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

        self.sess_macro = None
        if macro_onnx_path and Path(macro_onnx_path).is_file():
            self.sess_macro = ort.InferenceSession(str(macro_onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

        self.reset(0)

    def reset(self, seed: int) -> None:
        self.feats = Features()

    def actions(self, step: StepResult) -> list[int]:
        self.feats.observe(step.obs)
        if not step.pending:
            return []

        g = np.expand_dims(self.feats.g, axis=0)  # [1, GLOBAL_DIM]
        out: list[int] = []

        for req in step.pending:
            if req.kind == pb.DECIDE_ABILITY and self.sess_ability is not None:
                cand, mask = self.feats.candidates(req)
                cand_in = np.expand_dims(cand, axis=0)
                mask_in = np.expand_dims(mask, axis=0)
                ab_id = np.array([ABILITY_INDEX.get(req.queue, 0)], dtype=np.int64)
                logits = self.sess_ability.run(
                    None,
                    {"g": g, "cand": cand_in, "mask": mask_in, "ability_id": ab_id},
                )[0]
                a = int(np.argmax(logits[0]))
            elif req.kind == pb.DECIDE_MACRO_OPS and self.sess_macro is not None:
                mask_in = np.ones((1, 3), dtype=bool)
                logits = self.sess_macro.run(
                    None,
                    {"g": g, "mask": mask_in},
                )[0]
                a = int(np.argmax(logits[0]))
            elif is_tactical_request(req):
                cand, mask = self.feats.candidates(req)
                cand_in = np.expand_dims(cand, axis=0)  # [1, 16, CAND_DIM]
                mask_in = np.expand_dims(mask, axis=0)  # [1, 34]
                grp_in = np.array([tactical_channel_id(req)], dtype=np.int64)
                logits = self.sess_army.run(
                    None,
                    {"g": g, "cand": cand_in, "mask": mask_in, "group": grp_in},
                )[0]
                a = int(np.argmax(logits[0]))
            else:
                queue_in = np.array([QUEUE_INDEX.get(req.queue, 0)], dtype=np.int64)
                mask_in = np.expand_dims(self.feats.item_mask(req), axis=0)  # [1, 76]
                logits = self.sess_item.run(
                    None,
                    {"g": g, "queue": queue_in, "mask": mask_in},
                )[0]
                a = int(np.argmax(logits[0]))
            out.append(a)
        return out

    def decide(self, step: StepResult) -> list[pb.Decision]:
        out = []
        for req, a in zip(step.pending, self.actions(step), strict=True):
            d = self.feats.decision(req, a)
            self.feats.record(req, d)
            if d is not None:
                out.append(d)
        return out
