"""
PolypNet: a single lightweight multi-head model replacing the 3x ResNet18
cascade. One shared mobilenetv3_small_100 encoder (timm, pretrained,
num_classes=0 -- pooled features only) feeds three linear heads:

  type_h  (3): NORM / HP / ADENOMA
  shape_h (2): TA / TVA        -- only meaningful (and only trained) for adenoma patches
  grade_h (2): LG / HG         -- only meaningful (and only trained) for adenoma patches

timm's `num_features` attribute reports the pre-head channel count (576),
but this model's actual pooled forward output is 1024-dim (it still runs the
conv_head expansion even with num_classes=0) -- so the head input dim is
probed with a dummy forward pass rather than trusted from that attribute.
"""
import timm
import torch
import torch.nn as nn

ENCODER_NAME = "mobilenetv3_small_100"


class PolypNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(ENCODER_NAME, pretrained=pretrained, num_classes=0)
        with torch.no_grad():
            feat_dim = self.encoder(torch.randn(1, 3, 224, 224)).shape[1]
        self.feat_dim = feat_dim
        self.type_h = nn.Linear(feat_dim, 3)
        self.shape_h = nn.Linear(feat_dim, 2)
        self.grade_h = nn.Linear(feat_dim, 2)

    def forward(self, x):
        f = self.encoder(x)
        return self.type_h(f), self.shape_h(f), self.grade_h(f)

    def param_groups(self, encoder_lr, head_lr):
        """Separate LR for the pretrained encoder vs. the freshly-initialized
        heads, per the training spec (encoder 1e-4, heads 3e-4)."""
        return [
            {"params": self.encoder.parameters(), "lr": encoder_lr},
            {"params": list(self.type_h.parameters()) + list(self.shape_h.parameters()) +
                       list(self.grade_h.parameters()), "lr": head_lr},
        ]


TYPE_CLASSES = ["NORM", "HP", "ADENOMA"]
SHAPE_CLASSES = ["TA", "TVA"]
GRADE_CLASSES = ["LG", "HG"]
FULL_CLASSES = ["HP", "NORM", "TA.HG", "TA.LG", "TVA.HG", "TVA.LG"]


def derive_targets(orig_label):
    """orig_label: one of the 6 original UniToPatho class names.
    Returns (y_type, y_shape, y_grade) ints, per the exact spec:
      y_type:  NORM->0, HP->1, the four adenoma classes->2
      y_shape: 1 if label starts with 'TVA' else 0 (meaningful only when y_type==2)
      y_grade: 1 if label ends with 'HG' else 0 (meaningful only when y_type==2)
    """
    if orig_label == "NORM":
        y_type = 0
    elif orig_label == "HP":
        y_type = 1
    else:
        y_type = 2
    y_shape = int(orig_label.startswith("TVA"))
    y_grade = int(orig_label.endswith("HG"))
    return y_type, y_shape, y_grade


def combine_to_full_label(pred_type, pred_shape, pred_grade):
    """Cascade-style routing at inference: read type first; only if it
    predicts ADENOMA (2), read shape/grade and combine into the final
    6-class label."""
    if pred_type == 0:
        return "NORM"
    if pred_type == 1:
        return "HP"
    shape = SHAPE_CLASSES[pred_shape]
    grade = GRADE_CLASSES[pred_grade]
    return f"{shape}.{grade}"
