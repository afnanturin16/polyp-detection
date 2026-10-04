"""
Macenko H&E stain normalization, implemented directly in numpy (no SPAMS/
staintools dependency, which are painful to build on Windows).

H&E slides from different scanners/labs vary a lot in color -- a classifier
can pick up on stain color as a shortcut instead of tissue morphology, which
is a common source of hidden bias in histopathology models. Stain
normalization maps every image's hematoxylin/eosin stain vectors onto a
single reference image's vectors before the color/intensity augmentations
run, so downstream models see consistent staining.

Reference: Macenko et al., "A method for normalizing histology slides for
quantitative analysis", ISBI 2009.
"""
import numpy as np


class MacenkoNormalizer:
    def __init__(self, od_threshold=0.15, angular_percentile=1, target_max_conc_percentile=99):
        self.od_threshold = od_threshold
        self.angular_percentile = angular_percentile
        self.target_max_conc_percentile = target_max_conc_percentile
        self.stain_matrix = None  # (3, 2): columns are hematoxylin, eosin OD vectors
        self.max_concentrations = None  # (2,)

    @staticmethod
    def _rgb_to_od(image):
        image = image.astype(np.float64)
        image[image == 0] = 1.0
        return -np.log10(image / 255.0)

    def _estimate_stain_matrix(self, od):
        # Keep only pixels with enough optical density (i.e. not background).
        od_flat = od.reshape(-1, 3)
        od_flat = od_flat[np.all(od_flat > self.od_threshold, axis=1)]
        if od_flat.shape[0] < 10:
            raise ValueError("Not enough non-background pixels to estimate stain matrix")

        # Project onto the plane of the top 2 principal components.
        cov = np.cov(od_flat, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        top2 = eigvecs[:, np.argsort(eigvals)[-2:]]
        proj = od_flat @ top2

        angles = np.arctan2(proj[:, 1], proj[:, 0])
        min_angle = np.percentile(angles, self.angular_percentile)
        max_angle = np.percentile(angles, 100 - self.angular_percentile)

        v_min = top2 @ np.array([np.cos(min_angle), np.sin(min_angle)])
        v_max = top2 @ np.array([np.cos(max_angle), np.sin(max_angle)])

        # By convention, hematoxylin is the vector with the larger red
        # component so the two stains come out in a consistent order.
        if v_min[0] > v_max[0]:
            stain_matrix = np.stack([v_min, v_max], axis=1)
        else:
            stain_matrix = np.stack([v_max, v_min], axis=1)

        # Normalize each stain vector to unit length.
        stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0, keepdims=True)
        return stain_matrix

    def _estimate_concentrations(self, od, stain_matrix):
        od_flat = od.reshape(-1, 3).T  # (3, n_pixels)
        concentrations, *_ = np.linalg.lstsq(stain_matrix, od_flat, rcond=None)
        return concentrations  # (2, n_pixels)

    def fit(self, reference_image):
        """Fit the reference stain matrix and max concentrations from a
        representative RGB uint8 image (H, W, 3)."""
        od = self._rgb_to_od(reference_image)
        self.stain_matrix = self._estimate_stain_matrix(od)
        concentrations = self._estimate_concentrations(od, self.stain_matrix)
        self.max_concentrations = np.percentile(
            concentrations, self.target_max_conc_percentile, axis=1
        )
        return self

    def transform(self, image):
        """Normalize an RGB uint8 image (H, W, 3) to the fitted reference
        stain appearance."""
        if self.stain_matrix is None:
            raise RuntimeError("Call fit() (or load a saved reference) before transform()")
        h, w = image.shape[:2]
        od = self._rgb_to_od(image)
        source_stain_matrix = self._estimate_stain_matrix(od)
        concentrations = self._estimate_concentrations(od, source_stain_matrix)

        source_max = np.percentile(concentrations, self.target_max_conc_percentile, axis=1)
        source_max[source_max == 0] = 1e-6
        concentrations = concentrations * (self.max_concentrations / source_max)[:, None]

        od_normalized = self.stain_matrix @ concentrations
        rgb_normalized = 255.0 * np.exp(-od_normalized)
        rgb_normalized = np.clip(rgb_normalized, 0, 255).astype(np.uint8)
        return rgb_normalized.T.reshape(h, w, 3)

    def save(self, path):
        np.savez(path, stain_matrix=self.stain_matrix, max_concentrations=self.max_concentrations)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        normalizer = cls()
        normalizer.stain_matrix = data["stain_matrix"]
        normalizer.max_concentrations = data["max_concentrations"]
        return normalizer
