import cv2
import numpy as np
class ELADetector:
    def __init__(self, quality=90, block_variance_threshold=15):
        self.quality = quality
        self.threshold = block_variance_threshold
    def compute_ela(self, img_bgr):
        _, enc = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        recompressed = cv2.imdecode(enc, 1)
        return cv2.absdiff(img_bgr, recompressed) * 15
    def detect_boundaries(self, ela, block_size=32):
        gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        boundary_map = np.zeros_like(gray)
        for y in range(0, h-block_size, block_size//2):
            for x in range(0, w-block_size, block_size//2):
                patch = gray[y:y+block_size, x:x+block_size]
                if np.var(patch) > self.threshold:
                    boundary_map[y:y+block_size, x:x+block_size] = 255
        return boundary_map
    def is_spliced(self, img_bgr, boundary_ratio_thresh=0.02):
        ela = self.compute_ela(img_bgr)
        bmap = self.detect_boundaries(ela)
        ratio = np.sum(bmap > 0) / bmap.size
        return ratio > boundary_ratio_thresh, ela, bmap
