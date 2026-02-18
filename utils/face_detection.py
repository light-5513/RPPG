import cv2
import numpy as np
import mediapipe as mp


class FaceROIExtractor:
    """Extract face region of interest using MediaPipe Face Mesh."""

    def __init__(self, roi_size=72):
        self.roi_size = roi_size
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Landmark indices for different face regions
        self.FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                         288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                         150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
                         67, 109]

        self.LEFT_CHEEK = [116, 117, 118, 119, 100, 126, 209, 49, 48, 131,
                           198, 236, 3, 195, 235, 120, 47, 121, 128, 245]

        self.RIGHT_CHEEK = [345, 346, 347, 348, 329, 355, 429, 279, 278, 360,
                            420, 456, 248, 419, 399, 349, 277, 350, 357, 465]

    def get_roi_from_landmarks(self, frame, landmarks, region_indices):
        """Extract ROI from specific landmark indices."""
        h, w = frame.shape[:2]

        xs = [int(landmarks[i].x * w) for i in region_indices if i < len(landmarks)]
        ys = [int(landmarks[i].y * h) for i in region_indices if i < len(landmarks)]

        if not xs or not ys:
            return None

        x1 = max(0, min(xs))
        x2 = min(w, max(xs))
        y1 = max(0, min(ys))
        y2 = min(h, max(ys))

        if x2 - x1 < 5 or y2 - y1 < 5:
            return None

        return frame[y1:y2, x1:x2]

    def get_face_bbox(self, frame, landmarks):
        """Get full face bounding box from landmarks."""
        h, w = frame.shape[:2]

        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]

        x1 = max(0, min(xs))
        x2 = min(w, max(xs))
        y1 = max(0, min(ys))
        y2 = min(h, max(ys))

        # Add padding
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)

        x1 = max(0, x1 - pad_x)
        x2 = min(w, x2 + pad_x)
        y1 = max(0, y1 - pad_y)
        y2 = min(h, y2 + pad_y)

        return x1, y1, x2, y2

    def extract_face_roi(self, frame):
        """
        Extract face ROI from a frame.
        Returns: resized face crop, mean RGB of forehead, cheeks
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, None, None

        landmarks = results.multi_face_landmarks[0].landmark

        # Full face crop
        x1, y1, x2, y2 = self.get_face_bbox(frame, landmarks)
        face_crop = rgb_frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None, None, None

        face_resized = cv2.resize(face_crop, (self.roi_size, self.roi_size))

        # ROI mean RGB values (for signal processing methods)
        forehead_roi = self.get_roi_from_landmarks(rgb_frame, landmarks, self.FOREHEAD)
        left_cheek = self.get_roi_from_landmarks(rgb_frame, landmarks, self.LEFT_CHEEK)
        right_cheek = self.get_roi_from_landmarks(rgb_frame, landmarks, self.RIGHT_CHEEK)

        # Combine cheek ROIs
        mean_rgb = []
        for roi in [forehead_roi, left_cheek, right_cheek]:
            if roi is not None and roi.size > 0:
                mean_rgb.append(roi.mean(axis=(0, 1)))

        if mean_rgb:
            combined_rgb = np.mean(mean_rgb, axis=0)  # [R, G, B]
        else:
            combined_rgb = None

        # Face bbox for display
        bbox = (x1, y1, x2, y2)

        return face_resized, combined_rgb, bbox

    def extract_batch_frames(self, frames):
        """Process a batch of frames and return face crops + RGB signals."""
        face_crops = []
        rgb_signals = []

        for frame in frames:
            face, rgb, _ = self.extract_face_roi(frame)
            if face is not None and rgb is not None:
                face_crops.append(face)
                rgb_signals.append(rgb)

        return np.array(face_crops) if face_crops else None, \
               np.array(rgb_signals) if rgb_signals else None

    def close(self):
        self.face_mesh.close()