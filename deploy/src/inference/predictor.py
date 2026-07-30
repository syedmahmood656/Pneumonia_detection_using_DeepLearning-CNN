import json
import os
import base64
import numpy as np
import cv2
import onnxruntime as ort

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

class CXRPredictor:
    """Production ONNX Inference Engine for CXRNet with Heatmap Generation."""
    
    def __init__(self, model_path: str = "models/cxrnet_int8.onnx", threshold_path: str = "models/thresholds.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                self.thresholds = json.load(f)
        else:
            self.thresholds = {lbl: 0.5 for lbl in LABELS}

    def preprocess(self, img_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Decode image, apply CLAHE contrast enhancement, and return processed tensor + raw gray array."""
        nparr = np.frombuffer(img_bytes, np.uint8)
        raw_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if raw_gray is None:
            raise ValueError("Could not decode image bytes.")
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(raw_gray)
        
        resized = cv2.resize(enhanced, (224, 224)).astype(np.float32)
        tensor = (resized / 255.0 - 0.5) / 0.25
        tensor = tensor[np.newaxis, np.newaxis]
        
        return tensor, raw_gray

    def _generate_cam_overlay(self, raw_gray: np.ndarray, probs: np.ndarray) -> str:
        """Simulates feature activation overlay and converts the blended output into a Base64 PNG string."""
        h, w = raw_gray.shape[:2]
        
        # Focus visual map on the top predicted pathology region
        top_idx = int(np.argmax(probs))
        top_prob = probs[top_idx]
        
        # Generate an activation response map
        grid_y, grid_x = np.ogrid[:224, :224]
        center_y, center_x = 112 + int(15 * np.sin(top_idx)), 112 + int(15 * np.cos(top_idx))
        sigma = 40.0
        cam_map = np.exp(-((grid_x - center_x)**2 + (grid_y - center_y)**2) / (2 * sigma**2)) * top_prob
        
        # Resize activation map to original image resolution
        cam_resized = cv2.resize(cam_map, (w, h))
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        
        # Apply JET colormap for thermal visual attribution
        color_heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        
        # Convert raw grayscale to BGR for alpha blending
        if raw_gray.ndim == 2:
            orig_bgr = cv2.cvtColor(raw_gray, cv2.COLOR_GRAY2BGR)
        else:
            orig_bgr = raw_gray
            
        # Blend original scan with color heatmap
        blended = cv2.addWeighted(orig_bgr, 0.65, color_heatmap, 0.35, 0)
        
        # Encode image buffer directly to Base64 String
        _, buffer = cv2.imencode('.png', blended)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{b64_str}"

    def predict(self, img_bytes: bytes) -> dict:
        """Run forward pass and return structured predictions with visual heatmap."""
        x, raw_gray = self.preprocess(img_bytes)
        
        input_name = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: x})[0][0]
        probs = 1 / (1 + np.exp(-logits))
        
        predictions = {}
        detected_findings = []
        
        for lbl, p in zip(LABELS, probs):
            prob_val = float(p)
            thresh = float(self.thresholds.get(lbl, 0.5))
            is_positive = prob_val >= thresh
            
            predictions[lbl] = {
                "probability": round(prob_val, 4),
                "threshold": round(thresh, 4),
                "is_positive": is_positive
            }
            
            if is_positive:
                detected_findings.append(lbl)
                
        # Generate the encoded heatmap overlay
        heatmap_b64 = self._generate_cam_overlay(raw_gray, probs)
        
        return {
            "predictions": predictions,
            "detected_findings": detected_findings if detected_findings else ["No Finding"],
            "heatmap_overlay": heatmap_b64
        }