import gradio as gr
import requests
import base64
import numpy as np
import cv2

API_URL = "http://127.0.0.1:8000/predict"

def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decodes a Base64 data string back into an RGB NumPy image array for Gradio rendering."""
    header, encoded = b64_string.split(",", 1)
    img_data = base64.b64decode(encoded)
    nparr = np.frombuffer(img_data, np.uint8)
    bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

def analyze_xray(image_path):
    if image_path is None:
        return None, "Please upload a chest X-ray image.", ""

    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}
            response = requests.post(API_URL, files=files)
            
        if response.status_code != 200:
            return None, f"API Error: {response.text}", ""
            
        data = response.json()
        findings = data["detected_findings"]
        predictions = data["predictions"]
        heatmap_b64 = data["heatmap_overlay"]
        
        # Decode the Base64 heatmap for Gradio Image output
        heatmap_img = decode_base64_image(heatmap_b64)
        
        # Build Markdown summary output
        summary = f"### 🩺 **Primary Assessment:** {', '.join(findings)}\n\n"
        table_md = "| Pathology | Probability | Optimal Threshold | Assessment |\n"
        table_md += "| :--- | :---: | :---: | :---: |\n"
        
        for pathology, metrics in predictions.items():
            prob = metrics["probability"]
            thresh = metrics["threshold"]
            badge = "🔴 Positive" if metrics["is_positive"] else "🟢 Normal"
            table_md += f"| **{pathology}** | `{prob:.4f}` | `{thresh:.4f}` | {badge} |\n"
            
        return heatmap_img, summary, table_md

    except Exception as e:
        return None, f"Error connecting to backend API: {str(e)}", ""

# Build Gradio UI with Dual Column Image View
with gr.Blocks(title="ExplainX-CXR Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🫁 ExplainX-CXR: Explainable Chest X-Ray Classifier
    Upload a chest X-ray image to receive real-time ONNX pathography scoring and Grad-CAM visual attention overlays.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="filepath", label="Input Scan (Raw X-Ray)")
            btn_submit = gr.Button("Analyze Scan", variant="primary")
            
        with gr.Column(scale=1):
            heatmap_output = gr.Image(label="AI Focus Map (Grad-CAM Heatmap)", type="numpy")

    with gr.Row():
        with gr.Column():
            summary_output = gr.Markdown()
            table_output = gr.Markdown()

    btn_submit.click(
        fn=analyze_xray,
        inputs=[img_input],
        outputs=[heatmap_output, summary_output, table_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)