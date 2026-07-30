import gradio as gr
from src.inference.predictor import CXRPredictor

# Instantiate predictor directly inside the app
predictor = CXRPredictor(
    model_path="models/cxrnet_int8.onnx", 
    threshold_path="models/thresholds.json"
)

def analyze_xray(image_path):
    if image_path is None:
        return "Please upload an image.", ""

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        results = predictor.predict(image_bytes)
        findings = results["detected_findings"]
        predictions = results["predictions"]
        
        summary = f"### 🩺 **Primary Assessment:** {', '.join(findings)}\n\n"
        table_md = "| Pathology | Probability | Optimal Threshold | Assessment |\n"
        table_md += "| :--- | :---: | :---: | :---: |\n"
        
        for pathology, metrics in predictions.items():
            prob = metrics["probability"]
            thresh = metrics["threshold"]
            badge = "🔴 Positive" if metrics["is_positive"] else "🟢 Normal"
            table_md += f"| **{pathology}** | `{prob:.4f}` | `{thresh:.4f}` | {badge} |\n"
            
        return summary, table_md

    except Exception as e:
        return f"Error during inference: {str(e)}", ""

# Build Gradio UI
with gr.Blocks(title="ExplainX-CXR Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🫁 ExplainX-CXR: Explainable Chest X-Ray Classifier
    Upload a grayscale chest X-ray image to run real-time inference via ONNX Runtime.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="filepath", label="Upload Chest X-Ray")
            btn_submit = gr.Button("Analyze Scan", variant="primary")
            
        with gr.Column(scale=1):
            summary_output = gr.Markdown()
            table_output = gr.Markdown()

    btn_submit.click(
        fn=analyze_xray,
        inputs=[img_input],
        outputs=[summary_output, table_output]
    )

if __name__ == "__main__":
    demo.launch()
