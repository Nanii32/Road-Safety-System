import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os
from inference_sdk import InferenceHTTPClient

# Set the background image for the app
def set_background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://www.paybima.com/documents/778769/3030717/list-of-25-6th-11aaa.jpg");
            background-size: cover;
            background-attachment: fixed;
        }}
        .main {{
            background-color: rgba(89, 89, 255, 0.8);
            padding: 2rem;
            border-radius: 1rem;
        }}
        .stTitle {{
            font-size: 36px;
            font-weight: bold;
            color:;  /* Bright blue color for title */
            text-align: center;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3); /* Shadow for title */
        }}
        .summary-text {{
            font-size: 18px;
            color: #308;
            font-weight: 500;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 15px;
        }}
        .summary-text p {{
            margin: 0;
            padding: 5px 0;
        }}
        .summary-heading {{
            font-size: 22px;
            font-weight: 600;
            color: #007acc; /* Blue heading for summary */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="EZRRDT0bPs83rgIUMeGb"  # Replace with a valid API key if needed
)

model_1_id = "tvd-kp9qw/2"
model_2_id = "helmetdetectiondataset/2"

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area else 0

def remove_duplicate_detections(detections, iou_threshold=0.7):
    filtered = []
    for obj in detections:
        keep = True
        for f_obj in filtered:
            if iou((obj['x'], obj['y'], obj['width'], obj['height']),
                   (f_obj['x'], f_obj['y'], f_obj['width'], f_obj['height'])) > iou_threshold:

                if ("helmet" in obj["class"].lower() and "helmet" in f_obj["class"].lower()):
                    if obj["confidence"] > f_obj["confidence"]:
                        filtered.remove(f_obj)
                        filtered.append(obj)
                    keep = False
                    break
        if keep:
            filtered.append(obj)
    return filtered

def run_detection(image_path, output_dir):
    # Load image and run inference on both models
    image = mpimg.imread(image_path)
    result_1 = CLIENT.infer(image_path, model_id=model_1_id)
    result_2 = CLIENT.infer(image_path, model_id=model_2_id)

    # Merge and filter detections
    detections = result_1.get("predictions", []) + result_2.get("predictions", [])
    detections = [obj for obj in detections if obj["confidence"] > 0.5]
    detections = remove_duplicate_detections(detections)

    # Normalize labels
    for obj in detections:
        label = obj["class"].strip().lower()
        if "without helmet" in label or "no helmet" in label:
            obj["class"] = "no helmet"
        elif "with helmet" in label:
            obj["class"] = "with helmet"

    # Group detections
    triple_riding = [obj for obj in detections if "triple riding" in obj["class"].lower()]
    no_helmets = [obj for obj in detections if "no helmet" in obj["class"].lower()]
    with_helmets = [obj for obj in detections if "with helmet" in obj["class"].lower()]
    temp_mobile_usage = [obj for obj in detections if "mobile" in obj["class"].lower() or "phone" in obj["class"].lower()]
    mobile_usage = [obj for obj in temp_mobile_usage if obj["confidence"] > 0.85]
    if not mobile_usage:
        detections = [obj for obj in detections if "mobile" not in obj["class"].lower() and "phone" not in obj["class"].lower()]

    # Count violations and prepare summary
    violation_count = len(triple_riding) + len(no_helmets) + len(mobile_usage)
    summary_text = f"Violations: {violation_count}\n"
    summary_text += f"- Triple Riding: {len(triple_riding)}\n"
    summary_text += f"- ❌🪖No Helmet: {len(no_helmets)}\n"
    summary_text += f"- 📱🚫Mobile Usage: {len(mobile_usage)}\n"
    summary_text += f"Riders with Helmet: {len(with_helmets)}"
    
    # Emojis for violations
    emojis = []
    if violation_count > 0:
        if len(triple_riding) > 0:
            emojis.append("👥")  # Emoji for triple riding
        if len(no_helmets) > 0:
            emojis.append("❌🪖")  # Emoji for no helmet
        if len(mobile_usage) > 0:
            emojis.append("📱🚫")  # Emoji for mobile usage

    emoji_string = " ".join(emojis)

    if violation_count == 0:
        summary_text = "No rules violated."

    # Create a figure with bounding boxes and summary text
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(image)
    color_map = {"no helmet": "red", "triple riding": "red", "mobile usage": "red", "with helmet": "green"}
    for obj in detections:
        label = obj["class"].lower()
        confidence = obj["confidence"]
        x, y, w, h = obj["x"], obj["y"], obj["width"], obj["height"]
        x1, y1 = x - w / 2, y - h / 2  
        box_color = color_map.get(label, "red")
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=box_color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{obj['class']} ({confidence*100:.2f}%)", fontsize=10,
                color=box_color, bbox=dict(facecolor="white", alpha=0.6))
    plt.text(10, image.shape[0] - 50, summary_text + emoji_string, fontsize=12, color="black",
             bbox=dict(facecolor="white", alpha=0.8))

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return output_path, summary_text

# Streamlit App UI
def main():
    set_background()

    st.title("Traffic rules Violation Detection")
    st.markdown("Upload an image to detect violations related to helmet usage, triple riding, and mobile usage.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        if st.button("Detect Violation"):
            output_image_path, summary_text = run_detection(image_path, output_dir)

            st.image(output_image_path, caption="Detection Result", use_container_width=True)
            st.text(summary_text)

        if st.button("Try Another Image"):
            st.cache_data.clear()

            st.rerun()

if __name__ == "__main__":
    main()
