"""Plant Disease Recognition app (Streamlit).
Put the `app_model/` folder from the Colab results.zip next to this file, then run:
    pip install streamlit tensorflow pillow matplotlib
    streamlit run app.py
"""
import json
import numpy as np
from PIL import Image

MODEL_DIR = "app_model"


def load_model():
    import keras
    model = keras.models.load_model(f"{MODEL_DIR}/model.keras", compile=False)
    meta = json.load(open(f"{MODEL_DIR}/meta.json"))
    return model, meta


def gradcam(model, x):
    """Grad-CAM heat-map (Gap 6: explainability) for one image batch x of shape (1, H, W, 3)."""
    import keras, tensorflow as tf
    layer = next(l.name for l in reversed(model.layers) if len(l.output.shape) == 4)
    gm = keras.Model(model.inputs, [model.get_layer(layer).output, model.outputs[0]])
    with tf.GradientTape() as tape:
        conv, logits = gm(tf.convert_to_tensor(x, tf.float32), training=False)
        score = logits[:, int(tf.argmax(logits[0]))]
    g = tape.gradient(score, conv)
    cam = tf.nn.relu(tf.reduce_sum(tf.reduce_mean(g, axis=(1, 2), keepdims=True) * conv, -1))[0]
    cam = tf.image.resize(cam[..., None], x.shape[1:3])[..., 0].numpy()
    return cam / (cam.max() + 1e-8)


def analyse(model, meta, img: Image.Image):
    size = meta["img_size"]
    x = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), np.float32)[None]
    logits = model.predict(x, verbose=0)[0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    top = np.argsort(-p)[:3]
    import matplotlib.cm as cm
    heat = cm.jet(gradcam(model, x))[..., :3] * 255
    overlay = np.clip(0.55 * x[0] + 0.45 * heat, 0, 255).astype(np.uint8)
    return [(meta["class_names"][i], float(p[i])) for i in top], overlay


def pretty(name):
    crop, disease = name.split("___")
    return f"{crop.replace('_', ' ')}: {disease.replace('_', ' ').strip()}"


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Plant Disease Recognition", page_icon="🌿")
    get = st.cache_resource(load_model)
    page = st.sidebar.selectbox("Page", ["Disease Recognition", "About"])
    if page == "About":
        st.header("About")
        st.markdown("""
**Base paper:** S. M. Hassan and A. K. Maji, "Plant Disease Identification Using a Novel Convolutional
Neural Network," *IEEE Access*, vol. 10, pp. 5390–5401, 2022.

**Research gaps filled in this project**
1. **Leakage-free evaluation.** Data is split by leaf, not by image, so rotated or flipped copies of one leaf never appear in both train and test.
2. **Lab → field generalisation.** Tested on real field photos (PlantDoc), with field images mixed into training.
3. **Explainability.** A Grad-CAM heat-map shows which part of the leaf drove the prediction.

*This tool supports, but does not replace, an agricultural expert.*
""")
    else:
        st.header("🌿 Plant Disease Recognition")
        up = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
        if up and st.button("Predict"):
            model, meta = get()
            img = Image.open(up)
            preds, overlay = analyse(model, meta, img)
            c1, c2 = st.columns(2)
            c1.image(img, caption="Your photo", use_container_width=True)
            c2.image(overlay, caption="Grad-CAM: red = most important", use_container_width=True)
            name, conf = preds[0]
            st.success(f"**{pretty(name)}** ({conf:.0%} confidence)")
            st.write("Top-3 predictions")
            for n, c in preds:
                st.progress(c, text=f"{pretty(n)}: {c:.1%}")
