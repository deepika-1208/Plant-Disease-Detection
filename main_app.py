import numpy as np
import streamlit as st
import cv2
import tensorflow as tf
from keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Plant Doctor",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4faf5;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* Main headings */

h1, h2, h3, h4, h5, h6 {
    color: #176b2c !important;
}

p, label {
    color: #405947 !important;
}


/* Header */

.header-box {
    background-color: #e5f4e7;
    border: 1px solid #c9e4cc;
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 25px;
}

.header-title {
    color: #176b2c;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.header-text {
    color: #49604e;
    font-size: 17px;
}


/* Section */

.section-title {
    color: #176b2c;
    font-size: 25px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 15px;
}


/* Result */

.result-card {
    background-color: white;
    border: 1px solid #cfe5d2;
    border-radius: 18px;
    padding: 25px;
    margin-top: 10px;
    margin-bottom: 20px;
}

.result-label {
    color: #657267;
    font-size: 14px;
    font-weight: 700;
}

.result-name {
    color: #176b2c;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}


/* Cards */

.info-card {
    background-color: white;
    border: 1px solid #d5e7d7;
    border-radius: 16px;
    padding: 22px;
    min-height: 190px;
}

.info-card h3 {
    color: #247a35 !important;
}

.info-card p {
    color: #4f5f53 !important;
    line-height: 1.7;
}


/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #e5f3e7;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #244a2b !important;
}


/* Buttons */

.stButton > button {
    background-color: #237a36;
    color: white;
    border-radius: 10px;
    font-weight: 700;
    border: none;
    min-height: 45px;
}

.stButton > button:hover {
    background-color: #185d28;
    color: white;
}


/* Images */

[data-testid="stImage"] img {
    border-radius: 15px;
}


/* History */

.history-box {
    background-color: white;
    border: 1px solid #d5e7d7;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_plant_model():
    return load_model("plant_disease_model.h5")


try:

    model = load_plant_model()

except Exception as e:

    st.error("❌ Unable to load plant disease model.")
    st.exception(e)
    st.stop()


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = (
    "Tomato-Bacterial_spot",
    "Potato-Barly blight",
    "Corn-Common_rust"
)


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Tomato-Bacterial_spot": {

        "name": "Tomato Bacterial Spot",

        "plant": "Tomato",

        "about": (
            "Bacterial spot is a bacterial disease that can affect "
            "tomato leaves and fruits. Small dark spots may appear "
            "on affected plant parts."
        ),

        "treatment": (
            "Remove severely affected leaves where practical. "
            "Avoid handling plants while foliage is wet and follow "
            "locally recommended disease-management practices."
        ),

        "prevention": (
            "Use healthy planting material, maintain adequate spacing, "
            "avoid overhead watering and remove infected plant debris."
        ),

        "care": (
            "Provide sufficient sunlight, maintain consistent soil "
            "moisture and inspect new leaves regularly."
        )
    },


    "Potato-Barly blight": {

        "name": "Potato Early Blight",

        "plant": "Potato",

        "about": (
            "Early blight is a fungal disease commonly associated "
            "with dark lesions on potato foliage. Symptoms often "
            "appear first on older leaves."
        ),

        "treatment": (
            "Remove heavily affected foliage where practical and "
            "maintain good field hygiene. Fungicide use should follow "
            "local agricultural recommendations."
        ),

        "prevention": (
            "Practice crop rotation, remove infected plant debris, "
            "avoid prolonged leaf wetness and maintain good airflow."
        ),

        "care": (
            "Maintain balanced watering and nutrition. Avoid excessive "
            "nitrogen and regularly inspect older leaves."
        )
    },


    "Corn-Common_rust": {

        "name": "Corn Common Rust",

        "plant": "Corn",

        "about": (
            "Common rust is a fungal disease of corn that can produce "
            "reddish-brown or rust-colored pustules on leaves."
        ),

        "treatment": (
            "Monitor disease development and follow locally recommended "
            "disease-management practices."
        ),

        "prevention": (
            "Use resistant varieties when available, maintain good "
            "crop management and monitor plants during favorable conditions."
        ),

        "care": (
            "Provide adequate sunlight, balanced nutrients and "
            "appropriate soil moisture."
        )
    }
}


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


if "last_result" not in st.session_state:

    st.session_state.last_result = None


# ============================================================
# HEADER
# ============================================================

st.title("🌿 Smart Plant Doctor")

st.subheader(
    "Deep Learning-Based Plant Disease Detection & Plant Health Assistant"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌿 Smart Plant Doctor")

    st.write(
        "AI-powered plant disease detection "
        "and plant health assistant."
    )

    st.divider()

    st.subheader("🧠 Technologies")

    st.write("• Deep Learning")
    st.write("• TensorFlow / Keras")
    st.write("• Streamlit")
    st.write("• OpenCV")
    st.write("• Image Processing")
    st.write("• Plant Disease Classification")

    st.divider()

    st.subheader("🌱 Supported Plants")

    st.write("🍅 Tomato")
    st.write("🥔 Potato")
    st.write("🌽 Corn")

    st.divider()

    st.caption(
        "Model: CNN-based plant disease classifier"
    )


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📷 Upload Plant Leaf</div>',
    unsafe_allow_html=True
)


plant_image = st.file_uploader(
    "Choose a plant leaf image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a tomato, potato or corn leaf."
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔍 Analyze Plant",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if plant_image is None:

        st.warning(
            "⚠️ Please upload a plant leaf image first."
        )

    else:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        file_bytes = np.asarray(
            bytearray(
                plant_image.read()
            ),
            dtype=np.uint8
        )


        opencv_image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


        if opencv_image is None:

            st.error(
                "❌ Unable to read this image. "
                "Please upload another image."
            )

        else:

            # =================================================
            # UPLOADED IMAGE
            # =================================================

            st.markdown(
                '<div class="section-title">📷 Uploaded Leaf</div>',
                unsafe_allow_html=True
            )


            st.image(
                cv2.cvtColor(
                    opencv_image,
                    cv2.COLOR_BGR2RGB
                ),
                width=500
            )


            # =================================================
            # PREPROCESSING
            # =================================================

            resized_image = cv2.resize(
                opencv_image,
                (256, 256)
            )
            input_image=np.array(
                resized_image,
                dtype=np.float32
            )/255.0

            input_image = input_image.reshape(
                1,
                256,
                256,
                3
            )


            # =================================================
            # MODEL PREDICTION
            # =================================================

            with st.spinner(
                "🧠 AI is analyzing the plant leaf..."
            ):

                prediction = model.predict(
                    input_image,
                    verbose=0
                )
                


            # =================================================
            # MODEL OUTPUT
            # =================================================

            raw_scores = np.asarray(
                prediction[0],
                dtype=np.float32
            )


            # =================================================
            # CONVERT OUTPUT TO PROBABILITIES
            # =================================================

            if (

                np.all(
                    raw_scores >= 0
                )

                and

                np.all(
                    raw_scores <= 1
                )

                and

                np.isclose(
                    np.sum(raw_scores),
                    1.0,
                    atol=1e-3
                )

            ):

                probabilities = raw_scores

            else:

                probabilities = (
                    tf.nn.softmax(
                        raw_scores
                    ).numpy()
                )


            # =================================================
            # PREDICTED CLASS
            # =================================================

            predicted_class = int(
                np.argmax(
                    probabilities
                )
            )


            confidence = float(
                probabilities[
                    predicted_class
                ]
            ) * 100


            result = CLASS_NAMES[
                predicted_class
            ]


            disease = DISEASE_INFO[
                result
            ]


            # =================================================
            # SAVE RESULT
            # =================================================

            st.session_state.last_result = {

                "disease": disease["name"],

                "plant": disease["plant"],

                "confidence": confidence

            }


            st.session_state.history.insert(
                0,
                {

                    "disease": disease["name"],

                    "plant": disease["plant"],

                    "confidence": confidence

                }
            )


            st.session_state.history = (
                st.session_state.history[:10]
            )


            # =================================================
            # PREDICTION RESULT
            # =================================================

            st.markdown(
                '<div class="section-title">🦠 Prediction Result</div>',
                unsafe_allow_html=True
            )


            with st.container(border=True):

                st.caption(
                    "DISEASE DETECTED"
                )


                st.subheader(
                    f"🌿 {disease['name']}"
                )


                st.write(
                    f"🌱 **Plant:** {disease['plant']}"
                )


            # =================================================
            # CONFIDENCE
            # =================================================

            st.markdown(
                '<div class="section-title">🎯 Prediction Confidence</div>',
                unsafe_allow_html=True
            )


            st.progress(
                min(
                    confidence / 100,
                    1.0
                )
            )


            st.write(
                f"### {confidence:.2f}%"
            )


            if confidence >= 90:

                st.success(
                    "🟢 High model confidence"
                )

            elif confidence >= 70:

                st.info(
                    "🟡 Moderate model confidence"
                )

            else:

                st.warning(
                    "🟠 Low model confidence — "
                    "consider uploading a clearer leaf image."
                )


            # =================================================
            # TOP 3 PREDICTIONS
            # =================================================

            st.markdown(
                '<div class="section-title">📊 Model Prediction Breakdown</div>',
                unsafe_allow_html=True
            )


            sorted_indices = np.argsort(
                probabilities
            )[::-1]


            for rank, index in enumerate(
                sorted_indices[:3],
                start=1
            ):

                probability = float(
                    probabilities[index]
                ) * 100


                readable_name = (
                    DISEASE_INFO[
                        CLASS_NAMES[index]
                    ]["name"]
                )


                st.write(
                    f"**#{rank} {readable_name}**"
                )


                st.progress(
                    min(
                        probability / 100,
                        1.0
                    )
                )


                st.caption(
                    f"{probability:.2f}%"
                )


            # =================================================
            # DISEASE INFORMATION
            # =================================================

            st.markdown(
                '<div class="section-title">📖 Disease Information</div>',
                unsafe_allow_html=True
            )


            col1, col2 = st.columns(2)


            with col1:

                with st.container(border=True):

                    st.subheader(
                        "🦠 About"
                    )

                    st.write(
                        disease["about"]
                    )


            with col2:

                with st.container(border=True):

                    st.subheader(
                        "💊 Treatment"
                    )

                    st.write(
                        disease["treatment"]
                    )


            # =================================================
            # PLANT CARE
            # =================================================

            st.markdown(
                '<div class="section-title">🌱 Plant Care</div>',
                unsafe_allow_html=True
            )


            col1, col2 = st.columns(2)


            with col1:

                with st.container(border=True):

                    st.subheader(
                        "🛡️ Prevention"
                    )

                    st.write(
                        disease["prevention"]
                    )


            with col2:

                with st.container(border=True):

                    st.subheader(
                        "🌿 General Care"
                    )

                    st.write(
                        disease["care"]
                    )


            # =================================================
            # GENERAL DISCLAIMER
            # =================================================

            st.caption(
                "ℹ️ This tool provides AI-based predictions and "
                "general plant-care guidance. For serious crop "
                "disease problems, consult a qualified agricultural expert."
            )


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.history:

    st.markdown(
        '<div class="section-title">📜 Prediction History</div>',
        unsafe_allow_html=True
    )


    for index, item in enumerate(
        st.session_state.history,
        start=1
    ):

        with st.container(border=True):

            col1, col2, col3 = st.columns(
                [1, 5, 2]
            )


            with col1:

                st.write(
                    f"**#{index}**"
                )


            with col2:

                st.write(
                    f"🌿 **{item['disease']}**"
                )

                st.caption(
                    f"Plant: {item['plant']}"
                )


            with col3:

                st.write(
                    f"**{item['confidence']:.2f}%**"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌿 Smart Plant Doctor • Deep Learning • Plant Disease Detection • Plant Health Assistant"
)