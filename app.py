import torch
import streamlit as st
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Constants and Configurations
CLASS_LABELS = [
    'blackheads', 'dark spot', 'nodules', 'papules', 'pustules', 'whiteheads'
]

MODEL_NAMES = [
    "clip", "convnext", "mobilenet", "mobilevit", "siglip", "vit"
]

MODEL_INFO = {
    "mobilevit": {
        "company": "Apple",
        "accuracy": 0.774250,
        "f1_score": 0.631124,
        "recall": 0.596730,
        "precision": 0.669725,
        "parameters": "5.6 million",
        "description": "MobileViT is a lightweight model architecture that merges the benefits of transformers and convolutional networks, making it efficient for mobile and embedded applications.",
        "accuracy_explanation": "The accuracy of 77.4% reflects MobileViT’s ability to balance performance and efficiency. Its compact architecture allows it to generalize well on mobile devices but may not match the accuracy of larger models due to fewer parameters and less complex feature extraction."
    },
    "siglip": {
        "company": "Google",
        "accuracy": 0.788360,
        "f1_score": 0.661017,
        "recall": 0.637602,
        "precision": 0.686217,
        "parameters": "12 million",
        "description": "SigLIP is a robust image classification model designed by Google, known for its accurate multi-class predictions in various scenarios.",
        "accuracy_explanation": "SigLIP achieves 78.8% accuracy, driven by its robust feature extraction and multi-modal training. However, its accuracy can vary depending on the diversity of training data and the complexity of real-world scenarios."
    },
    "clip": {
        "company": "OpenAI",
        "accuracy": 0.709877,
        "f1_score": 0.538569,
        "recall": 0.523161,
        "precision": 0.554913,
        "parameters": "300 million",
        "description": "CLIP by OpenAI is a multi-modal model that connects vision and language. Although not primarily for skin diagnostics, it demonstrates versatile capabilities in visual understanding.",
        "accuracy_explanation": "With an accuracy of 70.9%, CLIP's performance is respectable, considering its general-purpose design. It wasn't specifically trained for skin condition classification, so its accuracy may be lower compared to more specialized models."
    },
    "convnext": {
        "company": "Meta",
        "accuracy": 0.782187,
        "f1_score": 0.622901,
        "recall": 0.555858,
        "precision": 0.708333,
        "parameters": "30 million",
        "description": "ConvNeXT is an evolution of the standard convolutional neural network, enhanced to match the performance of modern vision transformers.",
        "accuracy_explanation": "ConvNeXT achieves 78.2% accuracy by leveraging deep convolutional layers and residual connections, which help it generalize across different datasets. Its accuracy benefits from a mix of traditional CNN and modern enhancements, although it might still struggle with subtle variations in skin conditions."
    },
    "vit": {
        "company": "Google",
        "accuracy": 0.810406,
        "f1_score": 0.687045,
        "recall": 0.643052,
        "precision": 0.737500,
        "parameters": "86 million",
        "description": "Vision Transformer (ViT) is a model that applies transformer mechanisms directly to image patches, achieving high accuracy across multiple tasks.",
        "accuracy_explanation": "The 81.0% accuracy of ViT reflects its ability to capture global image features through self-attention, allowing it to excel in identifying subtle patterns. Its higher parameter count enables more complex feature extraction, leading to improved accuracy."
    },
    "mobilenet": {
        "company": "Google",
        "accuracy": 0.769841,
        "f1_score": 0.596600,
        "recall": 0.525886,
        "precision": 0.689286,
        "parameters": "4.2 million",
        "description": "MobileNet is a lightweight CNN architecture designed for mobile devices, providing a balance between accuracy and efficiency.",
        "accuracy_explanation": "With 76.9% accuracy, MobileNet strikes a balance between speed and performance. Its compact design allows efficient computation on mobile devices, but its smaller parameter count limits its ability to capture complex patterns, affecting overall accuracy."
    }
}

PRODUCT_RECOMMENDATIONS = {
    'blackheads': {
        "product": ("Dr. Dennis Gross Alpha Beta Universal Daily Peel", 
                   "Contains lactic acid to resurface pores, AHA and BHA to dissolve sebum. "
                   "Deactivator pad prevents over-exfoliation. Caution: Strong on sensitive skin.",
                   "https://www.sephora.com/product/alpha-beta-universal-daily-peel-P377533"),
        "explanation": "Blackheads are small, dark spots that appear on the skin due to clogged hair follicles. They are a type of mild acne caused by excess oil and dead skin cells.",
        "treatments": "Use products containing salicylic acid or benzoyl peroxide to help unclog pores. Regular exfoliation can also prevent buildup."
    },
    'whiteheads': {
        "product": ("Differin Gel", 
                   "Adapalene-based gel helps with whiteheads, restoring texture and tone. "
                   "Caution: May cause irritation for sensitive skin.", 
                   "https://differin.com/shop/differin-gel"),
        "explanation": "Whiteheads are small, white bumps that form when pores are clogged with oil, dead skin cells, and bacteria. They are a closed form of acne.",
        "treatments": "Use gentle cleansers, avoid picking or squeezing, and opt for products with retinoids or salicylic acid."
    },
    'pustules': {
        "product": ("Mighty Hero Pimple Correct Pen", 
                 "Contains salicylic acid and aloe. Effective on early, deeper pimples. "
                 "Caution: Use 1-3 times daily.", 
                 "https://www.herocosmetics.us/products/pimple-correct"),
        "explanation": "Pustules are inflamed pimples that contain pus. They are usually red at the base with a white or yellowish head.",
        "treatments": "Cleanse the area gently, use spot treatments with benzoyl peroxide or salicylic acid, and avoid popping the pustules."
    },
    'papules': {
        "product": ("Clearasil Rapid Rescue", 
                "Benzoyl peroxide-based. Effective for clearing papules within 4 hours. "
                "Caution: Use as a spot treatment.", 
                "https://www.clearasil.us/products/clearasil-ultra-rapid-action-vanishing-acne-treatment-cream-1-ounce"),
        "explanation": "Papules are small, red bumps that may be tender to touch. They occur when hair follicles are clogged and become inflamed.",
        "treatments": "Avoid touching or picking at the affected area. Use non-comedogenic products and consider topical treatments with benzoyl peroxide."
    },
    'dark spot': {
        "product": ("La Roche Posay MelaB3 Serum", 
                  "Contains Melasyl, effective against discoloration. "
                  "Caution: Use once daily.", 
                  "https://www.amazon.com/dp/B0CM4B43DZ"),
        "explanation": "Dark spots, also known as hyperpigmentation, occur when certain areas of the skin produce more melanin than usual. They can be caused by acne, sun damage, or hormonal changes.",
        "treatments": "Consider products with vitamin C, niacinamide, or retinoids to help lighten dark spots. Always wear sunscreen to prevent further darkening."
    },
    'nodules': {
        "product": ("Consult a dermatologist", 
                "For more severe cases, prescriptions with benzoyl peroxide, salicylic acid, and retinoids are recommended. "
                "Caution: Seek professional advice."),
        "explanation": "Nodules are large, painful lumps under the skin's surface. They are a more severe form of acne that occurs when clogged pores become deeply inflamed.",
        "treatments": "Seek advice from a dermatologist for prescription treatments. Oral medications and topical retinoids are often recommended."
    }
}

# Initialize Processor
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

def classify_acne(image_tensor, model_name):
    model = AutoModelForImageClassification.from_pretrained(f"bestmodels/{model_name}", from_tf=False, use_safetensors=True)
    output = model(image_tensor)
    logits = output.logits.detach().numpy()[0]
    predicted_classes = [CLASS_LABELS[i] for i, val in enumerate(logits) if val > 0.5]
    return predicted_classes

def display_recommendations(prediction):
    if prediction in PRODUCT_RECOMMENDATIONS:
        product, description, url = PRODUCT_RECOMMENDATIONS[prediction]["product"]
        explanation = PRODUCT_RECOMMENDATIONS[prediction]["explanation"]
        treatments = PRODUCT_RECOMMENDATIONS[prediction]["treatments"]
        
        st.markdown(f"### {prediction.capitalize()}")
        st.markdown(f"**Explanation**: {explanation}")
        st.markdown(f"**Treatments**: {treatments}")
        st.write(f"**Recommended Product**: {product}. {description}")
        st.markdown(f"[Product Link]({url})", unsafe_allow_html=True)
    else:
        st.write("**Most Likely**: No skin conditions detected!")
        st.write("Continue your good skin hygiene.")

def display_model_info(model_name):
    if model_name in MODEL_INFO:
        info = MODEL_INFO[model_name]
        st.header("Model Information")
        #st.markdown(f"**Model Name**: {info['company']} {model_name.upper()}")
        st.markdown(f"**Accuracy**: {info['accuracy'] * 100:.2f}%")
        st.markdown(f"**F1 Score**: {info['f1_score']:.2f}")
        st.markdown(f"**Recall**: {info['recall']:.2f}")
        st.markdown(f"**Precision**: {info['precision']:.2f}")
        st.markdown(f"**Parameters**: {info['parameters']}")
        st.markdown(f"**Description**: {info['description']}")
        st.markdown(f"**Why This Accuracy?**: {info['accuracy_explanation']}")
        st.caption("**Accuracy Explanation**: The accuracy indicates how often the model correctly classifies skin conditions based on its training data. While high accuracy suggests reliability, no model is perfect, and professional consultation is advised for severe cases.")

def process_and_display_image(selected_model):
    uploaded_file = st.file_uploader(label='Pick an image to test', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.header("Your Image:")
            st.image(image, caption="Uploaded Image")
        with col2:
            with st.spinner("Processing..."):
                inputs = image_processor(images=image, return_tensors='pt')
                tensor_object = inputs['pixel_values']
                predictions = classify_acne(tensor_object, selected_model)
                show_diagnosis(selected_model, predictions)

        st.header(f"{MODEL_INFO[selected_model]['company']} {selected_model.upper()}")
        display_model_info(selected_model)

def show_diagnosis(model_name, predictions):
    st.header('Diagnosis')
    for prediction in predictions:
        display_recommendations(prediction)
    if not predictions:
        display_recommendations('')

def show_intro():
    # Apply custom CSS to make radio buttons appear in a row
    st.markdown("""
        <style>
        .stRadio > div {
            display: flex;
            flex-direction: row;
            justify-content: flex-start;
        }
        .stRadio > div label {
            margin-right: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    selected_model = None
    with col1:
        st.title('DermaAI 🩺 Your Personalized Skin Advisor ')
        st.subheader('A suite of AI-based skin ailment classifiers leveraging Vision Transformers')
        st.markdown("""
            ### Why Choose DermaAI?
            - **Accurate**: Utilizes state-of-the-art image classification models.
            - **Simple to Use**: Just upload a picture, and we'll handle the rest.
            - **Personalized Recommendations**: Get product suggestions tailored to your specific skin condition.
        """)
        st.header('Acne Classifier')
        
        # Use radio buttons to automatically set the selected model in a row
        
        selected_model = st.radio("Choose Model", MODEL_NAMES, index=0)
        
        
        # Instructions for how to use the app
        st.markdown("### How to Use")
        st.write('Upload an image of your affected skin below. The diagnosis will appear underneath the Diagnosis header.')
        
        
    
    with col2:
        derm_image = Image.open("dermaphoto.jpeg")
        st.image(derm_image, use_column_width=True)
    
    # Process and display image based on the selected model
    if selected_model:
        process_and_display_image(selected_model)

def show_footer():
    st.markdown("""
        ---
        **Watch Our Demo Video**: [Demo Video](https://drive.google.com/file/d/1RdZXmp-HX30BvULKaN3BC7ymNggpLjWt/view?usp=sharing)  
        © 2024 DermaAI Inc. All Rights Reserved.
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide", page_title="DermaAI")
    show_intro()
    show_footer()

if __name__ == '__main__':
    main()
