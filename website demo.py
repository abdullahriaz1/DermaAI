import streamlit as st
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn import metrics
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

from io import StringIO

#from transformers import ViTForImageClassification, MobileViTImageProcessor, SiglipImageProcessor, ConvNextImageProcessor, 
from transformers import AutoModelForImageClassification
from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import AutoImageProcessor, ViTForImageClassification
import transformers
import os
import io
class_labels = ['blackheads', 'dark spot', 'nodules', 'papules', 'pustules', 'whiteheads']
#File Paths
model_names = ["clip", "convnext", "mobilenet", "mobilevit", "resnet", "siglip", "vit"]


#print(model)
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
def predict(img_path, ten, model_name):
    model = AutoModelForImageClassification.from_pretrained(f"bestmodels/{model_name}", from_tf=False, use_safetensors=True)
    output = model(ten)
    logits = output.logits.detach().numpy()[0]
    out = []
    for i in range(len(logits)):
        if logits[i] > 0.5:
            out.append(i)
            
    predicted_classes = [class_labels[i] for i in out]
    return predicted_classes

img = "01F3MMV6NBB013AEV2SH04SRQ9_jpeg.rf.a88bc38371062a738c5591586ce56307.jpg"
#print(predict(img))
#backend stuff... 
def response(prediction):
    if prediction == 'blackheads':
        st.write("Most Likely: Blackheads")
        st.write("Recommended Product: Dr. Dennis Gross Alpha Beta Universal Daily Peel. Contains Lactice acid to resurface pores, AHA and BHAs to disovle sebum and prevent oil buildup, and deactivator pad prevents over exfoliation")
        st.write("Caution: This product can be too strong on sensitive skin")
        url = "https://www.sephora.com/product/alpha-beta-universal-daily-peel-P377533?om_mmc=aff-linkshare-redirect-TnL5HPStwNw&c3ch=Linkshare&c3nid=TnL5HPStwNw&affid=TnL5HPStwNw-B9IdshF_Yz3SrxWtw1cDGA&ranEAID=TnL5HPStwNw&ranMID=2417&ranSiteID=TnL5HPStwNw-B9IdshF_Yz3SrxWtw1cDGA&ranLinkID=10-1&browserdefault=true"
        st.write("Check out this [link](%s)" % url)
    elif prediction == 'whiteheads':
        st.write("Most Likely: Whiteheads")
        st.write("Recommended Product: Differin Gel. An adaplane based gel has been clinicaly proven to help with whiteheads and restore orginal texture and tone to your skin")
        st.write("Caution: Can cause skin irritation to those with sensitive or dry skin")
        url = "https://differin.com/shop/differin-gel"
        st.write("Check out this [link](%s)" % url)
    elif prediction == 'pustules':
        st.write("Most Likely: Pustules")
        st.write("Recommended Product: Mighty Hero Pimple Correct Pen. These are often early, deeper pimples and this product can help remove them and the pain faster. It contains saylic acid which is a well known ingredient in acne medication that can help calm blemishes. Infused with Aloe to help prevent irritation on sensitive skin")
        st.write("Caution: Use 1-3x daily")
        url = "https://www.herocosmetics.us/products/pimple-correct?utm_source=google&utm_medium=shopping&utm_campaign=13073725863&gad_source=1&gclid=Cj0KCQjwsPCyBhD4ARIsAPaaRf1Rx61s3IzqbeothFHj3hoIfdONDDgHlnGt_8vOGHDATg1XLe43xJoaAoENEALw_wcB"
        st.write("Check out this [link](%s)" % url)
    elif prediction == 'papules':
        st.write("Most Likely: Papules")
        st.write("Recommended Product: Clearasil Rapid Rescue. Benzol-peroxide is most effective midication for papules. Helps calm and rid of pimples within 4 hours ")
        st.write("Caution: Use as spot treatment, not preventative treatment")
        url = "https://www.clearasil.us/products/clearasil-ultra-rapid-action-vanishing-acne-treatment-cream-1-ounce"
        st.write("Check out this [link](%s)" % url)
    elif prediction == 'dark spot':
        st.write("Most Likely: Dark spots")
        st.write("Recommended Product: La Roche Posay MelaB3 Serum. Conatins new ingrident, Melasyl, clinically proven to help with discoloration")
        st.write("Caution: Only use once daily")
        url = "https://www.amazon.com/dp/B0CM4B43DZ?ots=1&tag=thestrategistsite-20&ascsubtag=__st0602awd__cltzu71ay00000pdy8a3vbr1q__231589________1________google.com"
        st.write("Check out this [link](%s)" % url)
    elif prediction == 'nodules':
        st.write("Most Likely: Nodular acne")
        st.write("Recommended Product: Speak to dermotolgosit to recieve perscriptions for stronger dosage of benzol-peroxide, sailyic acid, and retnoids")
        st.write("Caution: A more severe form of acne, contact dermotologists for more expertise")
    else:
        st.write("Most Likely: We did not detect any skin conditions!")
        st.write("Continue your good skin hygenie")


def load_image(selected_model):
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image)

        # Process the image
        inputs = processor(images=image, return_tensors='pt')
        tensorObject = inputs['pixel_values']
        
        # Make predictions
        predictions = predict('', tensorObject, selected_model)
        for i in predictions:
            st.write(f"Predicted: {i}")
        st.header('Diagnosis ')
        for i in predictions:
            response(i)
        if len(predictions) == 0:
            response('')

def main():
    st.title('Image upload demo')
    
    st.title('Acne Detection')
    selected_model = st.selectbox("Choose Model", model_names)


    with st.title("How to use the website"):
        st.write('Upload an image of your affected skin at the top. The diagnosis will appear underneath the Diagnosis header. Read more about the process under "About the Algorithm"')
        
    st.header('About')
    st.write('About 50 million people in the US suffer from Acne. It is easy to spot but diffcult to differenatie between the many skin conditions that fall under the acne umbrella. We have developed a diagnositic tool that will help identify exactly what conditions you may have and recommmend product based on our results')
    preds = load_image(selected_model)

if __name__ == '__main__':
    main()
