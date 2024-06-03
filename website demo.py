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

st.sidebar.header('Upload Image here')
st.sidebar.subheader('Attempt to take photo in clear lighting from the side as best as possible')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)


#backend stuf... 
prediciton = ''

st.title('Acne Detection')

with st.beta_expander("How to use the website"):
    st.write('Upload an image of your affected skin at the top. The diagnosis will appear underneath the Diagnosis header. Read more about the process under "About the Algorithm"')
    
st.header('About')
st.write('About 50 million people in the US suffer from Acne. It is easy to spot but diffcult to differenatie between the many skin conditions that fall under the acne umbrella. We have developed a diagnositic tool that will help identify exactly what conditions you may have and recommmend product based on our results')

st.header('Diagnosis ')
if prediciton == ' blackheads':
    st.write("Most Likely: Blackheads")
    st.write("Recommended Product: Dr. Dennis Gross Alpha Beta Universal Daily Peel. Contains Lactice acid to resurface pores, AHA and BHAs to disovle sebum and prevent oil buildup, and deactivator pad prevents over exfoliation")
    st.write("Caution: This product can be too strong on sensitive skin")
    url = "https://www.sephora.com/product/alpha-beta-universal-daily-peel-P377533?om_mmc=aff-linkshare-redirect-TnL5HPStwNw&c3ch=Linkshare&c3nid=TnL5HPStwNw&affid=TnL5HPStwNw-B9IdshF_Yz3SrxWtw1cDGA&ranEAID=TnL5HPStwNw&ranMID=2417&ranSiteID=TnL5HPStwNw-B9IdshF_Yz3SrxWtw1cDGA&ranLinkID=10-1&browserdefault=true"
    st.write("Check out this [link](%s)" % url)
elif prediciton == ' whiteheads':
    st.write("Most Likely: Whiteheads")
    st.write("Recommended Product: Differin Gel. An adaplane based gel has been clinicaly proven to help with whiteheads and restore orginal texture and tone to your skin")
    st.write("Caution: Can cause skin irritation to those with sensitive or dry skin")
    url = "https://differin.com/shop/differin-gel"
    st.write("Check out this [link](%s)" % url)
elif prediciton == ' pustules':
    st.write("Most Likely: Pustules")
    st.write("Recommended Product: Mighty Hero Pimple Correct Pen. These are often early, deeper pimples and this product can help remove them and the pain faster. It contains saylic acid which is a well known ingredient in acne medication that can help calm blemishes. Infused with Aloe to help prevent irritation on sensitive skin")
    st.write("Caution: Use 1-3x daily")
    url = "https://www.herocosmetics.us/products/pimple-correct?utm_source=google&utm_medium=shopping&utm_campaign=13073725863&gad_source=1&gclid=Cj0KCQjwsPCyBhD4ARIsAPaaRf1Rx61s3IzqbeothFHj3hoIfdONDDgHlnGt_8vOGHDATg1XLe43xJoaAoENEALw_wcB"
    st.write("Check out this [link](%s)" % url)
elif prediciton == ' papules':
    st.write("Most Likely: Papules")
    st.write("Recommended Product: Clearasil Rapid Rescue. Benzol-peroxide is most effective midication for papules. Helps calm and rid of pimples within 4 hours ")
    st.write("Caution: Use as spot treatment, not preventative treatment")
    url = "https://www.clearasil.us/products/clearasil-ultra-rapid-action-vanishing-acne-treatment-cream-1-ounce"
    st.write("Check out this [link](%s)" % url)
elif prediciton == ' dark spots':
    st.write("Most Likely: Dark spots")
    st.write("Recommended Product: La Roche Posay MelaB3 Serum. Conatins new ingrident, Melasyl, clinically proven to help with discoloration")
    st.write("Caution: Only use once daily")
    url = "https://www.amazon.com/dp/B0CM4B43DZ?ots=1&tag=thestrategistsite-20&ascsubtag=__st0602awd__cltzu71ay00000pdy8a3vbr1q__231589________1________google.com"
    st.write("Check out this [link](%s)" % url)
elif prediciton == ' nodules':
    st.write("Most Likely: Nodular acne")
    st.write("Recommended Product: Speak to dermotolgosit to recieve perscriptions for stronger dosage of benzol-peroxide, sailyic acid, and retnoids")
    st.write("Caution: A more severe form of acne, contact dermotologists for more expertise")
else :
    st.write("Most Likely: We did not detect any skin conditions!")
    st.write("Continue your good skin hygenie")
