# Derma AI

Your Personalized Skin Advisor
A suite of AI-based skin ailment classifiers leveraging deep learning with new techniques, including Vision Transformers

## Components

- Training Framework
  - Use generate_bestmodels.ipynb to create a folder with your models (default folder is bestmodels).
  1. Open Google Colab
  2. Upload and open generate_bestmodels.ipynb
  3. Upload lump.zip to the /content folder
  4. Connect to a runtime with a GPU
  5. Adjust the models in the model list with your preferred ViTs
  6. Run the notebook

- Web App
  - Creates a web app that can diagnose skin ailments and recommend skincare products
  - Check Out the offical website at [https://derma-ai-pro.streamlit.app/](https://derma-ai-pro.streamlit.app)
  1. If you ran generate_bestmodels.ipynb and used models that are different from default, make sure to change the names in app.py
  2. In the directory where app.py is located, run:

    ```
    streamlit run app.py
    ```

  3. Select model
  4. Upload a photo of your face to view results
