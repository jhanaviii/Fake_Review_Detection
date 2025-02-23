# Fake Review Detection AI Agent

## Overview

The **Fake Review Detection AI Agent** is an advanced **AI-powered system** designed to **combat deceptive online reviews** using a state-of-the-art **DistilBERT-based deep learning model**. This intelligent agent **analyzes textual patterns**, understands sentiment, and **distinguishes between real and fake reviews** with **high accuracy**. Ideal for **e-commerce platforms, businesses, and researchers**, this tool ensures **authenticity, credibility, and trust** in online content.  


## 🔥 Key Features  

### ✅ **AI-Powered Fake Review Detection**  
Leverages **cutting-edge Natural Language Processing (NLP) and Transformers** to identify deceptive reviews with **high precision**.  

### 🚀 **End-to-End Automation**  
Automatically **processes, analyzes, and classifies** reviews—no manual intervention required.  

### 🔗 **Scalable & Fast API Integration**  
Built with **Flask**, enabling seamless integration with **web platforms, mobile apps, and business analytics tools**.  

### 📊 **Advanced Reporting & Insights**  
Generates **detailed classification reports**, providing **data-driven insights** on fake vs. genuine reviews.  

### 🔬 **Customizable & Extensible**  
Fine-tune the model on **custom datasets** for improved accuracy in domain-specific applications.  

### 🛡️ **Ethical AI for a Trustworthy Web**  
Promotes **authenticity and fairness** in online content by ensuring **genuine user feedback prevails**.  


## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/jhanaviii/Fake_Review_Detection.git
cd Fake_Review_Detection
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

## Dataset Acquisition

This project utilizes the https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset from Kaggle. Follow these steps to download the dataset:

### 1. Install Kaggle API
```bash
pip install kaggle
```

### 2. Authenticate with Kaggle
- Log in to your [Kaggle](https://www.kaggle.com/) account.
- Go to **Account** > **API** > Click **Create New API Token**.
- This will download a `kaggle.json` file.
- Move the `kaggle.json` file to `~/.kaggle/` (Linux/Mac) or `C:\Users\<YourUsername>\.kaggle\` (Windows).

### 3. Download the Dataset
```bash
kaggle datasets download -d mexwell/fake-reviews-dataset
unzip fake-reviews-dataset.zip -d data/
```

Ensure the dataset is in CSV format with 'text' and 'label' columns inside the `data/` folder.

## Usage

### Training the Model
```bash
python train.py --data_path data/your_dataset.csv
```
Replace `your_dataset.csv` with the actual dataset filename.

### Evaluating the Model
```bash
python evaluate.py --data_path data/your_dataset.csv
```
This will output a classification report and save results in `evaluation_results.csv`.

### Running the API Server
```bash
python web2.py
```

### Making Predictions
With the server running, send a POST request:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"review": "Your review text here"}'
```
The AI agent will respond with the classification result.

## Dependencies

- **Python 3.8+**
- **Transformers**
- **Torch**
- **Flask**
- **Pandas**
- **Scikit-learn**

