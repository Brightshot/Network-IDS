# Network Intrusion Detection System (IDS)

A comprehensive machine learning-based Intrusion Detection System that identifies and classifies network attacks using the NSL-KDD dataset. This project implements and compares three different machine learning models: Random Forest, XGBoost, and LSTM (Long Short-Term Memory) neural networks.

## Table of Contents

- [What is This Project?](#what-is-this-project)
- [Why This Project?](#why-this-project)
- [How It Works](#how-it-works)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contribution](#contribution)

## What is This Project?

This project is a **Network Intrusion Detection System (IDS)** that uses machine learning algorithms to automatically detect and classify network attacks. It analyzes network traffic patterns and identifies malicious activities such as:

- **DoS (Denial of Service)**: Attacks that overwhelm network resources
- **Probe**: Scanning attacks that gather information about network vulnerabilities
- **R2L (Remote to Local)**: Unauthorized access attempts from remote machines
- **U2R (User to Root)**: Privilege escalation attacks

The system processes network connection data, extracts meaningful features, and uses trained models to distinguish between normal network traffic and potential security threats.

## Why This Project?

### The Problem

In today's digital landscape, network security is paramount. Organizations face constant threats from:
- Increasingly sophisticated cyber attacks
- Large volumes of network traffic that are impossible to monitor manually
- The need for real-time threat detection
- High false positive rates in traditional rule-based systems

### The Solution

This project addresses these challenges by:

1. **Automated Detection**: Uses machine learning to automatically identify threats without manual rule configuration
2. **High Accuracy**: Achieves 95%+ accuracy in detecting intrusions
3. **Multiple Model Comparison**: Implements three different algorithms to find the best approach
4. **Comprehensive Analysis**: Provides detailed metrics and visualizations for model evaluation
5. **Scalability**: Can be adapted to process large volumes of network traffic

### Real-World Applications

- **Enterprise Security**: Protect corporate networks from unauthorized access
- **Data Centers**: Monitor and secure server infrastructure
- **IoT Security**: Detect attacks on Internet of Things devices
- **Cloud Security**: Enhance security in cloud computing environments
- **Research**: Study network attack patterns and develop better defense mechanisms

## How It Works

### 1. Data Preprocessing

The project follows a systematic approach to prepare the NSL-KDD dataset:

#### **Data Loading**
- Loads network connection records with 42 features including:
  - Connection duration, protocol type, service type
  - Source/destination bytes, flags
  - Error rates, connection counts
  - Host-based statistics

#### **Categorical Encoding**
- Converts categorical features (protocol_type, service, flag) to numerical values using Label Encoding
- Preserves the original label information for attack classification

#### **Label Grouping**
- Groups 22 different attack types into 4 main categories:
  - **DoS**: apache2, back, land, neptune, mailbomb, pod, processtable, smurf, teardrop, udpstorm, worm
  - **Probe**: ipsweep, mscan, nmap, portsweep, saint, satan
  - **R2L**: ftp_write, guess_passwd, httptunnel, imap, multihop, named, phf, sendmail, snmpgetattack, snmpguess, spy, warezclient, warezmaster, xlock, xsnoop
  - **U2R**: buffer_overflow, loadmodule, perl, ps, rootkit, sqlattack, xterm

#### **Binary Classification**
- Creates a binary target variable: `intrusion` (0 = normal, 1 = attack)
- Enables binary classification while maintaining multi-class attack information

#### **Feature Scaling**
- Applies StandardScaler to normalize numerical features
- Ensures all features are on the same scale for optimal model performance

### 2. Feature Selection

- **Correlation Analysis**: Calculates correlation between features and the target variable
- **High-Correlation Selection**: Selects features with correlation > 0.5 with the intrusion label
- **Dimensionality Reduction**: Reduces from 42 features to ~10 most relevant features
- **Visualization**: Creates correlation heatmaps to visualize feature relationships

### 3. Model Implementation

The project implements three different machine learning models:

#### **Random Forest Classifier**
- **Algorithm**: Ensemble learning using multiple decision trees
- **Advantages**: 
  - Handles non-linear relationships
  - Provides feature importance scores
  - Robust to overfitting
- **Configuration**: 100 estimators, random state for reproducibility

#### **XGBoost Classifier**
- **Algorithm**: Gradient boosting framework
- **Advantages**:
  - High performance and accuracy
  - Built-in regularization
  - Handles missing values
- **Configuration**: Optimized for binary classification with logloss metric

#### **LSTM Neural Network**
- **Algorithm**: Deep learning with Long Short-Term Memory layers
- **Architecture**:
  - LSTM layer (64 units) with ReLU activation
  - Dropout layer (0.2) for regularization
  - Dense layer (32 units)
  - Output layer with sigmoid activation for binary classification
- **Training**: 10 epochs with validation split, batch size of 64
- **Advantages**: Can capture temporal patterns in network traffic

### 4. Model Evaluation

Comprehensive evaluation includes:

#### **Performance Metrics**
- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to all positive predictions
- **Recall**: Ratio of true positives to all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

#### **Visualizations**
- **Confusion Matrices**: Shows true positives, false positives, true negatives, false negatives
- **ROC Curves**: Compares model performance across different thresholds
- **Feature Importance**: Displays which features each model considers most important
- **Training History**: LSTM training accuracy and loss over epochs
- **Training Time Comparison**: Efficiency analysis of all models

### 5. Results Comparison

The system provides:
- Side-by-side comparison of all three models
- Performance metrics table
- Training time analysis
- Detailed classification reports
- Visual comparison charts

## Dataset

### NSL-KDD Dataset

The project uses the **NSL-KDD** dataset, an improved version of the KDD Cup 1999 dataset:

- **Training Set**: 125,973 network connection records
- **Test Set**: 31,494 network connection records
- **Features**: 42 attributes per connection
- **Classes**: Normal traffic + 22 attack types grouped into 4 categories

### Dataset Structure

- **Basic Features**: duration, protocol_type, service, flag, src_bytes, dst_bytes
- **Content Features**: hot, num_failed_logins, logged_in, num_compromised
- **Time-based Features**: count, srv_count, serror_rate, rerror_rate
- **Host-based Features**: dst_host_count, dst_host_srv_count, dst_host_serror_rate

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Network-IDS
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirement.txt
```

### Key Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **matplotlib & seaborn**: Data visualization
- **scikit-learn**: Machine learning algorithms (Random Forest)
- **xgboost**: Gradient boosting framework
- **tensorflow/keras**: Deep learning (LSTM)
- **jupyter**: Interactive notebook environment

## Usage

### Running the Notebook

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the Main Notebook**:
   - Navigate to `main.ipynb`
   - Execute cells sequentially from top to bottom

3. **Data Preparation**:
   - Ensure the NSL-KDD dataset is in the `Data/` directory
   - Files needed: `KDDTrain+.txt`


## Project Structure

```
Network-IDS/
│
├── Data/
│   ├── KDDTrain+.txt          # Training dataset
│
├── main.ipynb                  # Main Jupyter notebook
│
├── requirement.txt             # Python dependencies
├── README.md                   # This file
└── LICENSE                     # License information
```

## Results

### Model Performance Summary

Based on the evaluation, the models achieve the following performance:

| Model | Accuracy | Precision | Recall | F1 Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Random Forest** | ~97% | ~97% | ~96% | ~97% | ~5-6 seconds |
| **XGBoost** | ~97% | ~98% | ~97% | ~97% | ~2-3 seconds |
| **LSTM** | ~95% | ~96% | ~94% | ~95% | ~30-40 seconds |

### Key Findings

1. **XGBoost** achieves the highest overall accuracy and precision
2. **Random Forest** provides excellent balance between accuracy and interpretability
3. **LSTM** captures temporal patterns but requires more training time
4. **XGBoost** is the most efficient in terms of training time vs. performance
5. All models successfully identify normal vs. attack traffic with high confidence

## 🛠️ Technologies Used

### Machine Learning Libraries
- **scikit-learn** (v1.8.0): Random Forest implementation
- **XGBoost** (v3.1.3): Gradient boosting classifier
- **TensorFlow/Keras** (v2.20.0): Deep learning framework for LSTM

### Data Processing
- **pandas** (v2.3.3): Data manipulation and analysis
- **numpy** (v2.4.0): Numerical computations

### Visualization
- **matplotlib** (v3.10.8): Plotting and visualization
- **seaborn** (v0.13.2): Statistical data visualization

### Development Environment
- **Jupyter Notebook**: Interactive development and analysis
- **Python 3.8+**: Programming language

## Contribution
Fazaz soliu 2022/11367



**Note**: This project is for educational and research purposes. For production deployment, additional considerations such as model retraining, monitoring, and security hardening should be implemented.
