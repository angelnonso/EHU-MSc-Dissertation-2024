# Predictive Modelling for Dementia Diagnosis: An Integrative Approach Using Machine Learning and Multi-Dataset Analysis

## Author
Chinonso Uche\
Supervisor: Professor Ella Pereira

## Project Overview
*This project explores the application of machine learning techniques to predict dementia using multiple datasets. By integrating OASIS Cross-Sectional, OASIS Longitudinal, and ADNI datasets, the study aims to enhance diagnostic accuracy and assess age-based classification impacts.*

## 📊 Datasets Used
- oasis_cross-sectional.csv: MRI and cognitive test results from cross-sectional studies.
- oasis_longitudinal.csv: Longitudinal data tracking cognitive decline over time.
- ADNI.csv: Imaging and clinical assessments from Alzheimer's Disease Neuroimaging Initiative.

## 📌Technologies Used
- Python (Pandas, NumPy, Scikit-learn, TensorFlow, Seaborn, Matplotlib, Plotly)
- Jupyter Notebook
- Microsoft Excel 365
- Agile Project Management

## Methodology
1. Data Preprocessing: Handling missing values, feature scaling, and encoding categorical variables.
2. Feature Engineering: Selecting key predictors using statistical and machine learning techniques.
3. Model Training: Implementing multiple algorithms including:
    - Random Forest Classifier
    - Support Vector Machine (SVM)
    - Logistic Regression
    - Gradient Boosting Classifier
    - Naive Bayes
    - Decision Tree Classifier
    - Multi-layer Perceptron (MLP)
4. Evaluation: Comparing models using Accuracy, F1-score, Precision, and Recall metrics.
5. Results Interpretation: Visualizing performance metrics using Power BI and Matplotlib.

## 📂 Repository Structure
```bash
├── datasets/
│   ├── oasis_cross-sectional.csv
│   ├── oasis_longitudinal.csv
│   ├── ADNI.csv
│
├── reports/
│   ├── EP25830651IPR.pdf (Interim Project Report)
│   ├── EP25830651ECL.pdf (Ethical Checklist)
│   ├── EP25830651final.pdf (Final Report)
│
├── src/
│   ├── Code Artifact - ML Dementia Multi-Dataset and Age-Class Analysis.ipynb
│
├── media/
│   ├── demo_video.webm 📹 (not available due to file size)
│
├── docs/
│   ├── Project Management -Agile Gantt chart.xlsx
│
├── README.md
```

## Installation & Setup
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```
## 🚀 Running the Project
1. Clone the repository:
```bash
git clone https://github.com/angelnonso/EHU-MSc-Dissertation-2024.git
```
2. Navigate to the directory:
```bash
cd EHU-MSc-Dissertation-2024
```
3. Open the Jupyter Notebook:
```bash
jupyter notebook "src/Code Artifact - ML Dementia Multi-Dataset and Age-Class Analysis.ipynb"
```

## Results & Findings
The project demonstrates the effectiveness of ensemble learning techniques in predicting dementia diagnoses. Key findings include:
- Feature importance analysis showed that hippocampus volume and cognitive test scores are strong predictors.
- The Random Forest Classifier & SVM outperformed other models with an accuracy of 95% (for 0-64 age group) and 96% (for 65+ age group).
- Age-grouped analysis revealed performance variation across different demographic categories.

## Future Work
Extend the model to incorporate deep learning techniques.
Improve interpretability through SHAP value analysis.
Deploy the model as an interactive web application.

## 📜 License
This project is open-source and available under the MIT License.

## Contact
For inquiries or collaboration opportunities, reach out via [LinkedIn](https://www.linkedin.com/in/angelnonso/).
