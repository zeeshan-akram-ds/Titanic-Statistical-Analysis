# Titanic Survival Prediction & Statistical Analysis

This project is a comprehensive case study on the famous Titanic dataset. It combines **statistical analysis**, **feature engineering**, **machine learning**, and an interactive **Streamlit app** to explore passenger survival predictions.

 **Live App**: [Titanic Prediction App](https://titanic-statistical-analysis-prediction-hok7vkjvuqgbjce6vxpbaa.streamlit.app/)  
 **Dataset**: [`titanicdataset.csv`](https://github.com/zeeshan-akram-ds/Titanic-Statistical-Analysis-prediction/blob/main/titanicdataset.csv)

---

## Project Objectives

- Conduct detailed statistical analysis to explore patterns in survival.
- Apply hypothesis testing and ANOVA to examine dependencies.
- Perform preprocessing & feature engineering.
- Build a machine learning model (Random Forest) to predict survival.
- Deploy a Streamlit app with interactive visualization and prediction interface.

---

## Statistical Analysis

- **Univariate, Bivariate & Multivariate** analysis using Seaborn, Matplotlib, and crosstab.
- **Chi-Square Tests** on categorical features (Sex, Embarked, Pclass vs Survived).
- **One-Way ANOVA**: Fare vs Pclass, Age vs Survived.
- **Tukey HSD** post-hoc test to compare mean fares between classes.
- Insights from all visualizations integrated into the app.

---

## Feature Engineering

- Created `family_size` (SibSp + Parch + 1).
- Created `is_alone` binary feature.
- One-hot encoding for `Embarked`.
- Scaling applied on `Fare` and `Age`.

---

## Machine Learning

- **Model Used**: Random Forest Classifier  
- **Final Accuracy**: ~84%
- **Important Features**:
  - `Fare`, `Age`, `Sex`, `Pclass`, `Family Size`, `Is Alone`, `Embarked (Q/S)`
- Model evaluated using classification report, confusion matrix, and feature importance.

---

## Streamlit App Features

- Predict survival using user input for all relevant features.
- Real-time model output with clear probability and prediction.
- Interactive plots:
  - Univariate (Age, Fare, etc.)
  - Bivariate (Survived vs Sex/Pclass/etc.)
  - Multivariate analysis selections.
- Clean, structured UI with dropdowns and sliders for better UX.

---

## Installation & Usage

1. **Clone the Repo**
   ```bash
   git clone https://github.com/zeeshan-akram-ds/Titanic-Statistical-Analysis-prediction.git
   cd Titanic-Statistical-Analysis-prediction  
2. **Install Requirements**  
   - pip install -r requirements.txt  
3. **Run Streamlit App**  
   - streamlit run titanic.py  
## Requirements  
See [requirements.txt](https://github.com/zeeshan-akram-ds/Titanic-Statistical-Analysis-prediction/blob/main/requirements.txt) for full list. Key libraries:  
pandas, numpy  
seaborn, matplotlib, plotly  
sklearn, statsmodels, scipy  
streamlit  
## Acknowledgments  
Thanks to the Kaggle Titanic Dataset and Streamlit for enabling interactive data apps!
