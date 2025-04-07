import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import chi2_contingency, f_oneway

st.set_page_config(page_title="Titanic Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
@st.cache_data
def load_data():
    url = "titanicdataset.csv"
    df = pd.read_csv(url)
    df.columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 
                  'Ticket', 'Fare', 'Cabin', 'Embarked']
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop('Cabin', axis=1, inplace=True)
    return df

df = load_data()

def feature_engineering(df):
    df_eng = df.copy()
    df_eng['FamilySize'] = df_eng['SibSp'] + df_eng['Parch'] + 1
    df_eng['IsAlone'] = (df_eng['FamilySize'] == 1).astype(int)
    df_eng['Title'] = df_eng['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df_eng['Title'] = df_eng['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 
                                               'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_eng['Title'] = df_eng['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df_eng['Title'] = df_eng['Title'].replace('Mme', 'Mrs')
    df_eng['Sex'] = df_eng['Sex'].map({'male': 0, 'female': 1})
    df_eng['Embarked'] = df_eng['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df_eng['Title'] = df_eng['Title'].map(title_mapping).fillna(0)
    return df_eng
@st.cache_resource
def train_model(df):
    df_eng = feature_engineering(df)
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']
    X = df_eng[features]
    y = df_eng['Survived']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model, scaler, features, X_train, X_test, y_train, y_test

model, scaler, feature_cols, X_train, X_test, y_train, y_test = train_model(df)
st.sidebar.title("Titanic Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Home", "Data Exploration", "Statistical Analysis", "Model Prediction"])
st.sidebar.markdown("---")
st.sidebar.write("Created by Zeeshan Akram | 2025")
if page == "Home":
    st.title("Titanic Analysis Dashboard")
    st.markdown("### Explore the Titanic Dataset with Advanced Analytics")
    st.write("""
    This dashboard offers:
    - **Data Exploration**: Insightful univariate, bivariate, and multivariate visualizations
    - **Statistical Analysis**: Chi-square and ANOVA tests
    - **Prediction**: RandomForest-based survival prediction
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Passengers", len(df), delta_color="off")
    with col2:
        st.metric("Survival Rate", f"{(df['Survived'].mean()*100):.1f}%", delta_color="off")
    with col3:
        st.metric("Average Fare", f"${df['Fare'].mean():.2f}", delta_color="off")
elif page == "Data Exploration":
    st.title("Data Exploration")
    st.markdown("### Uncover Patterns in the Titanic Dataset")
    
    st.subheader("Univariate Analysis")
    st.write("Analyze the distribution of individual variables.")
    uni_var = st.selectbox("Select Variable", ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], key="uni_var")
    
    if uni_var in ['Age', 'Fare']:
        fig = px.histogram(df, x=uni_var, color='Survived', histnorm='density', nbins=30,
                          marginal="violin", opacity=0.7,
                          color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        fig.update_layout(bargap=0.1, title=f"{uni_var} Distribution by Survival",
                         xaxis_title=uni_var, yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)
    elif uni_var in ['Pclass', 'Sex', 'Embarked']:
        fig = px.histogram(df, x=uni_var, color='Survived', barmode='group',
                          color_discrete_sequence=['#2ca02c', '#d62728'])
        fig.update_layout(title=f"{uni_var} Distribution by Survival",
                         xaxis_title=uni_var, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.histogram(df, x=uni_var, color='Survived', barmode='stack',
                          color_discrete_sequence=['#9467bd', '#e377c2'])
        fig.update_layout(title=f"{uni_var} Distribution by Survival",
                         xaxis_title=uni_var, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Bivariate Analysis")
    st.write("Explore relationships between pairs of variables.")
    x_var = st.selectbox("X Variable", ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], key="x_var")
    y_var = st.selectbox("Y Variable", ['Age', 'Fare', 'SibSp', 'Parch'], key="y_var")
    
    if x_var in ['Age', 'Fare'] and y_var in ['Age', 'Fare']:
        fig = px.scatter(df, x=x_var, y=y_var, color='Survived', size='Pclass',
                        hover_data=['Sex', 'Embarked'], opacity=0.6,
                        color_discrete_sequence=['#17becf', '#bcbd22'])
        fig.update_layout(title=f"{x_var} vs {y_var} by Survival",
                         xaxis_title=x_var, yaxis_title=y_var)
        st.plotly_chart(fig, use_container_width=True)
    elif x_var in ['Pclass', 'Sex', 'Embarked']:
        fig = px.box(df, x=x_var, y=y_var, color='Survived',
                    color_discrete_sequence=['#8c564b', '#e377c2'])
        fig.update_layout(title=f"{x_var} vs {y_var} by Survival",
                         xaxis_title=x_var, yaxis_title=y_var)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.violin(df, x=x_var, y=y_var, color='Survived', box=True,
                       color_discrete_sequence=['#ff9896', '#c5b0d5'])
        fig.update_layout(title=f"{x_var} vs {y_var} by Survival",
                         xaxis_title=x_var, yaxis_title=y_var)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Multivariate Analysis")
    st.write("Visualize interactions between three variables.")
    x_multi = st.selectbox("X Axis", ['Age', 'Fare'], key="x_multi")
    y_multi = st.selectbox("Y Axis", ['Fare', 'Age'], key="y_multi")
    color_multi = st.selectbox("Color By", ['Pclass', 'Sex', 'Survived'], key="color_multi")
    
    fig = px.scatter(df, x=x_multi, y=y_multi, color=color_multi, size='SibSp',
                    hover_data=['Parch', 'Embarked'], facet_col='Embarked',
                    color_discrete_sequence=px.colors.qualitative.D3)
    fig.update_layout(title=f"{x_multi} vs {y_multi} by {color_multi} (Faceted by Embarked)",
                     xaxis_title=x_multi, yaxis_title=y_multi)
    st.plotly_chart(fig, use_container_width=True)
elif page == "Statistical Analysis":
    st.title("Statistical Analysis")
    st.markdown("### Statistical Tests and Correlations")
    
    st.subheader("Chi-square Test")
    st.write("Test independence between categorical variables and survival.")
    cat_var = st.selectbox("Categorical Variable", ['Pclass', 'Sex', 'Embarked'], key="cat_var")
    contingency_table = pd.crosstab(df[cat_var], df['Survived'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    st.write(f"**{cat_var} vs Survived**")
    st.write(f"Chi-square Statistic: {chi2:.2f}")
    st.write(f"P-value: {p:.4f}")
    st.write(f"Degrees of Freedom: {dof}")
    st.write("Result:", "Significant (p < 0.05)" if p < 0.05 else "Not Significant (p >= 0.05)")
    
    st.subheader("ANOVA Test")
    st.write("Test differences in means across survival groups.")
    num_var = st.selectbox("Numerical Variable", ['Age', 'Fare', 'SibSp', 'Parch'], key="num_var")
    survived_0 = df[df['Survived'] == 0][num_var]
    survived_1 = df[df['Survived'] == 1][num_var]
    f_stat, p_val = f_oneway(survived_0, survived_1)
    st.write(f"**{num_var} across Survival**")
    st.write(f"F-Statistic: {f_stat:.2f}")
    st.write(f"P-value: {p_val:.4f}")
    st.write("Result:", "Significant (p < 0.05)" if p_val < 0.05 else "Not Significant (p >= 0.05)")
    
    st.subheader("Correlation Analysis")
    st.write("Correlation between numeric and engineered features.")
    df_eng = feature_engineering(df)
    numeric_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                    'FamilySize', 'IsAlone', 'IsAlone', 'Title']
    corr_df = df_eng[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_df, annot=True, cmap='RdBu', center=0, fmt='.2f', linewidths=0.5, ax=ax)
    plt.title("Correlation Heatmap", fontsize=16, pad=20)
    st.pyplot(fig)
elif page == "Model Prediction":
    st.title("Survival Prediction")
    st.markdown("### Predict Survival with RandomForest")
    
    st.subheader("Model Performance")
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        st.write(f"**Accuracy**: {accuracy:.2f}")
        st.markdown(f"**Classification Report**:\n```\n{report}\n```")
    except Exception as e:
        st.error(f"Error in model performance calculation: {str(e)}")
        st.write("This might indicate an issue with model training or test data. Please check the data loading and training steps.")
    
    st.subheader("Make a Prediction")
    st.write("Enter passenger details to predict survival probability.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.slider("Age", 0, 100, 30, help="Passenger's age in years")
            family_size = st.number_input("Family Size", 1, 20, 1, 
                                        help="Total number of family members aboard (including the passenger)")
            is_alone = st.selectbox("Is Alone?", [1, 0], 
                                  help="1 = Traveling alone, 0 = Traveling with family")
        with col2:
            fare = st.number_input("Fare", 0.0, 1000.0, 32.0, step=0.1, help="Ticket fare in pounds")
            embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], 
                                  help="S = Southampton, C = Cherbourg, Q = Queenstown")
            title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"], 
                                help="Extracted from passenger name")
        
        submit = st.form_submit_button("Predict Survival")
    
    if submit:
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [0 if sex == "male" else 1],
            'Age': [age],
            'Fare': [fare],
            'Embarked': [0 if embarked == "S" else 1 if embarked == "C" else 2],
            'FamilySize': [family_size],
            'IsAlone': [is_alone],
            'Title': [{"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}[title]]
        })
        
        if list(input_data.columns) != feature_cols:
            st.error(f"Feature mismatch! Expected: {feature_cols}, Got: {list(input_data.columns)}")
        else:
            try:
                input_scaled = scaler.transform(input_data)
                
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                
                st.markdown("---")
                st.success(f"**Prediction**: {'Survived' if prediction == 1 else 'Did Not Survive'}")
                st.info(f"**Survival Probability**: {probability:.2%}")
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                st.write("This might indicate an issue with the scaler or model. Please check the training step.")
