import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load Datasets
@st.cache_data
def load_data():
    """
    Load the policy and spending data from CSV files.
    """
    policy_data = pd.read_csv("data/insurance_policies_dataset.csv")
    spending_data = pd.read_csv("data/transactions.csv")
    return policy_data, spending_data

policy_data, spending_data = load_data()

# Data Preprocessing
def preprocess_data(spending_data, policy_data):
    spending_data.columns = spending_data.columns.str.strip()
    spending_data['Date'] = pd.to_datetime(spending_data['Date'])
    monthly_spending = spending_data.groupby(spending_data['Date'].dt.to_period("M"))['Amount'].sum().reset_index()
    monthly_spending.rename(columns={'Amount': 'Monthly Expense ($)', 'Date': 'Month'}, inplace=True)
    monthly_spending['Month'] = monthly_spending['Month'].dt.year * 100 + monthly_spending['Month'].dt.month
    monthly_spending['Monthly Expense ($)'] = pd.to_numeric(monthly_spending['Monthly Expense ($)'], errors='coerce')
    monthly_spending.dropna(subset=['Monthly Expense ($)'], inplace=True)
    monthly_spending['Spending Category'] = pd.cut(monthly_spending['Monthly Expense ($)'],
                                                   bins=[0, 500, 1500, np.inf],
                                                   labels=['Low', 'Medium', 'High'])
    le = LabelEncoder()
    policy_data['Policy Type'] = le.fit_transform(policy_data['Policy Type'])

    if 'Expected ROI' in policy_data.columns:
        policy_data['ROI Category'] = pd.cut(policy_data['Expected ROI'], 
                                             bins=[0, 5, 10, 15, np.inf], 
                                             labels=['Low', 'Medium', 'High', 'Very High'])
    else:
        st.error("Column 'Expected ROI' is missing from policy data.")
        return None, None

    required_columns = ['Policy Type', 'Expected ROI', 'Investment Horizon', 'Minimum Investment']
    missing_columns = [col for col in required_columns if col not in policy_data.columns]
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        return None, None

    return monthly_spending, policy_data

monthly_spending, policy_data = preprocess_data(spending_data, policy_data)

# Visualization Functions
def visualize_monthly_spending_trend(monthly_spending):
    if not monthly_spending.empty:
        monthly_spending['Readable Month'] = pd.to_datetime(monthly_spending['Month'].astype(str) + "01", format='%Y%m%d')
        plt.figure(figsize=(12, 6))
        sns.barplot(data=monthly_spending, x='Readable Month', y='Monthly Expense ($)', palette='coolwarm')
        plt.xticks(rotation=45)
        plt.title("Monthly Spending Trend", fontsize=16, weight='bold')
        plt.xlabel("Month", fontsize=14)
        plt.ylabel("Monthly Expense ($)", fontsize=14)
        st.pyplot(plt)

def visualize_spending_categories(monthly_spending):
    spending_category_counts = monthly_spending['Spending Category'].value_counts().sort_values()
    plt.figure(figsize=(10, 6))
    sns.barplot(y=spending_category_counts.index, x=spending_category_counts, palette='viridis')
    plt.title("Spending Category Distribution", fontsize=16, weight='bold')
    plt.xlabel("Count", fontsize=14)
    plt.ylabel("Spending Category", fontsize=14)
    st.pyplot(plt)

def visualize_roi_violin(policy_data):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=policy_data, x='ROI Category', y='Expected ROI', palette='pastel', inner='quartile')
    plt.title("Expected ROI Distribution by Policy Category", fontsize=16, weight='bold')
    plt.xlabel("ROI Category", fontsize=14)
    plt.ylabel("Expected ROI (%)", fontsize=14)
    st.pyplot(plt)

def visualize_policy_comparison(top_policies):
    if not top_policies.empty:
        plt.figure(figsize=(10, 6))
        categories = top_policies['Policy Type'].astype(str)
        x = np.arange(len(categories))
        width = 0.3

        plt.bar(x - width, top_policies['Expected ROI'], width, label='Expected ROI (%)', color='blue')
        plt.bar(x, top_policies['Investment Horizon'], width, label='Investment Horizon (years)', color='green')
        plt.bar(x + width, top_policies['Potential Return ($)'], width, label='Potential Return ($)', color='purple')

        plt.xticks(x, categories, rotation=45)
        plt.title("Top Policies Comparison", fontsize=16, weight='bold')
        plt.xlabel("Policy Type", fontsize=14)
        plt.ylabel("Values", fontsize=14)
        plt.legend()
        st.pyplot(plt)

def visualize_simplified_correlation(policy_data):
    key_columns = ['Expected ROI', 'Minimum Investment', 'Investment Horizon']
    numeric_data = policy_data[key_columns].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Key Policy Metrics Correlation", fontsize=16, weight='bold')
    st.pyplot(plt)

# Train Models and Evaluate Efficiency
def train_models(monthly_spending, policy_data):
    X_spending = monthly_spending[['Month']]
    y_spending = monthly_spending['Spending Category']
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_spending, y_spending, test_size=0.2, random_state=42)
    model_spending = RandomForestClassifier(random_state=42)
    model_spending.fit(X_train_s, y_train_s)
    acc_spending = accuracy_score(y_test_s, model_spending.predict(X_test_s))

    X_policy = policy_data[['Policy Type', 'Expected ROI', 'Investment Horizon', 'Minimum Investment']]
    X_policy = pd.get_dummies(X_policy, drop_first=True)
    y_policy = policy_data['ROI Category']
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_policy, y_policy, test_size=0.2, random_state=42)
    model_policy = RandomForestClassifier(random_state=42)
    model_policy.fit(X_train_p, y_train_p)
    acc_policy = accuracy_score(y_test_p, model_policy.predict(X_test_p))

    st.subheader("Model Efficiency")
    st.write(f"Spending Prediction Accuracy: {acc_spending * 100:.2f}%")
    st.write(f"Policy Prediction Accuracy: {acc_policy * 100:.2f}%")
    st.write("Classification Report for Policies:")
    st.text(classification_report(y_test_p, model_policy.predict(X_test_p)))

    return model_spending, model_policy

model_spending, model_policy = train_models(monthly_spending, policy_data)

# Policy Recommendation
def recommend_policy(user_investment, investment_duration, policy_data, spending_model):
    user_spending = np.array([[user_investment]])
    predicted_category = spending_model.predict(user_spending)[0]
    st.write(f"Predicted Spending Category: {predicted_category}")

    if predicted_category == 'Low':
        suitable_policies = policy_data[policy_data['ROI Category'] == 'Low']
    elif predicted_category == 'Medium':
        suitable_policies = policy_data[policy_data['ROI Category'] != 'Very High']
    else:
        suitable_policies = policy_data[policy_data['ROI Category'] == 'High']

    if not suitable_policies.empty:
        suitable_policies = suitable_policies.copy()
        suitable_policies['Potential Return ($)'] = (user_investment * investment_duration) * (suitable_policies['Expected ROI'] / 100)
        top_policies = suitable_policies.nlargest(3, 'Potential Return ($)')

        st.subheader("Top 3 Recommended Policies:")
        visualize_policy_comparison(top_policies)
    else:
        st.write("No suitable policies found for your spending category.")

# User Input for Investment
def get_user_input():
    st.header("Enter Your Investment Details")
    with st.form(key='investment_form'):
        monthly_investment = st.number_input("Enter your monthly investment amount ($):", min_value=0.0, value=100.0, step=10.0)
        investment_duration = st.number_input("Enter your investment duration (in months):", min_value=1, max_value=600, value=12)
        submit_button = st.form_submit_button(label='Submit Investment')
        if submit_button:
            st.session_state['monthly_investment'] = monthly_investment
            st.session_state['investment_duration'] = investment_duration
    return st.session_state.get('monthly_investment'), st.session_state.get('investment_duration')

# Main Function
def main():
    user_investment, investment_duration = get_user_input()
    if user_investment and investment_duration:
        recommend_policy(user_investment, investment_duration, policy_data, model_spending)

        # Visualizations
        visualize_monthly_spending_trend(monthly_spending)
        visualize_spending_categories(monthly_spending)
        visualize_roi_violin(policy_data)
        visualize_simplified_correlation(policy_data)

main()
