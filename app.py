import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
    
    # Convert 'Month' to a numeric value (e.g., YYYYMM format)
    monthly_spending['Month'] = monthly_spending['Month'].dt.year * 100 + monthly_spending['Month'].dt.month

    # Convert 'Monthly Expense ($)' to numeric, handling any non-numeric values
    monthly_spending['Monthly Expense ($)'] = pd.to_numeric(monthly_spending['Monthly Expense ($)'], errors='coerce')

    # Drop rows with missing 'Monthly Expense ($)'
    monthly_spending = monthly_spending.dropna(subset=['Monthly Expense ($)'])

    # Categorize monthly spending
    monthly_spending['Spending Category'] = pd.cut(monthly_spending['Monthly Expense ($)'],
                                                    bins=[0, 500, 1500, np.inf],
                                                    labels=['Low', 'Medium', 'High'])

    # Encoding policy types
    le = LabelEncoder()
    policy_data['Policy Type'] = le.fit_transform(policy_data['Policy Type'])

    # Check if 'Expected ROI' column exists and use it for categorization
    if 'Expected ROI' in policy_data.columns:
        policy_data['ROI Category'] = pd.cut(policy_data['Expected ROI'], bins=[0, 5, 10, 15, np.inf], labels=['Low', 'Medium', 'High', 'Very High'])
    else:
        st.error("Column 'Expected ROI' is missing from policy data.")
        return None, None

    # Check for required columns and adjust if needed
    required_columns = ['Policy Type', 'Expected ROI', 'Investment Horizon', 'Minimum Investment']
    missing_columns = [col for col in required_columns if col not in policy_data.columns]
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        return None, None

    return monthly_spending, policy_data

monthly_spending, policy_data = preprocess_data(spending_data, policy_data)

# Visualize the Monthly Spending Trend
def visualize_trends(monthly_spending):
    if monthly_spending is not None and not monthly_spending.empty:
        # Plotting the monthly spending trend
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        sns.lineplot(data=monthly_spending, x='Month', y='Monthly Expense ($)', marker='o', color='blue')
        
        plt.title('Monthly Spending Trend', fontsize=16, weight='bold')
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Monthly Expense ($)', fontsize=14)
        
        # Rotate the x-axis labels for better visibility
        plt.xticks(rotation=45)
        
        st.pyplot(plt)
    else:
        st.write("No data to visualize for monthly spending trends.")

# Visualization of ROI Distribution using Box Plot
def visualize_roi_distribution(policy_data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=policy_data, x='ROI Category', y='Expected ROI', palette='coolwarm')
    plt.title('Distribution of Expected ROI by Policy Type')
    plt.xlabel('ROI Category')
    plt.ylabel('Expected ROI (%)')
    st.pyplot(plt)

# Spending Distribution Visualization
def visualize_spending_distribution(monthly_spending):
    plt.figure(figsize=(10, 6))
    sns.histplot(monthly_spending['Monthly Expense ($)'], kde=True, bins=30, color='purple')
    plt.title('Distribution of Monthly Spending')
    plt.xlabel('Monthly Expense ($)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Correlation Matrix Visualization
def visualize_correlation_matrix(data):
    # Select only numeric columns for correlation calculation
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Check if there are numeric columns to compute correlation
    if numeric_data.empty:
        st.write("No numeric data available for correlation analysis.")
        return
    
    # Compute the correlation matrix
    correlation_matrix = numeric_data.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    st.pyplot(plt)

# Box Plot for ROI by Investment Horizon
def visualize_roi_by_horizon(policy_data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=policy_data, x='Investment Horizon', y='Expected ROI', palette='coolwarm')
    plt.title('Expected ROI by Investment Horizon')
    plt.xlabel('Investment Horizon (years)')
    plt.ylabel('Expected ROI (%)')
    st.pyplot(plt)

# Pie Chart for Spending Categories
def visualize_spending_category_distribution(monthly_spending):
    spending_category_counts = monthly_spending['Spending Category'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(spending_category_counts, labels=spending_category_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Spending Category Distribution')
    st.pyplot(plt)

# Train the models
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

    return model_spending, model_policy, acc_spending, acc_policy

model_spending, model_policy, acc_spending, acc_policy = train_models(monthly_spending, policy_data)

# User Input for investment
def get_user_input():
    st.header("Enter Your Investment Details")

    with st.form(key='investment_form'):
        monthly_investment = st.number_input("Enter your monthly investment amount ($):", min_value=0.0, value=100.0, step=10.0)
        investment_duration = st.number_input("Enter your investment duration (in months):", min_value=1, max_value=600, value=12)

        submit_button = st.form_submit_button(label='Submit Investment')
        
        if submit_button:
            st.session_state.monthly_investment = monthly_investment
            st.session_state.investment_duration = investment_duration
            st.session_state.input_submitted = True
            st.success("Investment details submitted successfully!")

    if 'monthly_investment' not in st.session_state or 'investment_duration' not in st.session_state:
        return None, None

    return st.session_state.monthly_investment, st.session_state.investment_duration

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
        for idx, policy in top_policies.iterrows():
            st.write(f"**Policy Type**: {policy['Policy Type']}")
            st.write(f"**Investment Horizon**: {policy['Investment Horizon']} years")
            st.write(f"**Expected ROI**: {policy['Expected ROI']}%")
            st.write(f"**Potential Return ($)**: ${policy['Potential Return ($)']:.2f}")
            st.write("---")
    else:
        st.write("No suitable policies found for your spending category.")

# Main function
def main():
    user_investment, investment_duration = get_user_input()
    
    if user_investment and investment_duration:
        recommend_policy(user_investment, investment_duration, policy_data, model_spending)
        
# Run the main function
if __name__ == "__main__":
    main()
