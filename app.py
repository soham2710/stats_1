import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from statsmodels.stats.proportion import proportions_ztest

# Function to perform the chosen statistical test
def perform_statistical_test(test_type, df, **kwargs):
    if test_type == 'Chi-Square Test':
        contingency_table = pd.crosstab(df[kwargs['col1']], df[kwargs['col2']])
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        return chi2_stat, p_value
    
    elif test_type == 'Z-Test for Proportions':
        successes = np.sum(df[kwargs['col']] == kwargs['value'])
        total = len(df)
        z_stat, p_value = proportions_ztest(successes, total, kwargs['p0'])
        return z_stat, p_value
    
    elif test_type == 'T-Test':
        group1 = df[df[kwargs['group_col']] == kwargs['group1']][kwargs['value_col']]
        group2 = df[df[kwargs['group_col']] == kwargs['group2']][kwargs['value_col']]
        t_stat, p_value = ttest_ind(group1, group2)
        return t_stat, p_value
    
    elif test_type == 'ANOVA':
        groups = [df[df[kwargs['group_col']] == group][kwargs['value_col']] for group in df[kwargs['group_col']].unique()]
        f_stat, p_value = f_oneway(*groups)
        return f_stat, p_value
    
    else:
        raise ValueError("Invalid test_type provided.")

# Streamlit Web App
def main():
    st.title("Statistical Hypothesis Testing Web App")
    
    st.sidebar.header("Step 1: Hypothesis Input")
    null_hypothesis = st.sidebar.text_input("Null Hypothesis (H0):")
    alt_hypothesis = st.sidebar.text_input("Alternative Hypothesis (H1):")

    st.sidebar.header("Step 2: Dataset Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        st.sidebar.header("Step 3: Select Statistical Test")
        test_type = st.sidebar.selectbox("Choose Test", ['Chi-Square Test', 'Z-Test for Proportions', 'T-Test', 'ANOVA'])
        
        st.sidebar.header("Step 4: Specify Test Parameters")
        
        # Request parameters based on test type
        if test_type == 'Chi-Square Test':
            col1 = st.sidebar.selectbox("Select Categorical Column 1", df.columns)
            col2 = st.sidebar.selectbox("Select Categorical Column 2", df.columns)
            params = {'col1': col1, 'col2': col2}
        
        elif test_type == 'Z-Test for Proportions':
            col = st.sidebar.selectbox("Select Column for Proportion Test", df.columns)
            value = st.sidebar.text_input("Category to Test Proportion Against")
            p0 = st.sidebar.number_input("Hypothesized Proportion (e.g., 0.2 for 20%)", min_value=0.0, max_value=1.0)
            params = {'col': col, 'value': value, 'p0': p0}
        
        elif test_type == 'T-Test':
            group_col = st.sidebar.selectbox("Select Group Column", df.columns)
            group1 = st.sidebar.text_input("First Group")
            group2 = st.sidebar.text_input("Second Group")
            value_col = st.sidebar.selectbox("Select Value Column for T-Test", df.columns)
            params = {'group_col': group_col, 'group1': group1, 'group2': group2, 'value_col': value_col}
        
        elif test_type == 'ANOVA':
            group_col = st.sidebar.selectbox("Select Group Column for ANOVA", df.columns)
            value_col = st.sidebar.selectbox("Select Value Column for ANOVA", df.columns)
            params = {'group_col': group_col, 'value_col': value_col}
        
        # Perform test
        if st.sidebar.button("Run Test"):
            test_stat, p_value = perform_statistical_test(test_type, df, **params)
            st.write(f"**{test_type} Results:**")
            st.write(f"Test Statistic: {test_stat:.4f}")
            st.write(f"P-Value: {p_value:.4f}")
            
            if p_value < 0.05:
                st.write(f"**Reject the Null Hypothesis**: {null_hypothesis}")
            else:
                st.write(f"**Fail to Reject the Null Hypothesis**: {null_hypothesis}")

if __name__ == '__main__':
    main()
