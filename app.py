import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from statsmodels.stats.proportion import proportions_ztest
import plotly.graph_objects as go

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

# Function to plot p-values
def plot_p_values(test_results):
    fig = go.Figure()
    for test_type, p_value in test_results.items():
        fig.add_trace(go.Bar(
            x=[test_type],
            y=[p_value],
            name=f'{test_type} p-value',
            text=f'p-value = {p_value:.4f}',
            textposition='auto'
        ))
    
    fig.update_layout(
        title="P-Values for Different Statistical Tests",
        xaxis_title="Test Type",
        yaxis_title="P-Value",
        yaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig)

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
        
        st.sidebar.header("Step 3: Select Test Set")
        test_set = st.sidebar.selectbox("Choose Test Set", ['Basic Tests', 'Advanced Tests'])

        st.sidebar.header("Step 4: Select Statistical Test(s)")
        if test_set == 'Basic Tests':
            test_types = ['Chi-Square Test', 'T-Test']
        else:  # Advanced Tests
            test_types = ['Chi-Square Test', 'Z-Test for Proportions', 'T-Test', 'ANOVA']
        
        selected_tests = st.sidebar.multiselect("Select Tests", test_types)
        
        st.sidebar.header("Step 5: Specify Test Parameters")
        test_results = {}
        
        for test_type in selected_tests:
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
            
            if st.sidebar.button(f"Run {test_type}"):
                test_stat, p_value = perform_statistical_test(test_type, df, **params)
                st.write(f"**{test_type} Results:**")
                st.write(f"Test Statistic: {test_stat:.4f}")
                st.write(f"P-Value: {p_value:.4f}")
                
                if p_value < 0.05:
                    st.write(f"**Reject the Null Hypothesis**: {null_hypothesis}")
                else:
                    st.write(f"**Fail to Reject the Null Hypothesis**: {null_hypothesis}")

                # Store the p-value for plotting
                test_results[test_type] = p_value

        # Plot p-values
        if test_results:
            plot_p_values(test_results)

if __name__ == '__main__':
    main()
