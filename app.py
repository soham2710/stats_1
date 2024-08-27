import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, norm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.anova import AnovaRM
import plotly.graph_objects as go

# Function to perform the chosen statistical test
def perform_statistical_test(test_type, df, **kwargs):
    if test_type == 'Chi-Square Test of Independence':
        contingency_table = pd.crosstab(df[kwargs['col1']], df[kwargs['col2']])
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        return chi2_stat, p_value
    
    elif test_type == 'Chi-Square Test of Homogeneity':
        contingency_table = pd.crosstab(df[kwargs['col1']], df[kwargs['col2']])
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        return chi2_stat, p_value
    
    elif test_type == 'Chi-Square Test for Goodness of Fit':
        observed_counts = df[kwargs['col']].value_counts()
        expected_counts = np.array([kwargs['expected_count']] * len(observed_counts))
        chi2_stat, p_value = chi2_contingency([observed_counts, expected_counts])[:2]
        return chi2_stat, p_value

    elif test_type == 'Z-Test for Proportions':
        successes = np.sum(df[kwargs['col']] == kwargs['value'])
        total = len(df)
        z_stat, p_value = proportions_ztest(successes, total, kwargs['p0'])
        return z_stat, p_value
    
    elif test_type == 'One-Sample Z-Test':
        mean = kwargs['mean']
        std_dev = kwargs['std_dev']
        sample_mean = df[kwargs['col']].mean()
        sample_size = len(df)
        z_stat = (sample_mean - mean) / (std_dev / np.sqrt(sample_size))
        p_value = 1 - norm.cdf(abs(z_stat))
        return z_stat, p_value
    
    elif test_type == 'Two-Sample Z-Test':
        mean1 = kwargs['mean1']
        mean2 = kwargs['mean2']
        std_dev1 = kwargs['std_dev1']
        std_dev2 = kwargs['std_dev2']
        n1 = kwargs['n1']
        n2 = kwargs['n2']
        pooled_std = np.sqrt((std_dev1**2 / n1) + (std_dev2**2 / n2))
        z_stat = (mean1 - mean2) / pooled_std
        p_value = 1 - norm.cdf(abs(z_stat))
        return z_stat, p_value
    
    elif test_type == 'T-Test':
        group1 = df[df[kwargs['group_col']] == kwargs['group1']][kwargs['value_col']]
        group2 = df[df[kwargs['group_col']] == kwargs['group2']][kwargs['value_col']]
        t_stat, p_value = ttest_ind(group1, group2)
        return t_stat, p_value
    
    elif test_type == 'ANOVA (One-Way)':
        groups = [df[df[kwargs['group_col']] == group][kwargs['value_col']] for group in df[kwargs['group_col']].unique()]
        f_stat, p_value = f_oneway(*groups)
        return f_stat, p_value
    
    elif test_type == 'ANOVA (Two-Way)':
        # Assuming 'df' has columns 'dependent_var', 'factor1', and 'factor2'
        model = AnovaRM(df, 'dependent_var', 'subject_id', within=['factor1', 'factor2'])
        res = model.fit()
        return res.anova_table['F'][0], res.anova_table['Pr > F'][0]
    
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

# Function to display assumptions
def display_assumptions():
    st.title("Statistical Test Assumptions")

    st.write("""
    | **Test Type**                | **Assumptions**                                                                                                      |
    |------------------------------|---------------------------------------------------------------------------------------------------------------------|
    | **Chi-Square Test of Independence** | 1. **Independence:** Observations should be independent of each other.                                               |
    |                              | 2. **Sample Size:** Expected frequency in each cell of the contingency table should be at least 5 (some say 10).      |
    |                              | 3. **Categorical Data:** Both variables should be categorical.                                                        |
    |                              | 4. **Sufficient Sample Size:** Overall sample size should be large enough to ensure valid results.                    |
    | **Chi-Square Test of Homogeneity** | 1. **Independence:** Observations should be independent.                                                               |
    |                              | 2. **Sample Size:** Expected frequency in each cell of the contingency table should be at least 5 (some say 10).      |
    |                              | 3. **Categorical Data:** Both samples should be categorical and drawn from different populations.                    |
    |                              | 4. **Sufficient Sample Size:** Overall sample size should be large enough to ensure valid results.                    |
    | **Chi-Square Test for Goodness of Fit** | 1. **Independence:** Observations should be independent.                                                               |
    |                              | 2. **Sample Size:** Expected frequency for each category should be at least 5 (some say 10).                         |
    |                              | 3. **Categorical Data:** The data should be categorical.                                                                |
    |                              | 4. **Sufficient Sample Size:** Overall sample size should be large enough to ensure valid results.                    |
    | **T-Test**                   | 1. **Independence:** Observations in each group should be independent of each other.                                   |
    |                              | 2. **Normality:** Data should be approximately normally distributed (especially important for small sample sizes).     |
    |                              | 3. **Homogeneity of Variances:** Variances in the two groups should be approximately equal (for independent t-test).   |
    |                              | 4. **Scale of Measurement:** Data should be measured on a continuous scale.                                           |
    | **ANOVA (One-Way)**          | 1. **Independence:** Observations should be independent of each other.                                                |
    |                              | 2. **Normality:** Data in each group should be approximately normally distributed (important for small sample sizes). |
    |                              | 3. **Homogeneity of Variances:** Variances among groups should be roughly equal.                                      |
    |                              | 4. **Scale of Measurement:** Data should be measured on a continuous scale.                                           |
    | **ANOVA (Two-Way)**          | 1. **Independence:** Observations should be independent of each other.                                                |
    |                              | 2. **Normality:** Data in each group should be approximately normally distributed (important for small sample sizes). |
    |                              | 3. **Homogeneity of Variances:** Variances among groups should be roughly equal.                                      |
    |                              | 4. **Scale of Measurement:** Data should be measured on a continuous scale.                                           |
    | **Z-Test for Proportions**   | 1. **Independence:** Observations should be independent.                                                               |
    |                              | 2. **Sample Size:** Both the number of successes and failures should be at least 5 (some guidelines suggest 10).       |
    |                              | 3. **Binomial Distribution:** The data should follow a binomial distribution.                                         |
    |                              | 4. **Normal Approximation:** For large sample sizes, the binomial distribution can be approximated by the normal distribution. |
    | **One-Sample Z-Test**        | 1. **Independence:** Observations should be independent.                                                               |
    |                              | 2. **Normality:** Data should be approximately normally distributed (important for small sample sizes).                |
    |                              | 3. **Known Variance:** Population variance should be known.                                                             |
    |                              | 4. **Scale of Measurement:** Data should be measured on a continuous scale.                                           |
    | **Two-Sample Z-Test**        | 1. **Independence:** Observations should be independent.                                                               |
    |                              | 2. **Normality:** Data should be approximately normally distributed (important for small sample sizes).                |
    |                              | 3. **Known Variance:** Population variances should be known for both groups.                                           |
    |                              | 4. **Scale of Measurement:** Data should be measured on a continuous scale.                                           |
    """)

# Main Streamlit app function
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Home", "Assumptions"])

    if page == "Home":
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
                test_types = ['Chi-Square Test of Independence', 'T-Test']
            else:  # Advanced Tests
                test_types = ['Chi-Square Test of Independence', 'Chi-Square Test of Homogeneity', 'Chi-Square Test for Goodness of Fit', 
                              'Z-Test for Proportions', 'One-Sample Z-Test', 'Two-Sample Z-Test', 'T-Test', 'ANOVA (One-Way)', 'ANOVA (Two-Way)']
            
            selected_tests = st.sidebar.multiselect("Select Tests", test_types)
            
            st.sidebar.header("Step 5: Specify Test Parameters")
            test_results = {}
            
            for test_type in selected_tests:
                if test_type == 'Chi-Square Test of Independence':
                    col1 = st.sidebar.selectbox("Select Categorical Column 1", df.columns)
                    col2 = st.sidebar.selectbox("Select Categorical Column 2", df.columns)
                    params = {'col1': col1, 'col2': col2}
                
                elif test_type == 'Chi-Square Test of Homogeneity':
                    col1 = st.sidebar.selectbox("Select Categorical Column 1", df.columns)
                    col2 = st.sidebar.selectbox("Select Categorical Column 2", df.columns)
                    params = {'col1': col1, 'col2': col2}
                
                elif test_type == 'Chi-Square Test for Goodness of Fit':
                    col = st.sidebar.selectbox("Select Column", df.columns)
                    expected_count = st.sidebar.number_input("Expected Count per Category", min_value=1)
                    params = {'col': col, 'expected_count': expected_count}
                
                elif test_type == 'Z-Test for Proportions':
                    col = st.sidebar.selectbox("Select Column for Proportion Test", df.columns)
                    value = st.sidebar.text_input("Category to Test Proportion Against")
                    p0 = st.sidebar.number_input("Hypothesized Proportion (e.g., 0.2 for 20%)", min_value=0.0, max_value=1.0)
                    params = {'col': col, 'value': value, 'p0': p0}
                
                elif test_type == 'One-Sample Z-Test':
                    col = st.sidebar.selectbox("Select Column", df.columns)
                    mean = st.sidebar.number_input("Population Mean", value=0.0)
                    std_dev = st.sidebar.number_input("Population Standard Deviation", value=1.0)
                    params = {'col': col, 'mean': mean, 'std_dev': std_dev}
                
                elif test_type == 'Two-Sample Z-Test':
                    col = st.sidebar.selectbox("Select Column", df.columns)
                    mean1 = st.sidebar.number_input("Mean of Group 1", value=0.0)
                    mean2 = st.sidebar.number_input("Mean of Group 2", value=0.0)
                    std_dev1 = st.sidebar.number_input("Standard Deviation of Group 1", value=1.0)
                    std_dev2 = st.sidebar.number_input("Standard Deviation of Group 2", value=1.0)
                    n1 = st.sidebar.number_input("Sample Size of Group 1", value=30)
                    n2 = st.sidebar.number_input("Sample Size of Group 2", value=30)
                    params = {'mean1': mean1, 'mean2': mean2, 'std_dev1': std_dev1, 'std_dev2': std_dev2, 'n1': n1, 'n2': n2}
                
                elif test_type == 'T-Test':
                    group_col = st.sidebar.selectbox("Select Group Column", df.columns)
                    group1 = st.sidebar.text_input("First Group")
                    group2 = st.sidebar.text_input("Second Group")
                    value_col = st.sidebar.selectbox("Select Value Column for T-Test", df.columns)
                    params = {'group_col': group_col, 'group1': group1, 'group2': group2, 'value_col': value_col}
                
                elif test_type == 'ANOVA (One-Way)':
                    group_col = st.sidebar.selectbox("Select Group Column for ANOVA", df.columns)
                    value_col = st.sidebar.selectbox("Select Value Column for ANOVA", df.columns)
                    params = {'group_col': group_col, 'value_col': value_col}
                
                elif test_type == 'ANOVA (Two-Way)':
                    dependent_var = st.sidebar.selectbox("Select Dependent Variable", df.columns)
                    factor1 = st.sidebar.selectbox("Select Factor 1", df.columns)
                    factor2 = st.sidebar.selectbox("Select Factor 2", df.columns)
                    params = {'dependent_var': dependent_var, 'factor1': factor1, 'factor2': factor2, 'subject_id': 'subject_id'}
                
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

    elif page == "Assumptions":
        display_assumptions()

if __name__ == '__main__':
    main()
