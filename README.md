# Statistical Hypothesis Testing Web App

## Overview

This web application provides an interactive platform for performing various statistical hypothesis tests. Users can upload datasets, input hypotheses, select statistical tests, and view the results, including p-values and plots. Additionally, the application includes a dedicated page for understanding the assumptions behind different statistical tests.

## Features

- **Upload Dataset:** Load your dataset in CSV or Excel format.
- **Input Hypotheses:** Define null and alternative hypotheses.
- **Select Tests:** Choose from basic and advanced statistical tests.
- **Specify Parameters:** Set parameters for the selected tests.
- **View Results:** See test statistics, p-values, and hypothesis test results.
- **Plot Results:** Visualize p-values for different tests in a bar chart.
- **Test Assumptions:** Access a detailed table of assumptions for each statistical test.

## Supported Tests

- **Chi-Square Test**
- **T-Test**
- **ANOVA (Analysis of Variance)**
- **Z-Test for Proportions**

## Assumptions

### Chi-Square Test
- **Independence:** Observations should be independent.
- **Sample Size:** Expected frequency in each cell should be at least 5.
- **Categorical Data:** Both variables should be categorical.
- **Sufficient Sample Size:** Overall sample size should be large enough.

### T-Test
- **Independence:** Observations in each group should be independent.
- **Normality:** Data should be approximately normally distributed (especially for small samples).
- **Homogeneity of Variances:** Variances in the two groups should be approximately equal.
- **Scale of Measurement:** Data should be measured on a continuous scale.

### ANOVA
- **Independence:** Observations should be independent.
- **Normality:** Data in each group should be approximately normally distributed (important for small samples).
- **Homogeneity of Variances:** Variances among groups should be roughly equal.
- **Scale of Measurement:** Data should be measured on a continuous scale.

### Z-Test for Proportions
- **Independence:** Observations should be independent.
- **Sample Size:** Both successes and failures should be at least 5.
- **Binomial Distribution:** Data should follow a binomial distribution.
- **Normal Approximation:** For large sample sizes, the binomial distribution can be approximated by the normal distribution.



## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
2. **Navigate to the Project Directory:**
   ```bash
   cd <project_directory>
3. **Install Dependencies: Ensure you have Python installed. Then, install the required packages:**
   ```bash
   pip install -r requirements.txt
4. **Run the Streamlit App:**
   ```bash
   streamlit run app.py  

## Usage

1. **Navigate to the Application:**
   Open a web browser and go to `http://localhost:8501` (default address for Streamlit apps).

2. **Upload Dataset:**
   Use the sidebar to upload your CSV or Excel file.

3. **Input Hypotheses:**
   Enter your null and alternative hypotheses in the provided fields.

4. **Select Tests:**
   Choose the test set (Basic or Advanced) and select the statistical tests you want to run.

5. **Specify Parameters:**
   Provide the necessary parameters for the selected tests.

6. **View Results:**
   Run the tests and view the results, including test statistics and p-values.

7. **Plot Results:**
   View the p-value bar chart for the selected tests.

8. **Understand Assumptions:**
   Switch to the "Assumptions" page to view the assumptions for each test.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact [your_email@example.com](mailto:your_email@example.com).
