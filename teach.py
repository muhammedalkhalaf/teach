import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Set page config
st.set_page_config(page_title="Econometrics Teaching App by Dr Merwan Roudane", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1565C0;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
    }
    .formula {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.3rem;
        font-family: "Courier New", monospace;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<div class='main-header'>Interactive Econometrics Teaching Tool</div>", unsafe_allow_html=True)
st.markdown("""
This app helps teach fundamental econometric concepts through interactive visualizations. 
Teachers can use this tool to demonstrate regression models, OLS assumptions, the Gauss-Markov theorem, 
and the effects of assumption violations.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to",
                        ["Introduction to Regression",
                         "Functional Forms",
                         "OLS Assumptions",
                         "Gauss-Markov Theorem",
                         "Assumption Violations",
                         "Practice Problems"])


# Function to generate sample data
def generate_data(n=100, relationship="linear", error_type="normal", heteroskedastic=False, outliers=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = np.random.uniform(0, 10, n)

    # Base error term
    if error_type == "normal":
        e = np.random.normal(0, 1, n)
    elif error_type == "t":
        e = np.random.standard_t(3, n)
    elif error_type == "uniform":
        e = np.random.uniform(-2, 2, n)

    # Heteroskedasticity
    if heteroskedastic:
        e = e * (0.5 + x / 5)

    # Different functional relationships
    if relationship == "linear":
        y = 2 + 1.5 * x + e
    elif relationship == "quadratic":
        y = 2 + 1.5 * x + 0.3 * x ** 2 + e
    elif relationship == "cubic":
        y = 2 + 1.5 * x - 0.2 * x ** 2 + 0.05 * x ** 3 + e
    elif relationship == "log-linear":
        y = 2 + 3 * np.log(x + 1) + e
    elif relationship == "exponential":
        y = 2 + np.exp(0.3 * x) + e
        y = np.minimum(y, 30)  # Limit max value for visualization
    elif relationship == "logarithmic":
        x = np.maximum(x, 0.1)  # Ensure positive x for log
        y = np.exp(1 + 0.3 * x + e / 10)

    # Add outliers
    if outliers:
        outlier_idx = np.random.choice(range(n), int(n * 0.05), replace=False)
        y[outlier_idx] = y[outlier_idx] + np.random.choice([-1, 1], size=len(outlier_idx)) * np.random.uniform(5, 10,
                                                                                                               len(outlier_idx))

    return x, y


# Function to plot regression
def plot_regression(x, y, relationship="linear", reg_type="linear", title="Linear Regression"):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot of data
    ax.scatter(x, y, alpha=0.6, color='#1E88E5', label='Data points')

    # Sort x for smooth curve plotting
    x_sorted = np.sort(x)

    # Fit and plot the regression line/curve
    if reg_type == "linear":
        model = LinearRegression()
        x_reshaped = x.reshape(-1, 1)
        model.fit(x_reshaped, y)
        y_pred = model.predict(x_sorted.reshape(-1, 1))
        ax.plot(x_sorted, y_pred, color='#D81B60', linewidth=2,
                label=f'Linear fit: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x')

    elif reg_type == "quadratic":
        X = np.column_stack((x, x ** 2))
        model = LinearRegression()
        model.fit(X, y)
        X_sorted = np.column_stack((x_sorted, x_sorted ** 2))
        y_pred = model.predict(X_sorted)
        eq = f'y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x + {model.coef_[1]:.2f}x²'
        ax.plot(x_sorted, y_pred, color='#D81B60', linewidth=2, label=f'Quadratic fit: {eq}')

    elif reg_type == "cubic":
        X = np.column_stack((x, x ** 2, x ** 3))
        model = LinearRegression()
        model.fit(X, y)
        X_sorted = np.column_stack((x_sorted, x_sorted ** 2, x_sorted ** 3))
        y_pred = model.predict(X_sorted)
        eq = f'y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x + {model.coef_[1]:.2f}x² + {model.coef_[2]:.2f}x³'
        ax.plot(x_sorted, y_pred, color='#D81B60', linewidth=2, label=f'Cubic fit: {eq}')

    elif reg_type == "log-linear":
        log_x = np.log(x + 1)  # Adding 1 to avoid log(0)
        log_x_sorted = np.log(x_sorted + 1)
        model = LinearRegression()
        model.fit(log_x.reshape(-1, 1), y)
        y_pred = model.predict(log_x_sorted.reshape(-1, 1))
        ax.plot(x_sorted, y_pred, color='#D81B60', linewidth=2,
                label=f'Log-linear fit: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}ln(x)')

    elif reg_type == "exponential":
        # For exponential, we fit log(y) = a + b*x, then transform back
        valid_mask = y > 0  # Avoid log of negative numbers
        if np.sum(valid_mask) > 10:  # Only proceed if we have enough valid points
            log_y = np.log(y[valid_mask])
            x_valid = x[valid_mask]
            model = LinearRegression()
            model.fit(x_valid.reshape(-1, 1), log_y)
            log_y_pred = model.predict(x_sorted.reshape(-1, 1))
            y_pred = np.exp(log_y_pred)
            ax.plot(x_sorted, y_pred, color='#D81B60', linewidth=2,
                    label=f'Exponential fit: y = {np.exp(model.intercept_):.2f} · e^({model.coef_[0]:.2f}x)')

    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig


# Function to generate residual plots
def plot_residuals(x, y, reg_type="linear"):
    # Fit model
    if reg_type == "linear":
        X = x.reshape(-1, 1)
    elif reg_type == "quadratic":
        X = np.column_stack((x, x ** 2))
    elif reg_type == "cubic":
        X = np.column_stack((x, x ** 2, x ** 3))
    elif reg_type == "log-linear":
        X = np.log(x + 1).reshape(-1, 1)
    elif reg_type == "exponential":
        # For exponential, we fit log(y) = a + b*x, then transform back
        valid_mask = y > 0
        if np.sum(valid_mask) > 10:
            X = x[valid_mask].reshape(-1, 1)
            y = np.log(y[valid_mask])
        else:
            X = x.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Create plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Residuals vs Fitted
    axs[0].scatter(y_pred, residuals, color='#1E88E5', alpha=0.6)
    axs[0].axhline(y=0, color='#D81B60', linestyle='--')
    axs[0].set_xlabel('Fitted values', fontsize=12)
    axs[0].set_ylabel('Residuals', fontsize=12)
    axs[0].set_title('Residuals vs Fitted', fontsize=14)
    axs[0].grid(True, alpha=0.3)

    # QQ plot for normality check
    stats.probplot(residuals, dist="norm", plot=axs[1])
    axs[1].set_title('Normal Q-Q Plot', fontsize=14)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, residuals


# Introduction to Regression
if page == "Introduction to Regression":
    st.markdown("<div class='section-header'>Introduction to Regression Analysis</div>", unsafe_allow_html=True)

    st.markdown("""
    Regression analysis is a statistical method that examines the relationship between a dependent variable (Y) 
    and one or more independent variables (X). It helps us understand how the dependent variable changes 
    when any independent variable varies.

    <div class='highlight'>
    <b>Key Concepts:</b>
    <ul>
        <li>Dependent variable (Y): The outcome we want to predict or explain</li>
        <li>Independent variable(s) (X): The factor(s) we believe influence Y</li>
        <li>Regression coefficient: Measures the effect of X on Y</li>
        <li>Ordinary Least Squares (OLS): Method to estimate parameters by minimizing squared residuals</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Interactive demonstration
    st.markdown("<div class='subsection-header'>Interactive Demonstration: Basic Linear Regression</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Parameters")
        n_points = st.slider("Number of data points", 20, 200, 100, 10)
        correlation = st.slider("Correlation strength", 0.0, 1.0, 0.7, 0.1)
        noise_level = st.slider("Noise level", 0.1, 5.0, 1.0, 0.1)
        include_outliers = st.checkbox("Include outliers", False)

        if st.button("Generate New Data"):
            st.session_state.seed = np.random.randint(0, 1000)

        # Initialize seed if not in session state
        if 'seed' not in st.session_state:
            st.session_state.seed = 42

    # Generate data based on parameters
    x = np.random.uniform(0, 10, n_points)
    e = np.random.normal(0, noise_level, n_points)
    y = 2 + correlation * 3 * x + e

    # Add outliers if selected
    if include_outliers:
        outlier_idx = np.random.choice(range(n_points), int(n_points * 0.05), replace=False)
        y[outlier_idx] = y[outlier_idx] + np.random.choice([-1, 1], size=len(outlier_idx)) * np.random.uniform(5, 10,
                                                                                                               len(outlier_idx))

    # Calculate regression
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    with col2:
        # Plot regression
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, alpha=0.6, color='#1E88E5')

        # Plot regression line
        xmin, xmax = x.min(), x.max()
        x_line = np.linspace(xmin, xmax, 100)
        y_line = model.params[0] + model.params[1] * x_line
        ax.plot(x_line, y_line, color='#D81B60', linewidth=2)

        # Add equation to plot
        equation = f"Y = {model.params[0]:.2f} + {model.params[1]:.2f}X"
        ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel("X variable", fontsize=12)
        ax.set_ylabel("Y variable", fontsize=12)
        ax.set_title("Simple Linear Regression", fontsize=14)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

    st.markdown("### Regression Results")
    st.code(model.summary().as_text())

    st.markdown("<div class='subsection-header'>Understanding the Model</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='formula'>
    Y = β₀ + β₁X + ε
    </div>

    Where:
    - Y is the dependent variable we're trying to predict
    - X is the independent variable (predictor)
    - β₀ is the intercept (value of Y when X=0)
    - β₁ is the slope (change in Y for a one-unit change in X)
    - ε is the error term (residual)

    The OLS method finds the values of β₀ and β₁ that minimize the sum of squared residuals:

    <div class='formula'>
    Minimize Σ(Yᵢ - (β₀ + β₁Xᵢ))²
    </div>
    """, unsafe_allow_html=True)

# Functional Forms
elif page == "Functional Forms":
    st.markdown("<div class='section-header'>Different Functional Forms in Regression</div>", unsafe_allow_html=True)

    st.markdown("""
    Regression models can take many functional forms depending on the relationship between variables.
    Understanding different forms helps in properly specifying models for different types of relationships.

    <div class='highlight'>
    <b>Common Functional Forms:</b>
    <ul>
        <li>Linear: Y = β₀ + β₁X + ε</li>
        <li>Quadratic: Y = β₀ + β₁X + β₂X² + ε</li>
        <li>Cubic: Y = β₀ + β₁X + β₂X² + β₃X³ + ε</li>
        <li>Log-linear: Y = β₀ + β₁ln(X) + ε</li>
        <li>Exponential: Y = e^(β₀ + β₁X + ε) or ln(Y) = β₀ + β₁X + ε</li>
        <li>Log-log (double-log): ln(Y) = β₀ + β₁ln(X) + ε</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Interactive demonstration
    st.markdown("<div class='subsection-header'>Interactive Demonstration: Exploring Functional Forms</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Data Settings")
        true_relationship = st.selectbox(
            "True underlying relationship",
            ["linear", "quadratic", "cubic", "log-linear", "exponential"],
            format_func=lambda x: x.capitalize() + " relationship"
        )

        fitted_model = st.selectbox(
            "Model to fit",
            ["linear", "quadratic", "cubic", "log-linear", "exponential"],
            format_func=lambda x: x.capitalize() + " model"
        )

        n_points = st.slider("Number of data points", 20, 200, 100)
        noise_level = st.slider("Noise level", 0.1, 3.0, 1.0, 0.1)

        if st.button("Generate New Data", key="func_form_gen"):
            st.session_state.functional_seed = np.random.randint(0, 1000)

        # Initialize seed if not in session state
        if 'functional_seed' not in st.session_state:
            st.session_state.functional_seed = 42

    # Generate data based on selected relationship
    x, y = generate_data(
        n=n_points,
        relationship=true_relationship,
        heteroskedastic=False,
        outliers=False,
        seed=st.session_state.functional_seed
    )

    with col2:
        # Plot the data with the fitted model
        fig = plot_regression(
            x, y,
            relationship=true_relationship,
            reg_type=fitted_model,
            title=f"True: {true_relationship.capitalize()}, Fitted: {fitted_model.capitalize()}"
        )
        st.pyplot(fig)

    # Plot residuals
    st.markdown("<div class='subsection-header'>Residual Analysis</div>", unsafe_allow_html=True)
    fig_resid, residuals = plot_residuals(x, y, reg_type=fitted_model)
    st.pyplot(fig_resid)

    # Calculate metrics
    X = sm.add_constant(x) if fitted_model == "linear" else None
    if fitted_model == "quadratic":
        X = sm.add_constant(np.column_stack((x, x ** 2)))
    elif fitted_model == "cubic":
        X = sm.add_constant(np.column_stack((x, x ** 2, x ** 3)))
    elif fitted_model == "log-linear":
        X = sm.add_constant(np.log(x + 1))
    elif fitted_model == "exponential":
        # Only use positive y values for log transform
        valid_mask = y > 0
        if np.sum(valid_mask) > 10:
            X = sm.add_constant(x[valid_mask])
            y_log = np.log(y[valid_mask])
            model = sm.OLS(y_log, X).fit()
            mse = np.mean(model.resid ** 2)
            r2 = model.rsquared
            residuals = model.resid
        else:
            X = sm.add_constant(x)
            mse = np.nan
            r2 = np.nan

    if fitted_model != "exponential" or not 'model' in locals():
        model = sm.OLS(y, X).fit()
        mse = np.mean(model.resid ** 2)
        r2 = model.rsquared

    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
    with col2:
        st.metric("R-squared", f"{r2:.4f}")

    # Educational content
    st.markdown("<div class='subsection-header'>Interpreting Different Functional Forms</div>", unsafe_allow_html=True)

    interpretation = {
        "linear": """
            <b>Linear model:</b> Y = β₀ + β₁X + ε
            <ul>
                <li>Interpretation: A one-unit increase in X is associated with a β₁ unit change in Y.</li>
                <li>When to use: When the relationship between X and Y is approximately constant throughout the range of X.</li>
                <li>Example: The relationship between height and weight might be approximately linear within a certain range.</li>
            </ul>
        """,
        "quadratic": """
            <b>Quadratic model:</b> Y = β₀ + β₁X + β₂X² + ε
            <ul>
                <li>Interpretation: The effect of X on Y depends on the value of X itself.</li>
                <li>When to use: When the relationship shows one curve or turning point.</li>
                <li>Example: The relationship between age and income often follows a quadratic pattern, increasing until middle age then declining.</li>
            </ul>
        """,
        "cubic": """
            <b>Cubic model:</b> Y = β₀ + β₁X + β₂X² + β₃X³ + ε
            <ul>
                <li>Interpretation: Allows for two turning points in the relationship.</li>
                <li>When to use: For more complex non-linear relationships with potentially multiple bends.</li>
                <li>Example: Growth patterns that show multiple phases of acceleration and deceleration.</li>
            </ul>
        """,
        "log-linear": """
            <b>Log-linear model:</b> Y = β₀ + β₁ln(X) + ε
            <ul>
                <li>Interpretation: A 1% increase in X is associated with a β₁/100 unit change in Y.</li>
                <li>When to use: When the effect of X diminishes as X gets larger (diminishing returns).</li>
                <li>Example: The relationship between years of education and earnings often shows diminishing returns.</li>
            </ul>
        """,
        "exponential": """
            <b>Exponential model:</b> Y = e^(β₀ + β₁X + ε) or ln(Y) = β₀ + β₁X + ε
            <ul>
                <li>Interpretation: A one-unit increase in X is associated with a (e^β₁ - 1) * 100% change in Y.</li>
                <li>When to use: When Y grows at a constant percentage rate as X increases.</li>
                <li>Example: Population growth, compound interest, or exponential decay phenomena.</li>
            </ul>
        """
    }

    st.markdown(f"<div class='highlight'>{interpretation[fitted_model]}</div>", unsafe_allow_html=True)

    st.markdown("<div class='subsection-header'>Model Specification</div>", unsafe_allow_html=True)
    st.markdown("""
    <b>Choosing the right functional form is crucial:</b>

    <ul>
        <li><b>Model misspecification</b> occurs when we choose the wrong functional form.</li>
        <li>Misspecification can lead to <b>biased estimates</b> and <b>incorrect inferences</b>.</li>
        <li>Always check residual plots for patterns that might indicate misspecification.</li>
        <li>Compare different models using metrics like R-squared, AIC, or BIC.</li>
        <li>Theory should guide functional form choices, not just statistical fit.</li>
    </ul>
    """, unsafe_allow_html=True)

# OLS Assumptions
elif page == "OLS Assumptions":
    st.markdown("<div class='section-header'>OLS Assumptions</div>", unsafe_allow_html=True)

    st.markdown("""
    For Ordinary Least Squares (OLS) regression to produce the Best Linear Unbiased Estimators (BLUE), 
    certain assumptions must be met. These assumptions are at the core of the Gauss-Markov Theorem.

    <div class='highlight'>
    <b>Key OLS Assumptions:</b>
    <ol>
        <li><b>Linearity</b>: The relationship between X and Y is linear in parameters</li>
        <li><b>Random Sampling</b>: Data is collected through random sampling</li>
        <li><b>No Perfect Multicollinearity</b>: No exact linear relationships among independent variables</li>
        <li><b>Zero Conditional Mean</b>: E(ε|X) = 0 (errors have zero mean, conditional on X)</li>
        <li><b>Homoskedasticity</b>: Var(ε|X) = σ² (constant error variance)</li>
        <li><b>Normality</b>: Errors are normally distributed (only needed for inference)</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    # Interactive visualization of assumptions
    st.markdown("<div class='subsection-header'>Interactive Visualization: OLS Assumptions</div>",
                unsafe_allow_html=True)

    assumption = st.selectbox(
        "Select an assumption to visualize",
        ["Linearity", "Homoskedasticity", "Normality of Errors", "Zero Conditional Mean"]
    )

    # Set up the demo based on selected assumption
    if assumption == "Linearity":
        st.markdown("""
        <div class='highlight'>
        <b>Linearity Assumption:</b> The relationship between X and Y is linear in parameters.
        This means that the expected value of Y is a linear function of the parameters (not necessarily the variables themselves).
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            true_relationship = st.selectbox(
                "True underlying relationship",
                ["linear", "quadratic", "cubic", "exponential"],
                format_func=lambda x: x.capitalize()
            )
            n_points = st.slider("Number of data points", min_value=20, max_value=200, value=100, key="lin_n")
            noise = st.slider("Noise level", min_value=0.1, max_value=3.0, value=1.0, step=0.1, key="lin_noise")

            if st.button("Generate New Data", key="lin_gen"):
                st.session_state.linearity_seed = np.random.randint(0, 1000)

            # Initialize seed if not in session state
            if 'linearity_seed' not in st.session_state:
                st.session_state.linearity_seed = 42

        # Generate data
        x, y = generate_data(
            n=n_points,
            relationship=true_relationship,
            error_type="normal",
            heteroskedastic=False,
            outliers=False,
            seed=st.session_state.linearity_seed
        )

        # Plot correct vs. misspecified models
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x, y, alpha=0.6, label="Data points", color='#1E88E5')

            # Fit linear model
            model_linear = LinearRegression()
            model_linear.fit(x.reshape(-1, 1), y)
            x_sorted = np.sort(x)
            y_linear = model_linear.predict(x_sorted.reshape(-1, 1))
            ax.plot(x_sorted, y_linear, color='#D81B60', linewidth=2, label="Linear fit")

            # Fit true relationship model
            if true_relationship == "quadratic":
                model_true = LinearRegression()
                X_true = np.column_stack((x, x ** 2))
                model_true.fit(X_true, y)
                X_true_sorted = np.column_stack((x_sorted, x_sorted ** 2))
                y_true = model_true.predict(X_true_sorted)
                ax.plot(x_sorted, y_true, color='#004D40', linewidth=2, linestyle='--', label="Quadratic fit")

            elif true_relationship == "cubic":
                model_true = LinearRegression()
                X_true = np.column_stack((x, x ** 2, x ** 3))
                model_true.fit(X_true, y)
                X_true_sorted = np.column_stack((x_sorted, x_sorted ** 2, x_sorted ** 3))
                y_true = model_true.predict(X_true_sorted)
                ax.plot(x_sorted, y_true, color='#004D40', linewidth=2, linestyle='--', label="Cubic fit")

            elif true_relationship == "exponential":
                valid_mask = y > 0
                if np.sum(valid_mask) > 10:
                    log_y = np.log(y[valid_mask])
                    x_valid = x[valid_mask]
                    model_true = LinearRegression()
                    model_true.fit(x_valid.reshape(-1, 1), log_y)
                    log_y_pred = model_true.predict(x_sorted.reshape(-1, 1))
                    y_true = np.exp(log_y_pred)
                    ax.plot(x_sorted, y_true, color='#004D40', linewidth=2, linestyle='--', label="Exponential fit")

            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Y", fontsize=12)
            ax.set_title(f"Linearity Assumption: {true_relationship.capitalize()} Relationship", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        # Residual plot
        st.markdown("### Residual Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Linear model residuals
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y)
            y_pred = model.predict(x.reshape(-1, 1))
            residuals = y - y_pred

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred, residuals, alpha=0.6, color='#1E88E5')
            ax.axhline(y=0, color='#D81B60', linestyle='--')
            ax.set_xlabel("Fitted values", fontsize=12)
            ax.set_ylabel("Residuals", fontsize=12)
            ax.set_title("Linear Model Residuals", fontsize=14)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        with col2:
            # True model residuals
            if true_relationship == "quadratic":
                X_true = np.column_stack((x, x ** 2))
                model_true = LinearRegression()
                model_true.fit(X_true, y)
                y_pred_true = model_true.predict(X_true)
            elif true_relationship == "cubic":
                X_true = np.column_stack((x, x ** 2, x ** 3))
                model_true = LinearRegression()
                model_true.fit(X_true, y)
                y_pred_true = model_true.predict(X_true)
            elif true_relationship == "exponential" and 'log_y' in locals():
                y_pred_true = np.zeros_like(y)
                y_pred_true[valid_mask] = np.exp(model_true.predict(x_valid.reshape(-1, 1)))
            else:
                y_pred_true = y_pred  # Fallback

            residuals_true = y - y_pred_true

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred_true, residuals_true, alpha=0.6, color='#004D40')
            ax.axhline(y=0, color='#D81B60', linestyle='--')
            ax.set_xlabel("Fitted values", fontsize=12)
            ax.set_ylabel("Residuals", fontsize=12)
            ax.set_title(f"{true_relationship.capitalize()} Model Residuals", fontsize=14)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        st.markdown("""
        <div class='highlight'>
        <b>Consequences of linearity violation:</b>
        <ul>
            <li>Biased coefficient estimates</li>
            <li>Incorrect predictions</li>
            <li>Residuals will show clear patterns</li>
            <li>R-squared will be lower than it could be with the correct functional form</li>
        </ul>

        <b>How to detect:</b> Look for patterns in residual plots against fitted values or predictors.

        <b>Possible remedies:</b>
        <ul>
            <li>Transform variables (log, square, etc.)</li>
            <li>Add polynomial terms</li>
            <li>Use nonlinear regression techniques</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    elif assumption == "Homoskedasticity":
        st.markdown("""
        <div class='highlight'>
        <b>Homoskedasticity Assumption:</b> The variance of the error term is constant across all values of the independent variables.
        In other words, the spread of residuals should be roughly the same across all values of X.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            heteroskedastic = st.radio("Error variance", ["Homoskedastic", "Heteroskedastic"])
            n_points = st.slider("Number of data points", min_value=20, max_value=200, value=100, key="het_n")
            base_noise = st.slider("Base noise level", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                                   key="het_noise")

            if st.button("Generate New Data", key="het_gen"):
                st.session_state.hetero_seed = np.random.randint(0, 1000)

            # Initialize seed if not in session state
            if 'hetero_seed' not in st.session_state:
                st.session_state.hetero_seed = 42

        # Generate data
        x, y = generate_data(
            n=n_points,
            relationship="linear",
            error_type="normal",
            heteroskedastic=(heteroskedastic == "Heteroskedastic"),
            seed=st.session_state.hetero_seed
        )

        # Fit model
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))
        residuals = y - y_pred

        with col2:
            # Plot regression with error bands
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x, y, alpha=0.6, color='#1E88E5')

            # Sort for smooth curve
            x_sorted_idx = np.argsort(x)
            x_sorted = x[x_sorted_idx]
            y_sorted = y[x_sorted_idx]
            y_pred_sorted = model.predict(x_sorted.reshape(-1, 1))

            # Plot regression line
            ax.plot(x_sorted, y_pred_sorted, color='#D81B60', linewidth=2)

            # Calculate and plot error bands
            if heteroskedastic == "Heteroskedastic":
                # For heteroskedastic, calculate local error variance
                window_size = max(10, n_points // 10)
                std_devs = []

                for i in range(len(x_sorted)):
                    lower = max(0, i - window_size // 2)
                    upper = min(len(x_sorted), i + window_size // 2)
                    local_residuals = y_sorted[lower:upper] - y_pred_sorted[lower:upper]
                    std_devs.append(np.std(local_residuals))

                std_devs = np.array(std_devs)

                # Smooth the std dev estimates
                from scipy.ndimage import gaussian_filter1d

                std_devs_smooth = gaussian_filter1d(std_devs, sigma=3)

                # Plot confidence bands
                ax.fill_between(x_sorted, y_pred_sorted - 1.96 * std_devs_smooth,
                                y_pred_sorted + 1.96 * std_devs_smooth,
                                alpha=0.2, color='#1E88E5')
            else:
                # For homoskedastic, use global error variance
                std_dev = np.std(residuals)
                ax.fill_between(x_sorted, y_pred_sorted - 1.96 * std_dev,
                                y_pred_sorted + 1.96 * std_dev,
                                alpha=0.2, color='#1E88E5')

            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Y", fontsize=12)
            ax.set_title(f"{heteroskedastic} Errors", fontsize=14)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        # Residual plots
        st.markdown("### Residual Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Residuals vs. fitted
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred, residuals, alpha=0.6, color='#1E88E5')
            ax.axhline(y=0, color='#D81B60', linestyle='--')
            ax.set_xlabel("Fitted values", fontsize=12)
            ax.set_ylabel("Residuals", fontsize=12)
            ax.set_title("Residuals vs Fitted", fontsize=14)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        with col2:
            # Residuals vs. X
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x, residuals, alpha=0.6, color='#1E88E5')
            ax.axhline(y=0, color='#D81B60', linestyle='--')
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Residuals", fontsize=12)
            ax.set_title("Residuals vs X", fontsize=14)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        # Breusch-Pagan test
        st.markdown("### Breusch-Pagan Test for Heteroskedasticity")

        X = sm.add_constant(x)
        model_sm = sm.OLS(y, X).fit()

        bp_test = het_breuschpagan(model_sm.resid, model_sm.model.exog)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("BP Test Statistic", f"{bp_test[0]:.4f}")
        with col2:
            st.metric("p-value", f"{bp_test[1]:.4f}")

        if bp_test[1] < 0.05:
            st.markdown(
                "**Result:** Reject null hypothesis of homoskedasticity. There is evidence of heteroskedasticity.")
        else:
            st.markdown("**Result:** Fail to reject null hypothesis. No strong evidence of heteroskedasticity.")

        st.markdown("""
        <div class='highlight'>
        <b>Consequences of heteroskedasticity:</b>
        <ul>
            <li>OLS estimators remain unbiased but are no longer efficient (minimum variance)</li>
            <li>Standard errors are biased, typically underestimated</li>
            <li>Confidence intervals and hypothesis tests are invalid</li>
            <li>Prediction intervals are incorrect</li>
        </ul>

        <b>How to detect:</b>
        <ul>
            <li>Visual inspection of residual plots (residuals vs fitted, residuals vs X)</li>
            <li>Formal tests: Breusch-Pagan, White, Goldfeld-Quandt</li>
        </ul>

        <b>Possible remedies:</b>
        <ul>
            <li>Transform the dependent variable (often using log)</li>
            <li>Use weighted least squares (WLS)</li>
            <li>Use robust standard errors (HC0, HC1, HC2, HC3)</li>
            <li>Bootstrap standard errors</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    elif assumption == "Normality of Errors":
        st.markdown("""
        <div class='highlight'>
        <b>Normality Assumption:</b> The error terms are normally distributed. This assumption is not needed for unbiasedness or consistency,
        but it's required for valid hypothesis testing and constructing confidence intervals.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            error_distribution = st.selectbox(
                "Error distribution",
                ["normal", "t", "uniform"],
                format_func=lambda x: {"normal": "Normal", "t": "T (heavy-tailed)", "uniform": "Uniform"}[x]
            )
            n_points = st.slider("Number of data points", min_value=20, max_value=500, value=100, key="norm_n")

            if st.button("Generate New Data", key="norm_gen"):
                st.session_state.norm_seed = np.random.randint(0, 1000)

            # Initialize seed if not in session state
            if 'norm_seed' not in st.session_state:
                st.session_state.norm_seed = 42

        # Generate data
        x, y = generate_data(
            n=n_points,
            relationship="linear",
            error_type=error_distribution,
            seed=st.session_state.norm_seed
        )

        # Fit model
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))
        residuals = y - y_pred

        with col2:
            # Distribution of residuals
            fig, ax = plt.subplots(figsize=(10, 6))

            # Histogram with KDE
            sns.histplot(residuals, kde=True, ax=ax, color='#1E88E5')

            # Overlay normal distribution
            xmin, xmax = ax.get_xlim()
            x_norm = np.linspace(xmin, xmax, 100)
            y_norm = stats.norm.pdf(x_norm, np.mean(residuals), np.std(residuals))
            ax.plot(x_norm, y_norm * len(residuals) * (xmax - xmin) / 10,
                    color='#D81B60', linewidth=2, label='Normal distribution')

            ax.set_xlabel("Residuals", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(f"Distribution of Residuals with {error_distribution.capitalize()} Errors", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        # QQ Plot
        st.markdown("### Normal Q-Q Plot")

        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q Plot", fontsize=14)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        # Statistical tests
        st.markdown("### Statistical Tests for Normality")

        col1, col2 = st.columns(2)

        with col1:
            # Jarque-Bera test
            jb_stat, jb_pval = stats.jarque_bera(residuals)
            st.metric("Jarque-Bera p-value", f"{jb_pval:.4f}")

            if jb_pval < 0.05:
                st.markdown("**Result:** Reject null hypothesis of normality.")
            else:
                st.markdown("**Result:** Fail to reject null hypothesis of normality.")

        with col2:
            # Shapiro-Wilk test
            if len(residuals) <= 5000:  # Shapiro-Wilk only works for n ≤ 5000
                sw_stat, sw_pval = stats.shapiro(residuals)
                st.metric("Shapiro-Wilk p-value", f"{sw_pval:.4f}")

                if sw_pval < 0.05:
                    st.markdown("**Result:** Reject null hypothesis of normality.")
                else:
                    st.markdown("**Result:** Fail to reject null hypothesis of normality.")
            else:
                st.markdown("Shapiro-Wilk test not available for large samples.")

        st.markdown("""
        <div class='highlight'>
        <b>Consequences of non-normality:</b>
        <ul>
            <li>OLS estimators are still unbiased and consistent</li>
            <li>For large samples, non-normality is less problematic due to the Central Limit Theorem</li>
            <li>For small samples, hypothesis tests (t-tests, F-tests) may not be valid</li>
            <li>Confidence intervals may not have the correct coverage</li>
        </ul>

        <b>How to detect:</b>
        <ul>
            <li>Visual inspection: Histogram of residuals, Q-Q plot</li>
            <li>Statistical tests: Shapiro-Wilk, Jarque-Bera, Kolmogorov-Smirnov</li>
        </ul>

        <b>Possible remedies:</b>
        <ul>
            <li>Transform the dependent variable (log, Box-Cox)</li>
            <li>For large samples, rely on asymptotic properties</li>
            <li>Bootstrap methods</li>
            <li>Robust regression methods</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    elif assumption == "Zero Conditional Mean":
        st.markdown("""
        <div class='highlight'>
        <b>Zero Conditional Mean Assumption:</b> E(ε|X) = 0. This means that the expected value of the error term, 
        conditional on the independent variables, is zero. In other words, the errors should have no systematic relationship with X.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            model_specification = st.selectbox(
                "Model specification",
                ["Correctly Specified", "Omitted Variable Bias", "Misspecified Functional Form"]
            )
            n_points = st.slider("Number of data points", min_value=20, max_value=200, value=100, key="zcm_n")

            if st.button("Generate New Data", key="zcm_gen"):
                st.session_state.zcm_seed = np.random.randint(0, 1000)

            # Initialize seed if not in session state
            if 'zcm_seed' not in st.session_state:
                st.session_state.zcm_seed = 42

        # Generate data based on specification
        np.random.seed(st.session_state.zcm_seed)

        if model_specification == "Correctly Specified":
            x = np.random.uniform(0, 10, n_points)
            e = np.random.normal(0, 1, n_points)
            y = 2 + 1.5 * x + e

            # Fit correct model
            X = x.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            residuals = y - y_pred

            # No omitted variable
            z = None

        elif model_specification == "Omitted Variable Bias":
            x = np.random.uniform(0, 10, n_points)
            # Generate correlated omitted variable
            z = 0.7 * x + np.random.normal(0, 2, n_points)
            e = np.random.normal(0, 1, n_points)

            # True model: y = b0 + b1*x + b2*z + e
            y = 2 + 1.2 * x + 0.8 * z + e

            # Fit misspecified model (omitting z)
            X = x.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            residuals = y - y_pred

        elif model_specification == "Misspecified Functional Form":
            x = np.random.uniform(0, 10, n_points)
            e = np.random.normal(0, 1, n_points)

            # True quadratic relationship
            y = 2 + 1.2 * x + 0.3 * x ** 2 + e

            # Fit misspecified linear model
            X = x.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            residuals = y - y_pred

            # No omitted variable
            z = None

        with col2:
            if model_specification == "Omitted Variable Bias":
                # Scatter plot with color indicating the omitted variable
                fig = px.scatter(
                    x=x, y=y,
                    color=z,
                    opacity=0.7,
                    labels={"x": "X", "y": "Y", "color": "Z (Omitted)"},
                    title="Data with Omitted Variable Z"
                )

                # Add regression line
                x_sorted = np.sort(x)
                y_pred_sorted = model.predict(x_sorted.reshape(-1, 1))
                fig.add_trace(
                    go.Scatter(
                        x=x_sorted, y=y_pred_sorted,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f'Linear fit: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x'
                    )
                )

                fig.update_layout(
                    height=500,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                # Regular scatter plot for other cases
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(x, y, alpha=0.6, color='#1E88E5')

                # Plot fitted line
                x_sorted = np.sort(x)
                y_pred_sorted = model.predict(x_sorted.reshape(-1, 1))
                ax.plot(x_sorted, y_pred_sorted, color='#D81B60', linewidth=2,
                        label=f'Linear fit: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x')

                # For misspecified functional form, also show true model
                if model_specification == "Misspecified Functional Form":
                    y_true = 2 + 1.2 * x_sorted + 0.3 * x_sorted ** 2
                    ax.plot(x_sorted, y_true, color='#004D40', linewidth=2, linestyle='--',
                            label=f'True relationship: y = 2 + 1.2x + 0.3x²')

                ax.set_xlabel("X", fontsize=12)
                ax.set_ylabel("Y", fontsize=12)
                ax.set_title(model_specification, fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

        # Residual plots
        st.markdown("### Residual Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Residuals vs. X
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x, residuals, alpha=0.6, color='#1E88E5')
            ax.axhline(y=0, color='#D81B60', linestyle='--')

            # Add smoothed line to help see patterns
            from scipy.stats import binned_statistic

            bins = min(20, n_points // 5)
            bin_means, bin_edges, _ = binned_statistic(x, residuals, statistic='mean', bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax.plot(bin_centers, bin_means, color='#004D40', linewidth=2)

            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Residuals", fontsize=12)
            ax.set_title("Residuals vs X", fontsize=14)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        with col2:
            if model_specification == "Omitted Variable Bias" and z is not None:
                # Residuals vs. omitted variable Z
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(z, residuals, alpha=0.6, color='#1E88E5')
                ax.axhline(y=0, color='#D81B60', linestyle='--')

                # Add smoothed line
                bin_means, bin_edges, _ = binned_statistic(z, residuals, statistic='mean', bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                ax.plot(bin_centers, bin_means, color='#004D40', linewidth=2)

                ax.set_xlabel("Z (Omitted Variable)", fontsize=12)
                ax.set_ylabel("Residuals", fontsize=12)
                ax.set_title("Residuals vs Omitted Variable Z", fontsize=14)
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)
            else:
                # Residuals vs. fitted values
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_pred, residuals, alpha=0.6, color='#1E88E5')
                ax.axhline(y=0, color='#D81B60', linestyle='--')

                # Add smoothed line
                bin_means, bin_edges, _ = binned_statistic(y_pred, residuals, statistic='mean', bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                ax.plot(bin_centers, bin_means, color='#004D40', linewidth=2)

                ax.set_xlabel("Fitted values", fontsize=12)
                ax.set_ylabel("Residuals", fontsize=12)
                ax.set_title("Residuals vs Fitted", fontsize=14)
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

        # Correlation test
        st.markdown("### Correlation between Residuals and X")

        corr, p_value = stats.pearsonr(x, residuals)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Correlation coefficient", f"{corr:.4f}")
        with col2:
            st.metric("p-value", f"{p_value:.4f}")

        if p_value < 0.05:
            st.markdown(
                "**Result:** Significant correlation between residuals and X. Zero conditional mean assumption is likely violated.")
        else:
            st.markdown("**Result:** No significant correlation between residuals and X.")

        if model_specification == "Omitted Variable Bias" and z is not None:
            st.markdown("### Correlation between Residuals and Z (Omitted Variable)")

            corr_z, p_value_z = stats.pearsonr(z, residuals)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Correlation coefficient", f"{corr_z:.4f}")
            with col2:
                st.metric("p-value", f"{p_value_z:.4f}")

            if p_value_z < 0.05:
                st.markdown(
                    "**Result:** Significant correlation between residuals and Z. The omitted variable is related to the error term.")
            else:
                st.markdown("**Result:** No significant correlation between residuals and Z.")

        st.markdown("""
        <div class='highlight'>
        <b>Consequences of violating zero conditional mean:</b>
        <ul>
            <li>Biased and inconsistent coefficient estimates</li>
            <li>Incorrect inferences about causal relationships</li>
            <li>Misleading predictions</li>
        </ul>

        <b>Common causes:</b>
        <ul>
            <li><b>Omitted Variable Bias:</b> Excluding a relevant variable that's correlated with included variables</li>
            <li><b>Misspecified Functional Form:</b> Using linear model when relationship is nonlinear</li>
            <li><b>Measurement Error:</b> Errors in measuring X variables</li>
            <li><b>Endogeneity:</b> When X is determined simultaneously with Y</li>
        </ul>

        <b>Potential remedies:</b>
        <ul>
            <li>Include omitted variables if data is available</li>
            <li>Use instrumental variables</li>
            <li>Apply appropriate transformations to capture nonlinear relationships</li>
            <li>Fixed effects or difference-in-differences for panel data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Gauss-Markov Theorem
elif page == "Gauss-Markov Theorem":
    st.markdown("<div class='section-header'>The Gauss-Markov Theorem</div>", unsafe_allow_html=True)

    st.markdown("""
    The Gauss-Markov theorem is a cornerstone of econometric theory. It states that under certain conditions, 
    the Ordinary Least Squares (OLS) estimator is the Best Linear Unbiased Estimator (BLUE).

    <div class='highlight'>
    <b>The Theorem States:</b><br>
    Under the following assumptions:
    <ol>
        <li>Linearity: The model is linear in parameters</li>
        <li>Random Sampling: Observations are randomly sampled</li>
        <li>No Perfect Multicollinearity: Independent variables are not perfectly correlated</li>
        <li>Zero Conditional Mean: E(ε|X) = 0</li>
        <li>Homoskedasticity: Var(ε|X) = σ² (constant error variance)</li>
    </ol>

    Then the OLS estimator is BLUE:
    <ul>
        <li><b>B</b>est: Has the smallest variance among all linear unbiased estimators</li>
        <li><b>L</b>inear: Is a linear function of the dependent variable</li>
        <li><b>U</b>nbiased: Expected value equals the true parameter value</li>
        <li><b>E</b>stimator: A rule for estimating the parameters</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Educational visualization
    st.markdown("<div class='subsection-header'>Interactive Visualization: Sampling Distribution of Estimators</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Simulation Parameters")
        assumption_violated = st.selectbox(
            "Assumption violation",
            ["None (All Satisfied)", "Heteroskedasticity", "Nonlinearity", "Measurement Error"]
        )
        n_samples = st.slider("Number of samples", min_value=100, max_value=1000, value=500, step=100)
        sample_size = st.slider("Sample size", min_value=20, max_value=100, value=30, step=10)

        true_beta = 1.5  # True coefficient value

        if st.button("Run Simulation"):
            st.session_state.gm_seed = np.random.randint(0, 1000)

        # Initialize seed if not in session state
        if 'gm_seed' not in st.session_state:
            st.session_state.gm_seed = 42

    # Run simulation
    np.random.seed(st.session_state.gm_seed)

    # Arrays to store coefficient estimates
    beta_ols = np.zeros(n_samples)
    beta_alt = np.zeros(n_samples)  # Alternative estimator

    for i in range(n_samples):
        # Generate X values
        x = np.random.uniform(0, 10, sample_size)

        # Generate errors based on violation type
        if assumption_violated == "Heteroskedasticity":
            e = np.random.normal(0, 0.5 + x / 5, sample_size)  # Heteroskedastic errors
        else:
            e = np.random.normal(0, 1, sample_size)  # Homoskedastic errors

        # Generate Y values
        if assumption_violated == "Nonlinearity":
            # True model is quadratic, but we'll fit linear
            y = 2 + true_beta * x + 0.2 * x ** 2 + e
        elif assumption_violated == "Measurement Error":
            # X is measured with error
            x_true = x.copy()
            x = x_true + np.random.normal(0, 0.5, sample_size)  # Add measurement error
            y = 2 + true_beta * x_true + e
        else:
            # Standard linear model
            y = 2 + true_beta * x + e

        # OLS estimator
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta_ols[i] = model.params[1]

        # Alternative estimator depends on violation
        if assumption_violated == "Heteroskedasticity":
            # Weighted least squares
            weights = 1 / (0.5 + x / 5)  # Inverse of variance function
            model_wls = sm.WLS(y, X, weights=weights).fit()
            beta_alt[i] = model_wls.params[1]
        elif assumption_violated == "Nonlinearity":
            # Include quadratic term
            X_quad = sm.add_constant(np.column_stack((x, x ** 2)))
            model_quad = sm.OLS(y, X_quad).fit()
            beta_alt[i] = model_quad.params[1]  # Still look at linear coefficient
        elif assumption_violated == "Measurement Error":
            # IV-like estimator (split sample approach for illustration)
            half = sample_size // 2
            x1, x2 = x[:half], x[half:]
            y1, y2 = y[:half], y[half:]

            # Use x1 as instrument for x2
            x1_mean = np.mean(x1)
            x2_mean = np.mean(x2)
            y2_mean = np.mean(y2)

            numerator = np.sum((x1 - x1_mean) * (y2 - y2_mean))
            denominator = np.sum((x1 - x1_mean) * (x2 - x2_mean))

            if denominator != 0:
                beta_alt[i] = numerator / denominator
            else:
                beta_alt[i] = np.nan
        else:
            # For no violation, use a less efficient but unbiased estimator
            # Subsample estimator
            subsample = np.random.choice(sample_size, sample_size // 2, replace=False)
            X_sub = sm.add_constant(x[subsample])
            y_sub = y[subsample]
            model_sub = sm.OLS(y_sub, X_sub).fit()
            beta_alt[i] = model_sub.params[1]

    # Clean any NaNs
    beta_ols = beta_ols[~np.isnan(beta_ols)]
    beta_alt = beta_alt[~np.isnan(beta_alt)]

    # Calculate summary statistics
    ols_mean = np.mean(beta_ols)
    ols_var = np.var(beta_ols)
    alt_mean = np.mean(beta_alt)
    alt_var = np.var(beta_alt)

    with col2:
        # Plot distributions
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histograms
        ax.hist(beta_ols, bins=30, alpha=0.5, color='#1E88E5', label=f'OLS: μ={ols_mean:.4f}, σ²={ols_var:.4f}')
        ax.hist(beta_alt, bins=30, alpha=0.5, color='#D81B60', label=f'Alternative: μ={alt_mean:.4f}, σ²={alt_var:.4f}')

        # True parameter line
        ax.axvline(x=true_beta, color='black', linestyle='--', linewidth=2, label=f'True β={true_beta}')

        # Mean lines
        ax.axvline(x=ols_mean, color='#1E88E5', linestyle='-', linewidth=2)
        ax.axvline(x=alt_mean, color='#D81B60', linestyle='-', linewidth=2)

        ax.set_xlabel("Estimated Coefficient (β₁)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Sampling Distribution: {assumption_violated}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

    # Summary table
    st.markdown("### Comparison of Estimators")

    # Calculate bias
    ols_bias = ols_mean - true_beta
    alt_bias = alt_mean - true_beta

    # Calculate MSE
    ols_mse = ols_var + ols_bias ** 2
    alt_mse = alt_var + alt_bias ** 2

    # Create comparison table
    comparison_data = {
        "Estimator": ["OLS", "Alternative"],
        "Mean": [f"{ols_mean:.4f}", f"{alt_mean:.4f}"],
        "Variance": [f"{ols_var:.4f}", f"{alt_var:.4f}"],
        "Bias": [f"{ols_bias:.4f}", f"{alt_bias:.4f}"],
        "Mean Squared Error": [f"{ols_mse:.4f}", f"{alt_mse:.4f}"]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

    # Interpretation
    st.markdown("<div class='subsection-header'>Interpretation</div>", unsafe_allow_html=True)

    if assumption_violated == "None (All Satisfied)":
        st.markdown("""
        <div class='highlight'>
        When all Gauss-Markov assumptions are satisfied:
        <ul>
            <li>Both estimators are unbiased (expected value equals the true parameter)</li>
            <li>OLS has lower variance than the alternative estimator</li>
            <li>OLS has lower MSE, making it more efficient</li>
            <li>This illustrates that OLS is BLUE: among unbiased estimators, it has the smallest variance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    elif assumption_violated == "Heteroskedasticity":
        st.markdown("""
        <div class='highlight'>
        When heteroskedasticity is present:
        <ul>
            <li>OLS remains unbiased but is no longer efficient</li>
            <li>Weighted Least Squares (WLS) can be more efficient if the weights correctly reflect the error variance structure</li>
            <li>OLS standard errors will be incorrect without robust standard error adjustments</li>
            <li>For inference, it's crucial to use heteroskedasticity-robust standard errors with OLS</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    elif assumption_violated == "Nonlinearity":
        st.markdown("""
        <div class='highlight'>
        When the true relationship is nonlinear but we fit a linear model:
        <ul>
            <li>OLS produces biased estimates due to model misspecification</li>
            <li>Including the appropriate nonlinear terms improves both bias and efficiency</li>
            <li>The BLUE property applies only when the model is correctly specified</li>
            <li>Always check for nonlinearity using residual plots and specification tests</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    elif assumption_violated == "Measurement Error":
        st.markdown("""
        <div class='highlight'>
        When independent variables are measured with error:
        <ul>
            <li>OLS estimates are biased and inconsistent (attenuation bias)</li>
            <li>The bias typically moves the coefficient toward zero</li>
            <li>Instrumental variables (IV) can help address measurement error</li>
            <li>The measurement error problem is particularly common in observational studies</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Theoretical explanation
    st.markdown("<div class='subsection-header'>Mathematical Intuition</div>", unsafe_allow_html=True)

    st.markdown("""
    The Gauss-Markov theorem can be proven mathematically, but the intuition is:

    1. **Unbiasedness**: Under the first four assumptions, OLS estimates are unbiased: E(β̂) = β

    2. **Efficiency**: Under all five assumptions, OLS has the minimum variance among all linear unbiased estimators.

    <div class='formula'>
    Var(β̂) ≤ Var(β̃)
    </div>

    Where β̂ is the OLS estimator and β̃ is any other linear unbiased estimator.

    3. **Trade-off**: When assumptions are violated, we often face a bias-variance trade-off. Sometimes a biased estimator with lower variance can have a lower mean squared error.

    <div class='formula'>
    MSE(β̂) = Var(β̂) + Bias(β̂)²
    </div>
    """, unsafe_allow_html=True)

# Assumption Violations
elif page == "Assumption Violations":
    st.markdown("<div class='section-header'>Violations of OLS Assumptions</div>", unsafe_allow_html=True)

    st.markdown("""
    When OLS assumptions are violated, the consequences can range from minor inefficiencies to severe bias. 
    Understanding these violations helps in diagnosing problems and applying appropriate remedies.
    """)

    # Select violation to explore
    violation = st.selectbox(
        "Select assumption violation to explore",
        ["Heteroskedasticity", "Autocorrelation", "Multicollinearity", "Endogeneity/Omitted Variable", "Non-Normality"]
    )

    if violation == "Heteroskedasticity":
        st.markdown("<div class='subsection-header'>Heteroskedasticity</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='highlight'>
        <b>Definition:</b> Heteroskedasticity occurs when the variance of the error term varies across observations.
        This violates the assumption of homoskedasticity (constant variance).
        </div>
        """, unsafe_allow_html=True)

        # Interactive demonstration
        col1, col2 = st.columns([1, 2])

        with col1:
            heteroskedasticity_type = st.selectbox(
                "Type of heteroskedasticity",
                ["Increasing with X", "U-shaped", "Grouped", "None (homoskedastic)"]
            )
            n_points = st.slider("Number of data points", 50, 500, 200, key="het_viz_n")
            effect_size = st.slider("Effect strength", 0.1, 3.0, 1.0, 0.1, key="het_effect")

            if st.button("Generate New Data", key="het_viz_gen"):
                st.session_state.het_viz_seed = np.random.randint(0, 1000)

            if 'het_viz_seed' not in st.session_state:
                st.session_state.het_viz_seed = 42

        # Generate data with heteroskedasticity
        np.random.seed(st.session_state.het_viz_seed)

        x = np.random.uniform(0, 10, n_points)

        if heteroskedasticity_type == "Increasing with X":
            e = np.random.normal(0, 0.5 + effect_size * x / 5, n_points)
        elif heteroskedasticity_type == "U-shaped":
            e = np.random.normal(0, 0.5 + effect_size * abs(x - 5) / 2, n_points)
        elif heteroskedasticity_type == "Grouped":
            # Create groups
            groups = np.random.randint(0, 3, n_points)
            var_multipliers = [0.5, 1.0, 0.5 + effect_size]
            e = np.array([np.random.normal(0, var_multipliers[g]) for g in groups])
        else:  # Homoskedastic
            e = np.random.normal(0, 1, n_points)

        # Generate y
        y = 2 + 1.5 * x + e

        # Fit OLS model
        X = sm.add_constant(x)
        model_ols = sm.OLS(y, X).fit()

        # Fit WLS model if heteroskedastic
        if heteroskedasticity_type != "None (homoskedastic)":
            if heteroskedasticity_type == "Increasing with X":
                weights = 1 / (0.5 + effect_size * x / 5) ** 2
            elif heteroskedasticity_type == "U-shaped":
                weights = 1 / (0.5 + effect_size * abs(x - 5) / 2) ** 2
            elif heteroskedasticity_type == "Grouped":
                weights = 1 / np.array([var_multipliers[g] ** 2 for g in groups])

            model_wls = sm.WLS(y, X, weights=weights).fit()

        # Plotting
        with col2:
            fig = px.scatter(
                x=x, y=y,
                opacity=0.7,
                labels={"x": "X", "y": "Y"},
                title=f"{heteroskedasticity_type} Error Variance"
            )

            # Add regression lines
            x_sorted = np.sort(x)
            X_sorted = sm.add_constant(x_sorted)
            y_ols_pred = model_ols.predict(X_sorted)

            fig.add_trace(
                go.Scatter(
                    x=x_sorted, y=y_ols_pred,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'OLS: y = {model_ols.params[0]:.2f} + {model_ols.params[1]:.2f}x'
                )
            )

            if heteroskedasticity_type != "None (homoskedastic)":
                y_wls_pred = model_wls.predict(X_sorted)
                fig.add_trace(
                    go.Scatter(
                        x=x_sorted, y=y_wls_pred,
                        mode='lines',
                        line=dict(color='green', width=2, dash='dash'),
                        name=f'WLS: y = {model_wls.params[0]:.2f} + {model_wls.params[1]:.2f}x'
                    )
                )

            fig.update_layout(
                height=500,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        # Residual analysis
        st.markdown("### Residual Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # OLS residuals vs fitted
            ols_resid = model_ols.resid
            ols_fitted = model_ols.fittedvalues

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(ols_fitted, ols_resid, alpha=0.6, color='#1E88E5')
            ax.axhline(y=0, color='#D81B60', linestyle='--')

            # Add smoothed line
            from scipy.stats import binned_statistic

            bins = min(20, n_points // 10)
            bin_means, bin_edges, _ = binned_statistic(ols_fitted, ols_resid, statistic='mean', bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax.plot(bin_centers, bin_means, color='#004D40', linewidth=2)

            ax.set_xlabel("Fitted values", fontsize=12)
            ax.set_ylabel("Residuals", fontsize=12)
            ax.set_title("OLS: Residuals vs Fitted", fontsize=14)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        with col2:
            # Absolute residuals vs X to check heteroskedasticity
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x, np.abs(ols_resid), alpha=0.6, color='#1E88E5')

            # Add smoothed line
            bin_means, bin_edges, _ = binned_statistic(x, np.abs(ols_resid), statistic='mean', bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax.plot(bin_centers, bin_means, color='#004D40', linewidth=2)

            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Absolute Residuals", fontsize=12)
            ax.set_title("Absolute Residuals vs X", fontsize=14)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        # Breusch-Pagan test
        st.markdown("### Breusch-Pagan Test for Heteroskedasticity")

        bp_test = het_breuschpagan(model_ols.resid, model_ols.model.exog)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("BP Test Statistic", f"{bp_test[0]:.4f}")
        with col2:
            st.metric("p-value", f"{bp_test[1]:.4f}")

        if bp_test[1] < 0.05:
            st.markdown(
                "**Result:** Reject null hypothesis of homoskedasticity. There is evidence of heteroskedasticity.")
        else:
            st.markdown("**Result:** Fail to reject null hypothesis. No strong evidence of heteroskedasticity.")

        # Compare OLS and WLS if heteroskedastic
        if heteroskedasticity_type != "None (homoskedastic)":
            st.markdown("### Comparison of OLS and WLS")

            # Get standard errors for both models
            ols_se = model_ols.bse[1]
            wls_se = model_wls.bse[1]

            # Compare estimators
            est_data = {
                "Estimator": ["OLS", "WLS", "OLS with Robust SE"],
                "Slope Estimate": [f"{model_ols.params[1]:.4f}", f"{model_wls.params[1]:.4f}",
                                   f"{model_ols.params[1]:.4f}"],
                "Standard Error": [f"{ols_se:.4f}", f"{wls_se:.4f}",
                                   f"{model_ols.get_robustcov_results('HC1').bse[1]:.4f}"],
                "t-statistic": [f"{model_ols.tvalues[1]:.4f}", f"{model_wls.tvalues[1]:.4f}",
                                f"{model_ols.params[1] / model_ols.get_robustcov_results('HC1').bse[1]:.4f}"],
                "p-value": [f"{model_ols.pvalues[1]:.4f}", f"{model_wls.pvalues[1]:.4f}",
                            f"{stats.t.sf(abs(model_ols.params[1] / model_ols.get_robustcov_results('HC1').bse[1]), model_ols.df_resid) * 2:.4f}"]
            }

            est_df = pd.DataFrame(est_data)
            st.table(est_df)

        # Educational content
        st.markdown("<div class='subsection-header'>Consequences and Remedies</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='highlight'>
        <b>Consequences of Heteroskedasticity:</b>
        <ul>
            <li>OLS coefficients remain unbiased but are no longer efficient (minimum variance)</li>
            <li>Standard errors are biased, typically underestimated</li>
            <li>t-statistics and p-values are invalid, leading to incorrect inference</li>
            <li>Confidence intervals have incorrect coverage</li>
        </ul>

        <b>Detection Methods:</b>
        <ul>
            <li>Visual inspection of residual plots (residuals vs. fitted, absolute residuals vs. X)</li>
            <li>Formal tests: Breusch-Pagan, White, Goldfeld-Quandt tests</li>
            <li>Plot of residual variance across groups or bins of X</li>
        </ul>

        <b>Remedies:</b>
        <ul>
            <li><b>Heteroskedasticity-Robust Standard Errors:</b> Adjust standard errors to account for heteroskedasticity (HC0, HC1, HC2, HC3)</li>
            <li><b>Weighted Least Squares (WLS):</b> Weight observations by the inverse of error variance (if variance structure is known)</li>
            <li><b>Transform the dependent variable:</b> Often a log transformation can stabilize variance</li>
            <li><b>Generalized Least Squares (GLS):</b> More general approach that can handle various variance structures</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    elif violation == "Autocorrelation":
        st.markdown("<div class='subsection-header'>Autocorrelation</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='highlight'>
        <b>Definition:</b> Autocorrelation (or serial correlation) occurs when the error terms are correlated across observations.
        This commonly occurs in time series data, where errors in one period may be related to errors in previous periods.
        </div>
        """, unsafe_allow_html=True)

        # Interactive demonstration
        col1, col2 = st.columns([1, 2])

        with col1:
            autocorrelation_type = st.selectbox(
                "Type of autocorrelation",
                ["Positive AR(1)", "Negative AR(1)", "Seasonal", "None"]
            )
            n_points = st.slider("Number of time periods", 50, 500, 100, key="auto_n")
            ar_param = st.slider("AR parameter magnitude", 0.1, 0.9, 0.7, 0.1, key="ar_param")

            if st.button("Generate New Data", key="auto_gen"):
                st.session_state.auto_seed = np.random.randint(0, 1000)

            if 'auto_seed' not in st.session_state:
                st.session_state.auto_seed = 42

        # Generate time series data with autocorrelation
        np.random.seed(st.session_state.auto_seed)

        time = np.arange(n_points)

        # Generate correlated error terms
        if autocorrelation_type == "Positive AR(1)":
            e = np.zeros(n_points)
            e[0] = np.random.normal(0, 1)
            for t in range(1, n_points):
                e[t] = ar_param * e[t - 1] + np.random.normal(0, np.sqrt(1 - ar_param ** 2))
        elif autocorrelation_type == "Negative AR(1)":
            e = np.zeros(n_points)
            e[0] = np.random.normal(0, 1)
            for t in range(1, n_points):
                e[t] = -ar_param * e[t - 1] + np.random.normal(0, np.sqrt(1 - ar_param ** 2))
        elif autocorrelation_type == "Seasonal":
            # AR(4) with seasonal pattern
            e = np.zeros(n_points)
            e[:4] = np.random.normal(0, 1, 4)
            for t in range(4, n_points):
                e[t] = ar_param * e[t - 4] + np.random.normal(0, np.sqrt(1 - ar_param ** 2))
        else:  # No autocorrelation
            e = np.random.normal(0, 1, n_points)

        # Generate trend component
        trend = 0.1 * time

        # Generate y with trend and autocorrelated errors
        y = 10 + trend + e

        # Fit OLS model
        X = sm.add_constant(time)
        model_ols = sm.OLS(y, X).fit()

        # Plotting
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot data and OLS fit
            ax.scatter(time, y, alpha=0.6, color='#1E88E5', label='Data')
            ax.plot(time, model_ols.fittedvalues, color='#D81B60', linewidth=2,
                    label=f'OLS: y = {model_ols.params[0]:.2f} + {model_ols.params[1]:.4f}t')

            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Y", fontsize=12)
            ax.set_title(f"Time Series with {autocorrelation_type} Autocorrelation", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        # Residual analysis
        st.markdown("### Residual Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Residuals vs time
            ols_resid = model_ols.resid

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(time, ols_resid, alpha=0.6, color='#1E88E5')
            ax.axhline(y=0, color='#D81B60', linestyle='--')

            # Add smoothed line
            from scipy.ndimage import gaussian_filter1d

            smoothed = gaussian_filter1d(ols_resid, sigma=3)
            ax.plot(time, smoothed, color='#004D40', linewidth=2)

            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Residuals", fontsize=12)
            ax.set_title("Residuals vs Time", fontsize=14)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        with col2:
            # Autocorrelation plot
            fig, ax = plt.subplots(figsize=(8, 6))

            lags = min(40, n_points // 5)
            acf = np.zeros(lags)

            # Calculate autocorrelation for each lag
            for lag in range(lags):
                # Correlation between e_t and e_{t-lag}
                if lag == 0:
                    acf[lag] = 1.0  # Correlation with itself is 1
                else:
                    acf[lag] = np.corrcoef(ols_resid[lag:], ols_resid[:-lag])[0, 1]

            # Plot autocorrelation function
            ax.bar(range(lags), acf, width=0.3, color='#1E88E5')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

            # Add confidence bands (approximately ±2/sqrt(T))
            conf_band = 2 / np.sqrt(n_points)
            ax.axhline(y=conf_band, color='#D81B60', linestyle='--', linewidth=1)
            ax.axhline(y=-conf_band, color='#D81B60', linestyle='--', linewidth=1)

            ax.set_xlabel("Lag", fontsize=12)
            ax.set_ylabel("Autocorrelation", fontsize=12)
            ax.set_title("Autocorrelation Function (ACF)", fontsize=14)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        # Durbin-Watson test
        st.markdown("### Durbin-Watson Test for First-Order Autocorrelation")

        from statsmodels.stats.stattools import durbin_watson

        dw_stat = durbin_watson(model_ols.resid)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Durbin-Watson Statistic", f"{dw_stat:.4f}")
        with col2:
            # Interpret DW statistic
            if dw_stat < 1.5:
                dw_interp = "Positive autocorrelation likely"
            elif dw_stat > 2.5:
                dw_interp = "Negative autocorrelation likely"
            else:
                dw_interp = "No strong evidence of autocorrelation"
            st.metric("Interpretation", dw_interp)
        with col3:
            # Reference: DW ~ 2 means no autocorrelation
            st.metric("Reference (No autocorrelation)", "2.0")

        # Compare with ARIMA if autocorrelated
        if autocorrelation_type != "None":
            st.markdown("### Comparison of OLS and ARIMA")

            try:
                from statsmodels.tsa.arima.model import ARIMA

                # Set up ARIMA model based on autocorrelation type
                if autocorrelation_type == "Positive AR(1)" or autocorrelation_type == "Negative AR(1)":
                    arima_order = (1, 0, 0)  # AR(1) model
                elif autocorrelation_type == "Seasonal":
                    arima_order = (4, 0, 0)  # AR(4) model to capture seasonal pattern

                # Add deterministic trend to ARIMA
                arima_model = ARIMA(y, order=arima_order, trend='c')
                arima_results = arima_model.fit()

                # Display parameter estimates
                est_data = {
                    "Model": ["OLS", "ARIMA"],
                    "Intercept": [f"{model_ols.params[0]:.4f}", f"{arima_results.params[0]:.4f}"],
                    "Trend Coefficient": [f"{model_ols.params[1]:.4f}", f"{arima_results.params[1]:.4f}"],
                    "AR Parameter": ["N/A", f"{arima_results.params[2]:.4f}"],
                    "Standard Error (Trend)": [f"{model_ols.bse[1]:.4f}", f"{arima_results.bse[1]:.4f}"],
                    "Residual Variance": [f"{model_ols.mse_resid:.4f}", f"{arima_results.mse:.4f}"]
                }

                est_df = pd.DataFrame(est_data)
                st.table(est_df)

                # Plot fitted values from both models
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.scatter(time, y, alpha=0.4, color='#1E88E5', label='Data')
                ax.plot(time, model_ols.fittedvalues, color='#D81B60', linewidth=2,
                        label='OLS fit')
                ax.plot(time, arima_results.fittedvalues, color='#004D40', linewidth=2, linestyle='--',
                        label='ARIMA fit')

                ax.set_xlabel("Time", fontsize=12)
                ax.set_ylabel("Y", fontsize=12)
                ax.set_title("OLS vs ARIMA Fitted Values", fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

            except Exception as e:
                st.warning(f"Could not fit ARIMA model: {e}")

        # Educational content
        st.markdown("<div class='subsection-header'>Consequences and Remedies</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='highlight'>
        <b>Consequences of Autocorrelation:</b>
        <ul>
            <li>OLS coefficients remain unbiased and consistent but are no longer efficient</li>
            <li>Standard errors are typically underestimated</li>
            <li>t-statistics are inflated, leading to incorrect inference</li>
            <li>Predictions are inefficient</li>
        </ul>

        <b>Detection Methods:</b>
        <ul>
            <li>Visual inspection of residuals over time</li>
            <li>Autocorrelation function (ACF) and partial autocorrelation function (PACF) plots</li>
            <li>Durbin-Watson test for first-order autocorrelation</li>
            <li>Breusch-Godfrey test for higher-order autocorrelation</li>
        </ul>

        <b>Remedies:</b>
        <ul>
            <li><b>HAC Standard Errors:</b> Heteroskedasticity and Autocorrelation Consistent standard errors (Newey-West)</li>
            <li><b>Generalized Least Squares (GLS):</b> Accounts for the correlation structure in residuals</li>
            <li><b>ARIMA Models:</b> Explicitly model the autocorrelation structure</li>
            <li><b>First-Differencing:</b> If the series has a unit root, taking first differences can help</li>
            <li><b>Include lagged variables:</b> Adding lagged dependent or independent variables can reduce autocorrelation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    elif violation == "Multicollinearity":
        st.markdown("<div class='subsection-header'>Multicollinearity</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='highlight'>
        <b>Definition:</b> Multicollinearity occurs when independent variables in a regression model are highly correlated with each other.
        Perfect multicollinearity (exact linear relationship) violates the full rank assumption, while high (but not perfect) multicollinearity 
        makes coefficient estimates unstable and their standard errors large.
        </div>
        """, unsafe_allow_html=True)

        # Interactive demonstration
        col1, col2 = st.columns([1, 2])

        with col1:
            correlation_level = st.slider("Correlation between X1 and X2", 0.0, 0.99, 0.8, 0.01, key="multi_corr")
            sample_size = st.slider("Sample size", 50, 500, 100, 10, key="multi_n")

            if st.button("Generate New Data", key="multi_gen"):
                st.session_state.multi_seed = np.random.randint(0, 1000)

            if 'multi_seed' not in st.session_state:
                st.session_state.multi_seed = 42

        # Generate data with multicollinearity
        np.random.seed(st.session_state.multi_seed)

        # Generate correlated predictors
        mean = [0, 0]
        cov = [[1, correlation_level], [correlation_level, 1]]
        x1, x2 = np.random.multivariate_normal(mean, cov, sample_size).T

        # Scale to 0-10 range
        x1 = (x1 - x1.min()) / (x1.max() - x1.min()) * 10
        x2 = (x2 - x2.min()) / (x2.max() - x2.min()) * 10

        # True coefficients
        beta0 = 2
        beta1 = 1.5
        beta2 = 0.7

        # Generate y
        e = np.random.normal(0, 1, sample_size)
        y = beta0 + beta1 * x1 + beta2 * x2 + e

        # Fit OLS model
        X = sm.add_constant(np.column_stack((x1, x2)))
        model = sm.OLS(y, X).fit()

        # Calculate VIF
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif1 = variance_inflation_factor(X, 1)  # For x1
        vif2 = variance_inflation_factor(X, 2)  # For x2

        # Plotting
        with col2:
            # Scatter plot of x1 vs x2
            fig = px.scatter(
                x=x1, y=x2,
                opacity=0.7,
                labels={"x": "X1", "y": "X2"},
                title=f"Correlation between X1 and X2: {correlation_level:.2f}"
            )

            # Add regression line
            x1_sorted = np.sort(x1)
            model_x1x2 = LinearRegression().fit(x1.reshape(-1, 1), x2)
            y_pred = model_x1x2.predict(x1_sorted.reshape(-1, 1))

            fig.add_trace(
                go.Scatter(
                    x=x1_sorted, y=y_pred,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'y = {model_x1x2.intercept_:.2f} + {model_x1x2.coef_[0]:.2f}x'
                )
            )

            fig.update_layout(height=500)

            st.plotly_chart(fig, use_container_width=True)

        # Regression results
        st.markdown("### Regression Results with Multicollinearity")

        # Also fit single-variable models for comparison
        model_x1 = sm.OLS(y, sm.add_constant(x1)).fit()
        model_x2 = sm.OLS(y, sm.add_constant(x2)).fit()

        # Create comparison table
        results_data = {
            "Model": ["Y ~ X1 + X2", "Y ~ X1", "Y ~ X2"],
            "Intercept": [f"{model.params[0]:.4f}", f"{model_x1.params[0]:.4f}", f"{model_x2.params[0]:.4f}"],
            "Coefficient X1": [f"{model.params[1]:.4f} (SE: {model.bse[1]:.4f})",
                               f"{model_x1.params[1]:.4f} (SE: {model_x1.bse[1]:.4f})", ""],
            "Coefficient X2": [f"{model.params[2]:.4f} (SE: {model.bse[2]:.4f})", "",
                               f"{model_x2.params[1]:.4f} (SE: {model_x2.bse[1]:.4f})"],
            "R-squared": [f"{model.rsquared:.4f}", f"{model_x1.rsquared:.4f}", f"{model_x2.rsquared:.4f}"]
        }

        results_df = pd.DataFrame(results_data)
        st.table(results_df)

        # VIF results
        st.markdown("### Variance Inflation Factors (VIF)")

        vif_data = {
            "Variable": ["X1", "X2"],
            "VIF": [f"{vif1:.4f}", f"{vif2:.4f}"],
            "Interpretation": [
                "Severe multicollinearity" if vif1 > 10 else "Moderate multicollinearity" if vif1 > 5 else "Low multicollinearity",
                "Severe multicollinearity" if vif2 > 10 else "Moderate multicollinearity" if vif2 > 5 else "Low multicollinearity"
            ]
        }

        vif_df = pd.DataFrame(vif_data)
        st.table(vif_df)

        # Coefficient stability with bootstrapping
        st.markdown("### Coefficient Stability: Bootstrap Sampling")

        # Run bootstrap simulation
        n_bootstrap = 200
        beta1_samples = np.zeros(n_bootstrap)
        beta2_samples = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            # Bootstrap sample
            sample_idx = np.random.choice(sample_size, sample_size, replace=True)
            y_boot = y[sample_idx]
            x1_boot = x1[sample_idx]
            x2_boot = x2[sample_idx]

            # Fit model
            X_boot = sm.add_constant(np.column_stack((x1_boot, x2_boot)))
            try:
                model_boot = sm.OLS(y_boot, X_boot).fit()
                beta1_samples[i] = model_boot.params[1]
                beta2_samples[i] = model_boot.params[2]
            except:
                # If singular matrix due to perfect multicollinearity in sample
                beta1_samples[i] = np.nan
                beta2_samples[i] = np.nan

        # Remove NaNs
        beta1_samples = beta1_samples[~np.isnan(beta1_samples)]
        beta2_samples = beta2_samples[~np.isnan(beta2_samples)]

        # Plot bootstrap distribution
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # X1 coefficient
        axs[0].hist(beta1_samples, bins=20, alpha=0.6, color='#1E88E5')
        axs[0].axvline(x=beta1, color='black', linestyle='--', linewidth=2, label=f'True β₁={beta1}')
        axs[0].axvline(x=model.params[1], color='#D81B60', linewidth=2, label=f'OLS β₁={model.params[1]:.4f}')
        axs[0].set_xlabel("Coefficient for X1", fontsize=12)
        axs[0].set_ylabel("Frequency", fontsize=12)
        axs[0].set_title(f"Bootstrap Distribution: β₁ (SD={np.std(beta1_samples):.4f})", fontsize=14)
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # X2 coefficient
        axs[1].hist(beta2_samples, bins=20, alpha=0.6, color='#1E88E5')
        axs[1].axvline(x=beta2, color='black', linestyle='--', linewidth=2, label=f'True β₂={beta2}')
        axs[1].axvline(x=model.params[2], color='#D81B60', linewidth=2, label=f'OLS β₂={model.params[2]:.4f}')
        axs[1].set_xlabel("Coefficient for X2", fontsize=12)
        axs[1].set_ylabel("Frequency", fontsize=12)
        axs[1].set_title(f"Bootstrap Distribution: β₂ (SD={np.std(beta2_samples):.4f})", fontsize=14)
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Educational content
        st.markdown("<div class='subsection-header'>Consequences and Remedies</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='highlight'>
        <b>Consequences of Multicollinearity:</b>
        <ul>
            <li>Coefficient estimates remain unbiased but have large variances</li>
            <li>Standard errors are inflated, reducing the precision of estimates</li>
            <li>t-statistics are reduced, making it harder to detect significant effects</li>
            <li>Coefficient estimates are highly sensitive to small changes in the data</li>
            <li>Signs of coefficients may not make theoretical sense</li>
            <li>Overall model fit (R-squared) can be high despite non-significant individual coefficients</li>
        </ul>

        <b>Detection Methods:</b>
        <ul>
            <li>Correlation matrix among independent variables</li>
            <li>Variance Inflation Factor (VIF): VIF > 10 indicates severe multicollinearity</li>
            <li>Condition number of X'X matrix</li>
            <li>Signs or magnitudes of coefficients contrary to theory</li>
            <li>Coefficients that change dramatically when variables are added or removed</li>
        </ul>

        <b>Remedies:</b>
        <ul>
            <li><b>Remove one of the correlated variables:</b> Keep the theoretically more important one</li>
            <li><b>Create a composite index:</b> Combine correlated variables (e.g., principal component analysis)</li>
            <li><b>Ridge regression or LASSO:</b> Regularization techniques that handle multicollinearity</li>
            <li><b>Increase sample size:</b> Can reduce standard errors despite multicollinearity</li>
            <li><b>Centering variables:</b> Subtract the mean from each variable, particularly helpful for polynomial terms</li>
            <li><b>Use theory:</b> Impose constraints on parameters based on theoretical knowledge</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    elif violation == "Endogeneity/Omitted Variable":
        st.markdown("<div class='subsection-header'>Endogeneity and Omitted Variable Bias</div>",
                    unsafe_allow_html=True)

        st.markdown("""
        <div class='highlight'>
        <b>Definition:</b> Endogeneity occurs when an independent variable is correlated with the error term. 
        A common cause is omitted variable bias, where a relevant variable is excluded from the model and affects both the included variable(s) and the dependent variable.
        </div>
        """, unsafe_allow_html=True)

        # Interactive demonstration
        col1, col2 = st.columns([1, 2])

        with col1:
            omitted_correlation = st.slider("Correlation of omitted variable with X", 0.0, 0.9, 0.7, 0.1,
                                            key="omit_corr_x")
            omitted_effect = st.slider("Effect of omitted variable on Y", 0.0, 3.0, 1.5, 0.1, key="omit_effect")
            sample_size = st.slider("Sample size", 50, 500, 100, 10, key="omit_n")

            if st.button("Generate New Data", key="omit_gen"):
                st.session_state.omit_seed = np.random.randint(0, 1000)

            if 'omit_seed' not in st.session_state:
                st.session_state.omit_seed = 42

        # Generate data with omitted variable
        np.random.seed(st.session_state.omit_seed)

        # Generate correlated variables
        mean = [0, 0]
        cov = [[1, omitted_correlation], [omitted_correlation, 1]]
        x, z = np.random.multivariate_normal(mean, cov, sample_size).T

        # Scale to 0-10 range
        x = (x - x.min()) / (x.max() - x.min()) * 10
        z = (z - z.min()) / (z.max() - z.min()) * 10

        # True coefficients
        beta0 = 2
        beta1 = 1.0
        beta2 = omitted_effect  # Effect of omitted variable

        # Generate y
        e = np.random.normal(0, 1, sample_size)
        y = beta0 + beta1 * x + beta2 * z + e

        # Fit misspecified model (omitting z)
        X_miss = sm.add_constant(x)
        model_miss = sm.OLS(y, X_miss).fit()

        # Fit correctly specified model (including z)
        X_corr = sm.add_constant(np.column_stack((x, z)))
        model_corr = sm.OLS(y, X_corr).fit()

        # Calculate correlation between residuals and omitted variable
        resid_z_corr = np.corrcoef(model_miss.resid, z)[0, 1]

        # Plotting
        with col2:
            # 3D plot to show relationship
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(x, z, y, c=z, cmap='viridis', alpha=0.7)

            # Create a meshgrid for the regression plane
            x_surf = np.linspace(x.min(), x.max(), 20)
            z_surf = np.linspace(z.min(), z.max(), 20)
            x_surf, z_surf = np.meshgrid(x_surf, z_surf)

            # Correctly specified model plane
            y_surf = (model_corr.params[0] +
                      model_corr.params[1] * x_surf +
                      model_corr.params[2] * z_surf)
            ax.plot_surface(x_surf, z_surf, y_surf, alpha=0.2, color='blue')

            # Misspecified model plane (with omitted z)
            y_surf_miss = model_miss.params[0] + model_miss.params[1] * x_surf
            ax.plot_surface(x_surf, z_surf, y_surf_miss, alpha=0.2, color='red')

            ax.set_xlabel('X (Included)', fontsize=10)
            ax.set_ylabel('Z (Omitted)', fontsize=10)
            ax.set_zlabel('Y', fontsize=10)
            ax.set_title('Omitted Variable Bias', fontsize=14)

            # Add a colorbar for z values
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label('Z (Omitted Variable) Value')

            # Add text annotation
            ax.text(x.min(), z.max(), y.max(),
                    f"Blue plane: Full model\nRed plane: Misspecified model",
                    color='black', fontsize=10)

            st.pyplot(fig)

        # Regression results
        st.markdown("### Regression Results: Impact of Omitted Variable")

        # Create comparison table
        results_data = {
            "Model": ["Misspecified (Omits Z)", "Correctly Specified"],
            "Intercept": [f"{model_miss.params[0]:.4f}", f"{model_corr.params[0]:.4f}"],
            "Coefficient X": [f"{model_miss.params[1]:.4f}", f"{model_corr.params[1]:.4f}"],
            "Coefficient Z": ["Omitted", f"{model_corr.params[2]:.4f}"],
            "R-squared": [f"{model_miss.rsquared:.4f}", f"{model_corr.rsquared:.4f}"]
        }

        results_df = pd.DataFrame(results_data)
        st.table(results_df)

        # Theoretical bias calculation
        st.markdown("### Theoretical Omitted Variable Bias")

        # Calculate theoretical bias
        X1 = sm.add_constant(np.column_stack((z, np.ones(sample_size))))
        aux_reg = sm.OLS(x, X1).fit()

        theoretical_bias = aux_reg.params[1] * beta2
        actual_bias = model_miss.params[1] - model_corr.params[1]

        bias_data = {
            "Measure": ["Theoretical Bias", "Actual Bias"],
            "Value": [f"{theoretical_bias:.4f}", f"{actual_bias:.4f}"],
            "Formula": [
                f"Bias = ({aux_reg.params[1]:.4f}) × ({beta2:.4f})",
                f"Bias = ({model_miss.params[1]:.4f}) - ({model_corr.params[1]:.4f})"
            ]
        }

        bias_df = pd.DataFrame(bias_data)
        st.table(bias_df)

        # Residual analysis
        st.markdown("### Residual Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Residuals vs X
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x, model_miss.resid, alpha=0.6, color='#1E88E5')
            ax.axhline(y=0, color='#D81B60', linestyle='--')

            # Add smoothed line
            from scipy.stats import binned_statistic

            bins = min(20, sample_size // 5)
            bin_means, bin_edges, _ = binned_statistic(x, model_miss.resid, statistic='mean', bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax.plot(bin_centers, bin_means, color='#004D40', linewidth=2)

            corr_x_resid = np.corrcoef(x, model_miss.resid)[0, 1]
            ax.text(0.05, 0.95, f"Correlation: {corr_x_resid:.4f}", transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Residuals", fontsize=12)
            ax.set_title("Misspecified Model: Residuals vs X", fontsize=14)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        with col2:
            # Residuals vs Z (omitted variable)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(z, model_miss.resid, alpha=0.6, color='#1E88E5')
            ax.axhline(y=0, color='#D81B60', linestyle='--')

            # Add smoothed line
            bin_means, bin_edges, _ = binned_statistic(z, model_miss.resid, statistic='mean', bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax.plot(bin_centers, bin_means, color='#004D40', linewidth=2)

            corr_z_resid = np.corrcoef(z, model_miss.resid)[0, 1]
            ax.text(0.05, 0.95, f"Correlation: {corr_z_resid:.4f}", transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel("Z (Omitted Variable)", fontsize=12)
            ax.set_ylabel("Residuals", fontsize=12)
            ax.set_title("Misspecified Model: Residuals vs Z", fontsize=14)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        # Instrumental Variables demonstration
        st.markdown("### Instrumental Variables Approach")

        # Generate an instrument
        np.random.seed(st.session_state.omit_seed + 1)

        # Instrument is related to x but not to error term
        w = 0.7 * x + np.random.normal(0, 1, sample_size)

        try:
            # First stage: Regress X on instrument W
            X_first = sm.add_constant(w)
            first_stage = sm.OLS(x, X_first).fit()
            x_hat = first_stage.predict(X_first)

            # Second stage: Use predicted X values
            X_second = sm.add_constant(x_hat)
            iv_model = sm.OLS(y, X_second).fit()

            # Display results
            iv_data = {
                "Model": ["OLS (Misspecified)", "OLS (Correct)", "IV Estimator"],
                "Coefficient X": [
                    f"{model_miss.params[1]:.4f} (SE: {model_miss.bse[1]:.4f})",
                    f"{model_corr.params[1]:.4f} (SE: {model_corr.bse[1]:.4f})",
                    f"{iv_model.params[1]:.4f} (SE: {iv_model.bse[1]:.4f})"
                ],
                "True Value": [f"β = {beta1}", f"β = {beta1}", f"β = {beta1}"],
                "Bias": [
                    f"{model_miss.params[1] - beta1:.4f}",
                    f"{model_corr.params[1] - beta1:.4f}",
                    f"{iv_model.params[1] - beta1:.4f}"
                ]
            }

            iv_df = pd.DataFrame(iv_data)
            st.table(iv_df)

            # First stage F-statistic
            from scipy import stats as spstats

            f_stat = first_stage.fvalue
            f_pvalue = first_stage.f_pvalue

            col1, col2 = st.columns(2)

            with col1:
                st.metric("First-stage F-statistic", f"{f_stat:.4f}")
            with col2:
                st.metric("p-value", f"{f_pvalue:.4f}")

            if f_stat < 10:
                st.warning("Weak instrument warning: First-stage F-statistic < 10")
            else:
                st.success("Strong instrument: First-stage F-statistic > 10")

        except Exception as e:
            st.warning(f"Could not run instrumental variables estimation: {e}")

        # Educational content
        st.markdown("<div class='subsection-header'>Consequences and Remedies</div>", unsafe_allow_html=True)

        st.markdown("""
                <div class='highlight'>
                <b>Consequences of Endogeneity/Omitted Variable Bias:</b>
                <ul>
                    <li>OLS estimators are biased and inconsistent</li>
                    <li>The bias can be positive or negative depending on correlations</li>
                    <li>R-squared is typically lower than it would be with a correctly specified model</li>
                    <li>Incorrect interpretation of causal relationships</li>
                    <li>Residuals are correlated with omitted variables, violating OLS assumptions</li>
                </ul>

                <b>Formula for Omitted Variable Bias:</b>
                <p>If the true model is Y = β₀ + β₁X + β₂Z + ε but we estimate Y = β₀ + β₁X + u, then:</p>
                <p>E(β̂₁) = β₁ + β₂ × δ, where δ is the coefficient from regressing Z on X</p>

                <b>Direction of Bias:</b>
                <ul>
                    <li>If β₂ and δ have the same sign, the bias is positive (overestimation)</li>
                    <li>If β₂ and δ have opposite signs, the bias is negative (underestimation)</li>
                </ul>

                <b>Detection Methods:</b>
                <ul>
                    <li>Theory and domain knowledge about potential omitted variables</li>
                    <li>Examining residuals for patterns or correlations with potential omitted variables</li>
                    <li>Testing for correlation between residuals and suspected omitted variables (if data are available)</li>
                    <li>Using specification tests like the Ramsey RESET test to detect model misspecification</li>
                    <li>Comparing coefficient estimates across models with and without suspected omitted variables</li>
                </ul>

                <b>Remedies for Endogeneity/Omitted Variable Bias:</b>
                <ul>
                    <li><b>Include the omitted variable:</b> Add the omitted variable to the model if data are available.</li>
                    <li><b>Instrumental Variables (IV) Estimation:</b> Use a variable (instrument) correlated with the endogenous variable but not the error term.</li>
                    <li><b>Panel Data Methods:</b> Apply fixed effects or random effects to control for unobserved heterogeneity.</li>
                    <li><b>Quasi-Experimental Methods:</b> Use techniques like difference-in-differences to account for confounders.</li>
                    <li><b>Matching Techniques:</b> Employ propensity score matching to balance observed covariates.</li>
                </ul>
                </div>
        """, unsafe_allow_html=True)
def main():
    st.title("Econometrics Teaching App by Dr Merwan Roudane")
    st.write("Welcome to the interactive econometrics teaching tool!")

if __name__ == "__main__":
    main()

