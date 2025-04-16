import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="PSYC250 - Statistical Tables Explorer", layout="wide")

st.title("PSYC250 - Statistical Tables Explorer")
st.markdown("""
This interactive tool visualises statistical distributions with rejection regions,
critical values, and calculated test statistics. Select your distribution below
to begin.
""")

# Sidebar - User selects distribution type
distribution = st.sidebar.selectbox(
    "Choose a distribution",
    ("t-Distribution", "z-Distribution", "F-Distribution", "Chi-Square",
     "Mann–Whitney U", "Wilcoxon Signed-Rank", "Binomial")
)

# Common helper function for plotting labels
def place_label(ax, x, y, text, color='blue'):
    ax.text(x, y, text, color=color, ha="left", va="bottom", fontsize=8)

# t-Distribution Tab
if distribution == "t-Distribution":
    st.header("t-Distribution")
    t_val = st.number_input("t statistic", value=2.87, format="%.4f")
    df = st.number_input("Degrees of freedom (df)", value=55, step=1)
    alpha = st.number_input("Alpha (α)", value=0.05, format="%.3f")
    tail_type = st.radio("Tail type", ["one-tailed", "two-tailed"])

    x = np.linspace(-4, 4, 500)
    y = stats.t.pdf(x, df)

    fig, ax = plt.subplots()
    ax.plot(x, y, color="black", label="t-distribution")
    ax.fill_between(x, y, color="lightgrey", alpha=0.2, label="Fail to Reject H₀")

    if tail_type == "one-tailed":
        t_crit = stats.t.ppf(1 - alpha, df)
        ax.axvline(t_crit, color="green", linestyle="--", label=f"t_crit = {t_crit:.4f}")
        ax.fill_between(x[x >= t_crit], y[x >= t_crit], color='red', alpha=0.3, label="Reject H₀")
        significant = t_val > t_crit
        result = f"t = {t_val:.4f} {'>' if significant else '≤'} t_crit = {t_crit:.4f} → {'Reject' if significant else 'Fail to Reject'} H₀"
    else:
        t_crit_r = stats.t.ppf(1 - alpha/2, df)
        t_crit_l = stats.t.ppf(alpha/2, df)
        ax.axvline(t_crit_r, color="green", linestyle="--", label=f"+t_crit = {t_crit_r:.4f}")
        ax.axvline(t_crit_l, color="green", linestyle="--", label=f"-t_crit = {t_crit_l:.4f}")
        ax.fill_between(x[x >= t_crit_r], y[x >= t_crit_r], color='red', alpha=0.3)
        ax.fill_between(x[x <= t_crit_l], y[x <= t_crit_l], color='red', alpha=0.3, label="Reject H₀")
        significant = abs(t_val) > t_crit_r
        result = f"|t| = {abs(t_val):.4f} {'>' if significant else '≤'} t_crit = {t_crit_r:.4f} → {'Reject' if significant else 'Fail to Reject'} H₀"

    ax.axvline(t_val, color='blue', linestyle='--', label=f"t_calc = {t_val:.4f}")
    ax.legend()
    ax.set_title(f"t-Distribution (df={df})")
    ax.set_xlabel("t value")
    ax.set_ylabel("Density")
    st.pyplot(fig)
    st.subheader("Result")
    st.markdown(f"**{result}**")

# Placeholder for other distributions
def show_placeholder(name):
    st.header(name)
    st.info(f"The interactive plot for {name} is coming soon. Stay tuned!")

if distribution == "z-Distribution":
    show_placeholder("z-Distribution")
elif distribution == "F-Distribution":
    show_placeholder("F-Distribution")
elif distribution == "Chi-Square":
    show_placeholder("Chi-Square")
elif distribution == "Mann–Whitney U":
    show_placeholder("Mann–Whitney U")
elif distribution == "Wilcoxon Signed-Rank":
    show_placeholder("Wilcoxon Signed-Rank")
elif distribution == "Binomial":
    show_placeholder("Binomial")
