import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")

st.set_page_config(
    page_title="PSYC250 - Streamlit Stats Explorer (No Duplicates)",
    layout="wide"
)

def place_label(ax, label_positions, x, y, text, color='blue'):
    offset_x = 0.0
    offset_y = 0.02
    for (xx, yy) in label_positions:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            offset_x += 0.06
            offset_y += 0.04
    final_x = x + offset_x
    final_y = y + offset_y
    ax.text(final_x, final_y, text, color=color, ha="left", va="bottom", fontsize=8)
    label_positions.append((final_x, final_y))

###############################################################################
# t-DISTRIBUTION TAB
###############################################################################

def show_t_distribution_tab():
    st.subheader("Tab 1: t-Distribution")

    col1, col2 = st.columns(2)
    with col1:
        t_val = st.number_input("Enter t statistic (Tab1)",
                                value=2.87,
                                key="tab1_tstat")
        df_val = st.number_input("df (Tab1)",
                                 value=55,
                                 key="tab1_df")
    with col2:
        alpha_val = st.number_input("Alpha for t-dist (Tab1)",
                                    value=0.05,
                                    step=0.01,
                                    key="tab1_alpha")
        tail_type = st.radio("Tail Type (Tab1 t-dist)",
                             ["one-tailed", "two-tailed"],
                             key="tab1_tail")

    if st.button("Update Plot (Tab1)", key="tab1_update"):
        fig = plot_t_distribution(t_val, df_val, alpha_val, tail_type)
        st.pyplot(fig)

    with st.expander("Show t-Table Lookup (±5) (Tab1)", key="tab1_expander"):
        st.write("Pretend we highlight ±5 around df etc.")

def plot_t_distribution(t_val, df, alpha, tail_s):
    # same function body as before...
    fig, ax = plt.subplots(figsize=(5,3))
    ...
    return fig

###############################################################################
# z-DISTRIBUTION TAB
###############################################################################
def show_z_distribution_tab():
    st.subheader("Tab 2: z-Distribution")
    ...
    # Make sure all inputs have key="tab2_..."

###############################################################################
# F-DISTRIBUTION TAB
###############################################################################
def show_f_distribution_tab():
    st.subheader("Tab 3: F-Distribution")

    c1, c2 = st.columns(2)
    with c1:
        f_val = st.number_input("F statistic (Tab3)",
                                value=3.49,
                                key="tab3_fval")
        df1 = st.number_input("df1 (Tab3)",
                              value=3,
                              key="tab3_df1")
        df2 = st.number_input("df2 (Tab3)",
                              value=12,
                              key="tab3_df2")
    with c2:
        alpha = st.number_input("Alpha (Tab3 F-dist)",
                                value=0.05,
                                step=0.01,
                                key="tab3_alpha")

    if st.button("Update Plot (Tab3)", key="tab3_update"):
        fig = plot_f_distribution(f_val, df1, df2, alpha)
        st.pyplot(fig)

    with st.expander("Show F Table Lookup ±5 (Tab3)", key="tab3_expander"):
        st.write("Pretend highlight row df1, col df2, etc.")

def plot_f_distribution(f_val, df1, df2, alpha):
    # same function body as before...
    fig, ax = plt.subplots(figsize=(5,3))
    ...
    return fig

# ...and so on for Chi-Square, Mann–Whitney, Wilcoxon, Binomial tabs...
# just keep each widget labeled uniquely with distinct keys.

def main():
    st.title("PSYC250 - Statistical Tables Explorer (No Duplicate IDs)")

    tab_labels = [
        "t-Distribution",
        "z-Distribution",
        "F-Distribution",
        "Chi-Square",
        "Mann–Whitney U",
        "Wilcoxon Signed-Rank",
        "Binomial"
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        show_t_distribution_tab()
    with tabs[1]:
        show_z_distribution_tab()
    with tabs[2]:
        show_f_distribution_tab()
    # ... etc.

if __name__=="__main__":
    main()
