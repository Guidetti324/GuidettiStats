################################################################################
#                           PSYC250 STREAMLIT APP                              #
################################################################################
# This file reproduces the ~1,700-line Tkinter-based "PSYC250 - Statistical
# Tables Explorer" code, but adapted for Streamlit. It includes:
#   - 7 "tabs" implemented via `st.tabs()`
#   - Param inputs for each distribution
#   - Matplotlib plots with shading
#   - "Show Table Lookup" in an expander, with a "step" slider to highlight 
#     row/column in the table
#   - Binomial entire-tail shading + legend
#
# Usage:
#   streamlit run psyc250_streamlit.py
################################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

# We might configure matplotlib for Agg, but Streamlit usually handles it well.
matplotlib.use("Agg")

################################################################################
# 1) Helper Functions: place_label, draw_red_highlight (adapted for plt only)  #
################################################################################

def place_label(ax, label_positions, x, y, text, color='blue'):
    """
    Similar to the Tkinter version: place text on Axes at (x,y), shifting if needed
    to avoid overlapping previously placed labels.
    """
    offset_x = 0.0
    offset_y = 0.02
    for (xx, yy) in label_positions:
        if abs(x - xx)<0.15 and abs(y - yy)<0.05:
            offset_x += 0.06
            offset_y += 0.04
    final_x = x + offset_x
    final_y = y + offset_y
    ax.text(final_x, final_y, text, color=color, ha="left", va="bottom", fontsize=8)
    label_positions.append((final_x, final_y))

def highlight_cell_html(x1, y1, x2, y2, color="red", width=3):
    """
    A placeholder function. In Tkinter, we used canvas.create_rectangle; 
    but in Streamlit, we'll approach table highlights by conditionally changing 
    background color in an HTML table. We'll just return a style or color for the cell.
    """
    # In real usage, you'd return a style string. We'll do a minimal approach:
    return f"background-color:{color}; border:{width}px solid {color};"

################################################################################
# 2) Table-Rendering Helpers for "Show Table Lookup" in Streamlit              #
################################################################################

def render_t_table_lookup(df_user, alpha_user, tail_s):
    """
    For demonstration, we replicate the "±5 around df" table approach,
    but in Streamlit. We let the user pick a step from a slider, 
    then highlight the row/col accordingly.
    """
    tail_key = "one" if tail_s.startswith("one") else "two"
    df_min = max(1, df_user-5)
    df_max = df_user+5
    df_list = list(range(df_min, df_max+1))

    columns = [
        ("df", None),
        ("one",0.10), ("one",0.05), ("one",0.01), ("one",0.001),
        ("two",0.10), ("two",0.05), ("two",0.01), ("two",0.001)
    ]

    # step approach
    steps = [
        f"1) Highlight row for df={df_user}",
        f"2) Highlight column for tail={tail_key}, α={alpha_user}",
        "3) Intersection => t_crit"
    ]
    # If alpha=0.05 & tail=one => highlight two_0.10
    show_equiv_step = (abs(alpha_user-0.05)<1e-12 and tail_key=="one")
    if show_equiv_step:
        steps.append("4) Notice one_0.05 ~ two_0.10 => highlight that, too!")
    step_max = len(steps)

    st.write("Table Steps:")
    step = st.slider("Step index", 0, step_max, 0, help="Use this slider to highlight row/col in the table.")
    if step>0 and step<= step_max:
        st.write("**Current Step:**", steps[step-1])
    elif step==0:
        st.write("_No highlights yet_")

    def compute_t_crit(dv, mode, a):
        if mode=="one":
            return stats.t.ppf(1- a, dv)
        else:
            return stats.t.ppf(1- a/2, dv)

    # Build table as HTML with highlights
    html = ['<table style="border-collapse:collapse;">']
    # headings
    html.append('<tr>')
    html.append('<th style="border:1px solid black;padding:4px;font-weight:bold;">df</th>')
    for i,(m,a) in enumerate(columns[1:], start=1):
        heading_txt = f"{m}_{a}"
        html.append(f'<th style="border:1px solid black;padding:4px;font-weight:bold;">{heading_txt}</th>')
    html.append('</tr>')

    for dfv in df_list:
        row_html = ['<tr>']
        # df cell
        style = "border:1px solid black;padding:4px;"
        # highlight row if step>=1 for df
        highlight_row = (step>=1 and dfv==df_user)
        if highlight_row:
            style += "background-color:yellow;"
        row_html.append(f'<td style="{style}">{dfv}</td>')

        for c_i,(mode,a) in enumerate(columns[1:],start=1):
            val = compute_t_crit(dfv, mode, a)
            cell_style = "border:1px solid black;padding:4px;"
            # step2 highlight col? step2 => highlight col if (m=tail_key & a=alpha_user).
            highlight_col = False
            highlight_intersect = False
            if step>=2:
                # find col_idx
                match = (mode==tail_key and abs(a-alpha_user)<1e-12)
                if match:
                    highlight_col = True
            if step>=3:
                # intersection => if dfv==df_user and col is the alpha
                if highlight_row and highlight_col:
                    highlight_intersect = True
            if step>=4 and show_equiv_step:
                # highlight two_0.10 as well
                # if alpha_user=0.05 and tail_key= "one", highlight (mode="two", a=0.10)
                eq_match = (mode=="two" and abs(a-0.10)<1e-12)
                if eq_match and dfv==df_user:
                    highlight_intersect = True
                    highlight_col = True
                    highlight_row = True

            if highlight_intersect:
                cell_style += "background-color:lightblue; border:2px solid blue;"
            else:
                if highlight_row:
                    cell_style += "background-color:yellow;"
                elif highlight_col:
                    cell_style += "background-color:yellow;"
            row_html.append(f'<td style="{cell_style}">{val:.4f}</td>')
        row_html.append('</tr>')
        html.append(''.join(row_html))

    html.append('</table>')
    st.markdown(''.join(html), unsafe_allow_html=True)


################################################################################
# 3) Create "Tabs" in Streamlit: st.tabs(...) for each distribution           #
################################################################################

def main():
    st.set_page_config(page_title="PSYC250 Stats Explorer", layout="wide")
    st.title("PSYC250 - Statistical Tables Explorer (Streamlit Version)")

    tabs = st.tabs(["t-Distribution", "z-Distribution", "F-Distribution",
                    "Chi-Square", "Mann–Whitney U", "Wilcoxon Signed-Rank",
                    "Binomial"])
    # We'll define each tab's content in separate functions for clarity

    with tabs[0]:
        show_t_distribution_tab()

    with tabs[1]:
        show_z_distribution_tab()

    with tabs[2]:
        show_f_distribution_tab()

    with tabs[3]:
        show_chi_square_tab()

    with tabs[4]:
        show_mann_whitney_tab()

    with tabs[5]:
        show_wilcoxon_tab()

    with tabs[6]:
        show_binomial_tab()


################################################################################
# 4) T-Distribution Tab (Example)                                             #
################################################################################

def show_t_distribution_tab():
    st.header("t-Distribution")
    st.write("Welcome to Dr Guidetti's Spooktacular Statistical Tool\n"
             "One- or Two-tailed t-tests with ±5 table lookups.")

    # input columns
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        t_val = st.number_input("t statistic", value=2.87)
        df_val = st.number_input("df", value=55)
    with c2:
        alpha_val = st.number_input("Alpha (α)", value=0.05, step=0.01, format="%.3f")
        tail_type = st.radio("Tail Type", ["one-tailed", "two-tailed"])

    # button to update plot
    if st.button("Update t-Plot"):
        fig = plot_t_distribution(t_val, df_val, alpha_val, tail_type)
        st.pyplot(fig)

    with st.expander("Show Table Lookup"):
        st.write("This replicates the ±5 df highlight. Adjust the steps below.")
        render_t_table_lookup(int(df_val), float(alpha_val), tail_type)


def plot_t_distribution(t_val, df, alpha, tail_s):
    fig, ax = plt.subplots(figsize=(5,3), dpi=100)
    label_positions=[]
    x = np.linspace(-4,4,400)
    y = stats.t.pdf(x, df)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey',alpha=0.2,label="Fail to Reject H₀")

    def labelme(xx,yy,txt,c='green'):
        place_label(ax, label_positions, xx, yy, txt, c)

    if tail_s.startswith("one"):
        t_crit= stats.t.ppf(1- alpha, df)
        note_txt=""
        if abs(alpha-0.05)<1e-9:
            note_txt=" (same as two-tailed α=0.10!)"
        rx= x[x>= t_crit]
        ax.fill_between(rx, y[x>=t_crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit, color='green', linestyle='--')
        labelme(t_crit, stats.t.pdf(t_crit, df)+0.02, f"t_crit={t_crit:.4f}{note_txt}", 'green')
        sig= (t_val> t_crit)
        final_crit= t_crit
    else:
        t_crit_r= stats.t.ppf(1- alpha/2, df)
        t_crit_l= stats.t.ppf(alpha/2, df)
        rx= x[x>= t_crit_r]
        lx= x[x<= t_crit_l]
        ax.fill_between(rx, y[x>=t_crit_r], color='red', alpha=0.3)
        ax.fill_between(lx, y[x<=t_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit_r, color='green', linestyle='--')
        ax.axvline(t_crit_l, color='green', linestyle='--')
        labelme(t_crit_r, stats.t.pdf(t_crit_r, df)+0.02, f"+t_crit={t_crit_r:.4f}", 'green')
        labelme(t_crit_l, stats.t.pdf(t_crit_l, df)+0.02, f"-t_crit={t_crit_l:.4f}", 'green')
        sig= (abs(t_val)> abs(t_crit_r))
        final_crit= abs(t_crit_r)

    ax.axvline(t_val, color='blue', linestyle='--')
    place_label(ax, label_positions, t_val, stats.t.pdf(t_val, df)+0.02,
                f"t_calc={t_val:.4f}", 'blue')

    if sig:
        msg= f"t={t_val:.4f} > t_crit={final_crit:.4f} → Reject H₀"
    else:
        msg= f"t={t_val:.4f} ≤ t_crit={final_crit:.4f} → Fail to Reject H₀"

    ax.set_title(f"t-Distribution (df={df})\n{msg}", fontsize=10)
    ax.set_xlabel("t value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig

################################################################################
# 5) Z-Distribution Tab
################################################################################

def show_z_distribution_tab():
    st.header("z-Distribution (One or Two-tailed)")
    z_val= st.number_input("z statistic", value=1.64)
    alpha_val= st.number_input("Alpha (α)", value=0.05, step=0.01)
    tail_type= st.radio("Tail Type", ["one-tailed", "two-tailed"])

    if st.button("Update z-Plot"):
        fig= plot_z_distribution(z_val, alpha_val, tail_type)
        st.pyplot(fig)

    with st.expander("Show Partial z-Table Lookup ±10 rows"):
        st.write("This replicates the partial ztable approach around the user's row.")
        # We'll do a simpler approach:
        z_in= z_val
        row_base= round(0.1* int(z_in*10),1)
        col_part= round(z_in-row_base,2)
        st.write(f"_We would highlight row={row_base}, col={col_part} in a custom table_")

def plot_z_distribution(z_val, alpha, tail_s):
    fig, ax= plt.subplots(figsize=(5,3), dpi=100)
    label_positions=[]
    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey',alpha=0.2,label="Fail to Reject H₀")

    def labelme(xx,yy,txt,c='blue'):
        place_label(ax, label_positions, xx, yy, txt, c)

    if tail_s=="one-tailed":
        z_crit= stats.norm.ppf(1- alpha)
        rx= x[x>= z_crit]
        ax.fill_between(rx, y[x>=z_crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(z_crit, color='green', linestyle='--')
        labelme(z_crit, stats.norm.pdf(z_crit), f"z_crit={z_crit:.4f}", 'green')
        sig= (z_val> z_crit)
        final_crit= z_crit
    else:
        z_crit_r= stats.norm.ppf(1- alpha/2)
        z_crit_l= -z_crit_r
        rx= x[x>=z_crit_r]
        lx= x[x<=z_crit_l]
        ax.fill_between(rx, y[x>=z_crit_r], color='red', alpha=0.3)
        ax.fill_between(lx, y[x<=z_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(z_crit_r, color='green', linestyle='--')
        ax.axvline(z_crit_l, color='green', linestyle='--')
        labelme(z_crit_r, stats.norm.pdf(z_crit_r), f"+z_crit={z_crit_r:.4f}", 'green')
        labelme(z_crit_l, stats.norm.pdf(z_crit_l), f"-z_crit={z_crit_r:.4f}", 'green')
        sig= (abs(z_val)> z_crit_r)
        final_crit= z_crit_r

    ax.axvline(z_val, color='blue', linestyle='--')
    labelme(z_val, stats.norm.pdf(z_val), f"z_calc={z_val:.4f}", 'blue')

    msg= (f"z={z_val:.4f} > z_crit={final_crit:.4f} → Reject H₀" if sig
          else f"z={z_val:.4f} ≤ z_crit={final_crit:.4f} → Fail to Reject H₀")

    ax.set_title(f"Z-Distribution\n{msg}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    return fig

################################################################################
# 6) F-Distribution Tab
################################################################################

def show_f_distribution_tab():
    st.header("F-Distribution (One-tailed)")
    f_val= st.number_input("F statistic", value=3.49)
    df1= st.number_input("df1", value=3)
    df2= st.number_input("df2", value=12)
    alpha= st.number_input("Alpha (α)", value=0.05, step=0.01)

    if st.button("Update F-Plot"):
        fig= plot_f_distribution(f_val, df1, df2, alpha)
        st.pyplot(fig)

    with st.expander("Show F Table Lookup (±5)"):
        st.write("We can highlight row df1, col df2, etc., but we'll do a minimal approach.")


def plot_f_distribution(f_val, df1, df2, alpha):
    fig, ax= plt.subplots(figsize=(5,3), dpi=100)
    label_positions=[]
    x= np.linspace(0,5,500)
    y= stats.f.pdf(x, df1, df2)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey',alpha=0.2,label="Fail to Reject H₀")

    f_crit= stats.f.ppf(1-alpha, df1, df2)
    rx= x[x>= f_crit]
    ry= y[x>= f_crit]
    ax.fill_between(rx, ry, color='red', alpha=0.3, label="Reject H₀")
    ax.axvline(f_crit, color='green', linestyle='--')
    place_label(ax, label_positions, f_crit, stats.f.pdf(f_crit,df1,df2)+0.02,
                f"F_crit={f_crit:.4f}", 'green')

    ax.axvline(f_val, color='blue', linestyle='--')
    place_label(ax, label_positions, f_val, stats.f.pdf(f_val,df1,df2)+0.02,
                f"F_calc={f_val:.4f}", 'blue')

    sig= (f_val> f_crit)
    msg= (f"F={f_val:.4f} > F_crit={f_crit:.4f} → Reject H₀"
          if sig else
          f"F={f_val:.4f} ≤ F_crit={f_crit:.4f} → Fail to Reject H₀")

    ax.set_title(f"F-Distribution (df1={df1}, df2={df2})\n{msg}", fontsize=10)
    ax.set_xlabel("F value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig

################################################################################
# 7) Chi-Square Tab
################################################################################

def show_chi_square_tab():
    st.header("Chi-Square (One-tailed)")
    chi_val= st.number_input("Chi-square stat", value=10.5)
    df_val= st.number_input("df", value=12)
    alpha= st.number_input("Alpha (α)", value=0.05, step=0.01)

    if st.button("Update χ²-Plot"):
        fig= plot_chi_square(chi_val, df_val, alpha)
        st.pyplot(fig)

    with st.expander("Show Chi-Square Table Lookup"):
        st.write("_Minimal demonstration of ±5 around df_")

def plot_chi_square(chi_val, df, alpha):
    fig, ax= plt.subplots(figsize=(5,3), dpi=100)
    label_positions=[]
    x_max= max(30, df*2)
    x= np.linspace(0,x_max,400)
    y= stats.chi2.pdf(x, df)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey',alpha=0.2,label="Fail to Reject H₀")

    chi_crit= stats.chi2.ppf(1-alpha, df)
    rx= x[x>= chi_crit]
    ry= y[x>= chi_crit]
    ax.fill_between(rx, ry, color='red', alpha=0.3, label="Reject H₀")
    ax.axvline(chi_crit, color='green', linestyle='--')
    place_label(ax, label_positions, chi_crit, stats.chi2.pdf(chi_crit, df)+0.02,
                f"chi2_crit={chi_crit:.4f}", 'green')

    ax.axvline(chi_val, color='blue', linestyle='--')
    place_label(ax, label_positions, chi_val, stats.chi2.pdf(chi_val, df)+0.02,
                f"chi2_calc={chi_val:.4f}", 'blue')

    sig= (chi_val> chi_crit)
    msg= (f"χ²={chi_val:.4f} > χ²_crit={chi_crit:.4f} → Reject H₀"
          if sig else
          f"χ²={chi_val:.4f} ≤ χ²_crit={chi_crit:.4f} → Fail to Reject H₀")

    ax.set_title(f"Chi-Square (df={df})\n{msg}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    return fig

################################################################################
# 8) Mann–Whitney U Tab
################################################################################

def show_mann_whitney_tab():
    st.header("Mann–Whitney U (one/two-tailed, approximate normal => z)")

    U_val= st.number_input("U statistic", value=5)
    n1= st.number_input("n1", value=5)
    n2= st.number_input("n2", value=6)
    alpha= st.number_input("Alpha (α)", value=0.05, step=0.01)
    tail_s= st.radio("Tail Type", ["one-tailed","two-tailed"])

    if st.button("Update MW Plot"):
        fig= plot_mann_whitney(U_val,n1,n2,alpha,tail_s)
        st.pyplot(fig)

    with st.expander("Show Mann–Whitney Table Lookup (±5)"):
        st.write("_Approx approach, minimal demonstration_")

def plot_mann_whitney(U_val,n1,n2,alpha,tail_s):
    fig, ax= plt.subplots(figsize=(5,3), dpi=100)
    label_positions=[]
    # approximate z
    meanU= n1*n2/2
    sdU= np.sqrt(n1*n2*(n1+n2+1)/12)
    z_val= (U_val-meanU)/ sdU
    # pick zCrit
    if tail_s=="one-tailed":
        zCrit= stats.norm.ppf(1-alpha)
        sig= (z_val> zCrit)
    else:
        zCrit= stats.norm.ppf(1-alpha/2)
        sig= (abs(z_val)> zCrit)

    # Plot normal
    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2,label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        rx= x[x>= zCrit]
        ry= y[x>= zCrit]
        ax.fill_between(rx, ry, color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit), f"z_crit={zCrit:.2f}", 'green')
    else:
        zCrit_r= zCrit
        zCrit_l= -zCrit
        rx= x[x>= zCrit_r]
        lx= x[x<= zCrit_l]
        ax.fill_between(rx, y[x>=zCrit_r], color='red', alpha=0.3)
        ax.fill_between(lx, y[x<=zCrit_l], color='red', alpha=0.3,label="Reject H₀")
        ax.axvline(zCrit_r, color='green', linestyle='--')
        ax.axvline(zCrit_l, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit_r, stats.norm.pdf(zCrit_r), f"+z_crit={zCrit_r:.2f}", 'green')
        place_label(ax, label_positions, zCrit_l, stats.norm.pdf(zCrit_l), f"-z_crit={zCrit_r:.2f}", 'green')

    ax.axvline(z_val, color='blue', linestyle='--')
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val), f"z_calc={z_val:.2f}", 'blue')

    msg= (f"U={U_val}, z={z_val:.2f} => Reject H₀" if sig else
          f"U={U_val}, z={z_val:.2f} => Fail to Reject H₀")
    ax.set_title(f"Mann–Whitney: n1={n1}, n2={n2}\n{msg}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    return fig

################################################################################
# 9) Wilcoxon Signed-Rank Tab
################################################################################

def show_wilcoxon_tab():
    st.header("Wilcoxon Signed-Rank (one/two-tailed, approx normal => z)")

    T_val= st.number_input("T statistic", value=5)
    N_val= st.number_input("N (non-zero diffs)", value=6)
    alpha= st.number_input("Alpha (α)", value=0.05, step=0.01)
    tail_s= st.radio("Tail Type", ["one-tailed","two-tailed"])

    if st.button("Update Wilcoxon Plot"):
        fig= plot_wilcoxon(T_val, N_val, alpha, tail_s)
        st.pyplot(fig)

    with st.expander("Show Wilcoxon Table Lookup"):
        st.write("_Approx T_crit approach, minimal demonstration_")

def plot_wilcoxon(T_val, N, alpha, tail_s):
    fig, ax= plt.subplots(figsize=(5,3), dpi=100)
    label_positions=[]
    meanT= N*(N+1)/4
    sdT= np.sqrt(N*(N+1)*(2*N+1)/24)
    z_val= (T_val- meanT)/ sdT

    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2,label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        zCrit= stats.norm.ppf(1- alpha)
        rx= x[x>= zCrit]
        ax.fill_between(rx, y[x>=zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit), f"z_crit={zCrit:.2f}", 'green')
        sig= (z_val> zCrit)
    else:
        zCrit= stats.norm.ppf(1- alpha/2)
        zCritR= zCrit
        zCritL= -zCrit
        rx= x[x>= zCritR]
        lx= x[x<= zCritL]
        ax.fill_between(rx, y[x>=zCritR], color='red', alpha=0.3)
        ax.fill_between(lx, y[x<=zCritL], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCritR, color='green', linestyle='--')
        ax.axvline(zCritL, color='green', linestyle='--')
        place_label(ax, label_positions, zCritR, stats.norm.pdf(zCritR), f"+z_crit={zCritR:.2f}", 'green')
        place_label(ax, label_positions, zCritL, stats.norm.pdf(zCritL), f"-z_crit={zCritR:.2f}", 'green')
        sig= (abs(z_val)> zCrit)

    ax.axvline(z_val, color='blue', linestyle='--')
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val), f"z_calc={z_val:.2f}", 'blue')

    msg= (f"T={T_val}, z={z_val:.2f} => Reject H₀" if sig else
          f"T={T_val}, z={z_val:.2f} => Fail to Reject H₀")
    ax.set_title(f"Wilcoxon (N={N})\n{msg}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    return fig

################################################################################
# 10) Binomial Tab
################################################################################

def show_binomial_tab():
    st.header("Binomial (one/two-tailed) with entire region shading + legend")

    n= st.number_input("n", value=10)
    x= st.number_input("x (successes)", value=3)
    p= st.number_input("p", value=0.5, step=0.01)
    alpha= st.number_input("Alpha (α)", value=0.05, step=0.01)
    tail_s= st.radio("Tail Type", ["one-tailed","two-tailed"])

    if st.button("Update Binomial Plot"):
        fig= plot_binomial(n,x,p,alpha,tail_s)
        st.pyplot(fig)

    with st.expander("Show Binomial Table Lookup (±5)"):
        st.write("_We highlight row => col => intersection similarly._")

def plot_binomial(n,x,p,alpha,tail_s):
    fig, ax= plt.subplots(figsize=(5,3), dpi=100)
    label_positions=[]
    k_vals= np.arange(n+1)
    pmf_vals= stats.binom.pmf(k_vals, n, p)
    bars= ax.bar(k_vals, pmf_vals, color='lightgrey', edgecolor='black')

    mean_ = n*p
    # compute p_value
    if tail_s=="one-tailed":
        if x<= mean_:
            p_val= stats.binom.cdf(x, n, p)
        else:
            p_val= 1- stats.binom.cdf(x-1,n,p)
    else:
        if x<= mean_:
            p_val= stats.binom.cdf(x,n,p)*2
        else:
            p_val= (1- stats.binom.cdf(x-1,n,p))*2
        p_val= min(p_val,1.0)

    sig= (p_val< alpha)
    msg= f"p-value={p_val:.4f} => " + ("Reject H₀" if sig else "Fail to Reject H₀")

    # shade region
    if tail_s=="one-tailed":
        if x< mean_:
            for i in range(x+1):
                bars[i].set_color('red')
        else:
            for i in range(x, n+1):
                bars[i].set_color('red')
    else:
        if x<= mean_:
            for i in range(x+1):
                bars[i].set_color('red')
            hi_start= max(0,n-x)
            for i in range(hi_start,n+1):
                bars[i].set_color('red')
        else:
            for i in range(x,n+1):
                bars[i].set_color('red')
            for i in range(0, (n-x)+1):
                bars[i].set_color('red')

    if 0<= x<= n:
        bars[x].set_color('blue')

    # custom legend
    import matplotlib.patches as mpatches
    legend_patches= [
        mpatches.Patch(facecolor='lightgrey', edgecolor='black', label='Fail to Reject H₀'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='Reject H₀ region'),
        mpatches.Patch(facecolor='blue', edgecolor='black', label='Observed x')
    ]
    ax.legend(handles=legend_patches, loc='best')

    ax.set_title(f"Binomial(n={n}, p={p:.2f})\n{msg}", fontsize=10)
    ax.set_xlabel("x (successes)")
    ax.set_ylabel("PMF")
    fig.tight_layout()
    return fig

################################################################################
# 11) Actually run the app
################################################################################

if __name__=="__main__":
    main()
