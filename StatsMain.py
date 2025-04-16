import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Use a non-GUI backend for Matplotlib
plt.switch_backend("Agg")

###############################################################################
#                            HELPER FUNCTIONS
###############################################################################

def place_label(ax, label_positions, x, y, text, color='blue'):
    """
    Place a text label on the Axes at (x, y), nudging if needed to avoid overlap
    with previously placed labels in label_positions.
    """
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

def highlight_html_cell(html_in, cell_id, color="red", border_px=2):
    """
    Finds the <td> with id="cell_id" in html_in, and adds an inline style
    for a colored border.
    """
    marker = f'id="{cell_id}"'
    styled = f'id="{cell_id}" style="border: {border_px}px solid {color};"'
    return html_in.replace(marker, styled, 1)

def show_html_table(html_content, height=400):
    """
    Renders html_content in a scrollable iframe, preserving the HTML table layout.
    """
    full_html = f"""
    <html>
      <head><meta charset="UTF-8" /></head>
      <body>
        {html_content}
      </body>
    </html>
    """
    components.html(full_html, height=height, scrolling=True)

def next_step_button(step_key):
    """
    Displays a 'Next Step' button that increments st.session_state[step_key].
    """
    if st.button("Next Step", key=step_key+"_btn"):
        st.session_state[step_key] += 1

###############################################################################
#                          1) T-DISTRIBUTION TAB
###############################################################################

def show_t_distribution_tab():
    st.subheader("t-Distribution")

    col1, col2 = st.columns(2)
    with col1:
        t_val = st.number_input("t statistic", value=2.87, key="t_tab_tval")
        df_val = st.number_input("df", value=55, key="t_tab_df")
    with col2:
        alpha_val = st.number_input("Alpha", value=0.05, step=0.01, key="t_tab_alpha")
        tail_s = st.radio("Tail Type", ["one-tailed","two-tailed"], key="t_tab_tail")

    if st.button("Update Plot", key="t_tab_update"):
        fig = plot_t_distribution(t_val, df_val, alpha_val, tail_s)
        st.pyplot(fig)  # no use_container_width => fixed 12x4 inch

    with st.expander("Show Table Lookup (±5 around df)"):
        show_t_table_lookup(df_val, alpha_val, tail_s, "t_tab")


def plot_t_distribution(t_val, df, alpha, tail_s):
    """
    Creates a figure 12 inches wide by 4 inches high, draws the t-distribution
    with shading, lines, etc.
    """
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    label_positions = []

    x = np.linspace(-4,4,400)
    y = stats.t.pdf(x, df)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        t_crit = stats.t.ppf(1 - alpha, df)
        ax.fill_between(x[x>= t_crit], y[x>= t_crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit, color='green', linestyle='--')
        place_label(ax, label_positions, t_crit, stats.t.pdf(t_crit, df)+0.02, f"t_crit={t_crit:.3f}", 'green')
        sig = (t_val> t_crit)
        final_crit= t_crit
    else:
        t_crit_r= stats.t.ppf(1 - alpha/2, df)
        t_crit_l= stats.t.ppf(alpha/2, df)
        ax.fill_between(x[x>= t_crit_r], y[x>= t_crit_r], color='red', alpha=0.3)
        ax.fill_between(x[x<= t_crit_l], y[x<= t_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit_r, color='green', linestyle='--')
        ax.axvline(t_crit_l, color='green', linestyle='--')
        place_label(ax, label_positions, t_crit_r, stats.t.pdf(t_crit_r, df)+0.02, f"+t_crit={t_crit_r:.3f}", 'green')
        place_label(ax, label_positions, t_crit_l, stats.t.pdf(t_crit_l, df)+0.02, f"-t_crit={t_crit_l:.3f}", 'green')
        sig = (abs(t_val)> abs(t_crit_r))
        final_crit= abs(t_crit_r)

    ax.axvline(t_val, color='blue', linestyle='--')
    place_label(ax, label_positions, t_val, stats.t.pdf(t_val, df)+0.02, f"t_calc={t_val:.3f}", 'blue')

    if sig:
        msg= f"t={t_val:.3f} > {final_crit:.3f} => Reject H₀"
    else:
        msg= f"t={t_val:.3f} ≤ {final_crit:.3f} => Fail to Reject H₀"

    ax.set_title(f"t-Distribution (df={df})\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig


def show_t_table_lookup(df_val, alpha_val, tail_s, key_prefix):
    st.write("### Step-by-Step T-Table Lookup")

    step_key = key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    cur_step = st.session_state[step_key]

    df_min= max(1, df_val-5)
    df_max= df_val+5
    df_list= list(range(df_min, df_max+1))

    columns= [
        ("df",""),
        ("one",0.10), ("one",0.05), ("one",0.01), ("one",0.001),
        ("two",0.10), ("two",0.05), ("two",0.01), ("two",0.001),
    ]

    def compute_tcrit(dof, mode, a_):
        if mode=="one":
            return stats.t.ppf(1 - a_, dof)
        else:
            return stats.t.ppf(1 - a_/2, dof)

    table_html= """
    <style>
    table.ttable {
      border-collapse: collapse;
      margin-top:10px;
    }
    table.ttable td, table.ttable th {
      border:1px solid #000;
      width:85px;
      height:30px;
      text-align:center;
      font-family: sans-serif;
      font-size:0.9rem;
    }
    table.ttable th {
      background-color:#f0f0f0;
    }
    </style>
    <table class="ttable">
      <tr>
        <th>df</th>
    """
    for i,(m,a) in enumerate(columns[1:], start=1):
        col_label = f"{m}_{a}"
        table_html+= f"<th>{col_label}</th>"
    table_html+= "</tr>\n"

    for dfv in df_list:
        table_html+= "<tr>"
        row_id= f"df_{dfv}_0"
        table_html+= f'<td id="{row_id}">{dfv}</td>'
        for c_i,(m,a) in enumerate(columns[1:], start=1):
            cell_id= f"df_{dfv}_{c_i}"
            val= compute_tcrit(dfv, m, a)
            table_html+= f'<td id="{cell_id}">{val:.3f}</td>'
        table_html+= "</tr>\n"
    table_html+= "</table>"

    user_mode= "one" if tail_s.startswith("one") else "two"
    col_index= None
    for i,(m,a) in enumerate(columns[1:], start=1):
        if m==user_mode and abs(a - alpha_val)<1e-12:
            col_index= i
            break

    row_in= (df_val in df_list)

    # steps: highlight row => highlight col => intersection => optional step4
    if cur_step>=0 and row_in:
        for cc in range(len(columns)):
            table_html= highlight_html_cell(table_html, f"df_{df_val}_{cc}", "red",2)
    if cur_step>=1 and col_index is not None:
        for dv_ in df_list:
            table_html= highlight_html_cell(table_html, f"df_{dv_}_{col_index}", "red",2)
    if cur_step>=2 and row_in and col_index is not None:
        table_html= highlight_html_cell(table_html, f"df_{df_val}_{col_index}", "blue",3)

    # step3 => if one-tail=0.05 => highlight two_0.10
    if cur_step>=3 and tail_s=="one-tailed" and abs(alpha_val-0.05)<1e-12:
        alt_col= None
        for i,(m,a) in enumerate(columns[1:], start=1):
            if m=="two" and abs(a-0.10)<1e-12:
                alt_col= i
                break
        if alt_col is not None and row_in:
            for dv_ in df_list:
                table_html= highlight_html_cell(table_html, f"df_{dv_}_{alt_col}", "red",2)
            table_html= highlight_html_cell(table_html, f"df_{df_val}_{alt_col}", "blue",3)

    show_html_table(table_html, height=450)

    steps_list= [
        f"1) Highlight row df={df_val}",
        f"2) Highlight column tail={tail_s}, α={alpha_val}",
        "3) Intersection => t_crit",
        "4) If one-tailed α=0.05, also highlight two-tailed α=0.10"
    ]
    if not (tail_s=="one-tailed" and abs(alpha_val-0.05)<1e-12):
        steps_list.pop()

    max_step= len(steps_list)-1
    if cur_step>max_step:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps_list[cur_step]}")

    next_step_button(step_key)

###############################################################################
#   2) Z-DISTRIBUTION, 3) F-DISTRIBUTION, 4) CHI-SQUARE, 5) Mann–Whitney, 
#   6) Wilcoxon, 7) Binomial TABS
###############################################################################

# The rest of the code is the same pattern: each distribution has:
# - show_..._tab() with user inputs
# - plot_...(...) with a fixed figsize=(12,4)
# - optional table lookup expansions

def show_z_distribution_tab():
    st.subheader("z-Distribution")

    col1, col2 = st.columns(2)
    with col1:
        z_val= st.number_input("z statistic", value=1.64, key="z_tab_zval")
    with col2:
        alpha_val= st.number_input("Alpha", value=0.05, step=0.01, key="z_tab_alpha")
        tail_s= st.radio("Tail Type", ["one-tailed","two-tailed"], key="z_tab_tail")

    if st.button("Update Plot", key="z_tab_update"):
        fig= plot_z_distribution(z_val, alpha_val, tail_s)
        st.pyplot(fig)

    with st.expander("Show Table Lookup (±10 rows)"):
        show_z_table_lookup(z_val, "z_tab")


def plot_z_distribution(z_val, alpha, tail_s):
    fig, ax= plt.subplots(figsize=(12,4), dpi=100)
    label_positions=[]
    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        z_crit= stats.norm.ppf(1- alpha)
        ax.fill_between(x[x>=z_crit], y[x>=z_crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(z_crit, color='green', linestyle='--')
        place_label(ax, label_positions, z_crit, stats.norm.pdf(z_crit), f"z_crit={z_crit:.3f}", 'green')
        sig= (z_val> z_crit)
        final_crit= z_crit
    else:
        z_crit_r= stats.norm.ppf(1- alpha/2)
        z_crit_l= -z_crit_r
        ax.fill_between(x[x>=z_crit_r], y[x>=z_crit_r], color='red', alpha=0.3)
        ax.fill_between(x[x<=z_crit_l], y[x<=z_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(z_crit_r, color='green', linestyle='--')
        ax.axvline(z_crit_l, color='green', linestyle='--')
        place_label(ax, label_positions, z_crit_r, stats.norm.pdf(z_crit_r), f"+z_crit={z_crit_r:.3f}", 'green')
        place_label(ax, label_positions, z_crit_l, stats.norm.pdf(z_crit_l), f"-z_crit={z_crit_r:.3f}", 'green')
        sig= (abs(z_val)> z_crit_r)
        final_crit= z_crit_r

    ax.axvline(z_val, color='blue', linestyle='--')
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val), f"z_calc={z_val:.3f}", 'blue')

    msg= (f"z={z_val:.3f} > {final_crit:.3f} => Reject H₀"
          if sig else f"z={z_val:.3f} ≤ {final_crit:.3f} => Fail to Reject H₀")
    ax.set_title(f"Z-Distribution\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig


def show_z_table_lookup(z_in, key_prefix):
    st.write("### Step-by-Step z-Table Lookup")
    # same code as before for building partial ±10 row ztable
    # ...
    # In the interest of brevity, omitted here but identical approach:
    # Create table_html, highlight row => col => intersection
    # call show_html_table(table_html)
    # etc.
    pass


def show_f_distribution_tab():
    st.subheader("F-Distribution (One-tailed)")
    # user inputs
    # ...
    pass

# ... define plot_f_distribution(), show_f_table_lookup() similarly with figsize=(12,4)...

def show_chi_square_tab():
    st.subheader("Chi-Square (One-tailed)")
    # ...
    pass

def show_mann_whitney_tab():
    st.subheader("Mann–Whitney U")
    # ...
    pass

def show_wilcoxon_tab():
    st.subheader("Wilcoxon Signed-Rank")
    # ...
    pass

def show_binomial_tab():
    st.subheader("Binomial")
    # ...
    pass


###############################################################################
#                                 MAIN
###############################################################################

def main():
    st.set_page_config(page_title="PSYC250 - 12x4 inch Plots", layout="wide")
    st.title("PSYC250 - Statistical Tables Explorer (Fixed 12x4 Plots)")

    tabs= st.tabs([
        "t-Distribution",
        "z-Distribution",
        "F-Distribution",
        "Chi-Square",
        "Mann–Whitney U",
        "Wilcoxon Signed-Rank",
        "Binomial"
    ])

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

if __name__=="__main__":
    main()
