import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Use a non-GUI backend for Matplotlib
plt.switch_backend("Agg")

###############################################################################
#                     HELPER FUNCTIONS
###############################################################################

def place_label(ax, label_positions, x, y, text, color='blue'):
    """
    Places a text label on the Axes at (x, y), shifting if needed
    to avoid overlapping previously placed labels.
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
    Renders html_content inside a scrollable iframe so that the table
    is displayed properly with all rows & columns.
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
#                     DYNAMIC FIGURE SIZE FROM SIDEBAR
###############################################################################

def get_figure_size():
    """
    Returns (plot_width, plot_height) in inches, as chosen by user in sidebar.
    """
    # If they're not set, initialize them
    if "plot_width" not in st.session_state:
        st.session_state["plot_width"] = 8
    if "plot_height" not in st.session_state:
        st.session_state["plot_height"] = 4

    # We'll read them from the sidebar. This way, each tab uses the same size.
    st.sidebar.title("Plot Size Controls")
    st.session_state["plot_width"] = st.sidebar.slider("Plot Width (inches)",
                                                       min_value=4, max_value=20,
                                                       value=st.session_state["plot_width"],
                                                       step=1)
    st.session_state["plot_height"] = st.sidebar.slider("Plot Height (inches)",
                                                        min_value=2, max_value=10,
                                                        value=st.session_state["plot_height"],
                                                        step=1)
    return (st.session_state["plot_width"], st.session_state["plot_height"])


###############################################################################
#                           1) T-DISTRIBUTION TAB
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
        w,h = get_figure_size()
        fig = plot_t_distribution(t_val, df_val, alpha_val, tail_s, w, h)
        st.pyplot(fig, use_container_width=True)

    with st.expander("Show Table Lookup (±5 around df)"):
        show_t_table_lookup(df_val, alpha_val, tail_s, "t_tab")


def plot_t_distribution(t_val, df, alpha, tail_s, width, height):
    fig, ax = plt.subplots(figsize=(width,height), dpi=100)
    label_positions = []

    x = np.linspace(-4,4,400)
    y = stats.t.pdf(x, df)
    ax.plot(x, y, color='black')
    ax.fill_between(x, y, color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        t_crit= stats.t.ppf(1 - alpha, df)
        ax.fill_between(x[x>=t_crit], y[x>=t_crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit, color='green', linestyle='--')
        place_label(ax, label_positions, t_crit, stats.t.pdf(t_crit, df)+0.02, f"t_crit={t_crit:.3f}", 'green')
        sig = (t_val > t_crit)
        final_crit = t_crit
    else:
        t_crit_r = stats.t.ppf(1 - alpha/2, df)
        t_crit_l = stats.t.ppf(alpha/2, df)
        ax.fill_between(x[x>=t_crit_r], y[x>=t_crit_r], color='red', alpha=0.3)
        ax.fill_between(x[x<=t_crit_l], y[x<=t_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit_r, color='green', linestyle='--')
        ax.axvline(t_crit_l, color='green', linestyle='--')
        place_label(ax, label_positions, t_crit_r, stats.t.pdf(t_crit_r, df)+0.02, f"+t_crit={t_crit_r:.3f}", 'green')
        place_label(ax, label_positions, t_crit_l, stats.t.pdf(t_crit_l, df)+0.02, f"-t_crit={t_crit_l:.3f}", 'green')
        sig = (abs(t_val) > abs(t_crit_r))
        final_crit = abs(t_crit_r)

    ax.axvline(t_val, color='blue', linestyle='--')
    place_label(ax, label_positions, t_val, stats.t.pdf(t_val, df)+0.02, f"t_calc={t_val:.3f}", 'blue')

    if sig:
        msg = f"t={t_val:.3f} > {final_crit:.3f} => Reject H₀"
    else:
        msg = f"t={t_val:.3f} ≤ {final_crit:.3f} => Fail to Reject H₀"

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
            return stats.t.ppf(1-a_, dof)
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
        table_html += f"<th>{col_label}</th>"
    table_html += "</tr>\n"

    for dfv in df_list:
        table_html += "<tr>"
        row_id= f"df_{dfv}_0"
        table_html += f'<td id="{row_id}">{dfv}</td>'
        for c_i,(m,a) in enumerate(columns[1:], start=1):
            cell_id= f"df_{dfv}_{c_i}"
            val= compute_tcrit(dfv, m, a)
            table_html += f'<td id="{cell_id}">{val:.3f}</td>'
        table_html += "</tr>\n"
    table_html += "</table>"

    user_mode = "one" if tail_s.startswith("one") else "two"
    col_index= None
    for i,(m,a) in enumerate(columns[1:], start=1):
        if m==user_mode and abs(a - alpha_val)<1e-12:
            col_index= i
            break

    row_in = (df_val in df_list)

    if cur_step>=0 and row_in:
        for cc in range(len(columns)):
            table_html= highlight_html_cell(table_html, f"df_{df_val}_{cc}", "red", 2)
    if cur_step>=1 and col_index is not None:
        for dv_ in df_list:
            table_html= highlight_html_cell(table_html, f"df_{dv_}_{col_index}", "red", 2)
    if cur_step>=2 and row_in and col_index is not None:
        table_html= highlight_html_cell(table_html, f"df_{df_val}_{col_index}", "blue", 3)

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

    max_step = len(steps_list)-1
    if cur_step>max_step:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps_list[cur_step]}")

    next_step_button(step_key)


###############################################################################
#                      2) Z-DISTRIBUTION TAB
###############################################################################

def show_z_distribution_tab():
    st.subheader("z-Distribution")

    c1, c2= st.columns(2)
    with c1:
        z_val= st.number_input("z statistic", value=1.64, key="z_tab_zval")
    with c2:
        alpha_val= st.number_input("Alpha", value=0.05, step=0.01, key="z_tab_alpha")
        tail_s= st.radio("Tail Type", ["one-tailed","two-tailed"], key="z_tab_tail")

    if st.button("Update Plot", key="z_tab_update"):
        w,h = get_figure_size()
        fig= plot_z_distribution(z_val, alpha_val, tail_s, w, h)
        st.pyplot(fig, use_container_width=True)

    with st.expander("Show Table Lookup (±10 rows)"):
        show_z_table_lookup(z_val, "z_tab")


def plot_z_distribution(z_val, alpha, tail_s, width, height):
    fig, ax= plt.subplots(figsize=(width,height), dpi=100)
    label_positions=[]
    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2,label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        z_crit= stats.norm.ppf(1- alpha)
        ax.fill_between(x[x>= z_crit], y[x>= z_crit], color='red', alpha=0.3, label="Reject H₀")
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

    if sig:
        msg= f"z={z_val:.3f} > {final_crit:.3f} => Reject H₀"
    else:
        msg= f"z={z_val:.3f} ≤ {final_crit:.3f} => Fail to Reject H₀"

    ax.set_title(f"Z-Distribution\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig

def show_z_table_lookup(z_in, key_prefix):
    st.write("### Step-by-Step z-Table Lookup")

    step_key= key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key]=0
    cur_step= st.session_state[step_key]

    if z_in<0:
        z_in=0
    if z_in>3.49:
        z_in=3.49

    row_base= round(np.floor(z_in*10)/10,1)
    col_part= round(z_in - row_base,2)

    row_vals= np.round(np.arange(0,3.5,0.1),1)
    if row_base not in row_vals:
        row_base= min(row_vals, key=lambda x:abs(x-row_base))

    row_idx= np.where(row_vals==row_base)[0]
    if len(row_idx)==0:
        row_idx= [0]
    row_idx= row_idx[0]

    row_start= max(0,row_idx-10)
    row_end= min(len(row_vals)-1, row_idx+10)
    sub_rows= row_vals[row_start: row_end+1]

    col_vals= np.round(np.arange(0,0.1,0.01),2)
    if col_part not in col_vals:
        col_part= min(col_vals, key=lambda x:abs(x-col_part))

    table_html= """
    <style>
    table.ztable {
      border-collapse: collapse;
      margin-top:10px;
    }
    table.ztable td, table.ztable th {
      border:1px solid #000; 
      width:70px; 
      height:30px; 
      text-align:center;
      font-family:sans-serif; 
      font-size:0.9rem;
    }
    table.ztable th {
      background-color:#f0f0f0;
    }
    </style>
    <table class="ztable">
    <tr><th>z.x</th>
    """
    for cv in col_vals:
        table_html+= f"<th>{cv:.2f}</th>"
    table_html+= "</tr>\n"

    for rv in sub_rows:
        table_html+= "<tr>"
        row_id= f"z_{rv:.1f}_0"
        table_html+= f'<td id="{row_id}">{rv:.1f}</td>'
        for cv in col_vals:
            cell_id= f"z_{rv:.1f}_{cv:.2f}"
            z_val= rv+ cv
            cdf_val= stats.norm.cdf(z_val)
            table_html+= f'<td id="{cell_id}">{cdf_val:.4f}</td>'
        table_html+= "</tr>\n"
    table_html+= "</table>"

    row_in= (row_base in sub_rows)
    col_in= (col_part in col_vals)

    if cur_step>=0 and row_in:
        for cv in col_vals:
            table_html= highlight_html_cell(table_html, f"z_{row_base:.1f}_{cv:.2f}", "red",2)
        table_html= highlight_html_cell(table_html, f"z_{row_base:.1f}_0", "red",2)

    if cur_step>=1 and col_in:
        for rv in sub_rows:
            table_html= highlight_html_cell(table_html, f"z_{rv:.1f}_{col_part:.2f}", "red",2)

    if cur_step>=2 and row_in and col_in:
        table_html= highlight_html_cell(table_html, f"z_{row_base:.1f}_{col_part:.2f}", "blue",3)

    show_html_table(table_html, height=450)

    steps_list= [
        f"1) Locate row {row_base:.1f}",
        f"2) Locate column {col_part:.2f}",
        "3) Intersection => CDF"
    ]
    max_step=2
    if cur_step>max_step:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps_list[cur_step]}")

    next_step_button(step_key)


###############################################################################
#                  3) F-DISTRIBUTION TAB
###############################################################################

def show_f_distribution_tab():
    st.subheader("F-Distribution (One-tailed)")

    c1, c2= st.columns(2)
    with c1:
        f_val= st.number_input("F statistic", value=3.49, key="f_tab_fval")
        df1= st.number_input("df1", value=3, key="f_tab_df1")
        df2= st.number_input("df2", value=12, key="f_tab_df2")
    with c2:
        alpha= st.number_input("Alpha", value=0.05, step=0.01, key="f_tab_alpha")

    if st.button("Update Plot", key="f_tab_update"):
        w,h = get_figure_size()
        fig= plot_f_distribution(f_val, df1, df2, alpha, w, h)
        st.pyplot(fig, use_container_width=True)

    with st.expander("Show F Table Lookup (±5 around df1, df2)"):
        show_f_table_lookup(df1, df2, alpha, "f_tab")


def plot_f_distribution(f_val, df1, df2, alpha, width, height):
    fig, ax= plt.subplots(figsize=(width,height), dpi=100)
    label_positions=[]
    x= np.linspace(0,5,500)
    y= stats.f.pdf(x, df1, df2)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey',alpha=0.2,label="Fail to Reject H₀")

    f_crit= stats.f.ppf(1-alpha, df1, df2)
    ax.fill_between(x[x>=f_crit], y[x>=f_crit], color='red', alpha=0.3, label="Reject H₀")
    ax.axvline(f_crit, color='green', linestyle='--')
    place_label(ax, label_positions, f_crit, stats.f.pdf(f_crit,df1,df2)+0.02,
                f"F_crit={f_crit:.3f}", 'green')

    ax.axvline(f_val, color='blue', linestyle='--')
    place_label(ax, label_positions, f_val, stats.f.pdf(f_val,df1,df2)+0.02,
                f"F_calc={f_val:.3f}", 'blue')

    sig= (f_val> f_crit)
    msg= (f"F={f_val:.3f} > {f_crit:.3f} => Reject H₀"
          if sig else
          f"F={f_val:.3f} ≤ {f_crit:.3f} => Fail to Reject H₀")

    ax.set_title(f"F-Distribution (df1={df1}, df2={df2})\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig


def show_f_table_lookup(df1_u, df2_u, alpha, key_prefix):
    st.write("### Step-by-Step F Table Lookup")

    step_key= key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key]=0
    cur_step= st.session_state[step_key]

    df1_min= max(1, df1_u-5)
    df1_max= df1_u+5
    df2_min= max(1, df2_u-5)
    df2_max= df2_u+5
    df1_list= range(df1_min, df1_max+1)
    df2_list= range(df2_min, df2_max+1)

    table_html= """
    <style>
    table.ftable {
      border-collapse: collapse;
      margin-top:10px;
    }
    table.ftable td, table.ftable th {
      border:1px solid #000; 
      width:70px; 
      height:30px; 
      text-align:center;
      font-family:sans-serif; 
      font-size:0.9rem;
    }
    table.ftable th {
      background-color:#f0f0f0;
    }
    </style>
    <table class="ftable">
    <tr><th>df1\\df2</th>
    """
    for d2 in df2_list:
        table_html+= f"<th>{d2}</th>"
    table_html+= "</tr>\n"

    def fcrit(d1,d2,a):
        return stats.f.ppf(1-a, d1,d2)

    for d1 in df1_list:
        table_html+= "<tr>"
        table_html+= f'<td id="f_{d1}_0">{d1}</td>'
        for d2 in df2_list:
            cell_id= f"f_{d1}_{d2}"
            val= fcrit(d1,d2,alpha)
            table_html+= f'<td id="{cell_id}">{val:.3f}</td>'
        table_html+= "</tr>\n"
    table_html+= "</table>"

    row_in= (df1_u in df1_list)
    col_in= (df2_u in df2_list)

    if cur_step>=0 and row_in:
        for d2 in df2_list:
            table_html= highlight_html_cell(table_html, f"f_{df1_u}_{d2}", "red",2)
        table_html= highlight_html_cell(table_html, f"f_{df1_u}_0", "red",2)

    if cur_step>=1 and col_in:
        for d1 in df1_list:
            table_html= highlight_html_cell(table_html, f"f_{d1}_{df2_u}", "red",2)

    if cur_step>=2 and row_in and col_in:
        table_html= highlight_html_cell(table_html, f"f_{df1_u}_{df2_u}", "blue",3)

    show_html_table(table_html, height=450)

    steps_list= [
        f"1) Highlight row df1={df1_u}",
        f"2) Highlight column df2={df2_u}",
        "3) Intersection => F_crit"
    ]
    max_step=2
    if cur_step>max_step:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps_list[cur_step]}")

    next_step_button(step_key)


###############################################################################
#                   4) CHI-SQUARE TAB
###############################################################################

def show_chi_square_tab():
    st.subheader("Chi-Square (One-tailed)")

    c1, c2= st.columns(2)
    with c1:
        chi_val= st.number_input("Chi-square stat", value=10.5, key="chi_tab_val")
        df_val= st.number_input("df", value=12, key="chi_tab_df")
    with c2:
        alpha= st.number_input("Alpha", value=0.05, step=0.01, key="chi_tab_alpha")

    if st.button("Update Plot", key="chi_tab_update"):
        w,h = get_figure_size()
        fig= plot_chi_square(chi_val, df_val, alpha, w, h)
        st.pyplot(fig, use_container_width=True)

    with st.expander("Show Chi-Square Table Lookup (±5 df)"):
        show_chi_table_lookup(df_val, alpha, "chi_tab")


def plot_chi_square(chi_val, df, alpha, width, height):
    fig, ax= plt.subplots(figsize=(width,height), dpi=100)
    label_positions=[]
    x_max= max(30, df*2)
    x= np.linspace(0,x_max,400)
    y= stats.chi2.pdf(x, df)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2,label="Fail to Reject H₀")

    chi_crit= stats.chi2.ppf(1-alpha, df)
    ax.fill_between(x[x>= chi_crit], y[x>= chi_crit], color='red', alpha=0.3, label="Reject H₀")
    ax.axvline(chi_crit, color='green', linestyle='--')
    place_label(ax, label_positions, chi_crit, stats.chi2.pdf(chi_crit,df)+0.02,
                f"chi²_crit={chi_crit:.3f}", 'green')

    ax.axvline(chi_val, color='blue', linestyle='--')
    place_label(ax, label_positions, chi_val, stats.chi2.pdf(chi_val,df)+0.02,
                f"chi²_calc={chi_val:.3f}", 'blue')

    sig= (chi_val> chi_crit)
    msg= (f"χ²={chi_val:.3f} > {chi_crit:.3f} => Reject H₀"
          if sig else
          f"χ²={chi_val:.3f} ≤ {chi_crit:.3f} => Fail to Reject H₀")
    ax.set_title(f"Chi-Square (df={df})\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig


def show_chi_table_lookup(df_val, alpha, key_prefix):
    st.write("### Step-by-Step Chi-Square Table Lookup")

    step_key= key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key]=0
    cur_step= st.session_state[step_key]

    df_min= max(1, df_val-5)
    df_max= df_val+5
    df_list= list(range(df_min, df_max+1))
    alpha_list= [0.10, 0.05, 0.01, 0.001]

    table_html= """
    <style>
    table.chitable {
      border-collapse: collapse; 
      margin-top:10px;
    }
    table.chitable td, table.chitable th {
      border:1px solid #000; 
      width:80px; 
      height:30px; 
      text-align:center;
      font-family:sans-serif; 
      font-size:0.9rem;
    }
    table.chitable th {
      background-color:#f0f0f0;
    }
    </style>
    <table class="chitable">
    <tr><th>df</th>
    """
    for a_ in alpha_list:
        table_html += f"<th>α={a_}</th>"
    table_html += "</tr>\n"

    def chi_crit(dof,a_):
        return stats.chi2.ppf(1-a_, dof)

    for dv in df_list:
        table_html += "<tr>"
        table_html += f'<td id="chi_{dv}_0">{dv}</td>'
        for c_i,a_ in enumerate(alpha_list, start=1):
            cell_id= f"chi_{dv}_{c_i}"
            val= chi_crit(dv,a_)
            table_html += f'<td id="{cell_id}">{val:.3f}</td>'
        table_html += "</tr>\n"
    table_html += "</table>"

    row_in= (df_val in df_list)
    try:
        c_idx= alpha_list.index(alpha)+1
        col_in= True
    except:
        c_idx= None
        col_in= False

    if cur_step>=0 and row_in:
        for cc in range(len(alpha_list)+1):
            table_html= highlight_html_cell(table_html, f"chi_{df_val}_{cc}", "red",2)
    if cur_step>=1 and col_in:
        for dv_ in df_list:
            table_html= highlight_html_cell(table_html, f"chi_{dv_}_{c_idx}", "red",2)
    if cur_step>=2 and row_in and col_in:
        table_html= highlight_html_cell(table_html, f"chi_{df_val}_{c_idx}", "blue",3)

    show_html_table(table_html, height=450)

    steps_list= [
        f"1) Highlight row df={df_val}",
        f"2) Highlight column α={alpha}",
        "3) Intersection => chi²_crit"
    ]
    max_step=2
    if cur_step>max_step:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps_list[cur_step]}")

    next_step_button(step_key)


###############################################################################
#                   5) MANN–WHITNEY U TAB
###############################################################################

def show_mann_whitney_tab():
    st.subheader("Mann–Whitney U")

    c1, c2 = st.columns(2)
    with c1:
        U_val= st.number_input("U statistic", value=5, key="mw_tab_uval")
        n1= st.number_input("n1", value=5, key="mw_tab_n1")
        n2= st.number_input("n2", value=6, key="mw_tab_n2")
    with c2:
        alpha= st.number_input("Alpha", value=0.05, step=0.01, key="mw_tab_alpha")
        tail_s= st.radio("Tail Type", ["one-tailed","two-tailed"], key="mw_tab_tail")

    if st.button("Update Plot", key="mw_tab_update"):
        w,h = get_figure_size()
        fig= plot_mann_whitney(U_val, n1, n2, alpha, tail_s, w, h)
        st.pyplot(fig, use_container_width=True)

    with st.expander("Show Mann–Whitney U Table Lookup"):
        st.write("Implement a step-by-step highlight similarly if desired.")


def plot_mann_whitney(U_val, n1, n2, alpha, tail_s, width, height):
    fig, ax= plt.subplots(figsize=(width,height), dpi=100)
    label_positions=[]
    meanU= n1*n2/2
    sdU= np.sqrt(n1*n2*(n1+n2+1)/12)
    z_val= (U_val- meanU)/sdU

    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey',alpha=0.2,label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        zCrit= stats.norm.ppf(1- alpha)
        ax.fill_between(x[x>=zCrit], y[x>=zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit), f"zCrit={zCrit:.3f}", 'green')
        sig= (z_val> zCrit)
    else:
        zCrit= stats.norm.ppf(1- alpha/2)
        ax.fill_between(x[x>=zCrit], y[x>=zCrit], color='red', alpha=0.3)
        ax.fill_between(x[x<=-zCrit], y[x<=-zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        ax.axvline(-zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit), f"+zCrit={zCrit:.3f}", 'green')
        place_label(ax, label_positions, -zCrit, stats.norm.pdf(-zCrit), f"-zCrit={zCrit:.3f}", 'green')
        sig= (abs(z_val)> zCrit)

    ax.axvline(z_val, color='blue', linestyle='--')
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val), f"z_calc={z_val:.3f}", 'blue')

    msg= (f"U={U_val}, z={z_val:.3f} => Reject H₀" if sig else f"U={U_val}, z={z_val:.3f} => Fail to Reject H₀")
    ax.set_title(f"Mann–Whitney U: n1={n1}, n2={n2}\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig


###############################################################################
#                   6) WILCOXON SIGNED-RANK TAB
###############################################################################

def show_wilcoxon_tab():
    st.subheader("Wilcoxon Signed-Rank")

    c1, c2= st.columns(2)
    with c1:
        T_val= st.number_input("T statistic", value=5, key="wil_tab_T")
        N_val= st.number_input("N (nonzero diffs)", value=6, key="wil_tab_N")
    with c2:
        alpha= st.number_input("Alpha", value=0.05, step=0.01, key="wil_tab_alpha")
        tail_s= st.radio("Tail Type", ["one-tailed","two-tailed"], key="wil_tab_tail")

    if st.button("Update Plot", key="wil_tab_update"):
        w,h = get_figure_size()
        fig= plot_wilcoxon(T_val, N_val, alpha, tail_s, w, h)
        st.pyplot(fig, use_container_width=True)

    with st.expander("Show Wilcoxon Table Lookup"):
        st.write("Same approach with step-by-step highlight if needed.")


def plot_wilcoxon(T_val, N, alpha, tail_s, width, height):
    fig, ax= plt.subplots(figsize=(width,height), dpi=100)
    label_positions=[]
    meanT= N*(N+1)/4
    sdT= np.sqrt(N*(N+1)*(2*N+1)/24)
    z_val= (T_val - meanT)/ sdT

    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2,label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        zCrit= stats.norm.ppf(1- alpha)
        ax.fill_between(x[x>=zCrit], y[x>=zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit), f"zCrit={zCrit:.3f}", 'green')
        sig= (z_val> zCrit)
    else:
        zCrit= stats.norm.ppf(1- alpha/2)
        ax.fill_between(x[x>= zCrit], y[x>= zCrit], color='red', alpha=0.3)
        ax.fill_between(x[x<= -zCrit], y[x<= -zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        ax.axvline(-zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit), f"+zCrit={zCrit:.3f}", 'green')
        place_label(ax, label_positions, -zCrit, stats.norm.pdf(-zCrit), f"-zCrit={zCrit:.3f}", 'green')
        sig= (abs(z_val)> zCrit)

    ax.axvline(z_val, color='blue', linestyle='--')
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val), f"z_calc={z_val:.3f}", 'blue')

    msg= (f"T={T_val}, z={z_val:.3f} => Reject H₀" if sig
          else f"T={T_val}, z={z_val:.3f} => Fail to Reject H₀")
    ax.set_title(f"Wilcoxon (N={N})\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig


###############################################################################
#                   7) BINOMIAL TAB
###############################################################################

def show_binomial_tab():
    st.subheader("Binomial")

    c1, c2= st.columns(2)
    with c1:
        n= st.number_input("n", value=10, key="bin_tab_n")
        x= st.number_input("x (successes)", value=3, key="bin_tab_x")
    with c2:
        p= st.number_input("p", value=0.5, step=0.01, key="bin_tab_p")
        alpha= st.number_input("Alpha", value=0.05, step=0.01, key="bin_tab_alpha")
        tail_s= st.radio("Tail Type", ["one-tailed","two-tailed"], key="bin_tab_tail")

    if st.button("Update Plot", key="bin_tab_update"):
        w,h = get_figure_size()
        fig= plot_binomial(n,x,p,alpha,tail_s, w, h)
        st.pyplot(fig, use_container_width=True)

    with st.expander("Show Binomial Table Lookup (±5 around n)"):
        show_binomial_table_lookup(n, alpha, tail_s, "bin_tab")


def plot_binomial(n,x,p,alpha,tail_s, width, height):
    fig, ax= plt.subplots(figsize=(width,height), dpi=100)
    k_vals= np.arange(n+1)
    pmf_vals= stats.binom.pmf(k_vals, n, p)
    bars= ax.bar(k_vals, pmf_vals, color='lightgrey', edgecolor='black')

    mean_= n*p
    if tail_s=="one-tailed":
        if x<=mean_:
            p_val= stats.binom.cdf(x, n, p)
        else:
            p_val= 1- stats.binom.cdf(x-1, n, p)
    else:
        if x<=mean_:
            p_val= stats.binom.cdf(x, n, p)*2
        else:
            p_val= (1- stats.binom.cdf(x-1, n, p))*2
        p_val= min(p_val,1.0)

    sig= (p_val< alpha)
    msg= f"p-value={p_val:.4f} => " + ("Reject H₀" if sig else "Fail to Reject H₀")

    if tail_s=="one-tailed":
        if x< mean_:
            for i in range(0, x+1):
                bars[i].set_color('red')
        else:
            for i in range(x, n+1):
                bars[i].set_color('red')
    else:
        if x<= mean_:
            for i in range(0, x+1):
                bars[i].set_color('red')
            hi_start= max(0, n-x)
            for i in range(hi_start, n+1):
                bars[i].set_color('red')
        else:
            for i in range(x, n+1):
                bars[i].set_color('red')
            for i in range(0, (n-x)+1):
                bars[i].set_color('red')

    if 0<= x<= n:
        bars[x].set_color('blue')

    import matplotlib.patches as mpatches
    legend_patches= [
        mpatches.Patch(facecolor='lightgrey', edgecolor='black', label='Fail to Reject H₀'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='Reject H₀ region'),
        mpatches.Patch(facecolor='blue', edgecolor='black', label='Observed x')
    ]
    ax.legend(handles=legend_patches)
    ax.set_title(f"Binomial(n={n}, p={p:.2f})\n{msg}")
    ax.set_xlabel("x (successes)")
    ax.set_ylabel("PMF")
    fig.tight_layout()
    return fig


def show_binomial_table_lookup(n_val, alpha, tail_s, key_prefix):
    st.write("### Step-by-Step Binomial Table Lookup")

    step_key= key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key]=0
    cur_step= st.session_state[step_key]

    n_min= max(1, n_val-5)
    n_max= n_val+5
    n_list= range(n_min, n_max+1)
    alpha_list= [0.10,0.05,0.01,0.001]

    def onetail_xcrit(nn,a_):
        cdf= [stats.binom.cdf(k,nn,0.5) for k in range(nn+1)]
        for k in range(nn+1):
            if cdf[k]>=a_:
                return k
        return nn

    def twotail_bounds(nn,a_):
        pmf= [stats.binom.pmf(k,nn,0.5) for k in range(nn+1)]
        cdf= np.cumsum(pmf)
        half= a_/2
        lo=0
        while lo<=nn and cdf[lo]<half:
            lo+=1
        hi= nn
        upper=1-cdf[hi-1] if hi>0 else 1
        while hi>=0 and upper<half:
            hi-=1
            if hi>=0:
                upper=1-cdf[hi]
        return (lo,hi)

    table_data={}
    for nn in n_list:
        row={}
        for a_ in alpha_list:
            if tail_s=="one-tailed":
                row[a_]= onetail_xcrit(nn,a_)
            else:
                row[a_]= twotail_bounds(nn,a_)
        table_data[nn]= row

    table_html= """
    <style>
    table.bintable {
      border-collapse: collapse;
      margin-top:10px;
    }
    table.bintable td, table.bintable th {
      border:1px solid #000; 
      width:80px; 
      height:30px; 
      text-align:center;
      font-family:sans-serif; 
      font-size:0.9rem;
    }
    table.bintable th {
      background-color:#f0f0f0;
    }
    </style>
    <table class="bintable">
    <tr><th>n</th>
    """
    for a_ in alpha_list:
        table_html+= f"<th>α={a_}</th>"
    table_html+= "</tr>\n"

    for nn in n_list:
        table_html+= "<tr>"
        row_id= f"bin_{nn}_0"
        table_html+= f'<td id="{row_id}">{nn}</td>'
        for c_i,a_ in enumerate(alpha_list, start=1):
            cell_id= f"bin_{nn}_{c_i}"
            val= table_data[nn][a_]
            table_html+= f'<td id="{cell_id}">{val}</td>'
        table_html+= "</tr>\n"
    table_html+= "</table>"

    row_in= (n_val in n_list)
    try:
        col_idx= alpha_list.index(alpha)+1
        col_in= True
    except:
        col_idx= None
        col_in= False

    if cur_step>=0 and row_in:
        for c_i in range(len(alpha_list)+1):
            table_html= highlight_html_cell(table_html, f"bin_{n_val}_{c_i}", "red",2)
    if cur_step>=1 and col_in:
        for nn_ in n_list:
            table_html= highlight_html_cell(table_html, f"bin_{nn_}_{col_idx}", "red",2)
    if cur_step>=2 and row_in and col_in:
        table_html= highlight_html_cell(table_html, f"bin_{n_val}_{col_idx}", "blue",3)

    show_html_table(table_html, height=450)

    steps_list= [
        f"1) Highlight row n={n_val}",
        f"2) Highlight column α={alpha}",
        "3) Intersection => xcrit or (lo,hi)"
    ]
    max_step=2
    if cur_step>max_step:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps_list[cur_step]}")

    next_step_button(step_key)


###############################################################################
#                                 MAIN
###############################################################################

def main():
    st.set_page_config(page_title="PSYC250 - Full Stats Explorer (Adjustable Plots)",
                       layout="wide")

    # Force the creation of the slider for plot width/height
    # so it's always visible on the sidebar
    get_figure_size()

    st.title("PSYC250 - Statistical Tables Explorer (Adjustable Figure Size)")

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
