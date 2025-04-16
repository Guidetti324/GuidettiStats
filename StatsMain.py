import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Use a non-GUI backend for Matplotlib
plt.switch_backend("Agg")

###############################################################################
#                            UTILITY FUNCTIONS
###############################################################################

def place_label(ax, label_positions, x, y, text, color='blue'):
    """
    Place a text label on the Axes at (x, y), shifting slightly if needed
    to avoid overlapping previously placed labels.
    """
    offset_x = 0.0
    offset_y = 0.02
    for (xx, yy) in label_positions:
        # If too close, nudge this new label
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            offset_x += 0.06
            offset_y += 0.04

    final_x = x + offset_x
    final_y = y + offset_y
    ax.text(final_x, final_y, text, color=color, ha="left", va="bottom", fontsize=8)
    label_positions.append((final_x, final_y))


def highlight_html_cell(html_in, cell_id, color="red", border_px=2):
    """
    In the 'html_in' string, find the <td> with id="cell_id" and apply an inline
    style to highlight it with a colored border.
    """
    marker = f'id="{cell_id}"'
    style = f'id="{cell_id}" style="border: {border_px}px solid {color};"'
    return html_in.replace(marker, style, 1)


def next_step_button(step_key):
    """
    Renders a 'Next Step' button that, when clicked, increments st.session_state[step_key].
    """
    if st.button("Next Step", key=step_key+"_btn"):
        st.session_state[step_key] += 1


###############################################################################
#                  1) T-DISTRIBUTION TAB
###############################################################################

def show_t_distribution_tab():
    st.subheader("t-Distribution")

    col1, col2 = st.columns(2)
    with col1:
        t_val = st.number_input("t statistic", value=2.87, key="t_tab_tval")
        df_val = st.number_input("df", value=55, key="t_tab_df")
    with col2:
        alpha_val = st.number_input("Alpha", value=0.05, step=0.01, key="t_tab_alpha")
        tail_type = st.radio("Tail Type", ["one-tailed","two-tailed"], key="t_tab_tail")

    if st.button("Update Plot", key="t_tab_update"):
        fig = plot_t_distribution(t_val, df_val, alpha_val, tail_type)
        st.pyplot(fig)

    with st.expander("Show Table Lookup (±5 around df)"):
        show_t_table_lookup(df_val, alpha_val, tail_type, "t_tab")


def plot_t_distribution(t_val, df, alpha, tail_s):
    fig, ax = plt.subplots(figsize=(5,3))
    label_positions = []

    x = np.linspace(-4,4,400)
    y = stats.t.pdf(x, df)
    ax.plot(x, y, color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        t_crit= stats.t.ppf(1-alpha, df)
        ax.fill_between(x[x>=t_crit], y[x>=t_crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit, color='green', linestyle='--')
        place_label(ax, label_positions, t_crit, stats.t.pdf(t_crit, df)+0.02,
                    f"t_crit={t_crit:.3f}", 'green')
        sig = (t_val> t_crit)
        final_crit= t_crit
    else:
        t_crit_r = stats.t.ppf(1 - alpha/2, df)
        t_crit_l = stats.t.ppf(alpha/2, df)
        ax.fill_between(x[x>=t_crit_r], y[x>=t_crit_r], color='red', alpha=0.3)
        ax.fill_between(x[x<=t_crit_l], y[x<=t_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit_r, color='green', linestyle='--')
        ax.axvline(t_crit_l, color='green', linestyle='--')
        place_label(ax, label_positions, t_crit_r, stats.t.pdf(t_crit_r, df)+0.02,
                    f"+t_crit={t_crit_r:.3f}", 'green')
        place_label(ax, label_positions, t_crit_l, stats.t.pdf(t_crit_l, df)+0.02,
                    f"-t_crit={t_crit_l:.3f}", 'green')
        sig = (abs(t_val)>abs(t_crit_r))
        final_crit= abs(t_crit_r)

    ax.axvline(t_val, color='blue', linestyle='--')
    place_label(ax, label_positions, t_val, stats.t.pdf(t_val, df)+0.02,
                f"t_calc={t_val:.3f}", 'blue')

    if sig:
        msg = f"t={t_val:.3f} > {final_crit:.3f} => Reject H₀"
    else:
        msg = f"t={t_val:.3f} ≤ {final_crit:.3f} => Fail to Reject H₀"

    ax.set_title(f"t-Distribution (df={df})\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig


def show_t_table_lookup(df_val, alpha_val, tail_s, key_prefix):
    """
    Step-by-step highlight of a partial t-table around df ±5. 
    We'll highlight row => column => intersection => (optionally) the one-tail=0.05 => two-tail=0.10 link.
    """
    st.write("### Step-by-Step T-Table Lookup")

    step_key = key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    cur_step = st.session_state[step_key]

    df_min= max(1, df_val-5)
    df_max= df_val+5
    df_list= list(range(df_min, df_max+1))

    # columns: (mode, alpha)
    columns= [
        ("df",""),
        ("one",0.10), ("one",0.05), ("one",0.01), ("one",0.001),
        ("two",0.10), ("two",0.05), ("two",0.01), ("two",0.001),
    ]
    def compute_tcrit(dof, m, a_):
        if m=="one":
            return stats.t.ppf(1-a_, dof)
        else:
            return stats.t.ppf(1-a_/2, dof)

    # Build HTML table
    table_html= """
    <style>
    table.ttable {border-collapse: collapse; margin-top:10px;}
    table.ttable td, table.ttable th {
      border:1px solid #000; width:85px; height:30px; text-align:center;
      font-family: sans-serif; font-size:0.9rem;
    }
    table.ttable th {background-color:#f0f0f0;}
    </style>
    <table class="ttable">
      <tr>
        <th>df</th>
    """
    for c_i,(m,a) in enumerate(columns[1:], start=1):
        table_html+= f"<th>{m}_{a}</th>"
    table_html+= "</tr>\n"

    for dfv in df_list:
        table_html+= "<tr>"
        row_id= f"df_{dfv}_0"
        table_html+= f'<td id="{row_id}">{dfv}</td>'
        for c_i,(m,a) in enumerate(columns[1:], start=1):
            cell_id= f"df_{dfv}_{c_i}"
            val= compute_tcrit(dfv,m,a)
            table_html+= f'<td id="{cell_id}">{val:.3f}</td>'
        table_html+="</tr>\n"
    table_html+="</table>"

    # Which column is user’s (tail_s, alpha_val)?
    user_mode= "one" if tail_s.startswith("one") else "two"
    col_index= None
    for i,(m,a) in enumerate(columns[1:], start=1):
        if m==user_mode and abs(a - alpha_val)<1e-12:
            col_index= i
            break

    # step logic:
    #   0 => highlight row for df_val
    #   1 => highlight column for tail_s, alpha_val
    #   2 => highlight intersection
    #   3 => if tail_s=one, alpha=0.05 => also highlight the "two_0.10" column
    row_in = (df_val in df_list)
    if row_in and cur_step>=0:
        # entire row
        row_cols= range(len(columns))
        for rc in row_cols:
            table_html= highlight_html_cell(table_html, f"df_{df_val}_{rc}", color="red", border_px=2)

    if col_index is not None and cur_step>=1:
        # entire column
        for dfv in df_list:
            table_html= highlight_html_cell(table_html, f"df_{dfv}_{col_index}", color="red", border_px=2)

    if row_in and col_index is not None and cur_step>=2:
        # intersection in blue
        table_html= highlight_html_cell(table_html, f"df_{df_val}_{col_index}",
                                        color="blue", border_px=3)

    # step 3 => if one-tail=0.05, highlight two_0.10
    if cur_step>=3 and tail_s=="one-tailed" and abs(alpha_val-0.05)<1e-12:
        alt_col= None
        for i,(m,a) in enumerate(columns[1:], start=1):
            if m=="two" and abs(a-0.10)<1e-12:
                alt_col= i
                break
        if alt_col is not None and row_in:
            # highlight entire alt column
            for dfv in df_list:
                table_html= highlight_html_cell(table_html, f"df_{dfv}_{alt_col}", color="red", border_px=2)
            # intersection in blue
            table_html= highlight_html_cell(table_html, f"df_{df_val}_{alt_col}",
                                            color="blue", border_px=3)

    st.markdown(table_html, unsafe_allow_html=True)

    # step descriptions
    steps_list= [
        f"1) Highlight row df={df_val}",
        f"2) Highlight column tail={tail_s}, α={alpha_val}",
        "3) Intersection => t_crit",
        "4) If one-tailed α=0.05, also highlight two-tailed α=0.10"
    ]
    # if not the special case, we only have 3 steps
    if not (tail_s=="one-tailed" and abs(alpha_val-0.05)<1e-12):
        steps_list.pop()  # remove the 4th step

    max_step= len(steps_list)-1
    if cur_step>max_step:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps_list[cur_step]}")

    next_step_button(step_key)


###############################################################################
#                  2) Z-DISTRIBUTION TAB
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
        fig= plot_z_distribution(z_val, alpha_val, tail_s)
        st.pyplot(fig)

    with st.expander("Show Table Lookup (±10 rows)"):
        show_z_table_lookup(z_val, "z_tab")


def plot_z_distribution(z_val, alpha, tail_s):
    fig, ax= plt.subplots(figsize=(5,3))
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
        ax.fill_between(x[x>= z_crit_r], y[x>= z_crit_r], color='red', alpha=0.3)
        ax.fill_between(x[x<= z_crit_l], y[x<= z_crit_l], color='red', alpha=0.3, label="Reject H₀")
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

    # row base => e.g. 1.6 if z_in=1.64
    row_base= round(np.floor(z_in*10)/10,1)
    col_part= round(z_in- row_base,2)

    # define all possible row bases from 0.0 to 3.4 in steps of 0.1
    row_vals= np.round(np.arange(0,3.5,0.1),1)
    if row_base not in row_vals:
        row_base= min(row_vals, key=lambda x:abs(x-row_base))

    # find row_idx
    row_idx= np.where(row_vals==row_base)[0]
    if len(row_idx)==0: row_idx=[0]
    row_idx= row_idx[0]

    # sub range ±10
    row_start= max(0,row_idx-10)
    row_end= min(len(row_vals)-1, row_idx+10)
    sub_rows= row_vals[row_start: row_end+1]

    # columns => 0.00..0.09
    col_vals= np.round(np.arange(0,0.1,0.01),2)
    if col_part not in col_vals:
        col_part= min(col_vals, key=lambda x:abs(x-col_part))

    # Build HTML
    table_html= """
    <style>
    table.ztable {border-collapse: collapse; margin-top:10px;}
    table.ztable td, table.ztable th {
      border:1px solid #000; width:70px; height:30px; text-align:center;
      font-family: sans-serif; font-size:0.9rem;
    }
    table.ztable th {background-color:#f0f0f0;}
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

    # Step logic: 0=highlight row, 1=highlight col, 2=intersection
    row_in= (row_base in sub_rows)
    col_in= (col_part in col_vals)

    if cur_step>=0 and row_in:
        # highlight entire row
        for cv in col_vals:
            table_html= highlight_html_cell(table_html, f"z_{row_base:.1f}_{cv:.2f}", "red",2)
        # highlight row label
        table_html= highlight_html_cell(table_html, f"z_{row_base:.1f}_0", "red",2)

    if cur_step>=1 and col_in:
        # highlight entire column
        for rv in sub_rows:
            table_html= highlight_html_cell(table_html, f"z_{rv:.1f}_{col_part:.2f}", "red",2)

    if cur_step>=2 and row_in and col_in:
        # intersection in blue
        table_html= highlight_html_cell(table_html, f"z_{row_base:.1f}_{col_part:.2f}", "blue",3)

    st.markdown(table_html, unsafe_allow_html=True)

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
#                3) F-DISTRIBUTION TAB
###############################################################################

def show_f_distribution_tab():
    st.subheader("F-Distribution (One-tailed)")

    c1, c2= st.columns(2)
    with c1:
        f_val= st.number_input("F statistic", value=3.49, key="f_tab_f")
        df1= st.number_input("df1", value=3, key="f_tab_df1")
        df2= st.number_input("df2", value=12, key="f_tab_df2")
    with c2:
        alpha= st.number_input("Alpha", value=0.05, step=0.01, key="f_tab_alpha")

    if st.button("Update Plot", key="f_tab_update"):
        fig= plot_f_distribution(f_val, df1, df2, alpha)
        st.pyplot(fig)

    with st.expander("Show F Table Lookup (±5 around df1, df2)"):
        show_f_table_lookup(df1, df2, alpha, "f_tab")


def plot_f_distribution(f_val, df1, df2, alpha):
    fig, ax= plt.subplots(figsize=(5,3))
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
    msg= (f"F={f_val:.3f} > {f_crit:.3f} => Reject H₀" if sig else
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
    table.ftable {border-collapse: collapse; margin-top:10px;}
    table.ftable td, table.ftable th {
      border:1px solid #000; width:70px; height:30px; text-align:center;
      font-family: sans-serif; font-size:0.9rem;
    }
    table.ftable th {background-color:#f0f0f0;}
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

    # steps: 0 => highlight row df1, 1 => highlight col df2, 2 => intersection
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

    st.markdown(table_html, unsafe_allow_html=True)

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
#                4) CHI-SQUARE TAB
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
        fig= plot_chi_square(chi_val, df_val, alpha)
        st.pyplot(fig)

    with st.expander("Show Chi-Square Table Lookup (±5 df)"):
        show_chi_table_lookup(df_val, alpha, "chi_tab")


def plot_chi_square(chi_val, df, alpha):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    x_max= max(30, df*2)
    x= np.linspace(0, x_max, 400)
    y= stats.chi2.pdf(x, df)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2,label="Fail to Reject H₀")

    chi_crit= stats.chi2.ppf(1-alpha, df)
    ax.fill_between(x[x>=chi_crit], y[x>=chi_crit], color='red', alpha=0.3, label="Reject H₀")
    ax.axvline(chi_crit, color='green', linestyle='--')
    place_label(ax, label_positions, chi_crit, stats.chi2.pdf(chi_crit,df)+0.02,
                f"chi²_crit={chi_crit:.3f}", 'green')

    ax.axvline(chi_val, color='blue', linestyle='--')
    place_label(ax, label_positions, chi_val, stats.chi2.pdf(chi_val,df)+0.02,
                f"chi²_calc={chi_val:.3f}", 'blue')

    sig= (chi_val> chi_crit)
    msg= f"χ²={chi_val:.3f} > {chi_crit:.3f} => Reject H₀" if sig else f"χ²={chi_val:.3f} ≤ {chi_crit:.3f} => Fail to Reject H₀"
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
    table.chitable {border-collapse: collapse; margin-top:10px;}
    table.chitable td, table.chitable th {
      border:1px solid #000; width:80px; height:30px; text-align:center;
      font-family: sans-serif; font-size:0.9rem;
    }
    table.chitable th {background-color:#f0f0f0;}
    </style>
    <table class="chitable">
    <tr><th>df</th>
    """
    for a_ in alpha_list:
        table_html+= f"<th>α={a_}</th>"
    table_html+= "</tr>\n"

    def chi_crit(dof,a_):
        return stats.chi2.ppf(1-a_, dof)

    for dv in df_list:
        table_html+= "<tr>"
        table_html+= f'<td id="chi_{dv}_0">{dv}</td>'
        for c_i,a_ in enumerate(alpha_list, start=1):
            cell_id= f"chi_{dv}_{c_i}"
            val= chi_crit(dv,a_)
            table_html+= f'<td id="{cell_id}">{val:.3f}</td>'
        table_html+="</tr>\n"
    table_html+="</table>"

    row_in= (df_val in df_list)
    try:
        c_idx= alpha_list.index(alpha)+1
        col_in= True
    except:
        c_idx= None
        col_in= False

    # steps: 0-> row highlight, 1-> column highlight, 2-> intersection
    if cur_step>=0 and row_in:
        for cc in range(len(alpha_list)+1):
            table_html= highlight_html_cell(table_html, f"chi_{df_val}_{cc}", "red",2)
    if cur_step>=1 and col_in:
        for dv_ in df_list:
            table_html= highlight_html_cell(table_html, f"chi_{dv_}_{c_idx}", "red",2)
    if cur_step>=2 and row_in and col_in:
        table_html= highlight_html_cell(table_html, f"chi_{df_val}_{c_idx}", "blue",3)

    st.markdown(table_html, unsafe_allow_html=True)

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
#             5) MANN–WHITNEY U TAB
###############################################################################

def show_mann_whitney_tab():
    st.subheader("Mann–Whitney U")

    c1, c2= st.columns(2)
    with c1:
        U_val= st.number_input("U statistic", value=5, key="mw_tab_U")
        n1= st.number_input("n1", value=5, key="mw_tab_n1")
        n2= st.number_input("n2", value=6, key="mw_tab_n2")
    with c2:
        alpha= st.number_input("Alpha", value=0.05, step=0.01, key="mw_tab_alpha")
        tail_s= st.radio("Tail Type", ["one-tailed","two-tailed"], key="mw_tab_tail")

    if st.button("Update Plot", key="mw_tab_update"):
        fig= plot_mann_whitney(U_val, n1, n2, alpha, tail_s)
        st.pyplot(fig)

    with st.expander("Show Mann–Whitney U Table Lookup"):
        st.write("Here you could implement a step-by-step highlight for U tables if desired.")


def plot_mann_whitney(U_val, n1, n2, alpha, tail_s):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    meanU= n1*n2/2
    sdU= np.sqrt(n1*n2*(n1+n2+1)/12)
    z_val= (U_val- meanU)/ sdU

    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey',alpha=0.2,label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        zCrit= stats.norm.ppf(1-alpha)
        ax.fill_between(x[x>=zCrit], y[x>=zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit),
                    f"zCrit={zCrit:.3f}", 'green')
        sig= (z_val> zCrit)
    else:
        zCrit= stats.norm.ppf(1- alpha/2)
        ax.fill_between(x[x>=zCrit], y[x>=zCrit], color='red', alpha=0.3)
        ax.fill_between(x[x<=-zCrit], y[x<=-zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        ax.axvline(-zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit),
                    f"+zCrit={zCrit:.3f}", 'green')
        place_label(ax, label_positions, -zCrit, stats.norm.pdf(-zCrit),
                    f"-zCrit={zCrit:.3f}", 'green')
        sig= (abs(z_val)> zCrit)

    ax.axvline(z_val, color='blue', linestyle='--')
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val),
                f"z_calc={z_val:.3f}", 'blue')

    msg= (f"U={U_val}, z={z_val:.3f} => Reject H₀" if sig else
          f"U={U_val}, z={z_val:.3f} => Fail to Reject H₀")
    ax.set_title(f"Mann–Whitney U: n1={n1}, n2={n2}\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig


###############################################################################
#             6) WILCOXON SIGNED-RANK TAB
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
        fig= plot_wilcoxon(T_val, N_val, alpha, tail_s)
        st.pyplot(fig)

    with st.expander("Show Wilcoxon Table Lookup"):
        st.write("Similar step-by-step approach can be implemented if needed.")


def plot_wilcoxon(T_val, N, alpha, tail_s):
    fig, ax= plt.subplots(figsize=(5,3))
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
        ax.fill_between(x[x>=zCrit], y[x>=zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit),
                    f"zCrit={zCrit:.3f}", 'green')
        sig= (z_val> zCrit)
    else:
        zCrit= stats.norm.ppf(1- alpha/2)
        ax.fill_between(x[x>= zCrit], y[x>= zCrit], color='red', alpha=0.3)
        ax.fill_between(x[x<= -zCrit], y[x<= -zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        ax.axvline(-zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit),
                    f"+zCrit={zCrit:.3f}", 'green')
        place_label(ax, label_positions, -zCrit, stats.norm.pdf(-zCrit),
                    f"-zCrit={zCrit:.3f}", 'green')
        sig= (abs(z_val)> zCrit)

    ax.axvline(z_val, color='blue', linestyle='--')
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val),
                f"z_calc={z_val:.3f}", 'blue')

    msg= (f"T={T_val}, z={z_val:.3f} => Reject H₀" if sig
          else f"T={T_val}, z={z_val:.3f} => Fail to Reject H₀")
    ax.set_title(f"Wilcoxon (N={N})\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig


###############################################################################
#                 7) BINOMIAL TAB
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
        fig= plot_binomial(n,x,p,alpha,tail_s)
        st.pyplot(fig)

    with st.expander("Show Binomial Table Lookup (±5 around n)"):
        show_binomial_table_lookup(n, alpha, tail_s, "bin_tab")


def plot_binomial(n,x,p,alpha,tail_s):
    fig, ax= plt.subplots(figsize=(5,3))
    k_vals= np.arange(n+1)
    pmf_vals= stats.binom.pmf(k_vals, n, p)
    bars= ax.bar(k_vals, pmf_vals, color='lightgrey', edgecolor='black')

    # compute p-value
    mean_= n*p
    if tail_s=="one-tailed":
        if x<=mean_:
            p_val= stats.binom.cdf(x,n,p)
        else:
            p_val= 1- stats.binom.cdf(x-1,n,p)
    else:
        if x<=mean_:
            p_val= stats.binom.cdf(x,n,p)*2
        else:
            p_val= (1- stats.binom.cdf(x-1,n,p))*2
        p_val= min(p_val,1.0)

    sig= (p_val< alpha)
    msg= f"p-value={p_val:.4f} => " + ("Reject H₀" if sig else "Fail to Reject H₀")

    # color bars
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

    # observed bar in blue
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

    n_min= max(1,n_val-5)
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

    # gather table data
    table_data={}
    for nn in n_list:
        row={}
        for a_ in alpha_list:
            if tail_s=="one-tailed":
                row[a_]= onetail_xcrit(nn,a_)
            else:
                row[a_]= twotail_bounds(nn,a_)
        table_data[nn]= row

    # build HTML
    table_html= """
    <style>
    table.bintable {border-collapse: collapse; margin-top:10px;}
    table.bintable td, table.bintable th {
      border:1px solid #000; width:80px; height:30px; text-align:center;
      font-family: sans-serif; font-size:0.9rem;
    }
    table.bintable th {background-color:#f0f0f0;}
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

    # steps: 0 => highlight row n_val, 1 => highlight col α, 2 => intersection
    if cur_step>=0 and row_in:
        for c_i in range(len(alpha_list)+1):
            table_html= highlight_html_cell(table_html, f"bin_{n_val}_{c_i}", "red",2)
    if cur_step>=1 and col_in:
        for nn_ in n_list:
            table_html= highlight_html_cell(table_html, f"bin_{nn_}_{col_idx}", "red",2)
    if cur_step>=2 and row_in and col_in:
        table_html= highlight_html_cell(table_html, f"bin_{n_val}_{col_idx}", "blue",3)

    st.markdown(table_html, unsafe_allow_html=True)
    steps_list= [
        f"1) Highlight row n={n_val}",
        f"2) Highlight column α={alpha}",
        "3) Intersection => xcrit or (lo, hi)"
    ]
    max_step=2
    if cur_step>max_step:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps_list[cur_step]}")

    next_step_button(step_key)


###############################################################################
#                               MAIN APP
###############################################################################

def main():
    st.set_page_config(page_title="PSYC250 - Full Stats Explorer", layout="wide")
    st.title("PSYC250 - Statistical Tables Explorer")

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
