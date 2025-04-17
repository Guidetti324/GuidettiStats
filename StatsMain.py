import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# We use a non‑interactive Matplotlib backend
plt.switch_backend("Agg")

###############################################################################
#                           HELPER FUNCTIONS
###############################################################################

def place_label(ax, placed_labels, x, y, text, color="blue"):
    """
    Places a text label at (x, y), shifting slightly to avoid overlap
    with previously placed labels.
    """
    dx = dy = 0.0
    for (xx, yy) in placed_labels:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            dx += 0.06
            dy += 0.04
    final_x = x + dx
    final_y = y + dy
    ax.text(final_x, final_y, text, color=color, ha="left", va="bottom", fontsize=8)
    placed_labels.append((final_x, final_y))

def style_cell(html_str, cell_id, color="red", px=2):
    """
    Finds <td id="cell_id"> in html_str and adds an inline style
    for a colored border.
    """
    marker = f'id="{cell_id}"'
    styled = f'id="{cell_id}" style="border:{px}px solid {color};"'
    return html_str.replace(marker, styled, 1)

def show_html_table(html_content, height=450):
    """
    Renders the given HTML in a scrollable iframe,
    so you can see the entire step-by-step table.
    """
    wrapped = f"""
    <html>
      <head><meta charset="UTF-8"></head>
      <body>
        {html_content}
      </body>
    </html>
    """
    components.html(wrapped, height=height, scrolling=True)

def next_step_button(step_key):
    """
    A button that increments st.session_state[step_key].
    """
    if st.button("Next Step", key=step_key + "_btn"):
        st.session_state[step_key] += 1

def wrap_html_table(css, table_body):
    return f"<style>{css}</style><table>{table_body}</table>"

###############################################################################
#                        1) T-DISTRIBUTION TAB
###############################################################################

def show_t_tab():
    st.subheader("Tab 1 • t-Distribution")

    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("t statistic", value=2.87, key="t_val")
        df_val = st.number_input("df", value=55, min_value=1, step=1, key="t_df")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01, min_value=0.0001, max_value=0.5, key="t_alpha")
        tail_s = st.radio("Tail", ["one-tailed","two-tailed"], key="t_tail")

    if st.button("Update Plot (t)", key="t_plot_btn"):
        fig = plot_t_distribution(t_val, df_val, alpha, tail_s)
        st.pyplot(fig)

    with st.expander("Step-by-step t-table"):
        show_t_table(df_val, alpha, tail_s, "t_tab")


def plot_t_distribution(t_val, df, alpha, tail_s):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    label_positions = []
    x = np.linspace(-4,4,400)
    y = stats.t.pdf(x, df)
    ax.plot(x, y, 'k')
    ax.fill_between(x, y, color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    if tail_s == "one-tailed":
        t_crit = stats.t.ppf(1 - alpha, df)
        ax.fill_between(x[x >= t_crit], y[x >= t_crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit, color='green', linestyle='--')
        place_label(ax, label_positions, t_crit, stats.t.pdf(t_crit, df)+0.02, f"t₍crit₎={t_crit:.3f}", "green")
        reject = (t_val > t_crit)
    else:
        t_crit_r = stats.t.ppf(1 - alpha/2, df)
        t_crit_l = -t_crit_r
        ax.fill_between(x[x >= t_crit_r], y[x >= t_crit_r], color='red', alpha=0.3)
        ax.fill_between(x[x <= t_crit_l], y[x <= t_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit_r, color='green', linestyle='--')
        ax.axvline(t_crit_l, color='green', linestyle='--')
        place_label(ax, label_positions, t_crit_r, stats.t.pdf(t_crit_r, df)+0.02, f"+t₍crit₎={t_crit_r:.3f}", "green")
        place_label(ax, label_positions, t_crit_l, stats.t.pdf(t_crit_l, df)+0.02, f"−t₍crit₎={t_crit_r:.3f}", "green")
        reject = (abs(t_val) > t_crit_r)

    ax.axvline(t_val, color='blue', linestyle='--')
    place_label(ax, label_positions, t_val, stats.t.pdf(t_val, df)+0.02, f"t₍calc₎={t_val:.3f}", "blue")
    msg = "Reject H₀" if reject else "Fail to Reject H₀"
    ax.set_title(f"t-Distribution (df={df}) – {msg}")
    ax.legend()
    fig.tight_layout()
    return fig


def show_t_table(df_val, alpha_val, tail_s, key_prefix):
    """
    Step-by-step highlight for t-table: row => col => intersection => 
    if one-tailed=0.05 => highlight two-tailed=0.10
    """
    step_key = key_prefix + "_step"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    step = st.session_state[step_key]

    df_min = max(1, df_val-5)
    df_max = df_val+5
    df_range = list(range(df_min, df_max+1))

    # columns: (mode, alpha)
    columns = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001),
    ]

    def t_crit(d, mode, a_):
        if mode == "one":
            return stats.t.ppf(1 - a_, d)
        else:
            return stats.t.ppf(1 - a_/2, d)

    # Build the HTML table
    header_html = "".join(f"<th>{m}_{a}</th>" for (m,a) in columns)
    table_body = "<tr><th>df</th>" + header_html + "</tr>\n"

    for d in df_range:
        row_html = f'<tr><td id="t_{d}_0">{d}</td>'
        for i,(m,a) in enumerate(columns, start=1):
            cell_id = f"t_{d}_{i}"
            val = t_crit(d, m, a)
            row_html += f'<td id="{cell_id}">{val:.3f}</td>'
        row_html += "</tr>\n"
        table_body += row_html

    css = """
    table {
      border-collapse: collapse;
      margin-top: 10px;
    }
    table td, table th {
      border:1px solid #000;
      width:80px;
      height:30px;
      text-align:center;
      font-size:0.9rem;
    }
    table th {
      background-color: #f0f0f0;
    }
    """
    table_html = wrap_html_table(css, table_body)

    # figure out which column matches tail_s, alpha_val
    mode_ = "one" if tail_s.startswith("one") else "two"
    col_index = None
    for i,(m,a) in enumerate(columns, start=1):
        if m==mode_ and abs(a - alpha_val)<1e-12:
            col_index = i
            break

    row_in = (df_val in df_range)

    # steps:
    # 0 => highlight entire row
    # 1 => highlight entire col
    # 2 => highlight intersection
    # 3 => if one-tailed alpha=0.05 => highlight the two_0.10 column as well
    if step >= 0 and row_in:
        for cc in range(len(columns)+1):
            table_html = style_cell(table_html, f"t_{df_val}_{cc}")
    if step >= 1 and col_index is not None:
        for d_ in df_range:
            table_html = style_cell(table_html, f"t_{d_}_{col_index}")
    if step >= 2 and row_in and col_index is not None:
        table_html = style_cell(table_html, f"t_{df_val}_{col_index}", "blue", 3)
    if step >= 3 and tail_s=="one-tailed" and abs(alpha_val-0.05)<1e-12:
        # highlight the two_0.10 column
        alt_index = None
        for i,(mm,aa) in enumerate(columns, start=1):
            if mm=="two" and abs(aa-0.10)<1e-12:
                alt_index=i
                break
        if alt_index is not None and row_in:
            for d_ in df_range:
                table_html = style_cell(table_html, f"t_{d_}_{alt_index}")
            table_html = style_cell(table_html, f"t_{df_val}_{alt_index}", "blue", 3)

    show_html_table(table_html, height=450)

    steps_list = [
        "Highlight df row",
        "Highlight α/tail column",
        "Intersection => t₍crit₎",
        "For one-tailed α=0.05, also highlight two-tailed α=0.10"
    ]
    if not (tail_s=="one-tailed" and abs(alpha_val-0.05)<1e-12):
        steps_list.pop()  # remove the 4th step if not relevant

    if step >= len(steps_list):
        st.write("All steps complete!")
    else:
        st.write(f"Step {step+1}: {steps_list[step]}")

    if st.button("Next Step", key=key_prefix+"_table_btn"):
        st.session_state[step_key] += 1


###############################################################################
#                      2) Z-DISTRIBUTION TAB
###############################################################################

def show_z_tab():
    st.subheader("Tab 2 • z-Distribution")

    c1,c2 = st.columns(2)
    with c1:
        z_val= st.number_input("z statistic", value=1.64, key="z_val")
    with c2:
        alpha= st.number_input("α", value=0.05, step=0.01, min_value=0.0001, max_value=0.5, key="z_alpha")
        tail_s= st.radio("Tail", ["one-tailed","two-tailed"], key="z_tail")

    if st.button("Update Plot (z)", key="z_plot_btn"):
        fig= plot_z_distribution(z_val, alpha, tail_s)
        st.pyplot(fig)

    with st.expander("Step-by-step z-table"):
        show_z_table(z_val, "z_tab")


def plot_z_distribution(z_val, alpha, tail_s):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    label_positions=[]
    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,'k')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        z_crit= stats.norm.ppf(1-alpha)
        ax.fill_between(x[x>=z_crit], y[x>=z_crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(z_crit, color='green', linestyle='--')
        place_label(ax, label_positions, z_crit, stats.norm.pdf(z_crit)+0.02, f"z₍crit₎={z_crit:.3f}", 'green')
        reject = (z_val > z_crit)
        final_crit= z_crit
    else:
        z_crit_r= stats.norm.ppf(1- alpha/2)
        z_crit_l= -z_crit_r
        ax.fill_between(x[x>= z_crit_r], y[x>= z_crit_r], color='red', alpha=0.3)
        ax.fill_between(x[x<= z_crit_l], y[x<= z_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(z_crit_r, color='green', linestyle='--')
        ax.axvline(z_crit_l, color='green', linestyle='--')
        place_label(ax, label_positions, z_crit_r, stats.norm.pdf(z_crit_r)+0.02, f"+z₍crit₎={z_crit_r:.3f}", 'green')
        place_label(ax, label_positions, z_crit_l, stats.norm.pdf(z_crit_l)+0.02, f"−z₍crit₎={z_crit_r:.3f}", 'green')
        reject = (abs(z_val) > z_crit_r)
        final_crit= z_crit_r

    ax.axvline(z_val, color='blue', linestyle='--')
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val)+0.02, f"z₍calc₎={z_val:.3f}", 'blue')

    msg = "Reject H₀" if reject else "Fail to Reject H₀"
    ax.set_title(f"z-Distribution – {msg}")
    ax.legend()
    fig.tight_layout()
    return fig


def show_z_table(z_in, key_prefix):
    step_key = key_prefix + "_step"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    step = st.session_state[step_key]

    z_in = max(0, min(3.49, z_in))
    row_base = round(np.floor(z_in*10)/10,1)
    col_part = round(z_in - row_base,2)

    row_vals = np.round(np.arange(0,3.5,0.1),1)
    col_vals = np.round(np.arange(0,0.1,0.01),2)

    if row_base not in row_vals:
        row_base= min(row_vals, key=lambda r:abs(r-row_base))
    if col_part not in col_vals:
        col_part= min(col_vals, key=lambda c:abs(c-col_part))

    # build table
    header = "".join(f"<th>{cv:.2f}</th>" for cv in col_vals)
    table_body = f"<tr><th>z.x</th>{header}</tr>\n"

    idx_r = np.where(row_vals==row_base)[0]
    if len(idx_r)==0:
        idx_r=[0]
    idx_r=idx_r[0]
    row_start=max(0, idx_r-10)
    row_end=min(len(row_vals)-1, idx_r+10)
    sub_rows=row_vals[row_start: row_end+1]

    for rv in sub_rows:
        row_html = f'<tr><td id="z_{rv:.1f}_0">{rv:.1f}</td>'
        for cv in col_vals:
            cell_id=f"z_{rv:.1f}_{cv:.2f}"
            zv= rv+cv
            cdf_val= stats.norm.cdf(zv)
            row_html+= f'<td id="{cell_id}">{cdf_val:.4f}</td>'
        row_html+="</tr>\n"
        table_body+= row_html

    css= """
    table {
      border-collapse: collapse;
      margin-top: 10px;
    }
    table td, table th {
      border:1px solid #000;
      width:70px; height:30px;
      text-align:center; font-size:0.9rem;
    }
    table th {
      background-color: #f0f0f0;
    }
    """
    html_out= wrap_html_table(css, table_body)

    row_in= (row_base in sub_rows)
    col_in= (col_part in col_vals)

    if step>=0 and row_in:
        for cv in col_vals:
            html_out= style_cell(html_out, f"z_{row_base:.1f}_{cv:.2f}")
        html_out= style_cell(html_out, f"z_{row_base:.1f}_0")

    if step>=1 and col_in:
        for rv in sub_rows:
            html_out= style_cell(html_out, f"z_{rv:.1f}_{col_part:.2f}")

    if step>=2 and row_in and col_in:
        html_out= style_cell(html_out, f"z_{row_base:.1f}_{col_part:.2f}", "blue",3)

    show_html_table(html_out, height=450)

    steps_text= [
        f"1) Highlight row {row_base:.1f}",
        f"2) Highlight column {col_part:.2f}",
        "3) Intersection => Φ(z)"
    ]
    if step>= len(steps_text):
        st.write("All steps complete!")
    else:
        st.write(steps_text[step])

    if st.button("Next Step", key=key_prefix+"_table_btn"):
        st.session_state[step_key]+=1

###############################################################################
#                        3) F-DISTRIBUTION TAB
###############################################################################

def show_f_tab():
    st.subheader("Tab 3 • F-Distribution")

    c1,c2 = st.columns(2)
    with c1:
        f_val = st.number_input("F statistic", value=4.32, key="f_val")
        df1   = st.number_input("df1 (numerator)", value=5, min_value=1, key="f_df1")
    with c2:
        df2   = st.number_input("df2 (denominator)", value=20, min_value=1, key="f_df2")
        alpha = st.number_input("α (F)", value=0.05, step=0.01, min_value=0.0001, max_value=0.5, key="f_alpha")

    if st.button("Update Plot (F)", key="f_plot_btn"):
        fig= plot_f_distribution(f_val, df1, df2, alpha)
        st.pyplot(fig)

    with st.expander("Step-by-step F-table"):
        show_f_table(df1, df2, alpha, "f_tab")


def plot_f_distribution(f_val, df1, df2, alpha):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    label_positions=[]
    # we set an upper x-limit to something like ppf(.995) * 1.1 so the distribution is visible
    x_max= stats.f.ppf(0.995, df1, df2)*1.2
    x= np.linspace(0, x_max, 400)
    y= stats.f.pdf(x, df1, df2)
    ax.plot(x,y,'k')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    f_crit= stats.f.ppf(1-alpha, df1, df2)
    ax.fill_between(x[x>= f_crit], y[x>= f_crit], color='red', alpha=0.3, label="Reject H₀")
    ax.axvline(f_crit, color='green', ls='--')
    place_label(ax, label_positions, f_crit, stats.f.pdf(f_crit, df1, df2)+0.02, f"F₍crit₎={f_crit:.3f}", 'green')

    ax.axvline(f_val, color='blue', ls='--')
    place_label(ax, label_positions, f_val, stats.f.pdf(f_val, df1, df2)+0.02, f"F₍calc₎={f_val:.3f}", 'blue')

    reject= (f_val> f_crit)
    msg= "Reject H₀" if reject else "Fail to Reject H₀"
    ax.set_title(f"F-Distribution (df1={df1}, df2={df2}) – {msg}")
    ax.legend()
    fig.tight_layout()
    return fig


def show_f_table(df1, df2, alpha, key_prefix):
    step_key= key_prefix+"_step"
    if step_key not in st.session_state:
        st.session_state[step_key]=0
    step= st.session_state[step_key]

    df1_min= max(1, df1-5)
    df1_max= df1+5
    df2_min= max(1, df2-5)
    df2_max= df2+5
    df1_list= list(range(df1_min, df1_max+1))
    df2_list= list(range(df2_min, df2_max+1))

    def f_crit(d1, d2, a_):
        return stats.f.ppf(1 - a_, d1, d2)

    header_html= "".join(f"<th>{col}</th>" for col in df2_list)
    table_body= f"<tr><th>df1\\df2</th>{header_html}</tr>\n"

    for r in df1_list:
        row_html= f'<tr><td id="f_{r}_0">{r}</td>'
        for idx,col in enumerate(df2_list, start=1):
            cell_id= f"f_{r}_{idx}"
            val= f_crit(r,col,alpha)
            row_html+= f'<td id="{cell_id}">{val:.3f}</td>'
        row_html+= "</tr>\n"
        table_body+= row_html

    css= """
    table {
      border-collapse: collapse;
      margin-top:10px;
    }
    table td, table th {
      border:1px solid #000;
      width:70px; height:30px;
      text-align:center;
      font-size:0.8rem;
    }
    table th {
      background-color:#f0f0f0;
    }
    """
    table_html= wrap_html_table(css, table_body)

    row_in= (df1 in df1_list)
    col_in= (df2 in df2_list)
    col_idx= None
    if col_in:
        col_idx= df2_list.index(df2)+1

    if step>=0 and row_in:
        for cc in range(len(df2_list)+1):
            table_html= style_cell(table_html, f"f_{df1}_{cc}")
    if step>=1 and col_in:
        for rr in df1_list:
            table_html= style_cell(table_html, f"f_{rr}_{col_idx}")
    if step>=2 and row_in and col_in:
        table_html= style_cell(table_html, f"f_{df1}_{col_idx}", "blue", 3)

    show_html_table(table_html)

    steps_text= [
        f"1) Highlight row df1={df1}",
        f"2) Highlight column df2={df2}",
        "3) Intersection => F₍crit₎"
    ]
    if step>= len(steps_text):
        st.write("All steps complete!")
    else:
        st.write(steps_text[step])

    if st.button("Next Step", key=key_prefix+"_table_btn"):
        st.session_state[step_key]+=1

###############################################################################
#                     4) CHI-SQUARE TAB
###############################################################################

def show_chi_tab():
    st.subheader("Tab 4 • Chi-Square (χ²)")

    c1,c2= st.columns(2)
    with c1:
        chi_val= st.number_input("χ² statistic", value=7.88, key="c_val")
        df_val= st.number_input("df (Chi)", value=3, min_value=1, step=1, key="c_df")
    with c2:
        alpha= st.selectbox("α", [0.10,0.05,0.01,0.001], index=1, key="c_alpha")

    if st.button("Update Plot (χ²)", key="chi_plot_btn"):
        fig= plot_chi_square(chi_val, df_val, alpha)
        st.pyplot(fig)

    with st.expander("Step-by-step χ²-table"):
        show_chi_table(df_val, alpha, "chi_tab")


def chi_crit(df, a):
    return stats.chi2.ppf(1-a, df)

def plot_chi_square(chi_val, df, alpha):
    fig, ax= plt.subplots(figsize=(12,4), dpi=100)
    label_positions=[]
    x_max= chi_crit(df, 0.001)*1.2
    x= np.linspace(0, x_max, 400)
    y= stats.chi2.pdf(x, df)
    ax.plot(x,y,'k')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    cc= chi_crit(df, alpha)
    ax.fill_between(x[x>=cc], y[x>=cc], color='red', alpha=0.3, label="Reject H₀")
    ax.axvline(cc, color='green', linestyle='--')
    place_label(ax, label_positions, cc, stats.chi2.pdf(cc,df)+0.02, f"χ²₍crit₎={cc:.3f}", 'green')

    ax.axvline(chi_val, color='blue', linestyle='--')
    place_label(ax, label_positions, chi_val, stats.chi2.pdf(chi_val,df)+0.02, f"χ²₍calc₎={chi_val:.3f}", 'blue')

    reject= (chi_val>cc)
    msg= "Reject H₀" if reject else "Fail to Reject H₀"
    ax.set_title(f"χ²-Distribution (df={df}) – {msg}")
    ax.legend()
    fig.tight_layout()
    return fig

def show_chi_table(df_val, alpha, key_prefix):
    step_key= key_prefix+"_step"
    if step_key not in st.session_state:
        st.session_state[step_key]=0
    step= st.session_state[step_key]

    df_min= max(1, df_val-5)
    df_max= df_val+5
    df_range= list(range(df_min, df_max+1))
    alpha_list= [0.10, 0.05, 0.01, 0.001]

    def chi_crit_func(d, a_):
        return stats.chi2.ppf(1-a_, d)

    header= "".join(f"<th>α={a_}</th>" for a_ in alpha_list)
    table_body= f"<tr><th>df</th>{header}</tr>\n"
    for d_ in df_range:
        row_html= f'<tr><td id="chi_{d_}_0">{d_}</td>'
        for idx,a_ in enumerate(alpha_list, start=1):
            cid= f"chi_{d_}_{idx}"
            val= chi_crit_func(d_, a_)
            row_html+= f'<td id="{cid}">{val:.3f}</td>'
        row_html+="</tr>\n"
        table_body+= row_html

    css= """
    table {
      border-collapse: collapse;
      margin-top:10px;
    }
    table td, table th {
      border:1px solid #000;
      width:80px; height:30px;
      text-align:center; font-size:0.85rem;
    }
    table th {
      background-color:#f0f0f0;
    }
    """
    html_out= wrap_html_table(css, table_body)

    row_in= (df_val in df_range)
    try:
        col_idx= alpha_list.index(alpha)+1
        col_in= True
    except:
        col_idx=None
        col_in=False

    if step>=0 and row_in:
        for cc in range(len(alpha_list)+1):
            html_out= style_cell(html_out, f"chi_{df_val}_{cc}")
    if step>=1 and col_in:
        for d_ in df_range:
            html_out= style_cell(html_out, f"chi_{d_}_{col_idx}")
    if step>=2 and row_in and col_in:
        html_out= style_cell(html_out, f"chi_{df_val}_{col_idx}", "blue", 3)

    show_html_table(html_out)

    steps_txt= [
        f"1) Highlight row df={df_val}",
        f"2) Highlight column α={alpha}",
        "3) Intersection => χ²₍crit₎"
    ]
    if step>= len(steps_txt):
        st.write("All steps complete!")
    else:
        st.write(steps_txt[step])

    if st.button("Next Step", key=key_prefix+"_table_btn"):
        st.session_state[step_key]+=1


###############################################################################
#                  5) MANN–WHITNEY U TAB
###############################################################################

def show_mannwhitney_tab():
    st.subheader("Tab 5 • Mann–Whitney U")

    c1,c2= st.columns(2)
    with c1:
        u_val= st.number_input("U statistic", value=23, min_value=1, key="u_val")
        n1= st.number_input("n1", value=10, min_value=2, step=1, key="u_n1")
    with c2:
        n2= st.number_input("n2", value=12, min_value=2, step=1, key="u_n2")
        alpha= st.number_input("α (Mann–Whitney)", value=0.05, step=0.01,
                               min_value=0.0001, max_value=0.5, key="u_alpha")
        tail_s= st.radio("Tail (U)", ["one-tailed","two-tailed"], key="u_tail")

    if st.button("Update Plot (U)", key="u_plot_btn"):
        fig= plot_mannwhitney(u_val, n1, n2, alpha, tail_s)
        st.pyplot(fig)

    with st.expander("Step-by-step U-table"):
        show_mannwhitney_table(n1, n2, alpha, tail_s, "u_tab")


def mw_u_crit(n1, n2, alpha, tail_s):
    """
    Minimal dummy formula. Typically you'd use tables or normal approx.
    We'll do a normal approximation approach here for demonstration.
    """
    mu= n1*n2/2.0
    sigma= np.sqrt(n1*n2*(n1+n2+1)/12.0)
    if tail_s=="one-tailed":
        z= stats.norm.ppf(alpha)
        return int(np.floor(mu+z*sigma))
    else:
        z= stats.norm.ppf(alpha/2)
        lower= int(np.floor(mu+ z*sigma))
        upper= int(np.ceil(mu - z*sigma))
        return (lower, upper)

def plot_mannwhitney(u_val, n1, n2, alpha, tail_s):
    mu= n1*n2/2
    sigma= np.sqrt(n1*n2*(n1+n2+1)/12)
    fig, ax= plt.subplots(figsize=(12,4), dpi=100)
    label_positions=[]

    x_min= mu-4*sigma
    x_max= mu+4*sigma
    x= np.linspace(x_min, x_max, 400)
    y= stats.norm.pdf(x, mu, sigma)
    ax.plot(x,y,'k')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        crit= mw_u_crit(n1, n2, alpha, "one-tailed")  # single integer
        ax.fill_between(x[x<=crit], y[x<=crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(crit, color='green', linestyle='--')
        place_label(ax, label_positions, crit, stats.norm.pdf(crit, mu, sigma)+0.02,
                    f"U₍crit₎={crit}", 'green')
        reject= (u_val<= crit)
    else:
        lohi= mw_u_crit(n1, n2, alpha, "two-tailed")  # (low, high)
        lo, hi= lohi
        ax.fill_between(x[x<=lo], y[x<=lo], color='red', alpha=0.3)
        ax.fill_between(x[x>=hi], y[x>=hi], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(lo, color='green', linestyle='--')
        ax.axvline(hi, color='green', linestyle='--')
        place_label(ax, label_positions, lo, stats.norm.pdf(lo, mu, sigma)+0.02,
                    f"U₍crit₎={lo}", 'green')
        reject= (u_val<= lo or u_val>= hi)
    ax.axvline(u_val, color='blue', linestyle='--')
    place_label(ax, label_positions, u_val, stats.norm.pdf(u_val,mu,sigma)+0.02,
                f"U₍calc₎={u_val}", 'blue')

    msg= "Reject H₀" if reject else "Fail to Reject H₀"
    ax.set_title(f"Mann–Whitney U: n1={n1}, n2={n2} – {msg}")
    ax.legend()
    fig.tight_layout()
    return fig

def show_mannwhitney_table(n1, n2, alpha, tail_s, key_prefix):
    """
    Step-by-step approach with row => n1±5, col => n2±5, intersection => U_crit.
    """
    step_key= key_prefix+"_step"
    if step_key not in st.session_state:
        st.session_state[step_key]=0
    step= st.session_state[step_key]

    n1_min= max(2, n1-5)
    n1_max= n1+5
    n1_list= list(range(n1_min, n1_max+1))

    n2_min= max(2, n2-5)
    n2_max= n2+5
    n2_list= list(range(n2_min, n2_max+1))

    def make_cell(r,c):
        val= mw_u_crit(r, c, alpha, tail_s)
        # for two-tailed, val is (lo,hi)
        if isinstance(val, tuple):
            return f"{val[0]},{val[1]}"
        return str(val)

    header= "".join(f"<th>{col}</th>" for col in n2_list)
    table_body= f"<tr><th>n1\\n2</th>{header}</tr>\n"

    for row_n1 in n1_list:
        row_html= f'<tr><td id="mw_{row_n1}_0">{row_n1}</td>'
        for idx, col_n2 in enumerate(n2_list, start=1):
            cid= f"mw_{row_n1}_{idx}"
            cell_val= make_cell(row_n1, col_n2)
            row_html+= f'<td id="{cid}">{cell_val}</td>'
        row_html+="</tr>\n"
        table_body+= row_html

    css= """
    table {
      border-collapse: collapse;
      margin-top:10px;
    }
    table td, table th {
      border:1px solid #000;
      width:80px; height:30px;
      text-align:center; font-size:0.85rem;
    }
    table th {
      background-color:#f0f0f0;
    }
    """
    html_out= wrap_html_table(css, table_body)

    # find col index
    col_in= (n2 in n2_list)
    if col_in:
        col_idx= n2_list.index(n2)+1
    else:
        col_idx= None

    row_in= (n1 in n1_list)

    # steps
    # 0 => highlight entire row n1
    # 1 => highlight entire column n2
    # 2 => highlight intersection => U_crit
    if step>=0 and row_in:
        for i in range(len(n2_list)+1):
            html_out= style_cell(html_out, f"mw_{n1}_{i}")
    if step>=1 and col_in:
        for r_ in n1_list:
            html_out= style_cell(html_out, f"mw_{r_}_{col_idx}")
    if step>=2 and row_in and col_in:
        html_out= style_cell(html_out, f"mw_{n1}_{col_idx}", "blue", 3)

    show_html_table(html_out)

    steps_text= [
        f"1) Highlight row n1={n1}",
        f"2) Highlight column n2={n2}",
        "3) Intersection => U₍crit₎"
    ]
    if step>= len(steps_text):
        st.write("All steps complete!")
    else:
        st.write(steps_text[step])

    if st.button("Next Step", key=key_prefix+"_table_btn"):
        st.session_state[step_key]+=1


###############################################################################
#                6) WILCOXON SIGNED-RANK TAB
###############################################################################

def show_wilcoxon_tab():
    st.subheader("Tab 6 • Wilcoxon Signed-Rank T")

    c1,c2= st.columns(2)
    with c1:
        T_val= st.number_input("T statistic", value=15, min_value=1, key="wil_T")
        N_val= st.number_input("N (non-zero diffs)", value=12, min_value=5, step=1, key="wil_N")
    with c2:
        alpha= st.number_input("α (Wilcoxon)", value=0.05, step=0.01, min_value=0.0001, max_value=0.5, key="wil_alpha")
        tail_s= st.radio("Tail (Wilcoxon)", ["one-tailed","two-tailed"], key="wil_tail")

    if st.button("Update Plot (Wilcoxon)", key="wil_plot_btn"):
        fig= plot_wilcoxon(T_val, N_val, alpha, tail_s)
        st.pyplot(fig)

    with st.expander("Step-by-step Wilcoxon T-table"):
        show_wilcoxon_table(N_val, alpha, tail_s, "wil_tab")


def wlx_t_crit(N, alpha, tail):
    """
    Minimal approach for demonstration. 
    Use normal approx: mu= N(N+1)/4, sigma= sqrt(N(N+1)(2N+1)/24), etc.
    We'll produce a single or pair of boundary values.
    """
    mu= N*(N+1)/4
    sigma= np.sqrt(N*(N+1)*(2*N+1)/24)
    if tail=="one-tailed":
        z= stats.norm.ppf(alpha)
        return int(np.floor(mu+ z*sigma))
    else:
        z= stats.norm.ppf(alpha/2)
        lo= int(np.floor(mu+ z*sigma))
        hi= int(np.ceil(mu- z*sigma))
        return (lo, hi)

def plot_wilcoxon(T_val, N_val, alpha, tail_s):
    mu= N_val*(N_val+1)/4
    sigma= np.sqrt(N_val*(N_val+1)*(2*N_val+1)/24)
    fig, ax= plt.subplots(figsize=(12,4), dpi=100)
    label_positions=[]
    x_min= mu - 4*sigma
    x_max= mu + 4*sigma
    x= np.linspace(x_min, x_max, 400)
    y= stats.norm.pdf(x, mu, sigma)
    ax.plot(x,y,'k')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        crit= wlx_t_crit(N_val, alpha, "one-tailed")
        ax.fill_between(x[x<=crit], y[x<=crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(crit, color='green', linestyle='--')
        place_label(ax, label_positions, crit, stats.norm.pdf(crit, mu, sigma)+0.02,
                    f"T₍crit₎={crit}", 'green')
        reject= (T_val<= crit)
    else:
        lohi= wlx_t_crit(N_val, alpha, "two-tailed")
        lo, hi= lohi
        ax.fill_between(x[x<=lo], y[x<=lo], color='red', alpha=0.3)
        ax.fill_between(x[x>=hi], y[x>=hi], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(lo, color='green', linestyle='--')
        ax.axvline(hi, color='green', linestyle='--')
        place_label(ax, label_positions, lo, stats.norm.pdf(lo, mu, sigma)+0.02, f"T₍crit₎={lo}", 'green')
        reject= (T_val<= lo or T_val>= hi)

    ax.axvline(T_val, color='blue', linestyle='--')
    place_label(ax, label_positions, T_val, stats.norm.pdf(T_val, mu, sigma)+0.02,
                f"T₍calc₎={T_val}", 'blue')

    msg= "Reject H₀" if reject else "Fail to Reject H₀"
    ax.set_title(f"Wilcoxon T (N={N_val}) – {msg}")
    ax.legend()
    fig.tight_layout()
    return fig

def show_wilcoxon_table(N_val, alpha, tail_s, key_prefix):
    """
    Step-by-step approach: row => N±5, col => alpha in [0.10, 0.05, 0.01, 0.001],
    intersection => T₍crit₎.
    """
    step_key= key_prefix + "_step"
    if step_key not in st.session_state:
        st.session_state[step_key]=0
    step= st.session_state[step_key]

    N_min= max(5, N_val-5)
    N_max= N_val+5
    N_list= range(N_min, N_max+1)
    alpha_list= [0.10, 0.05, 0.01, 0.001]

    def wlx_crit_func(nn, a_, tail_):
        return wlx_t_crit(nn, a_, tail_)

    # Build table
    header= "".join(f"<th>α={a_}</th>" for a_ in alpha_list)
    table_body= f"<tr><th>N</th>{header}</tr>\n"
    for nn in N_list:
        row_html= f'<tr><td id="wil_{nn}_0">{nn}</td>'
        for idx,a_ in enumerate(alpha_list, start=1):
            cell_id= f"wil_{nn}_{idx}"
            val= wlx_crit_func(nn, a_, tail_s)
            if isinstance(val, tuple):
                # two values
                row_html+= f'<td id="{cell_id}">{val[0]},{val[1]}</td>'
            else:
                row_html+= f'<td id="{cell_id}">{val}</td>'
        row_html+= "</tr>\n"
        table_body+= row_html

    css= """
    table {
      border-collapse: collapse;
      margin-top:10px;
    }
    table td, table th {
      border:1px solid #000;
      width:90px; height:30px;
      text-align:center;
      font-size:0.85rem;
    }
    table th {
      background-color:#f0f0f0;
    }
    """
    html_out= wrap_html_table(css, table_body)

    row_in= (N_val in N_list)
    try:
        col_idx= alpha_list.index(alpha)+1
        col_in=True
    except:
        col_idx=None
        col_in=False

    if step>=0 and row_in:
        for cc in range(len(alpha_list)+1):
            html_out= style_cell(html_out, f"wil_{N_val}_{cc}")
    if step>=1 and col_in:
        for nn_ in N_list:
            html_out= style_cell(html_out, f"wil_{nn_}_{col_idx}")
    if step>=2 and row_in and col_in:
        html_out= style_cell(html_out, f"wil_{N_val}_{col_idx}", "blue",3)

    show_html_table(html_out)

    steps_text= [
        f"1) Highlight row N={N_val}",
        f"2) Highlight column α={alpha}",
        "3) Intersection => T₍crit₎"
    ]
    if step>= len(steps_text):
        st.write("All steps complete!")
    else:
        st.write(steps_text[step])

    if st.button("Next Step", key=key_prefix+"_table_btn"):
        st.session_state[step_key]+=1


###############################################################################
#                 7) BINOMIAL TAB
###############################################################################

def show_binomial_tab():
    st.subheader("Tab 7 • Binomial")

    c1,c2= st.columns(2)
    with c1:
        n= st.number_input("n (trials)", value=20, min_value=1, step=1, key="b_n")
        p= st.number_input("p (null proportion)", value=0.5, step=0.01, min_value=0.01, max_value=0.99, key="b_p")
    with c2:
        k= st.number_input("k (successes)", value=12, min_value=0, step=1, key="b_k")
        alpha= st.number_input("α (two-tailed)", value=0.05, step=0.01, min_value=0.0001, max_value=0.5, key="b_alpha")

    if st.button("Update Plot (Binomial)", key="b_plot_btn"):
        fig= plot_binomial(n, p, k)
        st.pyplot(fig)

    with st.expander("Quick table (k ±5)"):
        show_binomial_table(n, p, k, alpha, "b_tab")


def plot_binomial(n, p, k):
    x= np.arange(n+1)
    pmf= stats.binom.pmf(x,n,p)
    fig, ax= plt.subplots(figsize=(12,4), dpi=100)
    ax.bar(x, pmf, color='lightgrey', label="P(X=k)")
    if 0<=k<=n:
        ax.bar(k, pmf[k], color='blue', label=f"k={k}")
    ax.set_title(f"Binomial (n={n}, p={p}) – k={k}")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X=k)")
    ax.legend()
    fig.tight_layout()
    return fig


def show_binomial_table(n, p, k, alpha, key_prefix):
    """
    Step-by-step highlight for a partial table ±5 around k,
    showing P(X=k), P(X≤k), P(X≥k).
    """
    step_key= key_prefix+"_step"
    if step_key not in st.session_state:
        st.session_state[step_key]=0
    step= st.session_state[step_key]

    k_min= max(0,k-5)
    k_max= min(n, k+5)
    k_list= range(k_min, k_max+1)

    # Build table
    header= "<th>P(X=k)</th><th>P(X≤k)</th><th>P(X≥k)</th>"
    rows_html=""
    for kk in k_list:
        pmf= stats.binom.pmf(kk,n,p)
        cdf= stats.binom.cdf(kk,n,p)
        sf= 1- stats.binom.cdf(kk-1,n,p) if kk>0 else 1.0
        rows_html+= (f'<tr><td id="bin_{kk}_0">{kk}</td>'
                     f'<td id="bin_{kk}_1">{pmf:.4f}</td>'
                     f'<td id="bin_{kk}_2">{cdf:.4f}</td>'
                     f'<td id="bin_{kk}_3">{sf:.4f}</td></tr>\n')

    css= """
    table {
      border-collapse: collapse;
      margin-top: 10px;
    }
    table td, table th {
      border:1px solid #000;
      width:80px; height:30px;
      text-align:center; font-size:0.8rem;
    }
    table th {
      background-color:#f0f0f0;
    }
    """
    table_body= f"<tr><th>k</th>{header}</tr>\n{rows_html}"
    html_out= wrap_html_table(css, table_body)

    if step>=0:
        # highlight entire row for k
        for cc in range(4):
            html_out= style_cell(html_out, f"bin_{k}_{cc}")
    if step>=1:
        # highlight the pmf cell specifically
        html_out= style_cell(html_out, f"bin_{k}_1", "blue", 3)

    show_html_table(html_out)

    steps_list= [
        f"1) Highlight k row={k}",
        "2) Highlight P(X=k)"
    ]
    if step>= len(steps_list):
        st.write("All steps complete!")
    else:
        st.write(steps_list[step])

    if st.button("Next Step", key=key_prefix+"_table_btn"):
        st.session_state[step_key]+=1


###############################################################################
#                                MAIN
###############################################################################

def main():
    st.set_page_config("PSYC250 – Statistical Tables Explorer (12×4 Figures)",
                       layout="wide")
    st.title("PSYC250 – Statistical Tables Explorer (12 × 4 figures)")

    tabs = st.tabs([
        "t-Dist",
        "z-Dist",
        "F-Dist",
        "Chi-Square",
        "Mann–Whitney U",
        "Wilcoxon T",
        "Binomial"
    ])

    with tabs[0]:
        show_t_tab()
    with tabs[1]:
        show_z_tab()
    with tabs[2]:
        show_f_tab()
    with tabs[3]:
        show_chi_tab()
    with tabs[4]:
        show_mannwhitney_tab()
    with tabs[5]:
        show_wilcoxon_tab()
    with tabs[6]:
        show_binomial_tab()


if __name__=="__main__":
    main()
