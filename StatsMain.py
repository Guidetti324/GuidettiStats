import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")

###############################################################################
#                            HELPER FUNCTIONS
###############################################################################

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

def highlight_html_cell(html_in, cell_id, color="red", border_px=2):
    needle = f'id="{cell_id}"'
    styled = f'id="{cell_id}" style="border:{border_px}px solid {color};"'
    return html_in.replace(needle, styled, 1)

def show_html_table(html_content, height=400):
    wrapped = f"""
    <html>
      <head><meta charset="UTF-8"></head>
      <body>{html_content}</body>
    </html>
    """
    components.html(wrapped, height=height, scrolling=True)

def next_step_button(step_key):
    if st.button("Next Step", key=step_key + "_btn"):
        st.session_state[step_key] += 1

###############################################################################
#                               T‑DISTRIBUTION
###############################################################################

def show_t_distribution_tab():
    st.subheader("Tab 1: t‑Distribution")
    col1, col2 = st.columns(2)
    with col1:
        t_val = st.number_input("t statistic", value=2.87, key="tab1_tval")
        df_val = st.number_input("df", min_value=1, value=55, key="tab1_df")
    with col2:
        alpha_val = st.number_input("α", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, key="tab1_alpha")
        tail_s   = st.radio("Tail Type", ["one‑tailed", "two‑tailed"], key="tab1_tail")

    if st.button("Update Plot", key="tab1_update"):
        st.pyplot(plot_t_distribution(t_val, df_val, alpha_val, tail_s))

    with st.expander("Show Table Lookup (±5 df)"):
        show_t_table_lookup(df_val, alpha_val, tail_s, "tab1")

def plot_t_distribution(t_val, df, alpha, tail_s):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    label_positions = []
    x = np.linspace(-4, 4, 400)
    y = stats.t.pdf(x, df)
    ax.plot(x, y, color='black')
    ax.fill_between(x, y, color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    if tail_s == "one‑tailed":
        t_crit = stats.t.ppf(1 - alpha, df)
        ax.fill_between(x[x >= t_crit], y[x >= t_crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit, color='green', linestyle='--')
        place_label(ax, label_positions, t_crit, stats.t.pdf(t_crit, df) + 0.02, f"t₍crit₎ = {t_crit:.3f}", 'green')
        sig = t_val > t_crit
        final_crit = t_crit
    else:
        t_crit_r = stats.t.ppf(1 - alpha / 2, df)
        t_crit_l = -t_crit_r
        ax.fill_between(x[x >= t_crit_r], y[x >= t_crit_r], color='red', alpha=0.3)
        ax.fill_between(x[x <= t_crit_l], y[x <= t_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit_r, color='green', linestyle='--')
        ax.axvline(t_crit_l, color='green', linestyle='--')
        place_label(ax, label_positions, t_crit_r, stats.t.pdf(t_crit_r, df) + 0.02, f"+t₍crit₎ = {t_crit_r:.3f}", 'green')
        place_label(ax, label_positions, t_crit_l, stats.t.pdf(t_crit_l, df) + 0.02, f"–t₍crit₎ = {t_crit_r:.3f}", 'green')
        sig = abs(t_val) > t_crit_r
        final_crit = t_crit_r

    ax.axvline(t_val, color='blue', linestyle='--')
    place_label(ax, label_positions, t_val, stats.t.pdf(t_val, df) + 0.02, f"t₍calc₎ = {t_val:.3f}", 'blue')
    verdict = "Reject H₀" if sig else "Fail to Reject H₀"
    ax.set_title(f"t‑Distribution (df = {df}) – {verdict}")
    ax.legend()
    fig.tight_layout()
    return fig

def show_t_table_lookup(df_val, alpha_val, tail_s, key_prefix):
    st.write("### Step‑by‑Step t‑Table Lookup")
    step_key = key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    cur_step = st.session_state[step_key]

    df_min, df_max = max(1, df_val - 5), df_val + 5
    df_list = list(range(df_min, df_max + 1))

    columns = [
        ("df", ""),
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001),
    ]
    def compute_tcrit(dof, mode, a_):
        return stats.t.ppf(1 - a_ / (1 if mode == "one" else 2), dof)

    style = """
    <style>
    table.ttable{border-collapse:collapse;margin-top:10px}
    table.ttable td,table.ttable th{border:1px solid #000;width:80px;height:30px;text-align:center;
    font-family:sans-serif;font-size:0.9rem}
    table.ttable th{background:#f0f0f0}
    </style>
    """
    table_html = f"{style}<table class='ttable'><tr><th>df</th>"
    for m, a in columns[1:]:
        table_html += f"<th>{m}_{a}</th>"
    table_html += "</tr>\n"

    for d in df_list:
        table_html += "<tr>"
        table_html += f'<td id="df_{d}_0">{d}</td>'
        for i, (m, a) in enumerate(columns[1:], start=1):
            table_html += f'<td id="df_{d}_{i}">{compute_tcrit(d, m, a):.3f}</td>'
        table_html += "</tr>\n"
    table_html += "</table>"

    mode = "one" if tail_s.startswith("one") else "two"
    col_idx = next(i for i, (m, a) in enumerate(columns[1:], 1) if m == mode and abs(a - alpha_val) < 1e-9)

    if cur_step >= 0:
        for i in range(len(columns)):
            table_html = highlight_html_cell(table_html, f"df_{df_val}_{i}", "red", 2)
    if cur_step >= 1:
        for d in df_list:
            table_html = highlight_html_cell(table_html, f"df_{d}_{col_idx}", "red", 2)
    if cur_step >= 2:
        table_html = highlight_html_cell(table_html, f"df_{df_val}_{col_idx}", "blue", 3)

    show_html_table(table_html, 450)
    steps = ["1) Highlight df row", "2) Highlight α/tail column", "3) Intersection → t₍crit₎"]
    if cur_step > 2:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps[cur_step]}")
    next_step_button(step_key)

###############################################################################
#                               Z‑DISTRIBUTION
###############################################################################

def show_z_distribution_tab():
    st.subheader("Tab 2: z‑Distribution")
    col1, col2 = st.columns(2)
    with col1:
        z_val = st.number_input("z statistic", value=1.64, key="tab2_zval")
    with col2:
        alpha_val = st.number_input("α", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, key="tab2_alpha")
        tail_s = st.radio("Tail Type", ["one‑tailed", "two‑tailed"], key="tab2_tail")

    if st.button("Update Plot", key="tab2_update"):
        st.pyplot(plot_z_distribution(z_val, alpha_val, tail_s))

    with st.expander("Show z‑Table Lookup (±10 rows)"):
        show_z_table_lookup(z_val, "tab2")

def plot_z_distribution(z_val, alpha, tail_s):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    label_positions = []
    x = np.linspace(-4, 4, 400)
    y = stats.norm.pdf(x)
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, color="lightgrey", alpha=0.2, label="Fail to Reject H₀")

    if tail_s == "one‑tailed":
        z_crit = stats.norm.ppf(1 - alpha)
        ax.fill_between(x[x >= z_crit], y[x >= z_crit], color="red", alpha=0.3, label="Reject H₀")
        ax.axvline(z_crit, color="green", linestyle="--")
        place_label(ax, label_positions, z_crit, stats.norm.pdf(z_crit) + 0.02, f"z₍crit₎ = {z_crit:.3f}", "green")
        sig = z_val > z_crit
        final_crit = z_crit
    else:
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ax.fill_between(x[x >= z_crit], y[x >= z_crit], color="red", alpha=0.3)
        ax.fill_between(x[x <= -z_crit], y[x <= -z_crit], color="red", alpha=0.3, label="Reject H₀")
        ax.axvline(z_crit, color="green", linestyle="--")
        ax.axvline(-z_crit, color="green", linestyle="--")
        place_label(ax, label_positions, z_crit, stats.norm.pdf(z_crit) + 0.02, f"+z₍crit₎ = {z_crit:.3f}", "green")
        place_label(ax, label_positions, -z_crit, stats.norm.pdf(-z_crit) + 0.02, f"–z₍crit₎ = {z_crit:.3f}", "green")
        sig = abs(z_val) > z_crit
        final_crit = z_crit

    ax.axvline(z_val, color="blue", linestyle="--")
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val) + 0.02, f"z₍calc₎ = {z_val:.3f}", "blue")
    verdict = "Reject H₀" if sig else "Fail to Reject H₀"
    ax.set_title(f"z‑Distribution – {verdict}")
    ax.legend()
    fig.tight_layout()
    return fig

def show_z_table_lookup(z_in, key_prefix):
    st.write("### Step‑by‑Step z‑Table Lookup")
    step_key = key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    cur_step = st.session_state[step_key]

    z_clamped = min(max(0, z_in), 3.49)
    row_base = np.floor(z_clamped * 10) / 10
    col_part = round(z_clamped - row_base, 2)

    row_vals = np.round(np.arange(0, 3.5, 0.1), 1)
    col_vals = np.round(np.arange(0, 0.1, 0.01), 2)
    sub_rows = row_vals[max(0, np.where(row_vals == row_base)[0][0] - 10):min(len(row_vals), np.where(row_vals == row_base)[0][0] + 11)]

    style = """
    <style>
    table.ztable{border-collapse:collapse;margin-top:10px}
    table.ztable td,table.ztable th{border:1px solid #000;width:70px;height:30px;text-align:center;
    font-family:sans-serif;font-size:0.9rem}
    table.ztable th{background:#f0f0f0}
    </style>
    """
    table_html = f"{style}<table class='ztable'><tr><th>z.x</th>"
    for cv in col_vals:
        table_html += f"<th>{cv:.2f}</th>"
    table_html += "</tr>\n"

    for rv in sub_rows:
        table_html += "<tr>"
        table_html += f'<td id="z_{rv:.1f}_0">{rv:.1f}</td>'
        for cv in col_vals:
            table_html += f'<td id="z_{rv:.1f}_{cv:.2f}">{stats.norm.cdf(rv + cv):.4f}</td>'
        table_html += "</tr>\n"
    table_html += "</table>"

    if cur_step >= 0:
        for cv in col_vals:
            table_html = highlight_html_cell(table_html, f"z_{row_base:.1f}_{cv:.2f}", "red", 2)
        table_html = highlight_html_cell(table_html, f"z_{row_base:.1f}_0", "red", 2)
    if cur_step >= 1:
        for rv in sub_rows:
            table_html = highlight_html_cell(table_html, f"z_{rv:.1f}_{col_part:.2f}", "red", 2)
    if cur_step >= 2:
        table_html = highlight_html_cell(table_html, f"z_{row_base:.1f}_{col_part:.2f}", "blue", 3)

    show_html_table(table_html, 450)
    steps = ["1) Locate row", "2) Locate column", "3) Intersection → Φ(z)"]
    if cur_step > 2:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps[cur_step]}")
    next_step_button(step_key)

###############################################################################
#                               F‑DISTRIBUTION
###############################################################################

def show_f_distribution_tab():
    st.subheader("Tab 3: F‑Distribution")
    col1, col2 = st.columns(2)
    with col1:
        f_val = st.number_input("F statistic", value=4.32, key="tab3_f")
        df1_val = st.number_input("df₁ (numerator)", min_value=1, value=5, key="tab3_df1")
    with col2:
        df2_val = st.number_input("df₂ (denominator)", min_value=1, value=20, key="tab3_df2")
        alpha_val = st.number_input("α (right‑tail)", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, key="tab3_alpha")

    if st.button("Update Plot", key="tab3_update"):
        st.pyplot(plot_f_distribution(f_val, df1_val, df2_val, alpha_val))

    with st.expander("Show F‑Table Lookup (±5 df)"):
        show_f_table_lookup(df1_val, df2_val, alpha_val, "tab3")

def plot_f_distribution(f_val, df1, df2, alpha):
    xmax = stats.f.ppf(0.995, df1, df2) * 1.1
    x = np.linspace(0, xmax, 400)
    y = stats.f.pdf(x, df1, df2)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, color="lightgrey", alpha=0.2, label="Fail to Reject H₀")
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    ax.fill_between(x[x >= f_crit], y[x >= f_crit], color="red", alpha=0.3, label="Reject H₀")
    ax.axvline(f_crit, color="green", linestyle="--")
    ax.axvline(f_val, color="blue", linestyle="--")
    place_label(ax, [], f_crit, stats.f.pdf(f_crit, df1, df2) + 0.02, f"F₍crit₎ = {f_crit:.3f}", "green")
    place_label(ax, [], f_val, stats.f.pdf(f_val, df1, df2) + 0.02, f"F₍calc₎ = {f_val:.3f}", "blue")
    verdict = "Reject H₀" if f_val > f_crit else "Fail to Reject H₀"
    ax.set_title(f"F‑Distribution (df₁ = {df1}, df₂ = {df2}) – {verdict}")
    ax.legend()
    fig.tight_layout()
    return fig

def show_f_table_lookup(df1_val, df2_val, alpha, key_prefix):
    st.write("### Step‑by‑Step F‑Table Lookup")
    step_key = key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    cur_step = st.session_state[step_key]

    df1_list = list(range(max(1, df1_val - 5), df1_val + 6))
    df2_list = list(range(max(1, df2_val - 5), df2_val + 6))

    style = """
    <style>
    table.ftable{border-collapse:collapse;margin-top:10px}
    table.ftable td,table.ftable th{border:1px solid #000;width:85px;height:30px;text-align:center;
    font-family:sans-serif;font-size:0.85rem}
    table.ftable th{background:#f0f0f0}
    </style>
    """
    table_html = f"{style}<table class='ftable'><tr><th>df₁＼df₂</th>"
    for d2 in df2_list:
        table_html += f"<th>{d2}</th>"
    table_html += "</tr>\n"

    for d1 in df1_list:
        table_html += "<tr>"
        table_html += f'<td id="f_{d1}_0">{d1}</td>'
        for idx, d2 in enumerate(df2_list, 1):
            table_html += f'<td id="f_{d1}_{idx}">{stats.f.ppf(1 - alpha, d1, d2):.3f}</td>'
        table_html += "</tr>\n"
    table_html += "</table>"

    col_idx = df2_list.index(df2_val) + 1
    if cur_step >= 0:
        for i in range(len(df2_list) + 1):
            table_html = highlight_html_cell(table_html, f"f_{df1_val}_{i}", "red", 2)
    if cur_step >= 1:
        for d1 in df1_list:
            table_html = highlight_html_cell(table_html, f"f_{d1}_{col_idx}", "red", 2)
    if cur_step >= 2:
        table_html = highlight_html_cell(table_html, f"f_{df1_val}_{col_idx}", "blue", 3)

    show_html_table(table_html, 450)
    steps = ["1) Highlight df₁ row", "2) Highlight df₂ column", "3) Intersection → F₍crit₎"]
    if cur_step > 2:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps[cur_step]}")
    next_step_button(step_key)

###############################################################################
#                               CHI‑SQUARE
###############################################################################

def show_chi_square_tab():
    st.subheader("Tab 4: Chi‑Square (χ²)")
    col1, col2 = st.columns(2)
    with col1:
        chi_val = st.number_input("χ² statistic", value=7.88, key="tab4_chi")
        df_val = st.number_input("df", min_value=1, value=3, key="tab4_df")
    with col2:
        alpha_val = st.selectbox("α (right‑tail)", [0.10, 0.05, 0.01, 0.001], key="tab4_alpha")

    if st.button("Update Plot", key="tab4_update"):
        st.pyplot(plot_chi_square_distribution(chi_val, df_val, alpha_val))

    with st.expander("Show χ²‑Table Lookup (±5 df)"):
        show_chi_square_lookup(df_val, alpha_val, "tab4")

def plot_chi_square_distribution(chi_val, df, alpha):
    xmax = stats.chi2.ppf(0.995, df) * 1.1
    x = np.linspace(0, xmax, 400)
    y = stats.chi2.pdf(x, df)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, color="lightgrey", alpha=0.2, label="Fail to Reject H₀")
    chi_crit = stats.chi2.ppf(1 - alpha, df)
    ax.fill_between(x[x >= chi_crit], y[x >= chi_crit], color="red", alpha=0.3, label="Reject H₀")
    ax.axvline(chi_crit, color="green", linestyle="--")
    ax.axvline(chi_val, color="blue", linestyle="--")
    place_label(ax, [], chi_crit, stats.chi2.pdf(chi_crit, df) + 0.02, f"χ²₍crit₎ = {chi_crit:.3f}", "green")
    place_label(ax, [], chi_val, stats.chi2.pdf(chi_val, df) + 0.02, f"χ²₍calc₎ = {chi_val:.3f}", "blue")
    verdict = "Reject H₀" if chi_val > chi_crit else "Fail to Reject H₀"
    ax.set_title(f"χ²‑Distribution (df = {df}) – {verdict}")
    ax.legend()
    fig.tight_layout()
    return fig

def show_chi_square_lookup(df_val, alpha, key_prefix):
    st.write("### Step‑by‑Step χ²‑Table Lookup")
    step_key = key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    cur_step = st.session_state[step_key]

    df_list = list(range(max(1, df_val - 5), df_val + 6))
    alphas = [0.10, 0.05, 0.01, 0.001]

    style = """
    <style>
    table.chitable{border-collapse:collapse;margin-top:10px}
    table.chitable td,table.chitable th{border:1px solid #000;width:80px;height:30px;text-align:center;
    font-family:sans-serif;font-size:0.85rem}
    table.chitable th{background:#f0f0f0}
    </style>
    """
    table_html = f"{style}<table class='chitable'><tr><th>df＼α</th>"
    for a in alphas:
        table_html += f"<th>{a}</th>"
    table_html += "</tr>\n"

    for d in df_list:
        table_html += "<tr>"
        table_html += f'<td id="chi_{d}_0">{d}</td>'
        for idx, a in enumerate(alphas, 1):
            table_html += f'<td id="chi_{d}_{idx}">{stats.chi2.ppf(1 - a, d):.3f}</td>'
        table_html += "</tr>\n"
    table_html += "</table>"

    col_idx = alphas.index(alpha) + 1
    if cur_step >= 0:
        for i in range(len(alphas) + 1):
            table_html = highlight_html_cell(table_html, f"chi_{df_val}_{i}", "red", 2)
    if cur_step >= 1:
        for d in df_list:
            table_html = highlight_html_cell(table_html, f"chi_{d}_{col_idx}", "red", 2)
    if cur_step >= 2:
        table_html = highlight_html_cell(table_html, f"chi_{df_val}_{col_idx}", "blue", 3)

    show_html_table(table_html, 450)
    steps = ["1) Highlight df row", "2) Highlight α column", "3) Intersection → χ²₍crit₎"]
    if cur_step > 2:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps[cur_step]}")
    next_step_button(step_key)

###############################################################################
#                            MANN–WHITNEY U
###############################################################################

def mann_whitney_crit(n1, n2, alpha, tail):
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if tail == "two‑tailed":
        z = stats.norm.ppf(alpha / 2)
    else:
        z = stats.norm.ppf(alpha)
    return max(0, int(np.floor(mu + z * sigma)))

def show_mann_whitney_tab():
    st.subheader("Tab 5: Mann–Whitney U")
    col1, col2 = st.columns(2)
    with col1:
        u_val = st.number_input("U statistic", value=23, key="tab5_u")
        n1_val = st.number_input("n₁", min_value=2, value=10, key="tab5_n1")
    with col2:
        n2_val = st.number_input("n₂", min_value=2, value=12, key="tab5_n2")
        alpha_val = st.number_input("α", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, key="tab5_alpha")
        tail_s = st.radio("Tail Type", ["one‑tailed", "two‑tailed"], key="tab5_tail")

    if st.button("Update Plot", key="tab5_update"):
        st.pyplot(plot_mann_whitney_distribution(u_val, n1_val, n2_val, alpha_val, tail_s))

    with st.expander("Show Mann–Whitney Table Lookup (±5 n₁ × ±5 n₂)"):
        show_mann_whitney_lookup(n1_val, n2_val, alpha_val, tail_s, "tab5")

def plot_mann_whitney_distribution(u_val, n1, n2, alpha, tail):
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    y = stats.norm.pdf(x, mu, sigma)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, color="lightgrey", alpha=0.2, label="Fail to Reject H₀")

    if tail == "one‑tailed":
        u_crit = mann_whitney_crit(n1, n2, alpha, tail)
        ax.fill_between(x[x <= u_crit], y[x <= u_crit], color="red", alpha=0.3, label="Reject H₀")
        ax.axvline(u_crit, color="green", linestyle="--")
        place_label(ax, [], u_crit, stats.norm.pdf(u_crit, mu, sigma) + 0.02, f"U₍crit₎ = {u_crit}", "green")
        sig = u_val <= u_crit
    else:
        u_crit = mann_whitney_crit(n1, n2, alpha, tail)
        u_crit_high = n1 * n2 - u_crit
        ax.fill_between(x[x <= u_crit], y[x <= u_crit], color="red", alpha=0.3)
        ax.fill_between(x[x >= u_crit_high], y[x >= u_crit_high], color="red", alpha=0.3, label="Reject H₀")
        ax.axvline(u_crit, color="green", linestyle="--")
        ax.axvline(u_crit_high, color="green", linestyle="--")
        place_label(ax, [], u_crit, stats.norm.pdf(u_crit, mu, sigma) + 0.02, f"U₍crit₎ = {u_crit}", "green")
        sig = u_val <= u_crit or u_val >= u_crit_high

    ax.axvline(u_val, color="blue", linestyle="--")
    place_label(ax, [], u_val, stats.norm.pdf(u_val, mu, sigma) + 0.02, f"U₍calc₎ = {u_val}", "blue")
    verdict = "Reject H₀" if sig else "Fail to Reject H₀"
    ax.set_title(f"Mann–Whitney U Distribution – {verdict}")
    ax.legend()
    fig.tight_layout()
    return fig

def show_mann_whitney_lookup(n1_val, n2_val, alpha, tail, key_prefix):
    st.write("### Step‑by‑Step Mann–Whitney Table Lookup")
    step_key = key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    cur_step = st.session_state[step_key]

    n1_list = list(range(max(2, n1_val - 5), n1_val + 6))
    n2_list = list(range(max(2, n2_val - 5), n2_val + 6))

    style = """
    <style>
    table.utable{border-collapse:collapse;margin-top:10px}
    table.utable td,table.utable th{border:1px solid #000;width:90px;height:30px;text-align:center;
    font-family:sans-serif;font-size:0.8rem}
    table.utable th{background:#f0f0f0}
    </style>
    """
    table_html = f"{style}<table class='utable'><tr><th>n₁＼n₂</th>"
    for n2 in n2_list:
        table_html += f"<th>{n2}</th>"
    table_html += "</tr>\n"

    for n1 in n1_list:
        table_html += "<tr>"
        table_html += f'<td id="u_{n1}_0">{n1}</td>'
        for idx, n2 in enumerate(n2_list, 1):
            table_html += f'<td id="u_{n1}_{idx}">{mann_whitney_crit(n1, n2, alpha, tail):d}</td>'
        table_html += "</tr>\n"
    table_html += "</table>"

    col_idx = n2_list.index(n2_val) + 1
    if cur_step >= 0:
        for i in range(len(n2_list) + 1):
            table_html = highlight_html_cell(table_html, f"u_{n1_val}_{i}", "red", 2)
    if cur_step >= 1:
        for n1 in n1_list:
            table_html = highlight_html_cell(table_html, f"u_{n1}_{col_idx}", "red", 2)
    if cur_step >= 2:
        table_html = highlight_html_cell(table_html, f"u_{n1_val}_{col_idx}", "blue", 3)

    show_html_table(table_html, 450)
    steps = ["1) Highlight n₁ row", "2) Highlight n₂ column", "3) Intersection → U₍crit₎"]
    if cur_step > 2:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps[cur_step]}")
    next_step_button(step_key)

###############################################################################
#                        WILCOXON SIGNED‑RANK T
###############################################################################

def wilcoxon_crit(n, alpha, tail):
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = stats.norm.ppf(alpha / 2 if tail == "two‑tailed" else alpha)
    return max(0, int(np.floor(mu + z * sigma)))

def show_wilcoxon_tab():
    st.subheader("Tab 6: Wilcoxon Signed‑Rank T")
    col1, col2 = st.columns(2)
    with col1:
        t_val = st.number_input("T statistic", value=15, key="tab6_t")
        n_val = st.number_input("N (paired differences)", min_value=5, value=12, key="tab6_n")
    with col2:
        alpha_val = st.number_input("α", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, key="tab6_alpha")
        tail_s = st.radio("Tail Type", ["one‑tailed", "two‑tailed"], key="tab6_tail")

    if st.button("Update Plot", key="tab6_update"):
        st.pyplot(plot_wilcoxon_distribution(t_val, n_val, alpha_val, tail_s))

    with st.expander("Show Wilcoxon Table Lookup (±5 N)"):
        show_wilcoxon_lookup(n_val, alpha_val, tail_s, "tab6")

def plot_wilcoxon_distribution(t_val, n, alpha, tail):
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    y = stats.norm.pdf(x, mu, sigma)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, color="lightgrey", alpha=0.2, label="Fail to Reject H₀")

    if tail == "one‑tailed":
        t_crit = wilcoxon_crit(n, alpha, tail)
        ax.fill_between(x[x <= t_crit], y[x <= t_crit], color="red", alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit, color="green", linestyle="--")
        place_label(ax, [], t_crit, stats.norm.pdf(t_crit, mu, sigma) + 0.02, f"T₍crit₎ = {t_crit}", "green")
        sig = t_val <= t_crit
    else:
        t_crit = wilcoxon_crit(n, alpha, tail)
        t_high = n * (n + 1) / 2 - t_crit
        ax.fill_between(x[x <= t_crit], y[x <= t_crit], color="red", alpha=0.3)
        ax.fill_between(x[x >= t_high], y[x >= t_high], color="red", alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit, color="green", linestyle="--")
        ax.axvline(t_high, color="green", linestyle="--")
        place_label(ax, [], t_crit, stats.norm.pdf(t_crit, mu, sigma) + 0.02, f"T₍crit₎ = {t_crit}", "green")
        sig = t_val <= t_crit or t_val >= t_high

    ax.axvline(t_val, color="blue", linestyle="--")
    place_label(ax, [], t_val, stats.norm.pdf(t_val, mu, sigma) + 0.02, f"T₍calc₎ = {t_val}", "blue")
    verdict = "Reject H₀" if sig else "Fail to Reject H₀"
    ax.set_title(f"Wilcoxon Signed‑Rank T Distribution – {verdict}")
    ax.legend()
    fig.tight_layout()
    return fig

def show_wilcoxon_lookup(n_val, alpha, tail, key_prefix):
    st.write("### Step‑by‑Step Wilcoxon Table Lookup")
    step_key = key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    cur_step = st.session_state[step_key]

    n_list = list(range(max(5, n_val - 5), n_val + 6))
    alphas = [0.10, 0.05, 0.01, 0.001]

    style = """
    <style>
    table.wtable{border-collapse:collapse;margin-top:10px}
    table.wtable td,table.wtable th{border:1px solid #000;width:80px;height:30px;text-align:center;
    font-family:sans-serif;font-size:0.8rem}
    table.wtable th{background:#f0f0f0}
    </style>
    """
    table_html = f"{style}<table class='wtable'><tr><th>N＼α</th>"
    for a in alphas:
        table_html += f"<th>{a}</th>"
    table_html += "</tr>\n"

    for n in n_list:
        table_html += "<tr>"
        table_html += f'<td id="w_{n}_0">{n}</td>'
        for idx, a in enumerate(alphas, 1):
            table_html += f'<td id="w_{n}_{idx}">{wilcoxon_crit(n, a, tail):d}</td>'
        table_html += "</tr>\n"
    table_html += "</table>"

    col_idx = alphas.index(alpha) + 1
    if cur_step >= 0:
        for i in range(len(alphas) + 1):
            table_html = highlight_html_cell(table_html, f"w_{n_val}_{i}", "red", 2)
    if cur_step >= 1:
        for n in n_list:
            table_html = highlight_html_cell(table_html, f"w_{n}_{col_idx}", "red", 2)
    if cur_step >= 2:
        table_html = highlight_html_cell(table_html, f"w_{n_val}_{col_idx}", "blue", 3)

    show_html_table(table_html, 450)
    steps = ["1) Highlight N row", "2) Highlight α column", "3) Intersection → T₍crit₎"]
    if cur_step > 2:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps[cur_step]}")
    next_step_button(step_key)

###############################################################################
#                                 BINOMIAL
###############################################################################

def show_binomial_tab():
    st.subheader("Tab 7: Binomial")
    col1, col2 = st.columns(2)
    with col1:
        n_val = st.number_input("n (trials)", min_value=1, value=20, key="tab7_n")
        p_val = st.number_input("p (null π)", min_value=0.01, max_value=0.99, value=0.50, step=0.01, key="tab7_p")
    with col2:
        k_val = st.number_input("k (successes)", min_value=0, value=12, key="tab7_k")
        alpha_val = st.number_input("α (two‑tail)", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, key="tab7_alpha")

    if st.button("Update Plot", key="tab7_update"):
        st.pyplot(plot_binomial_distribution(k_val, n_val, p_val, alpha_val))

    with st.expander("Show Binomial Table Lookup (±5 k)"):
        show_binomial_lookup(k_val, n_val, p_val, "tab7")

def plot_binomial_distribution(k, n, p, alpha):
    x = np.arange(0, n + 1)
    y = stats.binom.pmf(x, n, p)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.bar(x, y, color="lightgrey")
    ax.bar(k, stats.binom.pmf(k, n, p), color="blue")
    ax.set_title(f"Binomial Distribution (n = {n}, p = {p}) – k = {k}")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X = k)")
    fig.tight_layout()
    return fig

def show_binomial_lookup(k_val, n_val, p_val, key_prefix):
    st.write("### Step‑by‑Step Binomial Table Lookup")
    step_key = key_prefix + "_table_step"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    cur_step = st.session_state[step_key]

    k_list = [k for k in range(max(0, k_val - 5), min(n_val, k_val + 5) + 1)]
    cols = ["P(X = k)", "P(X ≤ k)", "P(X ≥ k)"]

    style = """
    <style>
    table.bintable{border-collapse:collapse;margin-top:10px}
    table.bintable td,table.bintable th{border:1px solid #000;width:120px;height:30px;text-align:center;
    font-family:sans-serif;font-size:0.8rem}
    table.bintable th{background:#f0f0f0}
    </style>
    """
    table_html = f"{style}<table class='bintable'><tr><th>k</th>"
    for c in cols:
        table_html += f"<th>{c}</th>"
    table_html += "</tr>\n"

    for k in k_list:
        table_html += "<tr>"
        table_html += f'<td id="bin_{k}_0">{k}</td>'
        pmf = stats.binom.pmf(k, n_val, p_val)
        cdf = stats.binom.cdf(k, n_val, p_val)
        sf = 1 - stats.binom.cdf(k - 1, n_val, p_val)
        table_html += f'<td id="bin_{k}_1">{pmf:.4f}</td>'
        table_html += f'<td id="bin_{k}_2">{cdf:.4f}</td>'
        table_html += f'<td id="bin_{k}_3">{sf:.4f}</td></tr>\n'
    table_html += "</table>"

    if cur_step >= 0:
        for i in range(4):
            table_html = highlight_html_cell(table_html, f"bin_{k_val}_{i}", "red", 2)
    if cur_step >= 1:
        table_html = highlight_html_cell(table_html, f"bin_{k_val}_1", "blue", 3)

    show_html_table(table_html, 450)
    steps = ["1) Highlight k row", "2) Highlight P(X = k)"]
    if cur_step > 1:
        st.write("All steps complete!")
    else:
        st.write(f"**Step {cur_step+1}**: {steps[cur_step]}")
    next_step_button(step_key)

###############################################################################
#                                   MAIN
###############################################################################

def main():
    st.set_page_config(page_title="PSYC250 – Statistical Tables Explorer", layout="wide")
    st.title("PSYC250 – Statistical Tables Explorer (12 × 4 Figures)")

    tabs = st.tabs([
        "t‑Distribution", "z‑Distribution", "F‑Distribution", "Chi‑Square",
        "Mann–Whitney U", "Wilcoxon Signed‑Rank", "Binomial"
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

if __name__ == "__main__":
    main()
