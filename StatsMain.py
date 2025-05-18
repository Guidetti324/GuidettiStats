###############################################################################
#  PSYC-250 – Statistical Tables Explorer
#  ---------------------------------------------------------------------------
#  Seven complete tabs:
#       1) t-Distribution        4) Chi-Square
#       2) z-Distribution        5) Mann-Whitney U
#       3) F-Distribution        6) Wilcoxon Signed-Rank T
#       7) Binomial
#
#  NEW FEATURES ADDED FOR (t, z, Mann-Whitney U, Wilcoxon T, Binomial):
#   1) A "Cumulative Table Note" explaining how to interpret the table for
#      one- vs. two-tailed tests.
#   2) A "P-Value Calculation Explanation" section next to the table,
#      showing how the table lookup leads to p, depending on one- vs. two-tailed.
#   3) Automatic plot shading based on sign of the test statistic for one-tailed
#      (negative → left tail, positive → right tail). For two-tailed, shade both tails.
#
#  F-Distribution and Chi-Square remain as before, since they are always one-tailed.
#
#  To run:   streamlit run app.py
###############################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")

###############################################################################
#                               COMMON SETUP
###############################################################################

def show_cumulative_note():
    st.info(
        "Note: The values in this table represent cumulative probabilities "
        "(i.e., the area under the curve to the left of a given value). For "
        "one-tailed tests, use the area directly. For two-tailed tests, you must "
        "double the area in the tail beyond your observed value (i.e., "
        "$p=2 \\times (1−P(Z \\le |z|))$). The same logic applies for t-distributions. The table "
        "itself does not change—only how you interpret it does."
    )

def place_label(ax, placed_list, x, y, txt, *, color="blue"):
    dx = dy = 0.0
    for (xx, yy) in placed_list:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05: # Simple collision avoidance
            dx += 0.06
            dy += 0.04
    ax.text(x + dx, y + dy, txt, color=color,
            ha="left", va="bottom", fontsize=8, clip_on=True)
    placed_list.append((x + dx, y + dy))

def style_cell(html: str, cid: str, color: str = "red", px: int = 2) -> str:
    return html.replace(
        f'id="{cid}"',
        f'id="{cid}" style="border:{px}px solid {color};"',
        1
    )

def wrap_table(css: str, table_html: str) -> str:
    return f"<style>{css}</style><table>{table_html}</table>"

def container(html_content: str, *, height: int = 460) -> str:
    return f'<div style="overflow:auto; max-height:{height}px;">{html_content}</div>'

CSS_BASE = (
    "table{border-collapse:collapse}"
    "th,td{border:1px solid #000;height:30px;text-align:center;"
    "font-family:sans-serif;font-size:0.9rem}"
    "th{background:#fafafa}"
)

###############################################################################
#                             TAB 1: t-Distribution
###############################################################################

def plot_t(t_calc, df, input_alpha, tail): # Use input_alpha for plot's critical value
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    if df <= 0:
        ax.text(0.5, 0.5, "Degrees of freedom (df) must be positive.", ha='center', va='center')
        return fig
        
    xs = np.linspace(-4.5, 4.5, 400)
    try:
        ys = stats.t.pdf(xs, df)
    except Exception:
        ax.text(0.5, 0.5, f"Invalid df ({df}) for t-distribution PDF.", ha='center', va='center')
        return fig

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")
    placed_labels = []

    # Critical value for the plot is based on the USER'S INPUT ALPHA
    plot_crit_val = np.nan
    try:
        if tail.startswith("one"):
            plot_crit_val_one_sided = stats.t.ppf(1 - input_alpha, df) # Positive critical value
            if t_calc >= 0: # Right tail test assumption for positive t_calc
                actual_plot_crit = plot_crit_val_one_sided
                ax.fill_between(xs[xs >= actual_plot_crit], ys[xs >= actual_plot_crit], color="red", alpha=0.3, label="Reject H0")
                ax.axvline(actual_plot_crit, color="green", ls="--")
                pdf_val = stats.t.pdf(actual_plot_crit, df)
                if not np.isnan(pdf_val): place_label(ax, placed_labels, actual_plot_crit, pdf_val + 0.02, f"tcrit={actual_plot_crit:.2f}", color="green")
            else: # Left tail test assumption for negative t_calc
                actual_plot_crit = -plot_crit_val_one_sided # Negative critical value
                ax.fill_between(xs[xs <= actual_plot_crit], ys[xs <= actual_plot_crit], color="red", alpha=0.3, label="Reject H0")
                ax.axvline(actual_plot_crit, color="green", ls="--")
                pdf_val = stats.t.pdf(actual_plot_crit, df)
                if not np.isnan(pdf_val): place_label(ax, placed_labels, actual_plot_crit, pdf_val + 0.02, f"tcrit={actual_plot_crit:.2f}", color="green")
        else: # two-tailed
            plot_crit_val_two_sided = stats.t.ppf(1 - input_alpha / 2, df)
            ax.fill_between(xs[xs >= plot_crit_val_two_sided], ys[xs >= plot_crit_val_two_sided], color="red", alpha=0.3)
            ax.fill_between(xs[xs <= -plot_crit_val_two_sided], ys[xs <= -plot_crit_val_two_sided], color="red", alpha=0.3, label="Reject H0")
            ax.axvline(plot_crit_val_two_sided, color="green", ls="--")
            ax.axvline(-plot_crit_val_two_sided, color="green", ls="--")
            pdf_val = stats.t.pdf(plot_crit_val_two_sided, df)
            if not np.isnan(pdf_val):
                place_label(ax, placed_labels, plot_crit_val_two_sided, pdf_val + 0.02, f"+tcrit={plot_crit_val_two_sided:.2f}", color="green")
                place_label(ax, placed_labels, -plot_crit_val_two_sided, pdf_val + 0.02, f"–tcrit={plot_crit_val_two_sided:.2f}", color="green")
        
        pdf_at_t_calc = stats.t.pdf(t_calc, df)
        if not np.isnan(pdf_at_t_calc): place_label(ax, placed_labels, t_calc, pdf_at_t_calc + 0.02, f"tcalc={t_calc:.2f}", color="blue")
        ax.axvline(t_calc, color="blue", ls="--")
    except Exception as e:
        st.warning(f"Could not draw critical regions/labels on plot: {e}")

    ax.set_xlabel("t")
    ax.set_ylabel("Density")
    ax.legend(loc='upper right')
    ax.set_title(f"t-Distribution (df={df:.0f}, input $\\alpha$={input_alpha:.4f})") # Show input alpha
    fig.tight_layout()
    return fig

def build_t_html(df: int, user_input_alpha: float, tail: str) -> str: # user_input_alpha is for table highlighting logic
    rows = list(range(max(1, df - 5), df + 6))
    heads = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001)
    ]
    mode = "one" if tail.startswith("one") else "two"

    col_idx_to_highlight = -1
    best_match_info_for_table = None 
    min_diff_for_table = float('inf')

    for i, (h_mode, h_alpha_val) in enumerate(heads, start=1):
        if h_mode == mode:
            if np.isclose(h_alpha_val, user_input_alpha):
                col_idx_to_highlight = i
                best_match_info_for_table = {'alpha_val': h_alpha_val, 'is_exact': True, 'index': i}
                break 
            current_diff = abs(h_alpha_val - user_input_alpha)
            if current_diff < min_diff_for_table:
                min_diff_for_table = current_diff
                best_match_info_for_table = {'index': i, 'alpha_val': h_alpha_val, 'is_exact': False}
            elif current_diff == min_diff_for_table:
                if best_match_info_for_table and h_alpha_val < best_match_info_for_table['alpha_val']:
                     best_match_info_for_table = {'index': i, 'alpha_val': h_alpha_val, 'is_exact': False}

    if col_idx_to_highlight == -1: 
        if best_match_info_for_table:
            col_idx_to_highlight = best_match_info_for_table['index']
            selected_alpha_for_table_highlight = best_match_info_for_table['alpha_val']
            if not best_match_info_for_table.get('is_exact', False): 
                 st.warning(
                    f"The entered alpha ({user_input_alpha:.4f}) is not a standard value in this t-table for a {mode}-tailed test. "
                    f"Highlighting the column for the closest standard alpha: {selected_alpha_for_table_highlight:.4f}."
                )
        else:
            st.error(f"Internal error: No columns found for mode '{mode}'. Defaulting to column 1 for highlighting.")
            col_idx_to_highlight = 1 
            if heads: st.info(f"Table highlight defaulted to col 1 (alpha={heads[0][1]:.3f} for mode='{heads[0][0]}').")

    if not (1 <= col_idx_to_highlight <= len(heads)):
        st.warning(f"Invalid column index {col_idx_to_highlight} for table highlight. Resetting to 1.")
        col_idx_to_highlight = 1
    
    head_html_parts = []
    for m_h, a_h in heads:
        head_html_parts.append(f"<th>{m_h}<br>$\\alpha$={a_h:.3f}</th>")
    head_html = "".join(head_html_parts)
    
    body_html = ""
    for r_val in rows:
        row_cells = f'<td id="t_{r_val}_0">{r_val}</td>'
        for i_cell, (m_cell, a_cell) in enumerate(heads, start=1):
            try:
                crit_val_cell = stats.t.ppf(1 - a_cell if m_cell == "one" else 1 - a_cell / 2, r_val)
                cell_text = f"{crit_val_cell:.3f}" # Using .3f for table values
            except Exception: cell_text = "N/A"
            row_cells += f'<td id="t_{r_val}_{i_cell}">{cell_text}</td>'
        body_html += f"<tr>{row_cells}</tr>"

    table_code = f"<tr><th>df</th>{head_html}</tr>{body_html}"
    html_output = wrap_table(CSS_BASE, table_code)

    if df in rows:
        for i_highlight in range(len(heads) + 1):
            html_output = style_cell(html_output, f"t_{df}_{i_highlight}")
    for rr_val in rows: # Highlight column based on col_idx_to_highlight
        html_output = style_cell(html_output, f"t_{rr_val}_{col_idx_to_highlight}")
    if df in rows:
        html_output = style_cell(html_output, f"t_{df}_{col_idx_to_highlight}", color="blue", px=3)
    
    return html_output

def t_table(df: int, user_input_alpha: float, tail: str):
    code = build_t_html(df, user_input_alpha, tail)
    st.markdown(container(code), unsafe_allow_html=True)

def t_apa(t_val: float, df: int, input_alpha: float, tail: str): # Uses input_alpha for calculations
    p_calc_val = np.nan
    crit_val_from_input_alpha = np.nan # Critical value based on user's alpha
    reject = False

    try:
        if tail.startswith("one"):
            p_calc_val = stats.t.sf(abs(t_val), df) 
            crit_val_from_input_alpha = stats.t.ppf(1 - input_alpha, df) 
            if t_val >= 0: 
                reject = t_val > crit_val_from_input_alpha
            else: 
                reject = t_val < -crit_val_from_input_alpha 
        else: # two-tailed
            p_calc_val = stats.t.sf(abs(t_val), df) * 2
            crit_val_from_input_alpha = stats.t.ppf(1 - input_alpha / 2, df)
            reject = abs(t_val) > crit_val_from_input_alpha
    except Exception as e:
        st.warning(f"Could not calculate p-value or critical value for APA: {e}")

    decision = "rejected" if reject else "failed to reject"
    reason_stats = "because $t_{calc}$ was in the rejection region" if reject else "because $t_{calc}$ was not in the rejection region"
    reason_p = f"because $p \\approx {p_calc_val:.3f} < \\alpha$" if reject else f"because $p \\approx {p_calc_val:.3f} \\ge \\alpha$"
    
    cdf_val = np.nan
    try: cdf_val = stats.t.cdf(t_val, df)
    except Exception: pass

    expl_parts = []
    expl_parts.append(f"For $t_{{calc}} = {t_val:.2f}$ with $df = {df}$:")
    expl_parts.append(f"The cumulative probability $P(T \\le {t_val:.2f}) \\approx {cdf_val:.4f}$ (from t-distribution CDF).")
    if tail.startswith("one"):
        if t_val >= 0: # Right tail
            expl_parts.append(f"For a **one-tailed** (right tail) test, the p-value is $1 - P(T \\le {t_val:.2f}) \\approx {1-cdf_val:.4f}$.")
        else: # Left tail
            expl_parts.append(f"For a **one-tailed** (left tail) test, the p-value is $P(T \\le {t_val:.2f}) \\approx {cdf_val:.4f}$.")
    else: # Two-tail
        expl_parts.append(f"For a **two-tailed** test, the p-value is $2 \\times P(T \\ge |{t_val:.2f}|)$. Using the CDF, this is $2 \\times \\text{{min}}(P(T \\le {t_val:.2f}), 1 - P(T \\le {t_val:.2f})) \\approx {2*min(cdf_val, 1-cdf_val):.4f}$.")
    expl_parts.append(f"The calculated p-value is $p \\approx {p_calc_val:.4f}$.")
    
    st.write("\n\n".join(expl_parts))

    st.markdown( # Using original APA structure but with crit_val_from_input_alpha
        "**APA interpretation**\n"
        f"Calculated statistic: *$t$({df}) = {t_val:.2f}, *$p$ $\\approx$ {p_calc_val:.3f}.\n"
        f"Critical statistic for user's $\\alpha={input_alpha:.4f}$ ({tail}): $t_{{crit}} \\approx {crit_val_from_input_alpha:.2f}$.\n" # Using user's alpha for crit
        f"Comparison of statistics ($t_{{calc}}$ vs $t_{{crit}}$ for $\\alpha={input_alpha:.4f}$) $\\rightarrow$ H₀ **{decision}** ({reason_stats}).\n"
        f"Comparison of *$p$*-values ($p$ vs $\\alpha={input_alpha:.4f}$) $\\rightarrow$ H₀ **{decision}** ({reason_p}).\n"
        f"**APA 7 report:** *$t$({df}) = {t_val:.2f}, *$p$ $\\approx$ {p_calc_val:.3f} ({tail}). The null hypothesis was **{decision}** at $\\alpha$={input_alpha:.2f}."
    )

def tab_t():
    st.subheader("Tab 1 • t-Distribution")
    c1, c2 = st.columns(2)
    with c1:
        t_val_in = st.number_input("t statistic", value=2.10, step=0.01, key="t_val_widget")
        df_in = st.number_input("df", min_value=1, value=10, step=1, key="t_df_widget")
    with c2:
        alpha_in = st.number_input("α (alpha level)", value=0.05, step=0.001, min_value=0.0001, max_value=0.99, format="%.4f", key="t_alpha_widget")
        tail_in = st.radio("Tail", ["one-tailed", "two-tailed"], key="t_tail_widget", horizontal=True)

    # Calculations are performed when the button is clicked or if inputs change and it's the first run
    # For simplicity, let's use a button as in your original implied structure for some tabs
    # Or, remove the button if you want it to update live (can be slow for complex plots/tables)

    # To make it update without a button, call these directly:
    # However, using a button can prevent excessive re-renders during typing.
    # For now, retaining a similar flow to original screenshots which had an "Update Plot" button
    
    if 't_button_clicked' not in st.session_state: # Initialize if not exists
        st.session_state.t_button_clicked = False

    if st.button("Generate Plot and Table for t-Distribution", key="t_generate_button_main"):
        st.session_state.t_button_clicked = True

    if st.session_state.t_button_clicked : # Only draw if button was clicked
        try:
            # Pass user's alpha_in to plot_t and t_apa for their critical value calculations
            fig_t_dist = plot_t(float(t_val_in), int(df_in), float(alpha_in), tail_in)
            if fig_t_dist: st.pyplot(fig_t_dist)
        except Exception as e:
            st.error(f"Error generating t-plot: {e}")

        st.write("**t-table** (highlighted based on input α, showing standard α columns)")
        ctable_t, cexp_t = st.columns([3, 2]) 
        with ctable_t:
            try:
                # Pass user's alpha_in to t_table for highlighting logic
                t_table(int(df_in), float(alpha_in), tail_in)
                show_cumulative_note()
            except Exception as e:
                st.error(f"Error generating t-table: {e}")
                st.exception(e) 
        with cexp_t:
            try:
                st.subheader("P-value Calculation Explanation")
                # Pass user's alpha_in to t_apa for its critical value calculations
                t_apa(float(t_val_in), int(df_in), float(alpha_in), tail_in)
            except Exception as e:
                st.error(f"Error in t-APA explanation: {e}")

# ----- Z-Distribution (Restoring original structure, ensuring consistency) -----
def plot_z(z_calc, input_alpha, tail):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    xs = np.linspace(-4,4,400)
    ys = stats.norm.pdf(xs)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")
    placed_z = []
    try:
        if tail.startswith("one"):
            plot_crit_z_one = stats.norm.ppf(1 - input_alpha)
            if z_calc >= 0:
                ax.fill_between(xs[xs>=plot_crit_z_one], ys[xs>=plot_crit_z_one], color="red", alpha=0.3, label="Reject H0")
                ax.axvline(plot_crit_z_one, color="green", ls="--")
                place_label(ax, placed_z, plot_crit_z_one, stats.norm.pdf(plot_crit_z_one)+0.02, f"z₍crit₎={plot_crit_z_one:.2f}", color="green")
            else:
                plot_crit_z_one_neg = -plot_crit_z_one
                ax.fill_between(xs[xs<=plot_crit_z_one_neg], ys[xs<=plot_crit_z_one_neg], color="red", alpha=0.3, label="Reject H0")
                ax.axvline(plot_crit_z_one_neg, color="green", ls="--")
                place_label(ax, placed_z, plot_crit_z_one_neg, stats.norm.pdf(plot_crit_z_one_neg)+0.02, f"z₍crit₎={plot_crit_z_one_neg:.2f}", color="green")
        else: # two-tailed
            plot_crit_z_two = stats.norm.ppf(1 - input_alpha/2)
            ax.fill_between(xs[xs>=plot_crit_z_two], ys[xs>=plot_crit_z_two], color="red", alpha=0.3)
            ax.fill_between(xs[xs<=-plot_crit_z_two], ys[xs<=-plot_crit_z_two], color="red", alpha=0.3, label="Reject H0")
            ax.axvline(plot_crit_z_two, color="green", ls="--")
            ax.axvline(-plot_crit_z_two, color="green", ls="--")
            place_label(ax, placed_z, plot_crit_z_two, stats.norm.pdf(plot_crit_z_two)+0.02, f"+z₍crit₎={plot_crit_z_two:.2f}", color="green")
            place_label(ax, placed_z, -plot_crit_z_two, stats.norm.pdf(-plot_crit_z_two)+0.02, f"–z₍crit₎={plot_crit_z_two:.2f}", color="green")
        ax.axvline(z_calc, color="blue", ls="--")
        place_label(ax, placed_z, z_calc, stats.norm.pdf(z_calc)+0.02, f"z₍calc₎={z_calc:.2f}", color="blue")
    except Exception as e: st.warning(f"Could not draw z-plot elements: {e}")
    ax.set_xlabel("z")
    ax.set_ylabel("Density")
    ax.legend(loc='upper right')
    ax.set_title(f"z-Distribution (input $\\alpha$={input_alpha:.4f})")
    fig.tight_layout()
    return fig

def build_z_html(z_val: float) -> str: # z-table structure is independent of alpha
    z_val_clipped = np.clip(z_val, -3.499, 3.499)
    row_label = np.sign(z_val_clipped) * np.floor(abs(z_val_clipped) * 10) / 10
    col_label = round(abs(z_val_clipped) - abs(row_label), 2)
    TableRows = np.round(np.arange(-3.4, 3.5, 0.1), 1)
    TableCols = np.round(np.arange(0.00, 0.10, 0.01), 2)
    row_label = min(TableRows, key=lambda r_disp: abs(r_disp - row_label))
    col_label = min(TableCols, key=lambda c_disp: abs(c_disp - col_label))
    try: center_row_idx_in_table = list(TableRows).index(row_label)
    except ValueError: center_row_idx_in_table = len(TableRows) // 2
    display_row_start_idx = max(0, center_row_idx_in_table - 10)
    display_row_end_idx = min(len(TableRows), center_row_idx_in_table + 11)
    rows_to_show_in_html = TableRows[display_row_start_idx:display_row_end_idx]
    head_html = "".join(f"<th>{c_head:.2f}</th>" for c_head in TableCols)
    body_html = ""
    for rr_html in rows_to_show_in_html:
        row_html_cells = f'<td id="z_{rr_html:.1f}_0">{rr_html:.1f}</td>'
        for cc_html in TableCols:
            current_z_for_cell = rr_html + cc_html
            cdf_cell_val = stats.norm.cdf(current_z_for_cell)
            row_html_cells += f'<td id="z_{rr_html:.1f}_{cc_html:.2f}">{cdf_cell_val:.4f}</td>'
        body_html += f"<tr>{row_html_cells}</tr>"
    table_code = f"<tr><th>z</th>{head_html}</tr>{body_html}"
    html_output = wrap_table(CSS_BASE, table_code)
    if row_label in rows_to_show_in_html:
        for cc_highlight in TableCols: html_output = style_cell(html_output, f"z_{row_label:.1f}_{cc_highlight:.2f}")
        html_output = style_cell(html_output, f"z_{row_label:.1f}_0")
        for rr_highlight in rows_to_show_in_html: html_output = style_cell(html_output, f"z_{rr_highlight:.1f}_{col_label:.2f}")
        html_output = style_cell(html_output, f"z_{row_label:.1f}_{col_label:.2f}", color="blue", px=3)
    return html_output

def z_table(z_val: float): # Alpha and tail are not needed for z-table display
    code = build_z_html(z_val)
    st.markdown(container(code), unsafe_allow_html=True)

def z_apa(z_val: float, input_alpha: float, tail: str): # Using input_alpha
    p_calc_val = np.nan; crit_val_z = np.nan; reject = False
    try:
        if tail.startswith("one"):
            p_calc_val = stats.norm.sf(abs(z_val))
            if z_val >= 0: crit_val_z = stats.norm.ppf(1 - input_alpha); reject = z_val > crit_val_z
            else: crit_val_z = stats.norm.ppf(input_alpha); reject = z_val < crit_val_z
        else:
            p_calc_val = stats.norm.sf(abs(z_val)) * 2
            crit_val_z = stats.norm.ppf(1 - input_alpha / 2)
            reject = abs(z_val) > crit_val_z
    except Exception as e: st.warning(f"Could not calc Z APA details: {e}")
    decision = "rejected" if reject else "failed to reject"
    reason_stats = f"because $|z_{{calc}}|$ ({abs(z_val):.2f}) {'exceeded' if reject else 'did not exceed'} $|z_{{crit}}|$ ({abs(crit_val_z):.2f})"
    reason_p = f"because $p \\approx {p_calc_val:.3f} {'<' if reject else '≥'} \\alpha$"
    table_val_cdf = np.nan
    try: table_val_cdf = stats.norm.cdf(z_val)
    except: pass
    expl_parts_z = [f"Lookup $P(Z \\le {z_val:.2f}) \\approx {table_val_cdf:.4f}$ (from Z-table/CDF)."]
    if tail.startswith("one"):
        if z_val >= 0: expl_parts_z.append(f"For **one-tailed** (right tail), $p = 1 - P(Z \\le {z_val:.2f}) \\approx {1-table_val_cdf:.4f}$.")
        else: expl_parts_z.append(f"For **one-tailed** (left tail), $p = P(Z \\le {z_val:.2f}) \\approx {table_val_cdf:.4f}$.")
    else: expl_parts_z.append(f"For **two-tailed**, $p = 2 \\times P(Z \\ge |{z_val:.2f}|) \\approx {2*min(table_val_cdf, 1-table_val_cdf):.4f}$.")
    expl_parts_z.append(f"Calculated $p \\approx {p_calc_val:.4f}$.")
    st.write("\n\n".join(expl_parts_z))
    st.markdown(
        "**APA interpretation**\n"
        f"Calculated statistic: *$z$* = {z_val:.2f}, *$p$ $\\approx$ {p_calc_val:.3f}.\n"
        f"Critical statistic for user's $\\alpha={input_alpha:.4f}$ ({tail}): $z_{{crit}} \\approx {crit_val_z:.2f}$.\n"
        f"Statistic comparison $\\rightarrow$ H₀ **{decision}** ({reason_stats}).\n"
        f"*$p$* comparison $\\rightarrow$ H₀ **{decision}** ({reason_p}).\n"
        f"**APA 7 report:** *$z$* = {z_val:.2f}, *$p$ $\\approx$ {p_calc_val:.3f} ({tail}). Null hypothesis was **{decision}** at $\\alpha$={input_alpha:.2f}."
    )

def tab_z():
    st.subheader("Tab 2 • z-Distribution")
    c1, c2 = st.columns(2)
    with c1: z_val_in = st.number_input("z statistic", value=1.64, step=0.01, key="z_val_widget")
    with c2:
        alpha_in = st.number_input("α (alpha level)", value=0.05, step=0.001, min_value=0.0001, max_value=0.99, format="%.4f", key="z_alpha_widget")
        tail_in = st.radio("Tail", ["one-tailed", "two-tailed"], key="z_tail_widget", horizontal=True)
    if 'z_button_clicked' not in st.session_state: st.session_state.z_button_clicked = False
    if st.button("Generate Plot and Table for z-Distribution", key="z_generate_button_main"):
        st.session_state.z_button_clicked = True
    if st.session_state.z_button_clicked:
        try:
            fig_z = plot_z(float(z_val_in), float(alpha_in), tail_in)
            if fig_z: st.pyplot(fig_z)
        except Exception as e: st.error(f"Error generating z-plot: {e}")
        st.write("**z-table** (highlighted based on z-statistic)")
        ct_z, ce_z = st.columns([3, 2])
        with ct_z:
            try: z_table(float(z_val_in)); show_cumulative_note()
            except Exception as e: st.error(f"Error for z-table: {e}"); st.exception(e)
        with ce_z:
            try: st.subheader("P-value Calculation Explanation"); z_apa(float(z_val_in), float(alpha_in), tail_in)
            except Exception as e: st.error(f"Error for z-APA: {e}")


# ----- F-Distribution (Restoring original structure) -----
def plot_f(f_calc, df1, df2, input_alpha): # Original plot_f, ensure input_alpha is used for crit
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    if df1 <= 0 or df2 <= 0: ax.text(0.5,0.5, "df1 and df2 must be positive.", ha='center', va='center'); return fig
    try:
        crit_val_f_plot = stats.f.ppf(1 - input_alpha, df1, df2)
        plot_upper_x = max(f_calc, crit_val_f_plot, 5) * 1.5 # Dynamic upper limit
        if stats.f.ppf(0.999, df1, df2) < plot_upper_x : plot_upper_x = stats.f.ppf(0.999, df1, df2) * 1.1

        xs = np.linspace(0, plot_upper_x, 400)
        ys = stats.f.pdf(xs, df1, df2)
        valid_ys = ~np.isnan(ys) & ~np.isinf(ys); xs, ys = xs[valid_ys], ys[valid_ys]
        if len(xs) < 2: ax.text(0.5,0.5, "Cannot generate F-PDF.", ha='center', va='center'); return fig
        ax.plot(xs, ys, "k"); ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")
        ax.fill_between(xs[xs>=crit_val_f_plot], ys[xs>=crit_val_f_plot], color="red", alpha=0.3, label="Reject H0")
        ax.axvline(crit_val_f_plot, color="green", ls="--"); ax.axvline(f_calc, color="blue", ls="--")
        placed_f = [] 
        pdf_crit = stats.f.pdf(crit_val_f_plot, df1, df2); pdf_calc = stats.f.pdf(f_calc, df1, df2)
        if not np.isnan(pdf_crit): place_label(ax, placed_f, crit_val_f_plot, pdf_crit + 0.02, f"F₍crit₎={crit_val_f_plot:.2f}", color="green")
        if not np.isnan(pdf_calc): place_label(ax, placed_f, f_calc, pdf_calc + 0.02, f"F₍calc₎={f_calc:.2f}", color="blue")
    except Exception as e: st.warning(f"Could not draw F-plot elements: {e}")
    ax.set_xlabel("F"); ax.set_ylabel("Density"); ax.legend(loc='upper right')
    ax.set_title(f"F-Distribution (df1={df1:.0f}, df2={df2:.0f}, input $\\alpha$={input_alpha:.4f})"); fig.tight_layout()
    return fig

def build_f_table(df1: int, df2: int, input_alpha: float) -> str: # F-table values depend on input_alpha
    rows = list(range(max(1,df1-2), df1+3+1)) # Focused range
    cols = list(range(max(1,df2-2), df2+3+1))
    col_idx_hl = -1
    if df2 in cols: col_idx_hl = cols.index(df2)+1
    head_html = "".join(f"<th>{c}</th>" for c in cols)
    body_html = ""
    for r_f in rows:
        row_cells_f = f'<td id="f_{r_f}_0">{r_f}</td>'
        for i_f,c_f in enumerate(cols, start=1):
            val_f = np.nan
            try: val_f = stats.f.ppf(1 - input_alpha, r_f, c_f) # Table values are Fcrit for input_alpha
            except: pass
            row_cells_f += f'<td id="f_{r_f}_{i_f}">{val_f:.2f if not np.isnan(val_f) else "N/A"}</td>'
        body_html += f"<tr>{row_cells_f}</tr>"
    code = f"<tr><th>df1\\df2</th>{head_html}</tr>{body_html}"
    html = wrap_table(CSS_BASE, code)
    if df1 in rows:
        for i in range(len(cols)+1): html = style_cell(html, f"f_{df1}_{i}")
    if col_idx_hl != -1:
        for rr_f in rows: html = style_cell(html, f"f_{rr_f}_{col_idx_hl}")
        if df1 in rows: html = style_cell(html, f"f_{df1}_{col_idx_hl}", color="blue", px=3)
    return html

def f_table(df1: int, df2: int, input_alpha: float):
    code = build_f_table(df1, df2, input_alpha)
    st.markdown(container(code, height=300), unsafe_allow_html=True)

def f_apa(f_val: float, df1: int, df2: int, input_alpha: float): # Original f_apa restored
    p_calc = stats.f.sf(f_val, df1, df2)
    crit = stats.f.ppf(1 - input_alpha, df1, df2) # Critical value from input_alpha
    reject = (f_val>crit)
    decision = "rejected" if reject else "failed to reject"
    reason_stats = ("because F₍calc₎ exceeded F₍crit₎"
                    if reject else "because F₍calc₎ did not exceed F₍crit₎")
    reason_p = ("because p < α" if reject else "because p ≥ α") # Original greek alpha

    st.markdown(
        "**APA interpretation** \n"
        f"Calculated statistic: *F*({df1},{df2})={f_val:.2f}, *p*={p_calc:.3f}.\n"
        f"Critical statistic for user's $\\alpha={input_alpha:.4f}$: F₍crit₎={crit:.2f}.\n" # Using input_alpha for crit label
        f"Statistic comparison → H0 **{decision}** ({reason_stats}).\n"
        f"*p* comparison → H0 **{decision}** ({reason_p}).\n"
        f"**APA 7 report:** *F*({df1},{df2})={f_val:.2f}, *p*={p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at $\\alpha$={input_alpha:.2f}."
    )

def tab_f():
    st.subheader("Tab 3 • F-Distribution")
    c1,c2=st.columns(2)
    with c1:
        f_val_in = st.number_input("F statistic", value=4.32, step=0.01, key="f_val_widget")
        df1_in = st.number_input("df1 (numerator)", min_value=1, value=5, step=1, key="f_df1_widget")
    with c2:
        df2_in = st.number_input("df2 (denominator)", min_value=1, value=20, step=1, key="f_df2_widget")
        alpha_in = st.number_input("α (alpha level)", value=0.05, step=0.001, min_value=0.0001, max_value=0.99, format="%.4f", key="f_alpha_widget")
    if 'f_button_clicked' not in st.session_state: st.session_state.f_button_clicked = False
    if st.button("Generate Plot and Table for F-Distribution", key="f_generate_button_main"):
        st.session_state.f_button_clicked = True
    if st.session_state.f_button_clicked:
        try:
            fig_f = plot_f(float(f_val_in), int(df1_in), int(df2_in), float(alpha_in))
            if fig_f: st.pyplot(fig_f)
        except Exception as e: st.error(f"Error generating F-plot: {e}")
        st.write("**F-table** (Values are F-critical for your input α. Always one-tailed.)")
        try:
            f_table(int(df1_in), int(df2_in), float(alpha_in))
            f_apa(float(f_val_in), int(df1_in), int(df2_in), float(alpha_in))
        except Exception as e: st.error(f"Error for F-table/APA: {e}"); st.exception(e)


# ----- Chi-Square Distribution (Restoring original structure) -----
def plot_chi(chi_calc, df, input_alpha): # Original plot_chi, ensure input_alpha is used
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    if df <= 0: ax.text(0.5,0.5, "df must be positive.", ha='center', va='center'); return fig
    try:
        crit_val_chi_plot = stats.chi2.ppf(1 - input_alpha, df)
        plot_upper_x = max(chi_calc, crit_val_chi_plot, 10) * 1.5
        if stats.chi2.ppf(0.999, df) < plot_upper_x : plot_upper_x = stats.chi2.ppf(0.999, df) * 1.1
        xs = np.linspace(0, plot_upper_x, 400); ys = stats.chi2.pdf(xs, df)
        valid_ys = ~np.isnan(ys) & ~np.isinf(ys); xs, ys = xs[valid_ys], ys[valid_ys]
        if len(xs) < 2: ax.text(0.5,0.5, "Cannot generate Chi-PDF.", ha='center', va='center'); return fig
        ax.plot(xs, ys, "k"); ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")
        ax.fill_between(xs[xs>=crit_val_chi_plot], ys[xs>=crit_val_chi_plot], color="red", alpha=0.3, label="Reject H0")
        ax.axvline(crit_val_chi_plot, color="green", ls="--"); ax.axvline(chi_calc, color="blue", ls="--")
        placed_c = []; pdf_crit_c = stats.chi2.pdf(crit_val_chi_plot, df); pdf_calc_c = stats.chi2.pdf(chi_calc, df)
        if not np.isnan(pdf_crit_c): place_label(ax, placed_c, crit_val_chi_plot, pdf_crit_c + 0.02, f"χ²₍crit₎={crit_val_chi_plot:.2f}", color="green")
        if not np.isnan(pdf_calc_c): place_label(ax, placed_c, chi_calc, pdf_calc_c + 0.02, f"χ²₍calc₎={chi_calc:.2f}", color="blue")
    except Exception as e: st.warning(f"Could not draw Chi-plot elements: {e}")
    ax.set_xlabel("χ²"); ax.set_ylabel("Density"); ax.legend(loc='upper right')
    ax.set_title(f"χ²-Distribution (df={df:.0f}, input $\\alpha$={input_alpha:.4f})"); fig.tight_layout()
    return fig

def build_chi_table(df: int, selected_alpha_from_selectbox: float) -> str: # Table uses alpha from selectbox
    rows = list(range(max(1,df-2), df+3+1))
    standard_alphas = [0.10,0.05,0.025,0.01,0.005,0.001] # Standard alphas for table display
    col_idx_hl = -1
    if selected_alpha_from_selectbox in standard_alphas:
        col_idx_hl = standard_alphas.index(selected_alpha_from_selectbox)+1
    else: # Should not happen if selectbox items match standard_alphas
        min_diff_c = float('inf')
        for i_c, std_a_c in enumerate(standard_alphas, start=1):
            if abs(std_a_c - selected_alpha_from_selectbox) < min_diff_c:
                min_diff_c = abs(std_a_c - selected_alpha_from_selectbox); col_idx_hl = i_c
        if col_idx_hl !=-1 : st.warning(f"Alpha {selected_alpha_from_selectbox} for table highlight not exact, using closest {standard_alphas[col_idx_hl-1]}.")
        else: col_idx_hl = 1
            
    head_html = "".join(f"<th>{a:.3f}</th>" for a in standard_alphas)
    body_html = ""
    for r_c in rows:
        row_cells_c = f'<td id="chi_{r_c}_0">{r_c}</td>'
        for i_c_cell,a_c_cell in enumerate(standard_alphas, start=1):
            val_c = np.nan; 
            try: val_c = stats.chi2.ppf(1 - a_c_cell, r_c)
            except: pass
            row_cells_c += f'<td id="chi_{r_c}_{i_c_cell}">{val_c:.2f if not np.isnan(val_c) else "N/A"}</td>'
        body_html += f"<tr>{row_cells_c}</tr>"
    code = f"<tr><th>df\\α</th>{head_html}</tr>{body_html}"
    html = wrap_table(CSS_BASE, code)
    if df in rows:
        for i in range(len(standard_alphas)+1): html = style_cell(html, f"chi_{df}_{i}")
    if col_idx_hl != -1:
        for rr_c in rows: html = style_cell(html, f"chi_{rr_c}_{col_idx_hl}")
        if df in rows: html = style_cell(html, f"chi_{df}_{col_idx_hl}", color="blue", px=3)
    return html

def chi_table(df: int, alpha_from_selectbox: float):
    code = build_chi_table(df, alpha_from_selectbox)
    st.markdown(container(code, height=300), unsafe_allow_html=True)

def chi_apa(chi_val: float, df: int, input_alpha: float): # Original chi_apa, uses input_alpha for user's test
    p_calc = stats.chi2.sf(chi_val, df)
    crit = stats.chi2.ppf(1 - input_alpha, df) # Crit value from user's input_alpha
    reject = (chi_val>crit)
    decision = "rejected" if reject else "failed to reject"
    reason_stats = "because χ²₍calc₎ exceeded χ²₍crit₎" if reject else "because χ²₍calc₎ did not exceed χ²₍crit₎"
    reason_p = "because p < α" if reject else "because p ≥ α"

    st.markdown(
        "**APA interpretation** \n"
        f"Calculated statistic: χ²({df})={chi_val:.2f}, *p*={p_calc:.3f}.\n"
        f"Critical statistic for user's $\\alpha={input_alpha:.4f}$: χ²₍crit₎={crit:.2f}.\n"
        f"Statistic comparison → H0 **{decision}** ({reason_stats}).\n"
        f"*p* comparison → H0 **{decision}** ({reason_p}).\n"
        f"**APA 7 report:** χ²({df})={chi_val:.2f}, *p*={p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at $\\alpha$={input_alpha:.2f}."
    )

def tab_chi():
    st.subheader("Tab 4 • Chi-Square Distribution")
    c1,c2=st.columns(2)
    with c1:
        chi_val_in = st.number_input("χ² statistic", value=7.88, step=0.01, key="chi_val_widget")
        df_in = st.number_input("df", min_value=1, value=3, step=1, key="chi_df_widget")
    with c2:
        # Alpha from selectbox for table consistency, but user might want arbitrary alpha for their test
        alpha_options_for_table = [0.10, 0.05, 0.025, 0.01, 0.005, 0.001]
        alpha_for_table_display = st.selectbox("Table's α for highlighting", alpha_options_for_table, index=1, key="chi_alpha_selectbox_widget",
                                          help="Select an alpha to highlight in the table. Your test's alpha can be different.")
        # Allow user to input their own alpha for calculation, separate from table display alpha
        alpha_for_test_calc = st.number_input("Your test's α (alpha level)", value=0.05, step=0.001, 
                                            min_value=0.0001, max_value=0.99, format="%.4f", key="chi_alpha_test_widget")

    if 'chi_button_clicked' not in st.session_state: st.session_state.chi_button_clicked = False
    if st.button("Generate Plot and Table for Chi-Square", key="chi_generate_button_main"):
        st.session_state.chi_button_clicked = True

    if st.session_state.chi_button_clicked:
        try: # Plot uses user's test alpha
            fig_c = plot_chi(float(chi_val_in), int(df_in), float(alpha_for_test_calc))
            if fig_c: st.pyplot(fig_c)
        except Exception as e: st.error(f"Error for Chi-plot: {e}")
        st.write(f"**χ²-table** (Table columns show standard α's. Column for selected table α={alpha_for_table_display} is highlighted.)")
        try: # Table uses selectbox alpha for highlighting
            chi_table(int(df_in), float(alpha_for_table_display))
            # APA uses user's test alpha
            chi_apa(float(chi_val_in), int(df_in), float(alpha_for_test_calc))
        except Exception as e: st.error(f"Error for Chi-table/APA: {e}"); st.exception(e)

# ----- Mann-Whitney U (Restoring original structure, revised APA template) -----
def plot_u(u_calc, n1, n2, input_alpha, tail): # Uses input_alpha for plot crit
    mu_u = n1*n2/2.0; sigma_u = np.sqrt(n1*n2*(n1+n2+1)/12.0)
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    if sigma_u <= 1e-9: ax.text(0.5, 0.5, "Cannot plot U: Variance is zero.", ha='center', va='center'); return fig
    plot_min_x = max(0, mu_u - 4.5 * sigma_u); plot_max_x = min(n1 * n2, mu_u + 4.5 * sigma_u)
    if plot_min_x >= plot_max_x : plot_min_x = max(0, u_calc -1); plot_max_x = u_calc +1 ; if plot_min_x >=plot_max_x: plot_max_x=plot_min_x+2
    xs = np.linspace(plot_min_x, plot_max_x, 400); ys = stats.norm.pdf(xs, mu_u, sigma_u)
    valid_ys_u = ~np.isnan(ys) & ~np.isinf(ys); xs, ys = xs[valid_ys_u], ys[valid_ys_u]
    if len(xs) < 2: ax.text(0.5,0.5, "Cannot generate U-PDF.", ha='center', va='center'); return fig
    ax.plot(xs, ys, "k"); ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")
    placed_u_plot = []; from math import floor, ceil
    try:
        if tail.startswith("one"):
            # Assuming U_calc is tested against appropriate tail based on its value vs mean for plotting
            if u_calc <= mu_u: # Test U <= U_L
                z_c_u = stats.norm.ppf(input_alpha); crit_u_p = floor(mu_u + z_c_u * sigma_u)
                ax.fill_between(xs[xs <= crit_u_p], ys[xs <= crit_u_p], color="red", alpha=0.3, label="Reject H0")
                ax.axvline(crit_u_p, color="green", ls="--"); 
                if not np.isnan(stats.norm.pdf(crit_u_p, mu_u, sigma_u)): place_label(ax, placed_u_plot, crit_u_p, stats.norm.pdf(crit_u_p, mu_u, sigma_u) + 0.005, f"Ucrit L≈{crit_u_p}", color="green")
            else: # Test U >= U_U (U_calc is large)
                z_c_u = stats.norm.ppf(1-input_alpha); crit_u_p = ceil(mu_u + z_c_u * sigma_u)
                ax.fill_between(xs[xs >= crit_u_p], ys[xs >= crit_u_p], color="red", alpha=0.3, label="Reject H0")
                ax.axvline(crit_u_p, color="green", ls="--");
                if not np.isnan(stats.norm.pdf(crit_u_p, mu_u, sigma_u)): place_label(ax, placed_u_plot, crit_u_p, stats.norm.pdf(crit_u_p, mu_u, sigma_u) + 0.005, f"Ucrit U≈{crit_u_p}", color="green")
        else:
            z_c_l_u = stats.norm.ppf(input_alpha / 2); crit_l_u_p = floor(mu_u + z_c_l_u * sigma_u)
            z_c_u_u = stats.norm.ppf(1-input_alpha/2); crit_u_u_p = ceil(mu_u + z_c_u_u * sigma_u) # More direct for upper
            ax.fill_between(xs[xs <= crit_l_u_p], ys[xs <= crit_l_u_p], color="red", alpha=0.3)
            ax.fill_between(xs[xs >= crit_u_u_p], ys[xs >= crit_u_u_p], color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_l_u_p, color="green", ls="--"); ax.axvline(crit_u_u_p, color="green", ls="--")
            if not np.isnan(stats.norm.pdf(crit_l_u_p, mu_u, sigma_u)): place_label(ax, placed_u_plot, crit_l_u_p, stats.norm.pdf(crit_l_u_p, mu_u, sigma_u) + 0.005, f"Ucrit L≈{crit_l_u_p}", color="green")
            if not np.isnan(stats.norm.pdf(crit_u_u_p, mu_u, sigma_u)): place_label(ax, placed_u_plot, crit_u_u_p, stats.norm.pdf(crit_u_u_p, mu_u, sigma_u) + 0.005, f"Ucrit U≈{crit_u_u_p}", color="green")
        pdf_ucalc = stats.norm.pdf(u_calc, mu_u, sigma_u)
        if not np.isnan(pdf_ucalc): place_label(ax, placed_u_plot, u_calc, pdf_ucalc + 0.005, f"Ucalc={u_calc}", color="blue")
        ax.axvline(u_calc, color="blue", ls="--")
    except Exception as e: st.warning(f"Could not draw U-plot elements: {e}")
    ax.set_xlabel("U (Normal Approx.)"); ax.set_ylabel("Approx. density"); ax.legend(loc='upper right')
    ax.set_title(f"Mann-Whitney U (Approx. $n_1$={n1}, $n_2$={n2}, input $\\alpha$={input_alpha:.4f})"); fig.tight_layout()
    return fig

def u_crit(n1:int, n2:int, input_alpha:float, tail:str)->any: # Original u_crit, table values based on input_alpha
    mu_u = n1*n2/2.0; sigma_u = np.sqrt(n1*n2*(n1+n2+1)/12.0)
    if sigma_u <= 1e-9: return np.nan
    from math import floor
    # Returns the lower critical U value for the given alpha/tail
    z_crit_for_table = stats.norm.ppf(input_alpha if tail.startswith("one") else input_alpha/2)
    return int(floor(mu_u + z_crit_for_table*sigma_u))

def build_u_table(n1:int, n2:int, input_alpha:float, tail:str)->str: # Table values based on input_alpha
    rows_u = list(range(max(2, n1 - 2), n1 + 3 + 1)); cols_u = list(range(max(2, n2 - 2), n2 + 3 + 1))
    col_idx_u_hl = -1
    if n2 in cols_u: col_idx_u_hl = cols_u.index(n2)+1
    head_html_u = "".join(f"<th>$n_2$={c_u}</th>" for c_u in cols_u)
    body_html_u = ""
    for r_u in rows_u:
        row_cells_u = f'<td id="u_{r_u}_0">$n_1$={r_u}</td>'
        for i_u,c_u_cell in enumerate(cols_u, start=1):
            val_u_cell = u_crit(r_u,c_u_cell,input_alpha,tail) # Lower critical U for this alpha
            row_cells_u += f'<td id="u_{r_u}_{i_u}">{val_u_cell if not np.isnan(val_u_cell) else "N/A"}</td>'
        body_html_u += f"<tr>{row_cells_u}</tr>"
    code_u = f"<tr><th>$n_1 \\setminus n_2$</th>{head_html_u}</tr>{body_html_u}"
    html_u = wrap_table(CSS_BASE, code_u)
    if n1 in rows_u:
        for i in range(len(cols_u)+1): html_u = style_cell(html_u, f"u_{n1}_{i}")
    if col_idx_u_hl != -1:
        for rr_u_h in rows_u: html_u = style_cell(html_u, f"u_{rr_u_h}_{col_idx_u_hl}")
        if n1 in rows_u: html_u = style_cell(html_u, f"u_{n1}_{col_idx_u_hl}", color="blue", px=3)
    return html_u

def u_table(n1:int, n2:int, input_alpha:float, tail:str):
    code = build_u_table(n1,n2,input_alpha,tail)
    st.markdown(container(code, height=300), unsafe_allow_html=True)

def u_apa(u_val: int, n1: int, n2: int, input_alpha: float, tail: str): # Revised APA explanation template
    mu_u = n1 * n2 / 2.0; sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z_approx_val, p_calc_u, crit_z_apa_u, reject_u = np.nan, np.nan, np.nan, False

    if sigma_u > 1e-9:
        u_corr = 0.5 if u_val < mu_u else (-0.5 if u_val > mu_u else 0)
        z_approx_val = (u_val - mu_u + u_corr) / sigma_u
        if tail.startswith("one"):
            p_calc_u = stats.norm.cdf(z_approx_val) if z_approx_val <=0 else stats.norm.sf(z_approx_val) # Match p to Z direction
            crit_z_apa_u = stats.norm.ppf(input_alpha) if p_calc_u < 0.5 else stats.norm.ppf(1-input_alpha) # Match Z_crit to expected Z direction
            reject_u = (z_approx_val < crit_z_apa_u) if p_calc_u < 0.5 else (z_approx_val > crit_z_apa_u)
        else:
            p_calc_u = 2 * stats.norm.sf(abs(z_approx_val))
            crit_z_apa_u = stats.norm.ppf(1 - input_alpha / 2)
            reject_u = abs(z_approx_val) > crit_z_apa_u
    else: st.warning("Cannot calculate Z approx for U: σ=0.")

    decision_u = "rejected" if reject_u else "failed to reject"
    reason_stats_u = f"$Z_{{approx}}$ ({z_approx_val:.2f}) in rejection region (vs $Z_{{crit}} \\approx {crit_z_apa_u:.2f}$)" if reject_u else f"$Z_{{approx}}$ not in rejection region"
    reason_p_u = f"$p \\approx {p_calc_u:.3f} < \\alpha$" if reject_u else f"$p \\approx {p_calc_u:.3f} \\ge \\alpha$"
    
    # P-Value Calculation Explanation part (template like t-test)
    st.subheader("P-value Calculation Explanation (Normal Approx.)") # Added subheader here for consistency
    expl_u_parts = [f"For $U_{{calc}} = {u_val}$ (with $n_1={n1}, n_2={n2}$):"]
    if sigma_u > 1e-9:
        expl_u_parts.append(f"Using Normal Approximation: $\\mu_U \\approx {mu_u:.2f}$, $\\sigma_U \\approx {sigma_u:.2f}$.")
        expl_u_parts.append(f"The Z-statistic (with continuity correction) is $Z_{{approx}} \\approx {z_approx_val:.2f}$.")
        cdf_of_z_approx = stats.norm.cdf(z_approx_val)
        expl_u_parts.append(f"The cumulative probability $P(Z \\le Z_{{approx}}) \\approx {cdf_of_z_approx:.4f}$.")
        if tail.startswith("one"):
            # Direction of p-value calculation explanation depends on direction of test implied by Z_approx
            if z_approx_val <=0: # Test for U being small
                 expl_u_parts.append(f"For a **one-tailed** test (e.g., testing if $U$ is significantly small), $p \\approx P(Z \\le Z_{{approx}}) \\approx {cdf_of_z_approx:.4f}$.")
            else: # Test for U being large (or U_other being small)
                 expl_u_parts.append(f"For a **one-tailed** test (e.g., testing if $U$ is significantly large), $p \\approx 1 - P(Z \\le Z_{{approx}}) \\approx {1-cdf_of_z_approx:.4f}$.")
        else: # two-tailed
            expl_u_parts.append(f"For a **two-tailed** test, $p \\approx 2 \\times P(Z \\ge |Z_{{approx}}|) \\approx {p_calc_u:.4f}$.")
        expl_u_parts.append(f"The calculated p-value is $p \\approx {p_calc_u:.4f}$.")
    else:
        expl_u_parts.append("Cannot provide detailed explanation as $\\sigma_U$ is zero.")
    st.write("\n\n".join(expl_u_parts))

    # APA Interpretation part (remains separate)
    st.markdown( # Original markdown structure restored, using calculated values
        "**APA interpretation (Normal Approximation for U)**\n"
        f"Calculated Mann-Whitney U = {u_val}. With $n_1={n1}, n_2={n2}$, this yields an approximate Z-statistic (with continuity correction) $Z \\approx {z_approx_val:.2f}$, "
        f"and an approximate *$p$* = {p_calc_u:.3f} ({tail}).\n"
        f"The critical Z-value for user's $\\alpha={input_alpha:.4f}$ ({tail}) is $Z_{{crit}} \\approx {crit_z_apa_u:.2f}$.\n"
        f"Based on comparison of Z-statistics: H₀ is **{decision_u}** ({reason_stats_u}).\n"
        f"Based on comparison of p-value to alpha: H₀ is **{decision_u}** ({reason_p_u}).\n"
        f"**APA 7 report (approximate):** A Mann-Whitney U test indicated that the null hypothesis was **{decision_u}**, "
        f"U = {u_val}, Z $\\approx$ {z_approx_val:.2f}, *p* $\\approx$ {p_calc_u:.3f} ({tail}), at an alpha level of {input_alpha:.2f}."
    )

def tab_u():
    st.subheader("Tab 5 • Mann-Whitney U Distribution")
    c1_u, c2_u = st.columns(2)
    with c1_u:
        u_val_in = st.number_input("U statistic value", min_value=0, value=23, step=1, key="u_val_widget")
        n1_in = st.number_input("n1 (sample 1 size)", min_value=1, value=8, step=1, key="u_n1_widget")
    with c2_u:
        n2_in = st.number_input("n2 (sample 2 size)", min_value=1, value=10, step=1, key="u_n2_widget")
        alpha_in = st.number_input("α (alpha level) for U", value=0.05, step=0.001, min_value=0.0001, max_value=0.99, format="%.4f", key="u_alpha_widget")
        tail_in = st.radio("Tail for U", ["one-tailed", "two-tailed"], key="u_tail_widget", horizontal=True)
    if 'u_button_clicked' not in st.session_state: st.session_state.u_button_clicked = False
    if st.button("Generate Plot and Table for Mann-Whitney U", key="u_generate_button_main"):
        st.session_state.u_button_clicked = True
    if st.session_state.u_button_clicked:
        if int(n1_in) < 1 or int(n2_in) < 1: st.error("Sample sizes n1 and n2 must be at least 1.")
        else:
            try:
                fig_u = plot_u(int(u_val_in), int(n1_in), int(n2_in), float(alpha_in), tail_in)
                if fig_u: st.pyplot(fig_u)
            except Exception as e: st.error(f"Error for U-plot: {e}")
            st.write("**U-table (Approx. Lower Critical U Values for your input α)**")
            ct_u, ce_u = st.columns([3,2]) # Keep table and explanation separate
            with ct_u:
                try:
                    u_table(int(n1_in), int(n2_in), float(alpha_in), tail_in)
                    st.info("U-table shows approx. lower critical U's for *your input α*. Reject H₀ if obs. U ≤ table U (for left-tail or corresponding two-tail test).")
                except Exception as e: st.error(f"Error for U-table: {e}"); st.exception(e)
            with ce_u: # This is where the P-value explanation and APA go.
                try:
                    # The u_apa function now contains the "P-value Calc Explanation" subheader
                    u_apa(int(u_val_in), int(n1_in), int(n2_in), float(alpha_in), tail_in)
                except Exception as e: st.error(f"Error for U-APA: {e}")

def tab_wilcoxon_t():
    st.subheader("Tab 6 • Wilcoxon Signed-Rank T")
    st.write("Wilcoxon T functionality to be implemented.")
def tab_binomial():
    st.subheader("Tab 7 • Binomial Distribution")
    st.write("Binomial functionality to be implemented.")

def main():
    st.set_page_config(layout="wide", page_title="Statistical Tables Explorer")
    st.title("Oli's - Statistical Table Explorer")
    tab_titles = ["t-Dist", "z-Dist", "F-Dist", "Chi-Square", "Mann-Whitney U", "Wilcoxon T", "Binomial"]
    tabs = st.tabs(tab_titles)
    with tabs[0]: tab_t()
    with tabs[1]: tab_z()
    with tabs[2]: tab_f()
    with tabs[3]: tab_chi()
    with tabs[4]: tab_u()
    with tabs[5]: tab_wilcoxon_t()
    with tabs[6]: tab_binomial()

if __name__ == "__main__":
    main()
