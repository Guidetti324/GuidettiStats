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

# Ensure matplotlib uses a non-interactive backend for Streamlit
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
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
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

def plot_t(t_calc, df, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    if df <= 0: # df must be positive for t-distribution
        ax.text(0.5, 0.5, "Degrees of freedom (df) must be positive.", horizontalalignment='center', verticalalignment='center')
        return fig
        
    xs = np.linspace(-4.5, 4.5, 400) # Adjusted range slightly
    try:
        ys = stats.t.pdf(xs, df)
    except Exception: # Catch potential errors with df (e.g. non-integer if not cast)
        ax.text(0.5, 0.5, f"Invalid df ({df}) for t-distribution PDF.", horizontalalignment='center', verticalalignment='center')
        return fig

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")
    placed_labels = []

    try:
        if tail.startswith("one"):
            crit_pos = stats.t.ppf(1 - alpha, df)
            crit_neg = -crit_pos
            pdf_at_crit_pos = stats.t.pdf(crit_pos, df)
            pdf_at_crit_neg = stats.t.pdf(crit_neg, df)

            if t_calc >= 0:
                ax.fill_between(xs[xs >= crit_pos], ys[xs >= crit_pos], color="red", alpha=0.3, label="Reject H0")
                ax.axvline(crit_pos, color="green", ls="--")
                if not np.isnan(pdf_at_crit_pos):
                    place_label(ax, placed_labels, crit_pos, pdf_at_crit_pos + 0.02, f"tcrit={crit_pos:.2f}", color="green")
            else:
                ax.fill_between(xs[xs <= crit_neg], ys[xs <= crit_neg], color="red", alpha=0.3, label="Reject H0")
                ax.axvline(crit_neg, color="green", ls="--")
                if not np.isnan(pdf_at_crit_neg):
                    place_label(ax, placed_labels, crit_neg, pdf_at_crit_neg + 0.02, f"tcrit={crit_neg:.2f}", color="green")
        else: # two-tailed
            crit = stats.t.ppf(1 - alpha / 2, df)
            pdf_at_crit = stats.t.pdf(crit, df)
            ax.fill_between(xs[xs >= crit], ys[xs >= crit], color="red", alpha=0.3)
            ax.fill_between(xs[xs <= -crit], ys[xs <= -crit], color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit, color="green", ls="--")
            ax.axvline(-crit, color="green", ls="--")
            if not np.isnan(pdf_at_crit):
                place_label(ax, placed_labels, crit, pdf_at_crit + 0.02, f"+tcrit={crit:.2f}", color="green")
                place_label(ax, placed_labels, -crit, pdf_at_crit + 0.02, f"–tcrit={crit:.2f}", color="green")
        
        pdf_at_t_calc = stats.t.pdf(t_calc, df)
        if not np.isnan(pdf_at_t_calc):
            place_label(ax, placed_labels, t_calc, pdf_at_t_calc + 0.02, f"tcalc={t_calc:.2f}", color="blue")
        ax.axvline(t_calc, color="blue", ls="--")

    except Exception as e:
        st.warning(f"Could not draw all critical regions/labels on plot: {e}")


    ax.set_xlabel("t")
    ax.set_ylabel("Density")
    ax.legend(loc='upper right')
    ax.set_title(f"t-Distribution (df={df:.0f})")
    fig.tight_layout()
    return fig

def build_t_html(df: int, alpha: float, tail: str) -> str:
    rows = list(range(max(1, df - 5), df + 6))
    heads = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001)
    ]
    mode = "one" if tail.startswith("one") else "two"

    col_idx = -1
    best_match_info = None
    min_diff = float('inf')

    for i, (h_mode, h_alpha_val) in enumerate(heads, start=1):
        if h_mode == mode:
            if np.isclose(h_alpha_val, alpha):
                col_idx = i
                best_match_info = {'alpha_val': h_alpha_val, 'is_exact': True, 'index': i}
                break
            current_diff = abs(h_alpha_val - alpha)
            if current_diff < min_diff:
                min_diff = current_diff
                best_match_info = {'index': i, 'alpha_val': h_alpha_val, 'is_exact': False}
            elif current_diff == min_diff:
                if best_match_info and h_alpha_val < best_match_info['alpha_val']:
                    best_match_info = {'index': i, 'alpha_val': h_alpha_val, 'is_exact': False}

    if col_idx == -1:
        if best_match_info:
            col_idx = best_match_info['index']
            selected_alpha_for_table = best_match_info['alpha_val']
            if not best_match_info.get('is_exact', False):
                st.warning(
                    f"Entered alpha ({alpha:.4f}) is not standard for {mode}-tailed. "
                    f"Highlighting closest standard alpha: {selected_alpha_for_table:.4f}."
                )
        else: # Fallback if no relevant heads found for mode (should not happen for "one"/"two")
            st.error(f"No columns for mode '{mode}'. Defaulting to column 1.")
            col_idx = 1
            if heads: st.info(f"Defaulted to col 1 (alpha={heads[0][1]:.3f} for mode='{heads[0][0]}').")

    if not (1 <= col_idx <= len(heads)): # Ensure col_idx is valid
        st.warning(f"Invalid column index {col_idx}. Resetting to 1.")
        col_idx = 1

    head_html_parts = []
    for m_h, a_h in heads:
        head_html_parts.append(f"<th>{m_h}<br>$\\alpha$={a_h:.3f}</th>") # Using LaTeX for alpha
    head_html = "".join(head_html_parts)
    
    body_html = ""
    for r_val in rows:
        row_cells = f'<td id="t_{r_val}_0">{r_val}</td>'
        for i_cell, (m_cell, a_cell) in enumerate(heads, start=1):
            try:
                crit_val_cell = stats.t.ppf(1 - a_cell if m_cell == "one" else 1 - a_cell / 2, r_val)
                cell_text = f"{crit_val_cell:.3f}"
            except Exception:
                cell_text = "N/A"
            row_cells += f'<td id="t_{r_val}_{i_cell}">{cell_text}</td>'
        body_html += f"<tr>{row_cells}</tr>"

    table_code = f"<tr><th>df</th>{head_html}</tr>{body_html}"
    html_output = wrap_table(CSS_BASE, table_code)

    if df in rows:
        for i_highlight in range(len(heads) + 1):
            html_output = style_cell(html_output, f"t_{df}_{i_highlight}")
    for rr_val in rows:
        html_output = style_cell(html_output, f"t_{rr_val}_{col_idx}")
    if df in rows:
        html_output = style_cell(html_output, f"t_{df}_{col_idx}", color="blue", px=3)
    
    return html_output

def t_table(df: int, alpha: float, tail: str):
    code = build_t_html(df, alpha, tail)
    st.markdown(container(code), unsafe_allow_html=True)

def t_apa(t_val: float, df: int, alpha: float, tail: str):
    p_calc_val = np.nan
    crit_val = np.nan
    reject = False

    try:
        if tail.startswith("one"):
            p_calc_val = stats.t.sf(abs(t_val), df)
            crit_val = stats.t.ppf(1 - alpha, df) # Positive crit for right tail
            if t_val >= 0: # Right tail hypothesis
                reject = t_val > crit_val
            else: # Left tail hypothesis
                reject = t_val < -crit_val # Compare t_val to negative of positive crit_val
        else: # two-tailed
            p_calc_val = stats.t.sf(abs(t_val), df) * 2
            crit_val = stats.t.ppf(1 - alpha / 2, df)
            reject = abs(t_val) > crit_val
    except Exception as e:
        st.warning(f"Could not calculate p-value or critical value: {e}")


    decision = "rejected" if reject else "failed to reject"
    reason_stats = "because $t_{calc}$ was in the rejection region" if reject else "because $t_{calc}$ was not in the rejection region"
    reason_p = f"because $p \\approx {p_calc_val:.3f} < \\alpha$" if reject else f"because $p \\approx {p_calc_val:.3f} \\ge \\alpha$"
    
    cdf_val = np.nan
    try:
        cdf_val = stats.t.cdf(t_val, df)
    except Exception: pass


    expl_parts = []
    expl_parts.append(f"Lookup: $P(T \\le {t_val:.2f}) \\approx {cdf_val:.4f}$ (if calculable).")
    if tail.startswith("one"):
        if t_val >= 0:
            expl_parts.append(f"For a **one-tailed** (right tail) test, $p = 1 - P(T \\le {t_val:.2f}) \\approx {1-cdf_val:.4f}$ (actual $p \\approx {p_calc_val:.4f}$).")
        else:
            expl_parts.append(f"For a **one-tailed** (left tail) test, $p = P(T \\le {t_val:.2f}) \\approx {cdf_val:.4f}$ (actual $p \\approx {p_calc_val:.4f}$).")
    else:
        expl_parts.append(f"For a **two-tailed** test, $p = 2 \\times P(T \\ge |{t_val:.2f}|) \\approx {p_calc_val:.4f}$.")
    
    st.write("\n\n".join(expl_parts))

    st.markdown(
        f"**APA interpretation**\n"
        f"Calculated statistic: *$t$({df}) = {t_val:.2f}, approximate *$p$ = {p_calc_val:.3f}.\n"
        f"Critical statistic for $\\alpha={alpha:.3f}$ ({tail}): $t_{{crit}} \\approx {crit_val:.2f}$.\n"
        f"Comparison of statistics $\\rightarrow$ H₀ **{decision}** ({reason_stats}).\n"
        f"Comparison of *$p$*-values $\\rightarrow$ H₀ **{decision}** ({reason_p}).\n"
        f"**APA 7 report:** *$t$({df}) = {t_val:.2f}, *$p$ $\\approx$ {p_calc_val:.3f} ({tail}). The null hypothesis was **{decision}** at $\\alpha$={alpha:.2f}."
    )

def tab_t():
    st.subheader("Tab 1 • t-Distribution")
    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("t statistic", value=2.10, step=0.01, key="t_val_tab_t")
        df = st.number_input("df", min_value=1, value=10, step=1, key="t_df_tab_t")
    with c2:
        alpha = st.number_input("α (alpha)", value=0.05, step=0.001, min_value=0.0001, max_value=0.99, format="%.4f", key="t_alpha_tab_t")
        tail = st.radio("Tail", ["one-tailed", "two-tailed"], key="t_tail_tab_t", horizontal=True)

    # Use a button to trigger plot and table updates to avoid too many redraws on input change
    if st.button("Generate Plot and Table for t-Distribution", key="t_generate_button"):
        try:
            fig_t_dist = plot_t(float(t_val), int(df), float(alpha), tail)
            if fig_t_dist: st.pyplot(fig_t_dist)
        except Exception as e:
            st.error(f"Error generating t-plot: {e}")

        st.write("**t-table** (highlighted)")
        ctable_t, cexp_t = st.columns([3, 2]) # Adjusted ratio
        with ctable_t:
            try:
                t_table(int(df), float(alpha), tail)
                show_cumulative_note()
            except Exception as e:
                st.error(f"Error generating t-table: {e}")
                st.exception(e) # Shows full traceback for debugging this part
        with cexp_t:
            try:
                st.subheader("P-value Calculation Explanation")
                t_apa(float(t_val), int(df), float(alpha), tail)
            except Exception as e:
                st.error(f"Error in t-APA explanation: {e}")

# Restore original button behavior if preferred by user:
# if st.button("Update Plot", key="t_plot"):
#     st.pyplot(plot_t(t_val, df, alpha, tail))
# t_table(df, alpha, tail) ... etc. (original flow)


###############################################################################
#                             TAB 2: z-Distribution
###############################################################################
# (Keeping z-distribution code as per your original, with minor key changes for inputs)
def plot_z(z_calc, alpha, tail):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    xs = np.linspace(-4,4,400)
    ys = stats.norm.pdf(xs)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H0")

    placed_z = [] # ensure placed is initialized for each plot

    if tail.startswith("one"):
        crit_pos = stats.norm.ppf(1 - alpha)
        crit_neg = -crit_pos # same as stats.norm.ppf(alpha)
        if z_calc >= 0: # Assuming right-tailed for positive z
            ax.fill_between(xs[xs>=crit_pos], ys[xs>=crit_pos],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_pos, color="green", ls="--")
            place_label(ax, placed_z, crit_pos, stats.norm.pdf(crit_pos)+0.02,
                        f"z₍crit₎={crit_pos:.2f}", color="green")
        else: # Assuming left-tailed for negative z
            ax.fill_between(xs[xs<=crit_neg], ys[xs<=crit_neg],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_neg, color="green", ls="--")
            place_label(ax, placed_z, crit_neg, stats.norm.pdf(crit_neg)+0.02,
                        f"z₍crit₎={crit_neg:.2f}", color="green")
    else: # two-tailed
        crit = stats.norm.ppf(1 - alpha/2) # Positive critical value
        ax.fill_between(xs[xs>=crit], ys[xs>=crit], color="red", alpha=0.3)
        ax.fill_between(xs[xs<=-crit], ys[xs<=-crit], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(-crit, color="green", ls="--")
        place_label(ax, placed_z, crit, stats.norm.pdf(crit)+0.02,
                    f"+z₍crit₎={crit:.2f}", color="green")
        place_label(ax, placed_z, -crit, stats.norm.pdf(-crit)+0.02,
                    f"–z₍crit₎={crit:.2f}", color="green")

    ax.axvline(z_calc, color="blue", ls="--")
    place_label(ax, placed_z, z_calc, stats.norm.pdf(z_calc)+0.02,
                f"z₍calc₎={z_calc:.2f}", color="blue")

    ax.set_xlabel("z")
    ax.set_ylabel("Density")
    ax.legend(loc='upper right')
    ax.set_title("z-Distribution")
    fig.tight_layout()
    return fig

def build_z_html(z_val: float, alpha: float, tail: str) -> str: # alpha, tail not used for z-table structure
    z_val_clipped = np.clip(z_val, -3.499, 3.499) # Clip to avoid issues at exact boundaries for floor/round
    
    # Determine row for z-table (e.g., z=1.64 -> row_label=1.6)
    row_label = np.sign(z_val_clipped) * np.floor(abs(z_val_clipped) * 10) / 10
    
    # Determine column for z-table (e.g., z=1.64 -> col_label=0.04)
    # Handle potential floating point inaccuracies carefully
    col_label = round(abs(z_val_clipped) - abs(row_label), 2)

    # Define the Z-table row and column headers to be displayed
    TableRows = np.round(np.arange(-3.4, 3.5, 0.1), 1)
    TableCols = np.round(np.arange(0.00, 0.10, 0.01), 2) # Standard 0.00 to 0.09 columns

    # Find the closest row_label in TableRows (in case z_val_clipped was on a boundary after rounding)
    row_label = min(TableRows, key=lambda r_disp: abs(r_disp - row_label))
    # Find the closest col_label in TableCols
    col_label = min(TableCols, key=lambda c_disp: abs(c_disp - col_label))

    # Determine which rows to show in the HTML table view
    try:
        center_row_idx_in_table = list(TableRows).index(row_label)
    except ValueError: # Fallback if row_label somehow not in TableRows
        center_row_idx_in_table = len(TableRows) // 2
    
    display_row_start_idx = max(0, center_row_idx_in_table - 10)
    display_row_end_idx = min(len(TableRows), center_row_idx_in_table + 11)
    rows_to_show_in_html = TableRows[display_row_start_idx:display_row_end_idx]

    head_html = "".join(f"<th>{c_head:.2f}</th>" for c_head in TableCols)
    body_html = ""
    for rr_html in rows_to_show_in_html:
        row_html_cells = f'<td id="z_{rr_html:.1f}_0">{rr_html:.1f}</td>' # Row header cell
        for cc_html in TableCols:
            # Cell value is P(Z <= z_cell_value)
            # z_cell_value for standard tables: if rr_html is negative (e.g. -1.6),
            # and cc_html is positive (e.g. 0.04), then z = -1.6 + 0.04 = -1.56. (This matches many tables)
            # Some tables might use |rr_html| + cc_html then adjust sign.
            # We assume P(Z <= rr_html + cc_html)
            current_z_for_cell = rr_html + cc_html # This interpretation makes sense for "body to the left" tables
            cdf_cell_val = stats.norm.cdf(current_z_for_cell)
            row_html_cells += f'<td id="z_{rr_html:.1f}_{cc_html:.2f}">{cdf_cell_val:.4f}</td>'
        body_html += f"<tr>{row_html_cells}</tr>"

    table_code = f"<tr><th>z</th>{head_html}</tr>{body_html}" # Changed z.x to z for header
    html_output = wrap_table(CSS_BASE, table_code)

    # Highlighting logic:
    if row_label in rows_to_show_in_html: # Ensure the target row is actually displayed
        # Highlight the entire row for row_label
        for cc_highlight in TableCols:
            html_output = style_cell(html_output, f"z_{row_label:.1f}_{cc_highlight:.2f}")
        html_output = style_cell(html_output, f"z_{row_label:.1f}_0") # Highlight row header

        # Highlight the entire column for col_label
        for rr_highlight in rows_to_show_in_html:
            html_output = style_cell(html_output, f"z_{rr_highlight:.1f}_{col_label:.2f}")
        
        # Highlight intersection cell
        html_output = style_cell(html_output, f"z_{row_label:.1f}_{col_label:.2f}", color="blue", px=3)
        
    return html_output

def z_table(z_val: float, alpha: float, tail: str):
    code = build_z_html(z_val, alpha, tail)
    st.markdown(container(code), unsafe_allow_html=True)

def z_apa(z_val: float, alpha: float, tail: str):
    p_calc_val = np.nan
    crit_val = np.nan
    reject = False

    try:
        if tail.startswith("one"):
            p_calc_val = stats.norm.sf(abs(z_val)) # P(Z > |z|)
            if z_val >= 0: # Right-tail H1
                crit_val = stats.norm.ppf(1 - alpha)
                reject = z_val > crit_val
            else: # Left-tail H1
                crit_val = stats.norm.ppf(alpha) # Negative critical value
                reject = z_val < crit_val
        else: # two-tailed
            p_calc_val = stats.norm.sf(abs(z_val)) * 2
            crit_val = stats.norm.ppf(1 - alpha/2) # Positive critical value
            reject = abs(z_val) > crit_val
    except Exception as e:
        st.warning(f"Could not calculate p-value or Z-critical: {e}")

    decision = "rejected" if reject else "failed to reject"
    reason_stats = f"because $|z_{{calc}}|$ ({abs(z_val):.2f}) {'exceeded' if reject else 'did not exceed'} $|z_{{crit}}|$ ({abs(crit_val):.2f})"
    reason_p = f"because $p \\approx {p_calc_val:.3f} {'<' if reject else '≥'} \\alpha$"
    
    table_val_cdf = np.nan
    try:
        table_val_cdf = stats.norm.cdf(z_val) # P(Z <= z_val)
    except Exception: pass

    expl_parts = []
    expl_parts.append(f"Lookup from Z-table (body to the left): $P(Z \\le {z_val:.2f}) \\approx {table_val_cdf:.4f}$.")
    if tail.startswith("one"):
        if z_val >= 0: # Right-tail
            expl_parts.append(f"For a **one-tailed** (right tail) test, $p = 1 - P(Z \\le {z_val:.2f}) \\approx {1-table_val_cdf:.4f}$ (actual $p \\approx {p_calc_val:.4f}$).")
        else: # Left-tail
            expl_parts.append(f"For a **one-tailed** (left tail) test, $p = P(Z \\le {z_val:.2f}) \\approx {table_val_cdf:.4f}$ (actual $p \\approx {p_calc_val:.4f}$).")
    else: # Two-tail
        expl_parts.append(f"For a **two-tailed** test, $p = 2 \\times P(Z \\ge |{z_val:.2f}|) \\approx {p_calc_val:.4f}$.")
    st.write("\n\n".join(expl_parts))

    st.markdown(
        f"**APA interpretation**\n"
        f"Calculated statistic: *$z$* = {z_val:.2f}, approximate *$p$ = {p_calc_val:.3f}.\n"
        f"Critical statistic for $\\alpha={alpha:.3f}$ ({tail}): $z_{{crit}} \\approx {crit_val:.2f}$.\n"
        f"Statistic comparison $\\rightarrow$ H₀ **{decision}** ({reason_stats}).\n"
        f"*$p$* comparison $\\rightarrow$ H₀ **{decision}** ({reason_p}).\n"
        f"**APA 7 report:** *$z$* = {z_val:.2f}, *$p$ $\\approx$ {p_calc_val:.3f} ({tail}). The null hypothesis was **{decision}** at $\\alpha$={alpha:.2f}."
    )

def tab_z():
    st.subheader("Tab 2 • z-Distribution")
    c1, c2 = st.columns(2)
    with c1:
        z_val = st.number_input("z statistic", value=1.64, step=0.01, key="z_val_tab_z")
    with c2:
        alpha = st.number_input("α (alpha)", value=0.05, step=0.001, min_value=0.0001, max_value=0.99, format="%.4f", key="z_alpha_tab_z")
        tail = st.radio("Tail", ["one-tailed", "two-tailed"], key="z_tail_tab_z", horizontal=True)

    if st.button("Generate Plot and Table for z-Distribution", key="z_generate_button"):
        try:
            fig_z_dist = plot_z(float(z_val), float(alpha), tail)
            if fig_z_dist: st.pyplot(fig_z_dist)
        except Exception as e:
            st.error(f"Error generating z-plot: {e}")

        st.write("**z-table** (highlighted)")
        ctable_z, cexp_z = st.columns([3, 2])
        with ctable_z:
            try:
                z_table(float(z_val), float(alpha), tail)
                show_cumulative_note()
            except Exception as e:
                st.error(f"Error generating z-table: {e}")
                st.exception(e)
        with cexp_z:
            try:
                st.subheader("P-value Calculation Explanation")
                z_apa(float(z_val), float(alpha), tail)
            except Exception as e:
                st.error(f"Error in z-APA explanation: {e}")


###############################################################################
#                       TAB 3: F-Distribution
###############################################################################
# (Keeping F-distribution code as per your original, with minor key changes and plot label init)
def plot_f(f_calc, df1, df2, alpha):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    if df1 <= 0 or df2 <= 0:
        ax.text(0.5,0.5, "df1 and df2 must be positive.", ha='center', va='center')
        return fig
    try:
        # Define a reasonable upper limit for x-axis based on F calc and crit
        crit_val_f = stats.f.ppf(1 - alpha, df1, df2)
        plot_upper_x = stats.f.ppf(0.999, df1, df2) # Default upper x
        if np.isnan(plot_upper_x) or np.isinf(plot_upper_x) or plot_upper_x > 5 * max(f_calc, crit_val_f, 5): # Cap extreme ppf
            plot_upper_x = max(f_calc, crit_val_f, 5) * 1.5 # Fallback based on f_calc and crit
        
        xs = np.linspace(0, plot_upper_x, 400)
        ys = stats.f.pdf(xs, df1, df2)
        
        # Filter out NaN/inf from PDF results if any
        valid_ys_indices = ~np.isnan(ys) & ~np.isinf(ys)
        xs, ys = xs[valid_ys_indices], ys[valid_ys_indices]
        if len(xs) < 2: # Not enough points to plot
            ax.text(0.5,0.5, "Could not generate F-dist PDF for these df.", ha='center', va='center')
            return fig

        ax.plot(xs, ys, "k")
        ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")

        ax.fill_between(xs[xs>=crit_val_f], ys[xs>=crit_val_f], color="red", alpha=0.3, label="Reject H0")
        ax.axvline(crit_val_f, color="green", ls="--")
        ax.axvline(f_calc, color="blue", ls="--")

        placed_f = [] # ensure placed is initialized for each plot
        pdf_at_crit = stats.f.pdf(crit_val_f, df1, df2)
        if not np.isnan(pdf_at_crit):
            place_label(ax, placed_f, crit_val_f, pdf_at_crit + 0.02, f"F₍crit₎={crit_val_f:.2f}", color="green")
        
        pdf_at_fcalc = stats.f.pdf(f_calc, df1, df2)
        if not np.isnan(pdf_at_fcalc):
            place_label(ax, placed_f, f_calc, pdf_at_fcalc + 0.02, f"F₍calc₎={f_calc:.2f}", color="blue")
            
    except Exception as e:
        st.warning(f"Could not draw all elements on F-plot: {e}")


    ax.set_xlabel("F")
    ax.set_ylabel("Density")
    ax.legend(loc='upper right')
    ax.set_title(f"F-Distribution (df1={df1:.0f}, df2={df2:.0f})")
    fig.tight_layout()
    return fig

def build_f_table(df1: int, df2: int, alpha: float) -> str:
    # More focused table display
    rows_display = list(range(max(1, df1 - 2), df1 + 3 + 1)) 
    cols_display = list(range(max(1, df2 - 2), df2 + 3 + 1))
    
    col_idx_for_highlight = -1
    if df2 in cols_display:
        col_idx_for_highlight = cols_display.index(df2) + 1 # 1-based index for HTML id

    head_html = "".join(f"<th>{c_val}</th>" for c_val in cols_display)
    body_html = ""
    for r_val_html in rows_display:
        row_cells_html = f'<td id="f_{r_val_html}_0">{r_val_html}</td>'
        for i_col_html, c_val_html in enumerate(cols_display, start=1):
            try:
                f_cell_val = stats.f.ppf(1 - alpha, r_val_html, c_val_html)
                cell_text_f = f"{f_cell_val:.2f}" if not np.isnan(f_cell_val) else "N/A"
            except Exception:
                cell_text_f = "Err"
            row_cells_html += f'<td id="f_{r_val_html}_{i_col_html}">{cell_text_f}</td>'
        body_html += f"<tr>{row_cells_html}</tr>"

    table_code = f"<tr><th>df1\\df2</th>{head_html}</tr>{body_html}"
    html_output = wrap_table(CSS_BASE, table_code)

    if df1 in rows_display:
        for i_highlight_f in range(len(cols_display) + 1):
            html_output = style_cell(html_output, f"f_{df1}_{i_highlight_f}")
    
    if col_idx_for_highlight != -1: # df2 was in displayed columns
        for rr_val_f in rows_display:
            html_output = style_cell(html_output, f"f_{rr_val_f}_{col_idx_for_highlight}")
    
    if df1 in rows_display and col_idx_for_highlight != -1 :
        html_output = style_cell(html_output, f"f_{df1}_{col_idx_for_highlight}", color="blue", px=3)
    return html_output

def f_table(df1: int, df2: int, alpha: float):
    code = build_f_table(df1, df2, alpha)
    st.markdown(container(code, height=300), unsafe_allow_html=True) # Adjusted height

def f_apa(f_val: float, df1: int, df2: int, alpha: float): # Original f_apa
    p_calc = stats.f.sf(f_val, df1, df2)
    crit = stats.f.ppf(1 - alpha, df1, df2)
    reject = (f_val>crit)
    decision = "rejected" if reject else "failed to reject"
    reason_stats = ("because F₍calc₎ exceeded F₍crit₎"
                    if reject else "because F₍calc₎ did not exceed F₍crit₎")
    reason_p = ("because p < α" if reject else "because p ≥ α")

    st.markdown(
        "**APA interpretation** \n"
        f"Calculated statistic: *F*({df1},{df2})={f_val:.2f}, *p*={p_calc:.3f}.  \n"
        f"Critical statistic: F₍crit₎={crit:.2f}, *p*={alpha:.3f}.  \n" # alpha is crit_p here
        f"Statistic comparison → H0 **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *F*({df1},{df2})={f_val:.2f}, *p*={p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at α={alpha:.2f}."
    )

def tab_f():
    st.subheader("Tab 3 • F-Distribution")
    c1, c2 = st.columns(2)
    with c1:
        f_val = st.number_input("F statistic", value=4.32, step=0.01, key="f_val_tab_f")
        df1 = st.number_input("df1 (numerator)", min_value=1, value=5, step=1, key="f_df1_tab_f")
    with c2:
        df2 = st.number_input("df2 (denominator)", min_value=1, value=20, step=1, key="f_df2_tab_f")
        alpha = st.number_input("α (alpha)", value=0.05, step=0.001, min_value=0.0001, max_value=0.99, format="%.4f", key="f_alpha_tab_f")

    if st.button("Generate Plot and Table for F-Distribution", key="f_generate_button"):
        try:
            fig_f_dist = plot_f(float(f_val), int(df1), int(df2), float(alpha))
            if fig_f_dist: st.pyplot(fig_f_dist)
        except Exception as e:
            st.error(f"Error generating F-plot: {e}")

        st.write("**F-table** (Values are F-critical for the given α. Always one-tailed.)")
        # F-table APA and explanation are combined as it's simpler
        try:
            f_table(int(df1), int(df2), float(alpha))
            f_apa(float(f_val), int(df1), int(df2), float(alpha))
        except Exception as e:
            st.error(f"Error generating F-table/APA: {e}")
            st.exception(e)


###############################################################################
#                       TAB 4: Chi-Square
###############################################################################
# (Keeping Chi-Square code as per your original, with selectbox for alpha and minor key changes)
def plot_chi(chi_calc, df, alpha):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    if df <= 0:
        ax.text(0.5,0.5, "df must be positive for Chi-square.", ha='center', va='center')
        return fig
    try:
        crit_val_chi = stats.chi2.ppf(1 - alpha, df)
        plot_upper_x_chi = stats.chi2.ppf(0.999, df)
        if np.isnan(plot_upper_x_chi) or np.isinf(plot_upper_x_chi) or plot_upper_x_chi > 5 * max(chi_calc, crit_val_chi, 5):
            plot_upper_x_chi = max(chi_calc, crit_val_chi, 5) * 1.5

        xs = np.linspace(0, plot_upper_x_chi, 400)
        ys = stats.chi2.pdf(xs, df)

        valid_ys_indices_chi = ~np.isnan(ys) & ~np.isinf(ys)
        xs, ys = xs[valid_ys_indices_chi], ys[valid_ys_indices_chi]
        if len(xs) < 2:
            ax.text(0.5,0.5, "Could not generate Chi-square PDF for this df.", ha='center', va='center')
            return fig
            
        ax.plot(xs, ys, "k")
        ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")
        
        ax.fill_between(xs[xs>=crit_val_chi], ys[xs>=crit_val_chi], color="red", alpha=0.3, label="Reject H0")
        ax.axvline(crit_val_chi, color="green", ls="--")
        ax.axvline(chi_calc, color="blue", ls="--")
        
        placed_chi = [] # ensure placed is initialized for each plot
        pdf_at_crit_chi = stats.chi2.pdf(crit_val_chi, df)
        if not np.isnan(pdf_at_crit_chi):
            place_label(ax, placed_chi, crit_val_chi, pdf_at_crit_chi + 0.02, f"χ²₍crit₎={crit_val_chi:.2f}", color="green")
        
        pdf_at_chicalc = stats.chi2.pdf(chi_calc, df)
        if not np.isnan(pdf_at_chicalc):
            place_label(ax, placed_chi, chi_calc, pdf_at_chicalc + 0.02, f"χ²₍calc₎={chi_calc:.2f}", color="blue")

    except Exception as e:
        st.warning(f"Could not draw all elements on Chi-square plot: {e}")

    ax.set_xlabel("χ²")
    ax.set_ylabel("Density")
    ax.legend(loc='upper right')
    ax.set_title(f"χ²-Distribution (df={df:.0f})")
    fig.tight_layout()
    return fig

def build_chi_table(df: int, alpha_input: float) -> str: # alpha_input is from selectbox
    rows_display_chi = list(range(max(1, df - 2), df + 3 + 1))
    # Standard alpha levels for Chi-square tables
    standard_alphas_chi = [0.10, 0.05, 0.025, 0.01, 0.005, 0.001] # Expanded slightly
    
    # Since alpha_input comes from a selectbox matching some of these, find its index
    # If it might not match, find closest (but selectbox ensures it matches if items are same)
    col_idx_chi = -1
    if alpha_input in standard_alphas_chi:
        col_idx_chi = standard_alphas_chi.index(alpha_input) + 1 # 1-based
    else: # Fallback if alpha_input somehow doesn't match selectbox items
        min_diff_chi = float('inf')
        for i_chi, std_a_chi in enumerate(standard_alphas_chi, start=1):
            if abs(std_a_chi - alpha_input) < min_diff_chi:
                min_diff_chi = abs(std_a_chi - alpha_input)
                col_idx_chi = i_chi
        if col_idx_chi != -1:
             st.warning(f"Alpha {alpha_input} not standard for table, using closest: {standard_alphas_chi[col_idx_chi-1]}")
        else: # Should not happen if standard_alphas_chi is not empty
            col_idx_chi = 1


    head_html_chi = "".join(f"<th>{a_val:.3f}</th>" for a_val in standard_alphas_chi)
    body_html_chi = ""
    for r_val_chi in rows_display_chi:
        row_cells_chi = f'<td id="chi_{r_val_chi}_0">{r_val_chi}</td>'
        for i_h_chi, a_h_chi in enumerate(standard_alphas_chi, start=1):
            try:
                chi_cell_val = stats.chi2.ppf(1 - a_h_chi, r_val_chi) # Chi is upper tail
                cell_text_chi = f"{chi_cell_val:.2f}" if not np.isnan(chi_cell_val) else "N/A"
            except Exception:
                cell_text_chi = "Err"
            row_cells_chi += f'<td id="chi_{r_val_chi}_{i_h_chi}">{cell_text_chi}</td>'
        body_html_chi += f"<tr>{row_cells_chi}</tr>"

    table_code_chi = f"<tr><th>df\\α</th>{head_html_chi}</tr>{body_html_chi}"
    html_output_chi = wrap_table(CSS_BASE, table_code_chi)

    if df in rows_display_chi:
        for i_chi_highlight in range(len(standard_alphas_chi) + 1):
            html_output_chi = style_cell(html_output_chi, f"chi_{df}_{i_chi_highlight}")
            
    if col_idx_chi != -1:
        for rr_val_chi_h in rows_display_chi:
            html_output_chi = style_cell(html_output_chi, f"chi_{rr_val_chi_h}_{col_idx_chi}")
    
    if df in rows_display_chi and col_idx_chi != -1:
        html_output_chi = style_cell(html_output_chi, f"chi_{df}_{col_idx_chi}", color="blue", px=3)
    return html_output_chi

def chi_table(df: int, alpha: float):
    code = build_chi_table(df, alpha)
    st.markdown(container(code, height=300), unsafe_allow_html=True)

def chi_apa(chi_val: float, df: int, alpha: float): # Original chi_apa
    p_calc = stats.chi2.sf(chi_val, df)
    crit = stats.chi2.ppf(1 - alpha, df)
    reject = (chi_val>crit)
    decision = "rejected" if reject else "failed to reject"
    reason_stats = "because χ²₍calc₎ exceeded χ²₍crit₎" if reject else "because χ²₍calc₎ did not exceed χ²₍crit₎"
    reason_p = "because p < α" if reject else "because p ≥ α"

    st.markdown(
        "**APA interpretation** \n"
        f"Calculated statistic: χ²({df})={chi_val:.2f}, *p*={p_calc:.3f}.  \n"
        f"Critical statistic: χ²₍crit₎={crit:.2f}, *p*={alpha:.3f}.  \n" # alpha is crit_p
        f"Statistic comparison → H0 **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** χ²({df})={chi_val:.2f}, *p*={p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at α={alpha:.2f}."
    )

def tab_chi():
    st.subheader("Tab 4 • Chi-Square Distribution")
    c1, c2 = st.columns(2)
    with c1:
        chi_val = st.number_input("χ² statistic", value=7.88, step=0.01, key="chi_val_tab_chi")
        df = st.number_input("df", min_value=1, value=3, step=1, key="chi_df_tab_chi")
    with c2:
        # Original uses selectbox, which is safer as alpha is guaranteed to be in the list for build_chi_table
        alpha_options_chi = [0.10, 0.05, 0.025, 0.01, 0.005, 0.001]
        alpha = st.selectbox("α (alpha)", alpha_options_chi, index=1, key="chi_alpha_tab_chi")

    if st.button("Generate Plot and Table for Chi-Square", key="chi_generate_button"):
        try:
            fig_chi_dist = plot_chi(float(chi_val), int(df), float(alpha))
            if fig_chi_dist: st.pyplot(fig_chi_dist)
        except Exception as e:
            st.error(f"Error generating Chi-square plot: {e}")

        st.write("**χ²-table** (Values are χ²-critical for the given α. Always one-tailed.)")
        try:
            chi_table(int(df), float(alpha))
            chi_apa(float(chi_val), int(df), float(alpha))
        except Exception as e:
            st.error(f"Error generating Chi-square table/APA: {e}")
            st.exception(e)


###############################################################################
#                       TAB 5: Mann-Whitney U
###############################################################################
# (Keeping Mann-Whitney U code as per your original, with syntax correction in u_apa)
def plot_u(u_calc, n1, n2, alpha, tail): # Original plot_u
    mu_u = n1*n2/2.0
    sigma_u = np.sqrt(n1*n2*(n1+n2+1)/12.0)

    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    if sigma_u <= 1e-9: 
        ax.text(0.5, 0.5, "Cannot plot U-dist: Variance is zero (n1/n2 too small).", ha='center', va='center')
        return fig
        
    # Define plot range dynamically
    plot_min_x = max(0, mu_u - 4.5 * sigma_u) # U is non-negative
    plot_max_x = min(n1 * n2, mu_u + 4.5 * sigma_u) # Max U is n1*n2
    if plot_min_x >= plot_max_x : # Handle cases where sigma is tiny or mu is at an edge
        plot_min_x = max(0, u_calc - 3 * sigma_u -1) # Ensure some range around u_calc
        plot_max_x = u_calc + 3 * sigma_u + 1
        if plot_min_x >= plot_max_x: plot_max_x = plot_min_x + 2 # Absolute fallback for range

    xs = np.linspace(plot_min_x, plot_max_x, 400)
    ys = stats.norm.pdf(xs, mu_u, sigma_u)
    
    valid_ys_indices_u = ~np.isnan(ys) & ~np.isinf(ys)
    xs, ys = xs[valid_ys_indices_u], ys[valid_ys_indices_u]
    if len(xs) < 2:
        ax.text(0.5,0.5, "Could not generate U-dist PDF points.", ha='center', va='center')
        return fig

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")
    placed_u = [] # ensure placed is initialized for each plot
    from math import floor, ceil # Keep import local if only used here

    try:
        if tail.startswith("one"):
            # Determine if left or right tail based on u_calc vs mu_u for plotting purposes
            # For U-test, U_calc is usually min(U1,U2), so significant is small U (left tail)
            if u_calc <= mu_u: # Assuming testing for significantly small U
                z_crit_u = stats.norm.ppf(alpha)
                crit_val_u = floor(mu_u + z_crit_u * sigma_u)
                ax.fill_between(xs[xs <= crit_val_u], ys[xs <= crit_val_u], color="red", alpha=0.3, label="Reject H0")
                ax.axvline(crit_val_u, color="green", ls="--")
                if not np.isnan(stats.norm.pdf(crit_val_u, mu_u, sigma_u)):
                     place_label(ax, placed_u, crit_val_u, stats.norm.pdf(crit_val_u, mu_u, sigma_u) + 0.005, f"Ucrit L={crit_val_u}", color="green")
            else: # u_calc > mu_u; this might mean user provided U_other or is testing for large U.
                  # Standard tables usually focus on small U. For plotting, show upper critical.
                z_crit_u = stats.norm.ppf(1 - alpha)
                crit_val_u = ceil(mu_u + z_crit_u * sigma_u)
                ax.fill_between(xs[xs >= crit_val_u], ys[xs >= crit_val_u], color="red", alpha=0.3, label="Reject H0")
                ax.axvline(crit_val_u, color="green", ls="--")
                if not np.isnan(stats.norm.pdf(crit_val_u, mu_u, sigma_u)):
                    place_label(ax, placed_u, crit_val_u, stats.norm.pdf(crit_val_u, mu_u, sigma_u) + 0.005, f"Ucrit U={crit_val_u}", color="green")
        else: # two-tailed
            z_crit_lower_u = stats.norm.ppf(alpha / 2)
            crit_val_lower_u = floor(mu_u + z_crit_lower_u * sigma_u)
            crit_val_upper_u = ceil(mu_u - z_crit_lower_u * sigma_u) # Symmetrical: mu_u + stats.norm.ppf(1-alpha/2)*sigma_u

            ax.fill_between(xs[xs <= crit_val_lower_u], ys[xs <= crit_val_lower_u], color="red", alpha=0.3)
            ax.fill_between(xs[xs >= crit_val_upper_u], ys[xs >= crit_val_upper_u], color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_val_lower_u, color="green", ls="--")
            ax.axvline(crit_val_upper_u, color="green", ls="--")
            if not np.isnan(stats.norm.pdf(crit_val_lower_u, mu_u, sigma_u)):
                place_label(ax, placed_u, crit_val_lower_u, stats.norm.pdf(crit_val_lower_u, mu_u, sigma_u) + 0.005, f"Ucrit L={crit_val_lower_u}", color="green")
            if not np.isnan(stats.norm.pdf(crit_val_upper_u, mu_u, sigma_u)):
                 place_label(ax, placed_u, crit_val_upper_u, stats.norm.pdf(crit_val_upper_u, mu_u, sigma_u) + 0.005, f"Ucrit U={crit_val_upper_u}", color="green")
        
        pdf_at_ucalc = stats.norm.pdf(u_calc, mu_u, sigma_u)
        if not np.isnan(pdf_at_ucalc):
            place_label(ax, placed_u, u_calc, pdf_at_ucalc + 0.005, f"Ucalc={u_calc}", color="blue")
        ax.axvline(u_calc, color="blue", ls="--")
            
    except Exception as e:
        st.warning(f"Could not draw all elements on U-plot: {e}")


    ax.set_xlabel("U (Normal Approx.)")
    ax.set_ylabel("Approx. density")
    ax.legend(loc='upper right')
    ax.set_title(f"Mann-Whitney U (Normal Approx. $n_1$={n1}, $n_2$={n2})")
    fig.tight_layout()
    return fig

def u_crit(n1:int, n2:int, alpha:float, tail:str)->any: # Original u_crit, returns one value or NaN
    # This returns the lower critical value for U (reject if U_calc <= U_crit)
    mu_u = n1*n2/2.0
    sigma_u = np.sqrt(n1*n2*(n1+n2+1)/12.0)
    if sigma_u <= 1e-9: return np.nan # Cannot calculate if variance is 0
    
    from math import floor
    # For U-test, tables provide critical value U_alpha such that P(U <= U_alpha) approx alpha.
    # So we use alpha directly for one-tailed (left), and alpha/2 for two-tailed (for the lower tail).
    alpha_level_for_ppf = alpha if tail.startswith("one") else alpha/2
    z_val_crit = stats.norm.ppf(alpha_level_for_ppf) 
    
    return int(floor(mu_u + z_val_crit*sigma_u)) # mu + z*sigma (z is negative for left tail)

def build_u_table(n1:int, n2:int, alpha:float, tail:str)->str:
    # Display a small focused table around n1, n2
    rows_u_display = list(range(max(2, n1 - 2), n1 + 3 + 1))
    cols_u_display = list(range(max(2, n2 - 2), n2 + 3 + 1))

    col_idx_u_highlight = -1
    if n2 in cols_u_display:
        col_idx_u_highlight = cols_u_display.index(n2) + 1

    head_html_u = "".join(f"<th>$n_2$={c_val}</th>" for c_val in cols_u_display)
    body_html_u = ""
    for r_val_u in rows_u_display:
        row_cells_u = f'<td id="u_{r_val_u}_0">$n_1$={r_val_u}</td>'
        for i_col_u, c_val_u in enumerate(cols_u_display, start=1):
            # Table shows approximate lower critical U values
            # For one-tailed, use alpha. For two-tailed, use alpha/2 for this lower crit value.
            # u_crit function handles this interpretation of alpha/tail.
            u_crit_cell_val = u_crit(r_val_u, c_val_u, alpha, tail) 
            cell_text_u = f"{u_crit_cell_val}" if not np.isnan(u_crit_cell_val) else "N/A"
            row_cells_u += f'<td id="u_{r_val_u}_{i_col_u}">{cell_text_u}</td>'
        body_html_u += f"<tr>{row_cells_u}</tr>"

    table_code_u = f"<tr><th>$n_1 \\setminus n_2$</th>{head_html_u}</tr>{body_html_u}"
    html_output_u = wrap_table(CSS_BASE, table_code_u)

    if n1 in rows_u_display:
        for i_u_highlight in range(len(cols_u_display) + 1):
            html_output_u = style_cell(html_output_u, f"u_{n1}_{i_u_highlight}")
    if col_idx_u_highlight != -1:
        for rr_val_u_h in rows_u_display:
            html_output_u = style_cell(html_output_u, f"u_{rr_val_u_h}_{col_idx_u_highlight}")
    if n1 in rows_u_display and col_idx_u_highlight != -1:
        html_output_u = style_cell(html_output_u, f"u_{n1}_{col_idx_u_highlight}", color="blue", px=3)
    return html_output_u

def u_table(n1:int, n2:int, alpha:float, tail:str):
    code = build_u_table(n1,n2,alpha,tail)
    st.markdown(container(code, height=300), unsafe_allow_html=True)

def u_apa(u_val: int, n1: int, n2: int, alpha: float, tail: str):
    mu_u = n1 * n2 / 2.0
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)

    z_approx = np.nan
    p_val_approx = np.nan
    crit_val_z_apa = np.nan
    reject = False # Default

    if sigma_u > 1e-9: # Check for non-zero sigma
        # Apply continuity correction for Z approximation
        if u_val < mu_u:
            z_approx = (u_val - mu_u + 0.5) / sigma_u
        elif u_val > mu_u:
            z_approx = (u_val - mu_u - 0.5) / sigma_u
        else: # u_val == mu_u
            z_approx = 0.0

        if tail.startswith("one"):
            # Assuming u_val is min(U1,U2) for a one-tailed test aiming for small U
            # Or, if using scipy, it returns U for 'less' or 'greater' as specified.
            # For this general APA, let's assume p-value corresponds to the Z direction.
            if z_approx <= 0: # Corresponds to u_val <= mu_u
                p_val_approx = stats.norm.cdf(z_approx)
                crit_val_z_apa = stats.norm.ppf(alpha) # Left-tail Z_crit
                reject = z_approx < crit_val_z_apa
            else: # z_approx > 0, corresponds to u_val > mu_u
                p_val_approx = stats.norm.sf(z_approx) # Right-tail P(Z > z_approx)
                crit_val_z_apa = stats.norm.ppf(1 - alpha) # Right-tail Z_crit
                reject = z_approx > crit_val_z_apa
        else: # two-tailed
            p_val_approx = 2 * stats.norm.sf(abs(z_approx))
            crit_val_z_apa = stats.norm.ppf(1-alpha / 2) # Positive Z_crit for |Z|
            reject = abs(z_approx) > crit_val_z_apa
    else:
        st.warning("Cannot calculate Z approximation for U-test: standard deviation is zero or too small.")

    decision = "rejected" if reject else "failed to reject"
    reason_stats = (f"because Z_approx ({z_approx:.2f}) was in rejection region (Z_crit≈{crit_val_z_apa:.2f})" 
                    if reject else f"Z_approx ({z_approx:.2f}) not in rejection region (Z_crit≈{crit_val_z_apa:.2f})")
    reason_p = (f"approx. p ({p_val_approx:.3f}) < α ({alpha:.3f})" 
                if reject else f"approx. p ({p_val_approx:.3f}) ≥ α ({alpha:.3f})")

    normal_cdf_for_u_val = np.nan
    if sigma_u > 1e-9:
        normal_cdf_for_u_val = stats.norm.cdf((u_val - mu_u) / sigma_u) # CDF without continuity correction for general P(U<=u) idea

    expl_parts_u = []
    if sigma_u > 1e-9:
        expl_parts_u.append(f"For $U_{{calc}} = {u_val}$ (with $n_1={n1}, n_2={n2}$):")
        expl_parts_u.append(f"Approx. mean $\\mu_U \\approx {mu_u:.2f}$, approx. std. dev. $\\sigma_U \\approx {sigma_u:.2f}$.")
        expl_parts_u.append(f"Z-statistic (with continuity correction) $Z_{{approx}} \\approx {z_approx:.2f}$.")
        expl_parts_u.append(f"Approx. $P(U \\le {u_val}) \\approx {normal_cdf_for_u_val:.4f}$ (based on simple normal approx.).")
        if tail.startswith("one"):
            expl_parts_u.append(f"For a **one-tailed** test, the p-value (from Z_approx with cont. corr.) is $\\approx {p_val_approx:.4f}$.")
        else:
            expl_parts_u.append(f"For a **two-tailed** test, the p-value (from Z_approx with cont. corr.) is $\\approx {p_val_approx:.4f}$.")
    else:
        expl_parts_u.append("Cannot provide detailed explanation as $\\sigma_U$ is zero.")
        
    st.write("\n\n".join(expl_parts_u))

    st.markdown(
         f"**APA interpretation (Normal Approximation for U)**\n"
         f"Calculated Mann-Whitney U = {u_val}. With $n_1={n1}, n_2={n2}$, this corresponds to an approximate Z-statistic $Z \\approx {z_approx:.2f}$, "
         f"yielding an approximate *$p$* = {p_val_approx:.3f} ({tail}).\n"
         f"The critical Z-value for $\\alpha={alpha:.3f}$ ({tail}) is $Z_{{crit}} \\approx {crit_val_z_apa:.2f}$.\n"
         f"Based on Z-statistics: H₀ is **{decision}** ({reason_stats}).\n"
         f"Based on p-value: H₀ is **{decision}** ({reason_p}).\n"
         f"**APA 7 Report (approx.):** A Mann-Whitney U test indicated that the null hypothesis was **{decision}**, "
         f"U = {u_val}, Z $\\approx$ {z_approx:.2f}, *p* $\\approx$ {p_val_approx:.3f} ({tail}), at an alpha level of {alpha:.2f}."
     )

def tab_u():
    st.subheader("Tab 5 • Mann-Whitney U Distribution")
    c1_u, c2_u = st.columns(2)
    with c1_u:
        u_val_mw = st.number_input("U statistic value", min_value=0, value=23, step=1, key="u_val_tab_u")
        n1_mw = st.number_input("n1 (sample size 1)", min_value=1, value=8, step=1, key="u_n1_tab_u")
    with c2_u:
        n2_mw = st.number_input("n2 (sample size 2)", min_value=1, value=10, step=1, key="u_n2_tab_u")
        alpha_mw = st.number_input("α (alpha) for U", value=0.05, min_value=0.0001, max_value=0.99, step=0.001, format="%.4f", key="u_alpha_tab_u")
        tail_mw = st.radio("Tail for U", ["one-tailed", "two-tailed"], key="u_tail_tab_u", horizontal=True)

    if st.button("Generate Plot and Table for Mann-Whitney U", key="u_generate_button"):
        if int(n1_mw) < 1 or int(n2_mw) < 1:
            st.error("Sample sizes n1 and n2 must be at least 1.")
        else:
            try:
                fig_u_dist = plot_u(int(u_val_mw), int(n1_mw), int(n2_mw), float(alpha_mw), tail_mw)
                if fig_u_dist: st.pyplot(fig_u_dist)
            except Exception as e:
                st.error(f"Error generating U-plot: {e}")

            st.write("**U-table (Approximate Lower Critical U Values)**")
            ctable_u_main, cexp_u_main = st.columns([3, 2])
            with ctable_u_main:
                try:
                    u_table(int(n1_mw), int(n2_mw), float(alpha_mw), tail_mw)
                    # The cumulative note for Z/T might be misleading for U-test interpretation from this table.
                    st.info("Note: U-table values are approximate lower critical U's. Reject H₀ if observed U ≤ table U (for left-tail or corresponding two-tail test).")
                except Exception as e:
                    st.error(f"Error generating U-table: {e}")
                    st.exception(e)
            with cexp_u_main:
                try:
                    st.subheader("P-value Calculation & APA (Normal Approx.)")
                    u_apa(int(u_val_mw), int(n1_mw), int(n2_mw), float(alpha_mw), tail_mw)
                except Exception as e:
                    st.error(f"Error in U-APA explanation: {e}")

def tab_wilcoxon_t():
    st.subheader("Tab 6 • Wilcoxon Signed-Rank T")
    st.write("Wilcoxon T functionality to be implemented.")
    st.info("Note: This tab is a placeholder. The Wilcoxon test is for matched pairs or single sample median tests.")

def tab_binomial():
    st.subheader("Tab 7 • Binomial Distribution")
    st.write("Binomial test functionality to be implemented.")
    st.info("Note: This tab is a placeholder. The binomial test is used for count data with two outcomes.")

def main():
    st.set_page_config(layout="wide", page_title="Statistical Tables Explorer")
    st.title("Oli's - Statistical Table Explorer")

    tab_titles = ["t-Dist", "z-Dist", "F-Dist", "Chi-Square", "Mann-Whitney U", "Wilcoxon T", "Binomial"]
    
    # Using st.sidebar for navigation can sometimes be more stable if st.tabs causes issues with complex content
    # selected_tab_name = st.sidebar.radio("Select Distribution:", tab_titles)
    # For consistency with your screenshots, using st.tabs:
    
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        tab_t()
    with tabs[1]:
        tab_z()
    with tabs[2]:
        tab_f()
    with tabs[3]:
        tab_chi()
    with tabs[4]:
        tab_u()
    with tabs[5]:
        tab_wilcoxon_t()
    with tabs[6]:
        tab_binomial()

if __name__ == "__main__":
    main()
