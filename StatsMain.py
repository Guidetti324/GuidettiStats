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

plt.switch_backend("Agg")  # headless backend

###############################################################################
#                               COMMON SETUP
###############################################################################

def show_cumulative_note():
    """
    Shows the standardized cumulative probability note (for z, t, Mann-Whitney, 
    Wilcoxon, and Binomial).
    """
    st.info(
        "Note: The values in this table represent cumulative probabilities "
        "(i.e., the area under the curve to the left of a given value). For "
        "one-tailed tests, use the area directly. For two-tailed tests, you must "
        "double the area in the tail beyond your observed value (i.e., "
        "$p=2 \\times (1−P(Z \\le |z|))$). The same logic applies for t-distributions. The table "
        "itself does not change—only how you interpret it does."
    )


def place_label(ax, placed, x, y, txt, *, color="blue"):
    """
    Place text on the plot, shifting it slightly if it would collide
    with previously placed labels.
    """
    dx = dy = 0.0
    for (xx, yy) in placed:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            dx += 0.06
            dy += 0.04
    ax.text(x + dx, y + dy, txt, color=color,
            ha="left", va="bottom", fontsize=8, clip_on=True)
    placed.append((x + dx, y + dy))


def style_cell(html: str, cid: str, color: str = "red", px: int = 2) -> str:
    """
    Give the <td id="cid"> a border of the specified color & thickness.
    If color="blue" and px=3, it indicates the final intersection cell.
    """
    return html.replace(
        f'id="{cid}"',
        f'id="{cid}" style="border:{px}px solid {color};"',
        1
    )


def wrap_table(css: str, table: str) -> str:
    return f"<style>{css}</style><table>{table}</table>"


def container(html: str, *, height: int = 460) -> str:
    """
    Scrollable container for large HTML tables so the page doesn't grow indefinitely.
    """
    return f'<div style="overflow:auto; max-height:{height}px;">{html}</div>'


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
    """
    Plot the t-distribution. For one-tailed, decide left vs. right tail
    based on sign of t_calc. For two-tailed, shade both tails.
    """
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(-4, 4, 400)
    ys = stats.t.pdf(xs, df)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H0")

    placed_labels = []

    if tail.startswith("one"):
        # For one-tailed: decide which side based on sign of t_calc
        crit_pos = stats.t.ppf(1 - alpha, df)
        crit_neg = -crit_pos

        if t_calc >= 0:
            # shade right side
            ax.fill_between(xs[xs >= crit_pos], ys[xs >= crit_pos],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_pos, color="green", ls="--")
            place_label(ax, placed_labels, crit_pos, stats.t.pdf(crit_pos, df)+0.02,
                        f"$t_{{crit}}$={crit_pos:.2f}", color="green")
        else:
            # shade left side
            ax.fill_between(xs[xs <= crit_neg], ys[xs <= crit_neg],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_neg, color="green", ls="--")
            place_label(ax, placed_labels, crit_neg, stats.t.pdf(crit_neg, df)+0.02,
                        f"$t_{{crit}}$={crit_neg:.2f}", color="green")
    else:
        # two-tailed: shade both tails
        crit = stats.t.ppf(1 - alpha/2, df)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit], color="red", alpha=0.3)
        ax.fill_between(xs[xs <= -crit], ys[xs <= -crit], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(-crit, color="green", ls="--")
        place_label(ax, placed_labels, crit, stats.t.pdf(crit, df)+0.02,
                    f"+$t_{{crit}}$={crit:.2f}", color="green")
        place_label(ax, placed_labels, -crit, stats.t.pdf(-crit, df)+0.02,
                    f"–$t_{{crit}}$={crit:.2f}", color="green") # Note: using en-dash '–' for minus

    ax.axvline(t_calc, color="blue", ls="--")
    place_label(ax, placed_labels, t_calc, stats.t.pdf(t_calc, df)+0.02,
                f"$t_{{calc}}$={t_calc:.2f}", color="blue")

    ax.set_xlabel("t")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("t-Distribution")
    fig.tight_layout()
    return fig


def build_t_html(df: int, alpha: float, tail: str) -> str:
    """
    Single-step highlight for t-table row & column + intersection.
    Handles arbitrary alpha values by finding the closest standard alpha in the table.
    """
    rows = list(range(max(1, df - 5), df + 6))
    heads = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001)
    ]
    mode = "one" if tail.startswith("one") else "two"

    col_idx = -1
    # Try to find an exact (or very close) match first
    for i, (h_mode, h_alpha) in enumerate(heads, start=1):
        if h_mode == mode and np.isclose(h_alpha, alpha):
            col_idx = i
            break

    if col_idx == -1:
        # If no exact/close match, find the numerically closest alpha for the given mode
        relevant_heads_with_indices = []
        for i, (h_mode, h_alpha) in enumerate(heads, start=1):
            if h_mode == mode:
                relevant_heads_with_indices.append({'index': i, 'alpha_val': h_alpha, 'diff': abs(h_alpha - alpha)})
        
        if not relevant_heads_with_indices:
            st.error(f"Internal error: No table columns defined for mode '{mode}'. Defaulting to first column for this mode.")
            try:
                col_idx = next(i for i, (m, _) in enumerate(heads, start=1) if m == mode)
            except StopIteration: # Should not happen with current heads
                col_idx = 1 
        else:
            # Sort by difference, then by alpha value (smaller alpha preferred in ties), then by index
            relevant_heads_with_indices.sort(key=lambda x: (x['diff'], x['alpha_val'], x['index']))
            best_match = relevant_heads_with_indices[0]
            col_idx = best_match['index']
            selected_alpha_for_table = best_match['alpha_val']
            
            st.warning(
                f"The entered alpha ({alpha:.4f}) is not a standard value in this table for a {mode}-tailed test. "
                f"Highlighting the column for the closest standard alpha: {selected_alpha_for_table:.4f}."
            )

    # Fallback if col_idx is still not valid (e.g. if relevant_heads was empty and mode was unknown)
    if not (1 <= col_idx <= len(heads) +1): # len(heads)+1 because first col is df
         st.error(f"Failed to determine a valid column index. Defaulting to column 1.")
         col_idx = 1 # Default to the first alpha column index (index 1 for the first (m,a) pair)

    head_html = "".join(f"<th>{m}_$\\alpha$={a}</th>" for m, a in heads) # Using LaTeX for alpha
    body_html = ""
    for r_val in rows:
        row_cells = f'<td id="t_{r_val}_0">{r_val}</td>'
        for i, (m_head, a_head) in enumerate(heads, start=1):
            # Calculate critical value for the cell
            crit_val_cell = stats.t.ppf(1 - a_head if m_head == "one" else 1 - a_head / 2, r_val)
            row_cells += f'<td id="t_{r_val}_{i}">{crit_val_cell:.3f}</td>' # Increased precision to .3f
        body_html += f"<tr>{row_cells}</tr>"

    table_code = f"<tr><th>df</th>{head_html}</tr>{body_html}"
    html = wrap_table(CSS_BASE, table_code)

    # Highlight entire row for the input df
    if df in rows:
        for i in range(len(heads) + 1): # +1 to include the df column
            html = style_cell(html, f"t_{df}_{i}")
    else:
        # This case might occur if df is very small and df-5 < 1
        # Or if df input is outside the generated `rows` for any reason.
        st.warning(f"Note: The input df ({df}) is at the edge or outside the displayed table excerpt. Row highlighting might reflect the nearest displayed df.")


    # Highlight entire column for the determined col_idx
    for rr_val in rows:
        html = style_cell(html, f"t_{rr_val}_{col_idx}")
    
    # Highlight intersection cell
    if df in rows:
        html = style_cell(html, f"t_{df}_{col_idx}", color="blue", px=3)
    
    return html


def t_table(df: int, alpha: float, tail: str):
    code = build_t_html(df, alpha, tail)
    st.markdown(container(code), unsafe_allow_html=True)


def t_apa(t_val: float, df: int, alpha: float, tail: str):
    """
    Show the dynamic explanation and final APA lines for the t-distribution.
    """
    # Calculate p and critical value
    if tail.startswith("one"):
        p_calc = stats.t.sf(abs(t_val), df)  # single-sided
        crit = stats.t.ppf(1 - alpha, df)
        # For one-tailed, rejection rule depends on the direction of the hypothesis and t_val
        if t_val >= 0: # Assuming testing for positive effect or right-tail
             reject = (t_val > crit)
        else: # Assuming testing for negative effect or left-tail (crit would be negative)
             reject = (t_val < -crit) # -crit because our crit is from 1-alpha (positive)
    else: # two-tailed
        p_calc = stats.t.sf(abs(t_val), df) * 2
        crit = stats.t.ppf(1 - alpha/2, df) # Positive critical value
        reject = (abs(t_val) > crit)

    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = f"because $|t_{{calc}}|$ ({abs(t_val):.2f}) exceeded $|t_{{crit}}|$ ({abs(crit):.2f})" if tail.startswith("two") else f"because $t_{{calc}}$ ({t_val:.2f}) was in the rejection region defined by $t_{{crit}}$ ({crit:.2f})"
        reason_p = f"because p ({p_calc:.3f}) < $\\alpha$ ({alpha:.3f})"
    else:
        reason_stats = f"because $|t_{{calc}}|$ ({abs(t_val):.2f}) did not exceed $|t_{{crit}}|$ ({abs(crit):.2f})" if tail.startswith("two") else f"because $t_{{calc}}$ ({t_val:.2f}) was not in the rejection region defined by $t_{{crit}}$ ({crit:.2f})"
        reason_p = f"because p ({p_calc:.3f}) $\\ge \\alpha$ ({alpha:.3f})"

    # "table" cdf value
    cdf_val = stats.t.cdf(t_val, df)

    # Explanation text
    if tail.startswith("one"):
        if t_val >= 0:
            expl = (
                f"Lookup: $P(T \\le {t_val:.2f}) = {cdf_val:.4f}$.\n\n"
                f"For a **one-tailed** test (right tail, positive statistic), "
                f"p = $1 - P(T \\le {t_val:.2f}) = 1 - {cdf_val:.4f} = {(1-cdf_val):.4f}$."
            )
        else:
            expl = (
                f"Lookup: $P(T \\le {t_val:.2f}) = {cdf_val:.4f}$.\n\n"
                f"For a **one-tailed** test (left tail, negative statistic), "
                f"p = $P(T \\le {t_val:.2f}) = {cdf_val:.4f}$."
            )
    else:
        expl = (
            f"Lookup: $P(T \\le {t_val:.2f}) = {cdf_val:.4f}$.\n\n"
            f"For a **two-tailed** test, p = $2 \\times P(T \\ge |{t_val:.2f}|)$. "
            f"If $t_{{calc}} = {t_val:.2f}$, then using the CDF: p = $2 \\times \\text{{min}}({cdf_val:.4f}, {1-cdf_val:.4f}) = {p_calc:.4f}$."
        )

    st.write(expl)

    st.markdown(
        "**APA interpretation** \n"
        f"Calculated statistic: *$t$*({df})={t_val:.2f}, *$p$={p_calc:.3f}.  \n"
        f"Critical statistic for $\\alpha={alpha:.3f}$ ({tail}): $t_{{crit}}$={crit:.2f}.  \n" # Displaying the relevant crit
        f"Comparison of statistics $\\rightarrow$ H0 **{decision}** ({reason_stats}).  \n"
        f"Comparison of *$p$*-values $\\rightarrow$ H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *$t$*({df}) = {t_val:.2f}, *$p$ = {p_calc:.3f} "
        f"({tail}). The null hypothesis was **{decision}** at $\\alpha$ = {alpha:.2f}."
    )


def tab_t():
    st.subheader("Tab 1 • t-Distribution")

    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("t statistic", value=2.10, step=0.01, key="t_val")
        df = st.number_input("df", min_value=1, value=10, step=1, key="t_df")
    with c2:
        alpha = st.number_input("$\\alpha$", value=0.05, step=0.0001, # finer step for Bonferroni
                                 min_value=0.00001, max_value=0.99, format="%.4f", key="t_alpha")
        tail = st.radio("Tail", ["one-tailed", "two-tailed"], key="t_tail")

    # Always display plot and table, remove button
    st.pyplot(plot_t(t_val, float(df), float(alpha), tail)) # Ensure inputs are float where needed

    st.write("**t-table** (single highlight)")
    # Table & note in left column, explanation in right column
    ctable, cexp = st.columns([2,1])
    with ctable:
        t_table(int(df), float(alpha), tail) # df must be int for range, alpha float
        show_cumulative_note()
    with cexp:
        st.subheader("P-value Calculation Explanation")
        t_apa(float(t_val), int(df), float(alpha), tail)


###############################################################################
#                             TAB 2: z-Distribution
###############################################################################

def plot_z(z_calc, alpha, tail):
    """
    For one-tailed, if z_calc>=0, shade the right tail; if negative, shade left tail.
    For two-tailed, shade both tails.
    """
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    xs = np.linspace(-4,4,400)
    ys = stats.norm.pdf(xs)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H0")

    placed = []

    if tail.startswith("one"):
        crit_pos = stats.norm.ppf(1 - alpha)
        crit_neg = -crit_pos # This is stats.norm.ppf(alpha)
        if z_calc >= 0: # Assuming right-tailed test for positive z_calc
            ax.fill_between(xs[xs>=crit_pos], ys[xs>=crit_pos],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_pos, color="green", ls="--")
            place_label(ax, placed, crit_pos, stats.norm.pdf(crit_pos)+0.02,
                        f"$z_{{crit}}$={crit_pos:.2f}", color="green")
        else: # Assuming left-tailed test for negative z_calc
            ax.fill_between(xs[xs<=crit_neg], ys[xs<=crit_neg], # crit_neg is negative
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_neg, color="green", ls="--")
            place_label(ax, placed, crit_neg, stats.norm.pdf(crit_neg)+0.02,
                        f"$z_{{crit}}$={crit_neg:.2f}", color="green")
    else: # two-tailed
        crit = stats.norm.ppf(1 - alpha/2) # Positive critical value
        ax.fill_between(xs[xs>=crit], ys[xs>=crit], color="red", alpha=0.3)
        ax.fill_between(xs[xs<=-crit], ys[xs<=-crit], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(-crit, color="green", ls="--")
        place_label(ax, placed, crit, stats.norm.pdf(crit)+0.02,
                    f"+$z_{{crit}}$={crit:.2f}", color="green")
        place_label(ax, placed, -crit, stats.norm.pdf(-crit)+0.02,
                    f"–$z_{{crit}}$={crit:.2f}", color="green")

    ax.axvline(z_calc, color="blue", ls="--")
    place_label(ax, placed, z_calc, stats.norm.pdf(z_calc)+0.02,
                f"$z_{{calc}}$={z_calc:.2f}", color="blue")

    ax.set_xlabel("z")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("z-Distribution")
    fig.tight_layout()
    return fig


def build_z_html(z_val: float, alpha: float, tail: str) -> str: # alpha, tail not used for table structure
    """
    Single-step highlight for z-table based on z_val.
    """
    z_val_clipped = np.clip(z_val, -3.499, 3.499) # Clip to avoid issues at exact boundaries like 3.5
    
    # Determine row: z score to one decimal place (e.g., 1.6 for z=1.64)
    row_label_val = np.round(np.sign(z_val_clipped) * np.floor(abs(z_val_clipped)*10)/10, 1)
    
    # Determine column: the hundredths digit (e.g., 0.04 for z=1.64)
    # Handle potential floating point inaccuracies carefully
    col_label_val = np.round(abs(z_val_clipped) - abs(row_label_val), 2)
    if z_val_clipped < 0 and abs(row_label_val) == 0 : # e.g. z = -0.03 -> row_label_val = 0.0, col_label_val = 0.03 (needs to be positive)
         pass # col_label_val is already positive
    elif z_val_clipped < 0 and row_label_val == 0.0 and z_val_clipped != 0.0: # e.g. z = -0.03, row_label_val is 0.0
        col_label_val = np.round(abs(z_val_clipped),2)


    Rows_display = np.round(np.arange(-3.4, 3.5, 0.1), 1)
    Cols_display = np.round(np.arange(0.00, 0.10, 0.01), 2)
    
    # Find closest row_label_val in Rows_display
    row_label_val = min(Rows_display, key=lambda r_disp: abs(r_disp - row_label_val))

    # Find closest col_label_val in Cols_display
    col_label_val = min(Cols_display, key=lambda c_disp: abs(c_disp - col_label_val))


    # Determine which rows to show in the table view
    try:
        irow_center = list(Rows_display).index(row_label_val)
    except ValueError: # Should not happen if row_label_val is from min()
        irow_center = len(Rows_display) // 2

    show_rows_values = Rows_display[max(0, irow_center - 10): min(len(Rows_display),irow_center + 11)]

    head = "".join(f"<th>{c:.2f}</th>" for c in Cols_display)
    body = ""
    for rr_val in show_rows_values:
        # Row header (z to one decimal)
        row_html_cells = f'<td id="z_{rr_val:.1f}_0">{rr_val:.1f}</td>'
        for cc_val in Cols_display:
            # Calculate z for the cell: row_label + col_label
            # For negative z-scores in standard tables, the row is negative (e.g., -1.6)
            # and you add the positive hundredths column (e.g., 0.04)
            # So, if rr_val is -1.6 and cc_val is 0.04, effective z is -1.6 + 0.04 = -1.56.
            # This is different from how positive z-scores usually sum (1.6 + 0.04 = 1.64)
            # Standard z-tables usually list |z| for negative values or provide separate tables.
            # The provided code calculates stats.norm.cdf(rr_val + cc_val), which means for rr_val = -1.6:
            # -1.6 + 0.00 = -1.60
            # -1.6 + 0.01 = -1.59
            # -1.6 + 0.09 = -1.51  -- This is how many "body of Z" tables work.
            # Let's assume this interpretation: the cell value is for z = rr_val + cc_val.
            current_z_for_cell = rr_val + cc_val
            cdf_val_cell = stats.norm.cdf(current_z_for_cell)
            row_html_cells += f'<td id="z_{rr_val:.1f}_{cc_val:.2f}">{cdf_val_cell:.4f}</td>'
        body += f"<tr>{row_html_cells}</tr>"

    table_code = f"<tr><th>z</th>{head}</tr>{body}" # Changed z.x to z
    html = wrap_table(CSS_BASE, table_code)

    # Highlight row corresponding to row_label_val
    if row_label_val in show_rows_values:
        for cc_val in Cols_display:
            html = style_cell(html, f"z_{row_label_val:.1f}_{cc_val:.2f}")
        html = style_cell(html, f"z_{row_label_val:.1f}_0") # Highlight the row header cell

    # Highlight column corresponding to col_label_val
    for rr_val_in_shown_rows in show_rows_values:
        html = style_cell(html, f"z_{rr_val_in_shown_rows:.1f}_{col_label_val:.2f}")

    # Intersection in blue
    if row_label_val in show_rows_values:
        html = style_cell(html, f"z_{row_label_val:.1f}_{col_label_val:.2f}", color="blue", px=3)
    
    return html


def z_table(z_val: float, alpha: float, tail: str):
    code = build_z_html(z_val, alpha, tail)
    st.markdown(container(code), unsafe_allow_html=True)


def z_apa(z_val: float, alpha: float, tail: str):
    """
    Show the dynamic explanation and final APA lines for the z-distribution.
    """
    # Actual p
    if tail.startswith("one"):
        p_calc = stats.norm.sf(abs(z_val)) # Survival function gives P(Z > |z_val|)
        # Determine critical value based on direction if it were a directional test
        if z_val >=0: # Assumed right-tailed
            crit = stats.norm.ppf(1 - alpha)
            reject = (z_val > crit)
        else: # Assumed left-tailed
            crit = stats.norm.ppf(alpha) # Negative critical value
            reject = (z_val < crit)
    else: # two-tailed
        p_calc = stats.norm.sf(abs(z_val))*2
        crit = stats.norm.ppf(1 - alpha/2) # Positive critical value
        reject = (abs(z_val) > crit)


    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = f"because $|z_{{calc}}|$ ({abs(z_val):.2f}) exceeded $|z_{{crit}}|$ ({abs(crit):.2f})" if tail.startswith("two") else f"because $z_{{calc}}$ ({z_val:.2f}) was in the rejection region defined by $z_{{crit}}$ ({crit:.2f})"
        reason_p = f"because p ({p_calc:.3f}) < $\\alpha$ ({alpha:.3f})"
    else:
        reason_stats = f"because $|z_{{calc}}|$ ({abs(z_val):.2f}) did not exceed $|z_{{crit}}|$ ({abs(crit):.2f})" if tail.startswith("two") else f"because $z_{{calc}}$ ({z_val:.2f}) was not in the rejection region defined by $z_{{crit}}$ ({crit:.2f})"
        reason_p = f"because p ({p_calc:.3f}) $\\ge \\alpha$ ({alpha:.3f})"


    table_val_cdf = stats.norm.cdf(z_val) # CDF P(Z <= z_val)
    # explanation
    if tail.startswith("one"):
        if z_val >= 0: # Right-tailed test assumed
            expl = (
                f"Lookup: $P(Z \\le {z_val:.2f}) = {table_val_cdf:.4f}$.\n\n"
                f"For a **one-tailed** test (right tail), p = $1 - P(Z \\le {z_val:.2f}) = 1 - {table_val_cdf:.4f} = {1-table_val_cdf:.4f}$."
            )
        else: # Left-tailed test assumed
            expl = (
                f"Lookup: $P(Z \\le {z_val:.2f}) = {table_val_cdf:.4f}$.\n\n"
                f"For a **one-tailed** test (left tail), p = $P(Z \\le {z_val:.2f}) = {table_val_cdf:.4f}$."
            )
    else: # two-tailed
        expl = (
            f"Lookup: $P(Z \\le {z_val:.2f}) = {table_val_cdf:.4f}$.\n\n"
            f"For a **two-tailed** test, p = $2 \\times P(Z \\ge |{z_val:.2f}|)$.\n"
            f"Using CDF: p = $2 \\times \\text{{min}}({table_val_cdf:.4f}, {1 - table_val_cdf:.4f}) = {p_calc:.4f}$."
        )

    st.write(expl)

    st.markdown(
        "**APA interpretation** \n"
        f"Calculated statistic: *$z$*={z_val:.2f}, *$p$={p_calc:.3f}.  \n"
        f"Critical statistic for $\\alpha={alpha:.3f}$ ({tail}): $z_{{crit}}$={crit:.2f}.  \n"
        f"Statistic comparison $\\rightarrow$ H₀ **{decision}** ({reason_stats}).  \n"
        f"*$p$* comparison $\\rightarrow$ H₀ **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *$z$*={z_val:.2f}, *$p$={p_calc:.3f} ({tail}). "
        f"The null hypothesis was **{decision}** at $\\alpha$={alpha:.2f}."
    )


def tab_z():
    st.subheader("Tab 2 • z-Distribution")

    c1, c2 = st.columns(2)
    with c1:
        z_val = st.number_input("z statistic", value=1.64, step=0.01, key="z_val")
    with c2:
        alpha = st.number_input("$\\alpha$", value=0.05, step=0.0001,
                                 min_value=0.00001, max_value=0.99, format="%.4f", key="z_alpha")
        tail = st.radio("Tail", ["one-tailed", "two-tailed"], key="z_tail")
    
    st.pyplot(plot_z(float(z_val), float(alpha), tail))

    st.write("**z-table** (single highlight)")
    ctable, cexp = st.columns([2,1])
    with ctable:
        z_table(float(z_val), float(alpha), tail)
        show_cumulative_note()
    with cexp:
        st.subheader("P-value Calculation Explanation")
        z_apa(float(z_val), float(alpha), tail)


###############################################################################
#                       TAB 3: F-Distribution (unchanged)
###############################################################################

def plot_f(f_calc, df1, df2, alpha):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    # Ensure ppf has reasonable upper bound for plotting, prevent extreme df1/df2 issues
    ppf_upper_plot_limit = stats.f.ppf(0.999, df1, df2)
    if np.isnan(ppf_upper_plot_limit) or np.isinf(ppf_upper_plot_limit) or ppf_upper_plot_limit > 500: # Cap plotting range
        ppf_upper_plot_limit = max(f_calc * 1.5, 10) # Fallback plotting range based on f_calc or fixed
        
    xs = np.linspace(0, ppf_upper_plot_limit * 1.1, 400)
    ys = stats.f.pdf(xs, df1, df2)

    # Filter out NaNs or Infs from ys if pdf results in them for extreme values
    valid_indices = ~np.isnan(ys) & ~np.isinf(ys)
    xs, ys = xs[valid_indices], ys[valid_indices]
    if len(xs) < 2 : # Not enough points to plot
        st.warning("Could not generate a valid F-distribution plot for the given parameters.")
        return fig # return empty fig

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")

    crit = stats.f.ppf(1 - alpha, df1, df2)
    ax.fill_between(xs[xs>=crit], ys[xs>=crit], color="red", alpha=0.3,
                    label="Reject H0")
    ax.axvline(crit, color="green", ls="--")
    ax.axvline(f_calc, color="blue", ls="--")

    # Ensure PDF values are valid for placing labels
    pdf_at_crit = stats.f.pdf(crit, df1, df2)
    pdf_at_fcalc = stats.f.pdf(f_calc, df1, df2)

    if not (np.isnan(pdf_at_crit) or np.isinf(pdf_at_crit)):
        place_label(ax, [], crit, pdf_at_crit+0.02, # Check placed list
                    f"$F_{{crit}}$={crit:.2f}", color="green")
    if not (np.isnan(pdf_at_fcalc) or np.isinf(pdf_at_fcalc)):
        place_label(ax, [], f_calc, pdf_at_fcalc+0.02, # Check placed list
                    f"$F_{{calc}}$={f_calc:.2f}", color="blue")


    ax.set_xlabel("F")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"F-Distribution (df1={df1}, df2={df2})")
    fig.tight_layout()
    return fig


def build_f_table(df1: int, df2: int, alpha: float) -> str:
    # Define a smaller, more focused range for display
    row_start = max(1, df1 - 2)
    row_end = df1 + 3
    col_start = max(1, df2 - 2)
    col_end = df2 + 3
    
    rows_to_display = list(range(row_start, row_end + 1))
    cols_to_display = list(range(col_start, col_end + 1))

    # Ensure df1 and df2 are within the displayed range for highlighting
    # If not, they won't be highlighted, which is acceptable.
    
    col_idx_relative_to_cols_display = -1
    if df2 in cols_to_display:
        col_idx_relative_to_cols_display = cols_to_display.index(df2) + 1 # 1-based for cell ID

    head = "".join(f"<th>{c}</th>" for c in cols_to_display)
    body = ""
    for r_val in rows_to_display:
        # Row header is df1
        row_html_cells = f'<td id="f_{r_val}_0">{r_val}</td>' 
        for i_col_disp, c_val in enumerate(cols_to_display, start=1):
            # Calculate critical value for F(alpha, r_val, c_val)
            f_crit_cell = stats.f.ppf(1 - alpha, r_val, c_val)
            row_html_cells += f'<td id="f_{r_val}_{i_col_disp}">{f_crit_cell:.2f}</td>'
        body += f"<tr>{row_html_cells}</tr>"

    table_code = f"<tr><th>df1\\df2</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, table_code)

    # Highlight row for df1 if it's in the displayed rows
    if df1 in rows_to_display:
        for i_col_disp in range(len(cols_to_display) + 1): # +1 for df1 header col
            html = style_cell(html, f"f_{df1}_{i_col_disp}")
            
    # Highlight col for df2 if it's in the displayed cols
    if col_idx_relative_to_cols_display != -1: # df2 is in cols_to_display
        for r_val_in_rows_disp in rows_to_display:
            html = style_cell(html, f"f_{r_val_in_rows_disp}_{col_idx_relative_to_cols_display}")
            
    # Intersection
    if df1 in rows_to_display and col_idx_relative_to_cols_display != -1:
        html = style_cell(html, f"f_{df1}_{col_idx_relative_to_cols_display}", color="blue", px=3)
        
    return html


def f_table(df1: int, df2: int, alpha: float):
    code = build_f_table(df1, df2, alpha)
    st.markdown(container(code), unsafe_allow_html=True)


def f_apa(f_val: float, df1: int, df2: int, alpha: float):
    p_calc = stats.f.sf(f_val, df1, df2)
    crit = stats.f.ppf(1 - alpha, df1, df2)
    reject = (f_val>crit)
    decision = "rejected" if reject else "failed to reject"
    reason_stats = (f"because $F_{{calc}}$ ({f_val:.2f}) exceeded $F_{{crit}}$ ({crit:.2f})"
                    if reject else f"because $F_{{calc}}$ ({f_val:.2f}) did not exceed $F_{{crit}}$ ({crit:.2f})")
    reason_p = (f"because p ({p_calc:.3f}) < $\\alpha$ ({alpha:.3f})" if reject else f"because p ({p_calc:.3f}) $\\ge \\alpha$ ({alpha:.3f})")

    st.markdown(
        "**APA interpretation** \n"
        f"Calculated statistic: *$F$*({df1},{df2})={f_val:.2f}, *$p$={p_calc:.3f}.  \n"
        f"Critical statistic for $\\alpha={alpha:.3f}$: $F_{{crit}}$({df1},{df2})={crit:.2f}.  \n"
        f"Statistic comparison $\\rightarrow$ H0 **{decision}** ({reason_stats}).  \n"
        f"*$p$* comparison $\\rightarrow$ H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *$F$*({df1},{df2})={f_val:.2f}, *$p$={p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at $\\alpha$={alpha:.2f}."
    )


def tab_f():
    st.subheader("Tab 3 • F-Distribution")

    c1, c2 = st.columns(2)
    with c1:
        f_val = st.number_input("F statistic", value=4.32, step=0.01, key="f_val")
        df1 = st.number_input("df1 (numerator)", min_value=1, value=5, step=1, key="f_df1")
    with c2:
        df2 = st.number_input("df2 (denominator)", min_value=1, value=20, step=1, key="f_df2")
        alpha = st.number_input("$\\alpha$", value=0.05, step=0.0001,
                                 min_value=0.00001, max_value=0.99, format="%.4f", key="f_alpha")

    st.pyplot(plot_f(float(f_val), int(df1), int(df2), float(alpha)))

    st.write("**F-table** (always one-tailed, values are $F_{crit}$ for given $\\alpha$)")
    f_table(int(df1), int(df2), float(alpha))
    f_apa(float(f_val), int(df1), int(df2), float(alpha))


###############################################################################
#                      TAB 4: Chi-Square (unchanged from original, uses selectbox)
###############################################################################

def plot_chi(chi_calc, df, alpha):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    
    ppf_upper_plot_limit = stats.chi2.ppf(0.999, df)
    if np.isnan(ppf_upper_plot_limit) or np.isinf(ppf_upper_plot_limit) or ppf_upper_plot_limit > 500:
        ppf_upper_plot_limit = max(chi_calc * 1.5, 10) 
        
    xs = np.linspace(0, ppf_upper_plot_limit*1.1, 400)
    ys = stats.chi2.pdf(xs, df)

    valid_indices = ~np.isnan(ys) & ~np.isinf(ys)
    xs, ys = xs[valid_indices], ys[valid_indices]
    if len(xs) < 2 :
        st.warning("Could not generate a valid Chi-square plot for the given parameters.")
        return fig 

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")

    crit = stats.chi2.ppf(1 - alpha, df)
    ax.fill_between(xs[xs>=crit], ys[xs>=crit], color="red", alpha=0.3,
                    label="Reject H0")
    ax.axvline(crit, color="green", ls="--")
    ax.axvline(chi_calc, color="blue", ls="--")
    
    pdf_at_crit = stats.chi2.pdf(crit, df)
    pdf_at_chicalc = stats.chi2.pdf(chi_calc, df)

    placed_chi = [] # Initialize placed_labels for this plot
    if not (np.isnan(pdf_at_crit) or np.isinf(pdf_at_crit)):
        place_label(ax, placed_chi, crit, pdf_at_crit+0.02,
                    f"$\\chi^2_{{crit}}$={crit:.2f}", color="green")
    if not (np.isnan(pdf_at_chicalc) or np.isinf(pdf_at_chicalc)):
        place_label(ax, placed_chi, chi_calc, pdf_at_chicalc+0.02,
                    f"$\\chi^2_{{calc}}$={chi_calc:.2f}", color="blue")


    ax.set_xlabel("$\\chi^2$")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"$\\chi^2$-Distribution (df={df})")
    fig.tight_layout()
    return fig

def build_chi_table(df: int, alpha: float) -> str: # alpha here comes from selectbox
    rows_to_display = list(range(max(1,df-2), df+3+1)) # More focused range
    defined_alphas = [0.10,0.05,0.01,0.001] # These are the P-values for chi-square (upper tail)

    # Alpha from selectbox is guaranteed to be in defined_alphas
    try:
        col_idx = defined_alphas.index(alpha)+1 # 1-based
    except ValueError: # Should not happen with selectbox
        st.error("Selected alpha not in defined list for Chi-Square table. Defaulting.")
        col_idx = defined_alphas.index(0.05)+1 if 0.05 in defined_alphas else 1


    head = "".join(f"<th>{a_val}</th>" for a_val in defined_alphas)
    body = ""
    for r_val in rows_to_display:
        row_html_cells = f'<td id="chi_{r_val}_0">{r_val}</td>'
        for i_alpha_col, current_alpha_col_val in enumerate(defined_alphas, start=1):
            val_cell = stats.chi2.ppf(1 - current_alpha_col_val, r_val)
            row_html_cells += f'<td id="chi_{r_val}_{i_alpha_col}">{val_cell:.2f}</td>'
        body += f"<tr>{row_html_cells}</tr>"

    table_code = f"<tr><th>df\\$\\alpha$</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, table_code)

    # Highlight row, col, intersection
    if df in rows_to_display:
        for i_alpha_col in range(len(defined_alphas)+1): # +1 for df col
            html = style_cell(html, f"chi_{df}_{i_alpha_col}")
            
    for rr_val_disp in rows_to_display:
        html = style_cell(html, f"chi_{rr_val_disp}_{col_idx}")
        
    if df in rows_to_display :
        html = style_cell(html, f"chi_{df}_{col_idx}", color="blue", px=3)
    return html

def chi_table(df: int, alpha: float):
    code = build_chi_table(df, alpha)
    st.markdown(container(code), unsafe_allow_html=True)

def chi_apa(chi_val: float, df: int, alpha: float):
    p_calc = stats.chi2.sf(chi_val, df)
    crit = stats.chi2.ppf(1 - alpha, df)
    reject = (chi_val>crit)
    decision = "rejected" if reject else "failed to reject"
    reason_stats = f"because $\\chi^2_{{calc}}$ ({chi_val:.2f}) exceeded $\\chi^2_{{crit}}$ ({crit:.2f})" if reject else f"because $\\chi^2_{{calc}}$ ({chi_val:.2f}) did not exceed $\\chi^2_{{crit}}$ ({crit:.2f})"
    reason_p = f"because p ({p_calc:.3f}) < $\\alpha$ ({alpha:.3f})" if reject else f"because p ({p_calc:.3f}) $\\ge \\alpha$ ({alpha:.3f})"

    st.markdown(
        "**APA interpretation** \n"
        f"Calculated statistic: $\\chi^2$({df})={chi_val:.2f}, *$p$={p_calc:.3f}.  \n"
        f"Critical statistic for $\\alpha={alpha:.3f}$: $\\chi^2_{{crit}}$({df})={crit:.2f}.  \n"
        f"Statistic comparison $\\rightarrow$ H0 **{decision}** ({reason_stats}).  \n"
        f"*$p$* comparison $\\rightarrow$ H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** $\\chi^2$({df})={chi_val:.2f}, *$p$={p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at $\\alpha$={alpha:.2f}."
    )


def tab_chi():
    st.subheader("Tab 4 • Chi-Square")

    c1, c2 = st.columns(2)
    with c1:
        chi_val = st.number_input("$\\chi^2$ statistic", value=7.88, step=0.01, key="chi_val")
        df = st.number_input("df ", min_value=1, value=3, step=1, key="chi_df") # Added space to df key
    with c2:
        # Alpha is a selectbox as per original code for chi-square
        # If this were a number_input, build_chi_table would need modification like build_t_html
        alpha = st.selectbox("$\\alpha$", [0.10,0.05,0.01,0.001],
                             index=1, key="chi_alpha")

    st.pyplot(plot_chi(float(chi_val), int(df), float(alpha)))

    st.write("**$\\chi^2$-table** (values are $\\chi^2_{crit}$ for given $\\alpha$)")
    chi_table(int(df), float(alpha))
    chi_apa(float(chi_val), int(df), float(alpha))


###############################################################################
#                       TAB 5: Mann-Whitney U (updated)
###############################################################################

def plot_u(u_calc, n1, n2, alpha, tail):
    """
    Normal approx. We'll interpret sign around the midpoint (mu) to decide
    shading for one-tailed. For two-tailed, shade both tails.
    """
    mu_u = n1*n2/2
    sigma_u = np.sqrt(n1*n2*(n1+n2+1)/12)

    if sigma_u == 0: # Avoid division by zero if n1 or n2 is too small (e.g. 0 or 1)
        st.warning("Cannot plot Mann-Whitney U: Standard deviation is zero (n1 or n2 may be too small).")
        fig, ax = plt.subplots(figsize=(12,4), dpi=100) # Empty plot
        ax.text(0.5, 0.5, "Cannot generate plot for these n1/n2 values.", ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    # Define plot range based on mu and sigma, ensuring it's reasonable
    plot_range_min = max(0, mu_u - 4 * sigma_u) # U cannot be negative
    plot_range_max = min(n1*n2, mu_u + 4 * sigma_u) # U max is n1*n2
    if plot_range_min >= plot_range_max: # if sigma is very small or mu is at boundary
        plot_range_min = max(0, u_calc - 2)
        plot_range_max = u_calc + 2

    xs = np.linspace(plot_range_min, plot_range_max, 400)
    ys = stats.norm.pdf(xs, mu_u, sigma_u)
    
    valid_indices = ~np.isnan(ys) & ~np.isinf(ys) # Filter non-finite values
    xs, ys = xs[valid_indices], ys[valid_indices]
    if len(xs) < 2:
        st.warning("Could not generate a valid Mann-Whitney U plot for the given parameters.")
        return fig

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H0")

    from math import floor, ceil

    # Critical U values from normal approximation
    # For U-test, small U values are typically significant.
    if tail.startswith("one"):
        # One-tailed: depends on direction. Typically U_calc is compared to U_crit_lower.
        # If U_calc corresponds to U_L (smaller ranks in group 1), critical region is U <= U_crit.
        # If U_calc corresponds to U_L' (larger ranks in group 1, so U_L is small), critical is U <= U_crit.
        # Let's assume we always calculate min(U1, U2) as U_calc and test if U_calc <= U_crit_lower.
        z_crit_one_tail = stats.norm.ppf(alpha) # For left tail (U_calc <= U_crit_lower)
        ucrit_lower_approx = mu_u + z_crit_one_tail * sigma_u
        # U values are integers. Often tables give U_crit such that P(U <= U_crit) <= alpha.
        # Using floor for critical value if we test U_calc <= U_crit_lower
        # Or ceil if U_calc >= U_crit_upper
        
        # Plotting decision based on u_calc relative to mean
        if u_calc < mu_u: # Testing if U is significantly small
            crit_val_plot = floor(ucrit_lower_approx)
            ax.fill_between(xs[xs<=crit_val_plot], ys[xs<=crit_val_plot], color="red", alpha=0.3,
                            label="Reject H0 (U small)")
            ax.axvline(crit_val_plot, color="green", ls="--")
            place_label(ax, [], crit_val_plot, stats.norm.pdf(crit_val_plot, mu_u, sigma_u)+0.002, # adjusted y offset
                        f"$U_{{critL}}$={crit_val_plot}", color="green")
        else: # Testing if U is significantly large (u_calc is n1n2 - U_small)
            z_crit_one_tail_upper = stats.norm.ppf(1-alpha)
            ucrit_upper_approx = mu_u + z_crit_one_tail_upper * sigma_u
            crit_val_plot = ceil(ucrit_upper_approx)
            ax.fill_between(xs[xs>=crit_val_plot], ys[xs>=crit_val_plot], color="red", alpha=0.3,
                            label="Reject H0 (U large)")
            ax.axvline(crit_val_plot, color="green", ls="--")
            place_label(ax, [], crit_val_plot, stats.norm.pdf(crit_val_plot, mu_u, sigma_u)+0.002,
                        f"$U_{{critU}}$={crit_val_plot}", color="green")
    else: # two-tailed
        z_crit_two_tail = stats.norm.ppf(alpha/2) # For lower tail
        ucrit_lower_approx = mu_u + z_crit_two_tail * sigma_u
        ucrit_upper_approx = mu_u - z_crit_two_tail * sigma_u # Symmetrical: mu + (-z_crit_two_tail)*sigma

        crit_val_lower_plot = floor(ucrit_lower_approx)
        crit_val_upper_plot = ceil(ucrit_upper_approx)

        ax.fill_between(xs[xs<=crit_val_lower_plot], ys[xs<=crit_val_lower_plot], color="red", alpha=0.3)
        ax.fill_between(xs[xs>=crit_val_upper_plot], ys[xs>=crit_val_upper_plot], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit_val_lower_plot, color="green", ls="--")
        ax.axvline(crit_val_upper_plot, color="green", ls="--")
        place_label(ax, [], crit_val_lower_plot, stats.norm.pdf(crit_val_lower_plot, mu_u, sigma_u)+0.002,
                    f"$U_{{critL}}$={crit_val_lower_plot}", color="green")
        place_label(ax, [], crit_val_upper_plot, stats.norm.pdf(crit_val_upper_plot, mu_u, sigma_u)+0.002,
                    f"$U_{{critU}}$={crit_val_upper_plot}", color="green")


    ax.axvline(u_calc, color="blue", ls="--")
    place_label(ax, [], u_calc, stats.norm.pdf(u_calc, mu_u, sigma_u)+0.002,
                f"$U_{{calc}}$={u_calc}", color="blue")

    ax.set_xlabel("U statistic (Normal Approx.)")
    ax.set_ylabel("Approx. density")
    ax.legend()
    ax.set_title("Mann-Whitney U Distribution (Normal Approximation)")
    fig.tight_layout()
    return fig


def u_crit(n1:int, n2:int, alpha:float, tail:str, return_both_for_two_tailed=False)->any:
    # Returns approximate critical U value(s) using normal approximation
    # For one-tailed, returns U_crit_lower (reject if U_calc <= U_crit_lower)
    # For two-tailed, if return_both_for_two_tailed, returns (U_crit_lower, U_crit_upper)
    # else returns U_crit_lower for two-tailed (common table format)
    
    mu_u = n1*n2/2
    sigma_u = np.sqrt(n1*n2*(n1+n2+1)/12)
    from math import floor, ceil

    if sigma_u == 0: return np.nan # Cannot calculate

    if tail.startswith("one"):
        # Standard U-test: reject for small U (left tail)
        z_critical = stats.norm.ppf(alpha) 
        u_critical_approx = mu_u + z_critical * sigma_u
        return floor(u_critical_approx) # U values are integers
    else: # two-tailed
        z_critical_lower = stats.norm.ppf(alpha/2)
        u_critical_lower_approx = mu_u + z_critical_lower * sigma_u
        
        if return_both_for_two_tailed:
            # z_critical_upper = stats.norm.ppf(1 - alpha/2) # or -z_critical_lower
            u_critical_upper_approx = mu_u - z_critical_lower * sigma_u # mu + (-z_lower)*sigma
            return floor(u_critical_lower_approx), ceil(u_critical_upper_approx)
        else: # Common table lookup is for the lower critical value
            return floor(u_critical_lower_approx)


def build_u_table(n1:int, n2:int, alpha:float, tail:str)->str:
    # U-tables often show critical values for min(n1, n2) and max(n1, n2)
    # This table will be for n1 (rows) vs n2 (cols)
    # Values in table are U_crit_lower (reject if U_obs <= U_crit_lower)
    # For two-tailed, alpha is split. For one-tailed, use full alpha.
    
    row_start = max(2, n1 - 2) # n needs to be at least 2 for U-test usually
    row_end = n1 + 3
    col_start = max(2, n2 - 2)
    col_end = n2 + 3

    rows_to_display = list(range(row_start, row_end + 1))
    cols_to_display = list(range(col_start, col_end + 1))
    
    # Determine if current n2 is in the displayed columns for highlighting
    col_idx_relative_to_cols_display = -1
    if n2 in cols_to_display:
        col_idx_relative_to_cols_display = cols_to_display.index(n2) + 1

    head = "".join(f"<th>n2={c}</th>" for c in cols_to_display)
    body = ""
    for r_val_n1 in rows_to_display:
        row_html_cells = f'<td id="u_{r_val_n1}_0">n1={r_val_n1}</td>'
        for i_col_disp, c_val_n2 in enumerate(cols_to_display, start=1):
            # For table display, usually the lower critical U is shown.
            # For one-tailed, alpha is used directly. For two-tailed, alpha/2 is used for each tail.
            # The u_crit function handles this logic for alpha based on `tail` for lower crit.
            # However, standard U-tables are often for a specific alpha (e.g. 0.05 two-tailed).
            # Here, we're dynamic.
            # Let's assume table values are U_crit such that P(U <= U_crit) is approx. alpha (one-tail) or alpha/2 (two-tail)
            
            # If table alpha is fixed, e.g. 0.05 for one-tailed test:
            # val_cell = u_crit(r_val_n1, c_val_n2, 0.05, "one-tailed") 
            # But here alpha is from user input.
            # The u_crit function by default returns the lower critical value.
            val_cell = u_crit(r_val_n1, c_val_n2, alpha, tail)
            
            row_html_cells += f'<td id="u_{r_val_n1}_{i_col_disp}">{val_cell if not np.isnan(val_cell) else "N/A"}</td>'
        body += f"<tr>{row_html_cells}</tr>"

    table_code = f"<tr><th>n1\\n2</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, table_code)

    # Highlight row for n1
    if n1 in rows_to_display:
        for i_col_disp in range(len(cols_to_display) + 1):
            html = style_cell(html, f"u_{n1}_{i_col_disp}")
            
    # Highlight col for n2
    if col_idx_relative_to_cols_display != -1:
        for r_val_n1_disp in rows_to_display:
            html = style_cell(html, f"u_{r_val_n1_disp}_{col_idx_relative_to_cols_display}")
            
    # Intersection
    if n1 in rows_to_display and col_idx_relative_to_cols_display != -1:
        html = style_cell(html, f"u_{n1}_{col_idx_relative_to_cols_display}", color="blue", px=3)
        
    return html


def u_table(n1:int, n2:int, alpha:float, tail:str):
    code = build_u_table(n1,n2,alpha,tail)
    st.markdown(container(code, height=300), unsafe_allow_html=True) # Shorter height for U table


def u_apa(u_val: int, n1: int, n2: int, alpha: float, tail: str):
    """
    Show dynamic explanation and final APA lines for Mann-Whitney U.
    Assumes u_val is the smaller of U1 and U2 if calculated manually,
    or the direct output from scipy.stats.mannwhitneyu(..., alternative=...).
    """
    mu_u = n1 * n2 / 2.0
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)

    if sigma_u == 0:
        st.error("Cannot perform APA interpretation for Mann-Whitney U: standard deviation is zero.")
        return

    # Z transformation for u_val
    # Apply continuity correction: +0.5 if u_val < mu_u, -0.5 if u_val > mu_u
    if u_val < mu_u:
        z_calc = (u_val - mu_u + 0.5) / sigma_u
    elif u_val > mu_u:
        z_calc = (u_val - mu_u - 0.5) / sigma_u
    else: # u_val == mu_u
        z_calc = 0.0


    # P-value calculation from this z_calc
    if tail.startswith("one"):
        # If original u_val was expected to be small (left tail of U dist)
        # then small z_calc (large negative) is significant.
        # If original u_val was expected to be large (right tail of U dist, meaning U_other is small)
        # then large z_calc (large positive) is significant.
        # Let's assume the 'alternative' in mannwhitneyu would handle this.
        # For simplicity with z_calc from U:
        if u_val <= mu_u: # Corresponds to left-tail or smaller U
            p_calc_from_z = stats.norm.cdf(z_calc)
        else: # Corresponds to right-tail or larger U (U' would be small)
            p_calc_from_z = stats.norm.sf(z_calc) # 1 - cdf
    else: # two-tailed
        p_calc_from_z = 2 * stats.norm.sf(abs(z_calc))

    # Critical U values (lower U_L, upper U_U)
    crit_l, crit_u = None, None
    if tail.startswith("one"):
        # Typically, for U-test, reject if U_calc <= U_L (lower critical) for left-tail
        # or U_calc >= U_U (upper critical) for right-tail.
        # Our u_crit returns U_L by default for one-tailed.
        crit_l = u_crit(n1, n2, alpha, "one-tailed") 
        # We need to determine if it's a "less" or "greater" hypothesis for U_calc
        # For simplicity, if u_val is small, compare to U_L. If large, compare to U_U.
        if u_val <= mu_u: # Test "less"
            reject = (u_val <= crit_l) if crit_l is not None and not np.isnan(crit_l) else False
            crit_display = crit_l
        else: # Test "greater"
            # Need U_U which is n1*n2 - U_L(for alpha)
            # This is tricky with direct u_crit.
            # Alternative: get Z_crit and compare Z_calc.
            z_crit_apa = stats.norm.ppf(1-alpha) if u_val > mu_u else stats.norm.ppf(alpha)
            reject = (z_calc > z_crit_apa) if u_val > mu_u else (z_calc < z_crit_apa)
            crit_display = f"Z_crit={z_crit_apa:.2f}"


    else: # two-tailed
        crit_l, crit_u = u_crit(n1, n2, alpha, "two-tailed", return_both_for_two_tailed=True)
        reject = ((u_val <= crit_l) or (u_val >= crit_u)) \
            if crit_l is not None and crit_u is not None and not (np.isnan(crit_l) or np.isnan(crit_u)) \
            else False
        crit_display = f"({crit_l}, {crit_u})"


    decision = "rejected" if reject else "failed to reject"
    reason_p = f"because approximate p-value from Z ({p_calc_from_z:.3f}) {'<' if reject else '≥'} $\\alpha$ ({alpha:.3f})"
    reason_stats = f"because $U_{{calc}}$ ({u_val}) fell {'in' if reject else 'outside'} the rejection region defined by $U_{{crit}}$ ~{crit_display}"


    # Table-based explanation using normal approx's CDF for Z_calc
    normal_cdf_for_z_calc = stats.norm.cdf(z_calc)
    
    if tail.startswith("one"):
        if u_val <= mu_u: # Assumed "less than" alternative for U
            expl = (
                f"$U_{{calc}} = {u_val}$ (n1={n1}, n2={n2}). Mean $\\mu_U = {mu_u:.2f}$. $Z_{{calc}} \\approx {z_calc:.2f}$.\n\n"
                f"Approx. $P(Z \\le {z_calc:.2f}) = {normal_cdf_for_z_calc:.4f}$.\n"
                f"For a **one-tailed** test (expecting small U), p-value $\\approx {p_calc_from_z:.4f}$."
            )
        else: # Assumed "greater than" alternative for U (meaning U_other is small)
             expl = (
                f"$U_{{calc}} = {u_val}$ (n1={n1}, n2={n2}). Mean $\\mu_U = {mu_u:.2f}$. $Z_{{calc}} \\approx {z_calc:.2f}$.\n\n"
                f"Approx. $P(Z \\ge {z_calc:.2f}) = {1-normal_cdf_for_z_calc:.4f}$.\n"
                f"For a **one-tailed** test (expecting large U), p-value $\\approx {p_calc_from_z:.4f}$."
            )
    else: # two-tailed
        expl = (
            f"$U_{{calc}} = {u_val}$ (n1={n1}, n2={n2}). Mean $\\mu_U = {mu_u:.2f}$. $Z_{{calc}} \\approx {z_calc:.2f}$.\n\n"
            f"Approx. $P(Z \\le {z_calc:.2f}) = {normal_cdf_for_z_calc:.4f}$.\n"
            f"For a **two-tailed** test, p-value $\\approx 2 \\times P(Z \\ge |{z_calc:.2f}|) = {p_calc_from_z:.4f}$."
        )
    st.write(expl)

    st.markdown(
        "**APA interpretation (using Normal Approx.)** \n"
        f"Calculated statistic: U={u_val} (Z $\\approx$ {z_calc:.2f}), approx. *$p$={p_calc_from_z:.3f}.  \n"
        f"Critical U value(s) for $\\alpha$={alpha:.3f} ({tail}): $U_{{crit}} \\approx$ {crit_display}.  \n"
        f"Statistic comparison $\\rightarrow$ H₀ **{decision}** ({reason_stats}).  \n"
        f"*$p$* comparison $\\rightarrow$ H₀ **{decision}** ({reason_p}).  \n"
        f"**APA 7 report (approx.):** U={u_val}, Z $\\approx$ {z_calc:.2f}, *$p$={p_calc_from_z:.3f} ({tail}). "
        f"The null hypothesis was **{decision}** at $\\alpha$={alpha:.2f}."
    )

# --- Stub for tab_u and onwards as they were not fully provided ---
def tab_u():
    st.subheader("Tab 5 • Mann-Whitney U")
    
    c1, c2 = st.columns(2)
    with c1:
        u_val_input = st.number_input("U statistic value", min_value=0, value=23, step=1, key="u_val_input")
        n1_input = st.number_input("n1 (sample size 1)", min_value=1, value=8, step=1, key="u_n1")
    with c2:
        n2_input = st.number_input("n2 (sample size 2)", min_value=1, value=10, step=1, key="u_n2")
        alpha_input_u = st.number_input("$\\alpha$ ", value=0.05, step=0.001, format="%.3f", key="u_alpha_input")
        tail_input_u = st.radio("Tail ", ["one-tailed", "two-tailed"], key="u_tail_input", help="For one-tailed, test assumes U_calc is appropriately chosen (e.g., min of U1,U2 for a 'less' alternative, or use scipy's alternative).")

    if n1_input > 0 and n2_input > 0: # Basic check
        st.pyplot(plot_u(int(u_val_input), int(n1_input), int(n2_input), float(alpha_input_u), tail_input_u))
        
        st.write("**U-table (Normal Approx. Critical Values)**")
        ctable_u, cexp_u = st.columns([2,1]) # Adjusted column ratio
        with ctable_u:
            u_table(int(n1_input), int(n2_input), float(alpha_input_u), tail_input_u)
            show_cumulative_note() # Note might need adjustment for U-test interpretation
        with cexp_u:
            st.subheader("P-value & APA (Normal Approx.)")
            u_apa(int(u_val_input), int(n1_input), int(n2_input), float(alpha_input_u), tail_input_u)
    else:
        st.warning("Please enter valid sample sizes n1 and n2 (must be > 0).")


def tab_wilcoxon_t():
    st.subheader("Tab 6 • Wilcoxon Signed-Rank T")
    st.write("Wilcoxon T functionality to be implemented.")
    st.info("Note: This tab is a placeholder.")

def tab_binomial():
    st.subheader("Tab 7 • Binomial")
    st.write("Binomial test functionality to be implemented.")
    st.info("Note: This tab is a placeholder.")


def main():
    st.set_page_config(layout="wide") # Use wider layout
    st.title("Oli's - Statistical Table Explorer")
    
    tab_names = [
        "t-Dist", "z-Dist", "F-Dist", "Chi-Square", 
        "Mann-Whitney U", "Wilcoxon T", "Binomial"
    ]
    
    # Create tabs using st.tabs
    selected_tab = st.sidebar.radio("Select Distribution:", tab_names)

    # Placeholder for tab content display based on selected_tab
    if selected_tab == "t-Dist":
        tab_t()
    elif selected_tab == "z-Dist":
        tab_z()
    elif selected_tab == "F-Dist":
        tab_f()
    elif selected_tab == "Chi-Square":
        tab_chi()
    elif selected_tab == "Mann-Whitney U":
        tab_u()
    elif selected_tab == "Wilcoxon T":
        tab_wilcoxon_t()
    elif selected_tab == "Binomial":
        tab_binomial()

if __name__ == '__main__':
    main()
