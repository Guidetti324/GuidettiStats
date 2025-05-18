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
        "$p=2 \\times (1−P(Z \\le |z|))$). The same logic applies for t-distributions. The table " # Using LaTeX for math
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
        crit_pos = stats.t.ppf(1 - alpha, df)
        crit_neg = -crit_pos

        if t_calc >= 0:
            ax.fill_between(xs[xs >= crit_pos], ys[xs >= crit_pos],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_pos, color="green", ls="--")
            place_label(ax, placed_labels, crit_pos, stats.t.pdf(crit_pos, df)+0.02,
                        f"tcrit={crit_pos:.2f}", color="green")
        else:
            ax.fill_between(xs[xs <= crit_neg], ys[xs <= crit_neg],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_neg, color="green", ls="--")
            place_label(ax, placed_labels, crit_neg, stats.t.pdf(crit_neg, df)+0.02,
                        f"tcrit={crit_neg:.2f}", color="green")
    else:
        crit = stats.t.ppf(1 - alpha/2, df)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit], color="red", alpha=0.3)
        ax.fill_between(xs[xs <= -crit], ys[xs <= -crit], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(-crit, color="green", ls="--")
        place_label(ax, placed_labels, crit, stats.t.pdf(crit, df)+0.02,
                    f"+tcrit={crit:.2f}", color="green")
        place_label(ax, placed_labels, -crit, stats.t.pdf(-crit, df)+0.02,
                    f"–tcrit={crit:.2f}", color="green") # Using en-dash as per original

    ax.axvline(t_calc, color="blue", ls="--")
    place_label(ax, placed_labels, t_calc, stats.t.pdf(t_calc, df)+0.02,
                f"tcalc={t_calc:.2f}", color="blue")

    ax.set_xlabel("t")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("t-Distribution")
    fig.tight_layout()
    return fig


# ===== MINIMALLY CORRECTED build_t_html FUNCTION =====
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
    best_match_info = None 
    min_diff = float('inf')

    # Find the best matching column
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
            elif current_diff == min_diff: # Tie-breaking for equal differences
                if best_match_info and h_alpha_val < best_match_info['alpha_val']: # Prefer smaller alpha in ties
                     best_match_info = {'index': i, 'alpha_val': h_alpha_val, 'is_exact': False}

    if col_idx == -1: 
        if best_match_info:
            col_idx = best_match_info['index']
            selected_alpha_for_table = best_match_info['alpha_val']
            if not best_match_info.get('is_exact', False): 
                 st.warning(
                    f"The entered alpha ({alpha:.4f}) is not a standard value in this t-table for a {mode}-tailed test. "
                    f"Highlighting the column for the closest standard alpha: {selected_alpha_for_table:.4f}."
                )
        else: # Fallback if no heads matched the mode (should not happen for "one"/"two")
            st.error(f"Internal error: No columns found for mode '{mode}'. Defaulting to column 1.")
            col_idx = 1 
            if heads: st.info(f"Defaulted to col 1 (alpha={heads[0][1]:.3f} for mode='{heads[0][0]}').")

    if not (1 <= col_idx <= len(heads)):
        st.warning(f"Calculated column index ({col_idx}) was invalid. Resetting to 1.")
        col_idx = 1
    
    # Original HTML generation, using original formatting for headers and values
    head_html = "".join(f"<th>{m}_{a}</th>" for m,a in heads) # Original header format
    body_html = ""
    for r_val in rows:
        row_cells = f'<td id="t_{r_val}_0">{r_val}</td>'
        for i,(m_head,a_head) in enumerate(heads, start=1):
            val = stats.t.ppf(1 - a_head if m_head=="one" else 1 - a_head/2, r_val)
            row_cells += f'<td id="t_{r_val}_{i}">{val:.2f}</td>' # Original .2f precision
        body_html += f"<tr>{row_cells}</tr>"

    table_code = f"<tr><th>df</th>{head_html}</tr>{body_html}"
    html_output = wrap_table(CSS_BASE, table_code)

    if df in rows:
        for i in range(len(heads)+1):
            html_output = style_cell(html_output, f"t_{df}_{i}")
    for rr_val in rows:
        html_output = style_cell(html_output, f"t_{rr_val}_{col_idx}")
    if df in rows:
        html_output = style_cell(html_output, f"t_{df}_{col_idx}", color="blue", px=3)
    return html_output
# ===== END OF MINIMALLY CORRECTED build_t_html =====

def t_table(df: int, alpha: float, tail: str):
    code = build_t_html(df, alpha, tail)
    st.markdown(container(code), unsafe_allow_html=True)


def t_apa(t_val: float, df: int, alpha: float, tail: str): # Original t_apa
    """
    Show the dynamic explanation and final APA lines for the t-distribution.
    """
    if tail.startswith("one"):
        p_calc = stats.t.sf(abs(t_val), df) 
        crit = stats.t.ppf(1 - alpha, df)
        reject = (abs(t_val) > abs(crit)) if t_val < 0 else (t_val > crit)
    else:
        p_calc = stats.t.sf(abs(t_val), df) * 2
        crit = stats.t.ppf(1 - alpha/2, df)
        reject = (abs(t_val) > crit)

    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = "because t(calc) exceeded t(crit)"
        reason_p = "because p < α" # Using Greek alpha as in original
    else:
        reason_stats = "because t(calc) did not exceed t(crit)"
        reason_p = "because p ≥ α" # Using Greek alpha as in original

    cdf_val = stats.t.cdf(t_val, df)

    if tail.startswith("one"):
        if t_val >= 0:
            expl = (
                f"Lookup: P(T ≤ {t_val:.2f}) = {cdf_val:.4f}.\n\n"
                f"For a **one-tailed** test with a positive statistic, "
                f"p = 1 − {cdf_val:.4f} = {(1-cdf_val):.4f}."
            )
        else:
            expl = (
                f"Lookup: P(T ≤ {t_val:.2f}) = {cdf_val:.4f}.\n\n"
                f"For a **one-tailed** test with a negative statistic, "
                f"p = {cdf_val:.4f} (left tail)."
            )
    else:
        expl = (
            f"Lookup: P(T ≤ {t_val:.2f}) = {cdf_val:.4f}.\n\n"
            f"For a **two-tailed** test, p = 2 × min({cdf_val:.4f}, "
            f"{1-cdf_val:.4f}) = {2*min(cdf_val,1-cdf_val):.4f}."
        )
    st.write(expl)

    st.markdown( # Original markdown structure and content
        "**APA interpretation** \n"
        f"Calculated statistic: *t*({df})={t_val:.2f}, *p*={p_calc:.3f}.  \n"
        f"Critical statistic: t(crit)={crit:.2f}, *p*={alpha:.3f}.  \n"
        f"Comparison of statistics → H0 **{decision}** ({reason_stats}).  \n"
        f"Comparison of *p*-values → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *t*({df})={t_val:.2f}, *p*={p_calc:.3f} "
        f"({tail}). The null hypothesis was **{decision}** at α={alpha:.2f}."
    )


def tab_t(): # Original tab_t structure
    st.subheader("Tab 1 • t-Distribution")

    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("t statistic", value=2.10, key="t_val")
        df = st.number_input("df", min_value=1, value=10, step=1, key="t_df")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                 min_value=0.0001, max_value=0.5, key="t_alpha") # Original step
        tail = st.radio("Tail", ["one-tailed", "two-tailed"], key="t_tail")

    if st.button("Update Plot", key="t_plot"): # Original button
        try:
            fig_t_dist = plot_t(float(t_val), int(df), float(alpha), tail)
            if fig_t_dist: st.pyplot(fig_t_dist)
        except Exception as e:
            st.error(f"Error generating t-plot: {e}")

    # Table and APA are always displayed/updated when inputs change or button (if any) is pressed
    # For safety, let's ensure they are called after inputs are stable
    try:
        st.write("**t-table** (single highlight)")
        ctable, cexp = st.columns([2,1])
        with ctable:
            t_table(int(df), float(alpha), tail)
            show_cumulative_note()
        with cexp:
            st.subheader("P-value Calculation Explanation")
            t_apa(float(t_val), int(df), float(alpha), tail)
    except Exception as e:
        st.error(f"Error displaying t-table or APA explanation: {e}")
        st.exception(e)


###############################################################################
#                             TAB 2: z-Distribution
###############################################################################
# (Restoring to original code, only changing keys for inputs if needed for consistency)
def plot_z(z_calc, alpha, tail): # Original plot_z
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    xs = np.linspace(-4,4,400)
    ys = stats.norm.pdf(xs)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H0")
    placed = []

    if tail.startswith("one"):
        crit_pos = stats.norm.ppf(1 - alpha)
        crit_neg = -crit_pos
        if z_calc >= 0:
            ax.fill_between(xs[xs>=crit_pos], ys[xs>=crit_pos],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_pos, color="green", ls="--")
            place_label(ax, placed, crit_pos, stats.norm.pdf(crit_pos)+0.02,
                        f"z₍crit₎={crit_pos:.2f}", color="green")
        else:
            ax.fill_between(xs[xs<=crit_neg], ys[xs<=crit_neg],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_neg, color="green", ls="--")
            place_label(ax, placed, crit_neg, stats.norm.pdf(crit_neg)+0.02,
                        f"z₍crit₎={crit_neg:.2f}", color="green")
    else:
        crit = stats.norm.ppf(1 - alpha/2)
        ax.fill_between(xs[xs>=crit], ys[xs>=crit], color="red", alpha=0.3)
        ax.fill_between(xs[xs<=-crit], ys[xs<=-crit], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(-crit, color="green", ls="--")
        place_label(ax, placed, crit, stats.norm.pdf(crit)+0.02,
                    f"+z₍crit₎={crit:.2f}", color="green")
        place_label(ax, placed, -crit, stats.norm.pdf(-crit)+0.02,
                    f"–z₍crit₎={crit:.2f}", color="green")

    ax.axvline(z_calc, color="blue", ls="--")
    place_label(ax, placed, z_calc, stats.norm.pdf(z_calc)+0.02,
                f"z₍calc₎={z_calc:.2f}", color="blue")

    ax.set_xlabel("z")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("z-Distribution")
    fig.tight_layout()
    return fig

def build_z_html(z_val: float, alpha: float, tail: str) -> str: # Original build_z_html
    z_val = np.clip(z_val, -3.49, 3.49)
    row = np.floor(z_val*10)/10
    col = round(z_val-row, 2)

    Rows = np.round(np.arange(-3.4, 3.5, 0.1), 1)
    Cols = np.round(np.arange(0, 0.1, 0.01), 2)

    col = min(Cols, key=lambda c: abs(c - col))

    idx_arr = np.where(Rows == row)[0]
    if len(idx_arr) > 0:
        irow = idx_arr[0]
    else:
        irow = len(Rows)//2
    show_rows = Rows[max(0, irow-10): irow+11]

    head = "".join(f"<th>{c:.2f}</th>" for c in Cols)
    body = ""
    for rr in show_rows:
        row_html = f'<td id="z_{rr:.1f}_0">{rr:.1f}</td>'
        for cc in Cols:
            cdf_val = stats.norm.cdf(rr + cc)
            row_html += f'<td id="z_{rr:.1f}_{cc:.2f}">{cdf_val:.4f}</td>'
        body += f"<tr>{row_html}</tr>"

    table_code = f"<tr><th>z.x</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, table_code)

    for cc in Cols:
        html = style_cell(html, f"z_{row:.1f}_{cc:.2f}")
    html = style_cell(html, f"z_{row:.1f}_0")

    for rr in show_rows:
        html = style_cell(html, f"z_{rr:.1f}_{col:.2f}")

    html = style_cell(html, f"z_{row:.1f}_{col:.2f}", color="blue", px=3)
    return html

def z_table(z_val: float, alpha: float, tail: str): # Original z_table
    code = build_z_html(z_val, alpha, tail)
    st.markdown(container(code), unsafe_allow_html=True)

def z_apa(z_val: float, alpha: float, tail: str): # Original z_apa
    if tail.startswith("one"):
        p_calc = stats.norm.sf(abs(z_val))
        crit_pos = stats.norm.ppf(1 - alpha)
        crit_neg = -crit_pos
        crit = crit_pos if z_val>0 else crit_neg
        reject = (abs(z_val) > abs(crit))
    else:
        p_calc = stats.norm.sf(abs(z_val))*2
        crit = stats.norm.ppf(1 - alpha/2)
        reject = (abs(z_val) > crit)

    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = "because z₍calc₎ exceeded z₍crit₎"
        reason_p = "because p < α"
    else:
        reason_stats = "because z₍calc₎ did not exceed z₍crit₎"
        reason_p = "because p ≥ α"

    table_val = stats.norm.cdf(z_val)
    if tail.startswith("one"):
        if z_val >= 0:
            expl = (
                f"Lookup: P(Z ≤ {z_val:.2f}) = {table_val:.4f}\n\n"
                f"For a **one-tailed** test with positive z, p = 1 − {table_val:.4f}"
            )
        else:
            expl = (
                f"Lookup: P(Z ≤ {z_val:.2f}) = {table_val:.4f}\n\n"
                f"For a **one-tailed** test with negative z, p = {table_val:.4f}"
            )
    else:
        expl = (
            f"Lookup: P(Z ≤ {z_val:.2f}) = {table_val:.4f}\n\n"
            f"For a **two-tailed** test, p = 2 × min({table_val:.4f}, "
            f"{1 - table_val:.4f})"
        )
    st.write(expl)

    st.markdown(
        "**APA interpretation** \n"
        f"Calculated statistic: *z*={z_val:.2f}, *p*={p_calc:.3f}.  \n"
        f"Critical statistic: z₍crit₎={crit:.2f}, *p*={alpha:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H₀ **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *z*={z_val:.2f}, *p*={p_calc:.3f} ({tail}). "
        f"The null hypothesis was **{decision}** at α={alpha:.2f}."
    )

def tab_z(): # Original tab_z structure
    st.subheader("Tab 2 • z-Distribution")
    c1, c2 = st.columns(2)
    with c1:
        z_val = st.number_input("z statistic", value=1.64, key="z_val") # Original key
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                 min_value=0.0001, max_value=0.5, key="z_alpha") # Original key
        tail = st.radio("Tail", ["one-tailed", "two-tailed"], key="z_tail") # Original key

    if st.button("Update Plot", key="z_plot"): # Original button
        try:
            fig_z_dist = plot_z(float(z_val), float(alpha), tail)
            if fig_z_dist: st.pyplot(fig_z_dist)
        except Exception as e:
            st.error(f"Error generating z-plot: {e}")
            
    # Table and APA are always displayed
    try:
        st.write("**z-table** (single highlight)")
        ctable, cexp = st.columns([2,1])
        with ctable:
            z_table(float(z_val), float(alpha), tail)
            show_cumulative_note()
        with cexp:
            st.subheader("P-value Calculation Explanation")
            z_apa(float(z_val), float(alpha), tail)
    except Exception as e:
        st.error(f"Error displaying z-table or APA: {e}")
        st.exception(e)


###############################################################################
#                       TAB 3: F-Distribution
###############################################################################
# (Restoring to original code, only changing keys for inputs if needed for consistency)
def plot_f(f_calc, df1, df2, alpha): # Original plot_f
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    xs = np.linspace(0, stats.f.ppf(0.995, df1, df2)*1.1, 400)
    ys = stats.f.pdf(xs, df1, df2)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")

    crit = stats.f.ppf(1 - alpha, df1, df2)
    ax.fill_between(xs[xs>=crit], ys[xs>=crit], color="red", alpha=0.3,
                    label="Reject H0")
    ax.axvline(crit, color="green", ls="--")
    ax.axvline(f_calc, color="blue", ls="--")

    placed_f = [] 
    place_label(ax, placed_f, crit, stats.f.pdf(crit, df1, df2)+0.02,
                f"F₍crit₎={crit:.2f}", color="green")
    place_label(ax, placed_f, f_calc, stats.f.pdf(f_calc, df1, df2)+0.02,
                f"F₍calc₎={f_calc:.2f}", color="blue")

    ax.set_xlabel("F")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"F-Distribution (df1={df1}, df2={df2})")
    fig.tight_layout()
    return fig

def build_f_table(df1: int, df2: int, alpha: float) -> str: # Original build_f_table
    rows = list(range(max(1,df1-5), df1+6))
    cols = list(range(max(1,df2-5), df2+6))
    try: # Original logic for col_idx
        col_idx = cols.index(df2)+1
    except ValueError: # df2 might not be in the default range of cols
        st.warning(f"df2 value {df2} is outside the displayed F-table column range. Highlighting may be affected.")
        # Fallback: highlight the first column if df2 is not in the range
        col_idx = 1 


    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        row_html = f'<td id="f_{r}_0">{r}</td>'
        for i,c in enumerate(cols, start=1):
            val = stats.f.ppf(1 - alpha, r, c)
            row_html += f'<td id="f_{r}_{i}">{val:.2f}</td>'
        body += f"<tr>{row_html}</tr>"

    code = f"<tr><th>df1\\df2</th>{head}</tr>{body}" # Original header
    html = wrap_table(CSS_BASE, code)

    if df1 in rows: # Check if df1 is in the displayed range for safety
        for i in range(len(cols)+1):
            html = style_cell(html, f"f_{df1}_{i}")

    # Ensure col_idx is valid for highlighting
    if 1 <= col_idx <= len(cols):
        for rr in rows:
            html = style_cell(html, f"f_{rr}_{col_idx}")
        if df1 in rows: # Intersection
            html = style_cell(html, f"f_{df1}_{col_idx}", color="blue", px=3)
    return html

def f_table(df1: int, df2: int, alpha: float): # Original f_table
    code = build_f_table(df1, df2, alpha)
    st.markdown(container(code), unsafe_allow_html=True)

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
        f"Critical statistic: F₍crit₎={crit:.2f}, *p*={alpha:.3f}.  \n"
        f"Statistic comparison → H0 **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *F*({df1},{df2})={f_val:.2f}, *p*={p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at α={alpha:.2f}."
    )

def tab_f(): # Original tab_f structure
    st.subheader("Tab 3 • F-Distribution")
    c1, c2 = st.columns(2)
    with c1:
        f_val = st.number_input("F statistic", value=4.32, key="f_val") # Original key
        df1 = st.number_input("df1 (numerator)", min_value=1, value=5, step=1, key="f_df1") # Original key
    with c2:
        df2 = st.number_input("df2 (denominator)", min_value=1, value=20, step=1, key="f_df2") # Original key
        alpha = st.number_input("α", value=0.05, step=0.01,
                                 min_value=0.0001, max_value=0.5, key="f_alpha") # Original key
    if st.button("Update Plot", key="f_plot"): # Original button
        try:
            fig_f_dist = plot_f(float(f_val), int(df1), int(df2), float(alpha))
            if fig_f_dist: st.pyplot(fig_f_dist)
        except Exception as e:
            st.error(f"Error generating F-plot: {e}")

    # Table and APA always displayed
    try:
        st.write("**F-table** (always one-tailed, no new cumulative note or p expl.)")
        f_table(int(df1), int(df2), float(alpha))
        f_apa(float(f_val), int(df1), int(df2), float(alpha))
    except Exception as e:
        st.error(f"Error displaying F-table or APA: {e}")
        st.exception(e)

###############################################################################
#                       TAB 4: Chi-Square
###############################################################################
# (Restoring to original code, only changing keys for inputs if needed for consistency)
def plot_chi(chi_calc, df, alpha): # Original plot_chi
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    xs = np.linspace(0, stats.chi2.ppf(0.995, df)*1.1, 400)
    ys = stats.chi2.pdf(xs, df)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")

    crit = stats.chi2.ppf(1 - alpha, df)
    ax.fill_between(xs[xs>=crit], ys[xs>=crit], color="red", alpha=0.3,
                    label="Reject H0")
    ax.axvline(crit, color="green", ls="--")
    ax.axvline(chi_calc, color="blue", ls="--")
    
    placed_chi = []
    place_label(ax, placed_chi, crit, stats.chi2.pdf(crit, df)+0.02,
                f"χ²₍crit₎={crit:.2f}", color="green") # Using Greek chi as in original
    place_label(ax, placed_chi, chi_calc, stats.chi2.pdf(chi_calc, df)+0.02,
                f"χ²₍calc₎={chi_calc:.2f}", color="blue")

    ax.set_xlabel("χ²") # Using Greek chi
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"χ²-Distribution (df={df})") # Using Greek chi
    fig.tight_layout()
    return fig

def build_chi_table(df: int, alpha: float) -> str: # Original build_chi_table
    rows = list(range(max(1,df-5), df+6))
    alphas_list = [0.10,0.05,0.01,0.001] # Renamed to avoid conflict
    try: # alpha from selectbox is guaranteed to be in alphas_list
        col_idx = alphas_list.index(alpha)+1
    except ValueError: # Should not happen
        st.warning("Selected alpha for Chi-square not in standard list. Defaulting highlighting.")
        col_idx = 1 # Fallback

    head = "".join(f"<th>{a}</th>" for a in alphas_list)
    body = ""
    for r in rows:
        row_html = f'<td id="chi_{r}_0">{r}</td>'
        for i,a_val in enumerate(alphas_list, start=1): # Renamed a to a_val
            val = stats.chi2.ppf(1 - a_val, r)
            row_html += f'<td id="chi_{r}_{i}">{val:.2f}</td>'
        body += f"<tr>{row_html}</tr>"

    table_code = f"<tr><th>df\\α</th>{head}</tr>{body}" # Original header with Greek alpha
    html = wrap_table(CSS_BASE, table_code)

    if df in rows: # Check if df in displayed rows
        for i in range(len(alphas_list)+1):
            html = style_cell(html, f"chi_{df}_{i}")
    
    # Ensure col_idx is valid before using for styling all rows in that column
    if 1 <= col_idx <= len(alphas_list):
        for rr in rows:
            html = style_cell(html, f"chi_{rr}_{col_idx}")
        if df in rows: # Intersection
            html = style_cell(html, f"chi_{df}_{col_idx}", color="blue", px=3)
    return html

def chi_table(df: int, alpha: float): # Original chi_table
    code = build_chi_table(df, alpha)
    st.markdown(container(code), unsafe_allow_html=True)

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
        f"Critical statistic: χ²₍crit₎={crit:.2f}, *p*={alpha:.3f}.  \n"
        f"Statistic comparison → H0 **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** χ²({df})={chi_val:.2f}, *p*={p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at α={alpha:.2f}."
    )

def tab_chi(): # Original tab_chi structure
    st.subheader("Tab 4 • Chi-Square")
    c1, c2 = st.columns(2)
    with c1:
        chi_val = st.number_input("χ² statistic", value=7.88, key="chi_val") # Original key
        df = st.number_input("df", min_value=1, value=3, step=1, key="chi_df") # Original key
    with c2:
        alpha = st.selectbox("α", [0.10,0.05,0.01,0.001], # Original selectbox for alpha
                           index=1, key="chi_alpha") # Original key
    if st.button("Update Plot", key="chi_plot"): # Original button
        try:
            fig_chi_dist = plot_chi(float(chi_val), int(df), float(alpha))
            if fig_chi_dist: st.pyplot(fig_chi_dist)
        except Exception as e:
            st.error(f"Error generating Chi-square plot: {e}")

    # Table and APA always displayed
    try:
        st.write("**χ²-table**")
        chi_table(int(df), float(alpha))
        chi_apa(float(chi_val), int(df), float(alpha))
    except Exception as e:
        st.error(f"Error displaying Chi-square table or APA: {e}")
        st.exception(e)


###############################################################################
#                       TAB 5: Mann-Whitney U
###############################################################################
# (Restoring to original code, with syntax correction in u_apa)
def plot_u(u_calc, n1, n2, alpha, tail): # Original plot_u
    μ = n1*n2/2
    σ = np.sqrt(n1*n2*(n1+n2+1)/12)

    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    if σ == 0: # Handle cases where variance is zero
        ax.text(0.5,0.5, "Cannot plot Mann-Whitney U: Variance is zero.", ha='center', va='center')
        return fig

    xs = np.linspace(μ-4*σ, μ+4*σ, 400)
    ys = stats.norm.pdf(xs, μ, σ)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H0")
    placed_u = []
    from math import floor # Original import location
    
    # Critical values based on normal approximation
    # This part was significantly changed in my previous versions, reverting to a simpler interpretation
    # consistent with original structure if possible, or a standard one.
    # U-test typically tests if U_calc <= U_crit_lower (for left-tail)
    # or U_calc <= U_crit_lower OR U_calc >= U_crit_upper (for two-tail)
    
    if tail.startswith("one"):
        # Assuming a left-tailed test as U is often min(U1,U2)
        zcrit_one_tail = stats.norm.ppf(alpha) # Negative Z for left tail
        crit_val_approx = floor(μ + zcrit_one_tail * σ)
        
        ax.fill_between(xs[xs<=crit_val_approx], ys[xs<=crit_val_approx], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit_val_approx, color="green", ls="--")
        place_label(ax, placed_u, crit_val_approx, stats.norm.pdf(crit_val_approx, μ, σ)+0.02,
                    f"Ucrit≈{crit_val_approx}", color="green")
    else: # two-tailed
        zcrit_lower = stats.norm.ppf(alpha/2) # Negative Z
        crit_val_lower = floor(μ + zcrit_lower * σ)
        
        # Upper critical value for U (U_U) for two-tailed test
        # U_U = n1*n2 - U_L where U_L is critical value for alpha/2 from lower tail.
        # So, U_U_approx = n1*n2 - crit_val_lower
        # Or, using Z:
        zcrit_upper = stats.norm.ppf(1-alpha/2) # Positive Z
        crit_val_upper = floor(μ + zcrit_upper*σ) # Some tables might use ceil for upper.

        ax.fill_between(xs[xs<=crit_val_lower], ys[xs<=crit_val_lower], color="red", alpha=0.3)
        ax.fill_between(xs[xs>=crit_val_upper], ys[xs>=crit_val_upper], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit_val_lower, color="green", ls="--")
        ax.axvline(crit_val_upper, color="green", ls="--")
        place_label(ax, placed_u, crit_val_lower, stats.norm.pdf(crit_val_lower, μ, σ)+0.02,
                    f"UcritL≈{crit_val_lower}", color="green")
        place_label(ax, placed_u, crit_val_upper, stats.norm.pdf(crit_val_upper, μ, σ)+0.02,
                    f"UcritU≈{crit_val_upper}", color="green")

    ax.axvline(u_calc, color="blue", ls="--")
    place_label(ax, placed_u, u_calc, stats.norm.pdf(u_calc, μ, σ)+0.02,
                f"Ucalc={u_calc}", color="blue")

    ax.set_xlabel("U (Normal Approx.)")
    ax.set_ylabel("Approx. density")
    ax.legend()
    ax.set_title("Mann-Whitney U (Normal Approximation)")
    fig.tight_layout()
    return fig

def u_crit(n1:int, n2:int, alpha:float, tail:str)->int: # Original u_crit
    μ = n1*n2/2
    σ = np.sqrt(n1*n2*(n1+n2+1)/12)
    if σ == 0: return np.nan # Original handles sigma=0 by returning NaN
    from math import floor
    # This calculates the lower critical U value.
    # For one-tailed, use alpha directly (P(U <= Ucrit) = alpha)
    # For two-tailed, use alpha/2 for the lower tail (P(U <= Ucrit) = alpha/2)
    z = stats.norm.ppf(alpha if tail.startswith("one") else alpha/2)
    return int(floor(μ + z*σ))

def build_u_table(n1:int, n2:int, alpha:float, tail:str)->str: # Original build_u_table
    rows = list(range(max(2,n1-5), n1+6))
    cols = list(range(max(2,n2-5), n2+6))
    try:
        col_idx = cols.index(n2)+1
    except ValueError: # n2 not in displayed range for cols
        st.warning(f"n2={n2} is outside the displayed U-table range. Highlighting might be affected.")
        col_idx = 1 # Fallback: highlight first column.

    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        row_html = f'<td id="u_{r}_0">{r}</td>'
        for i,c in enumerate(cols, start=1):
            val = u_crit(r,c,alpha,tail) # This returns lower critical U.
            row_html += f'<td id="u_{r}_{i}">{val if not np.isnan(val) else "N/A"}</td>'
        body += f"<tr>{row_html}</tr>"

    code = f"<tr><th>n1\\n2</th>{head}</tr>{body}" # Original header
    html = wrap_table(CSS_BASE, code)

    if n1 in rows: # Ensure n1 is in displayed rows
        for i in range(len(cols)+1):
            html = style_cell(html, f"u_{n1}_{i}")
    
    # Ensure col_idx is valid for highlighting
    if 1 <= col_idx <= len(cols):
        for rr in rows:
            html = style_cell(html, f"u_{rr}_{col_idx}")
        if n1 in rows: # Intersection
            html = style_cell(html, f"u_{n1}_{col_idx}", color="blue", px=3)
    return html

def u_table(n1:int, n2:int, alpha:float, tail:str): # Original u_table
    code = build_u_table(n1,n2,alpha,tail)
    st.markdown(container(code), unsafe_allow_html=True)

def u_apa(u_val: int, n1: int, n2: int, alpha: float, tail: str): # Corrected u_apa
    """
    Show dynamic explanation and final APA lines for Mann-Whitney U.
    """
    mu_u = n1*n2/2.0 # Use float for mu
    sigma_u = np.sqrt(n1*n2*(n1+n2+1)/12.0)

    # Default values in case sigma_u is zero or calculations fail
    p_calc_val = np.nan
    z_approx_val = np.nan
    reject = False
    crit_display = "N/A"


    if sigma_u > 1e-9: # Calculations only if sigma is valid
        # Z transformation with continuity correction
        if u_val < mu_u:
            z_approx_val = (u_val - mu_u + 0.5) / sigma_u
        elif u_val > mu_u:
            z_approx_val = (u_val - mu_u - 0.5) / sigma_u
        else: # u_val == mu_u
            z_approx_val = 0.0

        # P-value from this z_approx_val
        if tail.startswith("one"):
            # Assuming u_val is appropriately chosen for the direction of the test
            # e.g., if testing for "less", u_val is U1 where group 1 ranks are smaller.
            # Scipy's mannwhitneyu handles 'alternative' hypotheses correctly.
            # For this APA, we base p-value on the Z score's tail.
            if z_approx_val <=0: # Corresponds to U_calc being on the "smaller" side of mean
                p_calc_val = stats.norm.cdf(z_approx_val)
                crit_z_apa = stats.norm.ppf(alpha) # Left-tail Z_crit
                reject = z_approx_val < crit_z_apa
                crit_display = f"Z_crit ≈ {crit_z_apa:.2f} (left tail)"
            else: # z_approx_val > 0
                p_calc_val = stats.norm.sf(z_approx_val) # 1 - cdf for right tail
                crit_z_apa = stats.norm.ppf(1-alpha) # Right-tail Z_crit
                reject = z_approx_val > crit_z_apa
                crit_display = f"Z_crit ≈ {crit_z_apa:.2f} (right tail)"

        else: # two-tailed
            p_calc_val = 2 * stats.norm.sf(abs(z_approx_val))
            crit_z_apa = stats.norm.ppf(1-alpha/2) # Positive Z_crit for |Z| comparison
            reject = abs(z_approx_val) > crit_z_apa
            crit_display = f"|Z_crit| ≈ {crit_z_apa:.2f} (for two tails)"
    else:
        st.warning("Cannot calculate Z approximation for U-test: standard deviation is zero or too small.")


    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = f"because U statistic (Z≈{z_approx_val:.2f}) was in the rejection region"
        reason_p = f"because approx. p ({p_calc_val:.3f}) < α ({alpha:.3f})"
    else:
        reason_stats = f"because U statistic (Z≈{z_approx_val:.2f}) was not in the rejection region"
        reason_p = f"because approx. p ({p_calc_val:.3f}) ≥ α ({alpha:.3f})"


    # Table-based explanation (CDF of U from normal approx, no cont. corr. for this general P(U<=u) idea)
    normal_cdf_for_u_expl = np.nan
    if sigma_u > 1e-9:
        normal_cdf_for_u_expl = stats.norm.cdf((u_val-mu_u)/sigma_u)

    expl = "" # Initialize expl string
    if sigma_u > 1e-9 :
        if tail.startswith("one"):
            if u_val <= mu_u : # Consistent with a left-tail test on U
                expl = (
                    f"Lookup: P(U ≤ {u_val}) approx. from normal CDF ~ {normal_cdf_for_u_expl:.4f}.\n\n"
                    f"For a **one-tailed** test (testing if U is significantly small), "
                    f"p (from Z with cont. corr.) ≈ {p_calc_val:.4f}."
                )
            else: # u_val > mu_u, consistent with a right-tail test on U (or U' being small)
                expl = (
                    f"Lookup: P(U ≤ {u_val}) approx. from normal CDF ~ {normal_cdf_for_u_expl:.4f}.\n\n"
                    f"For a **one-tailed** test (testing if U is significantly large), "
                    f"p (from Z with cont. corr.) ≈ {p_calc_val:.4f} (this p is $1-P(Z \\le Z_{{approx}})$)."
                )
        else: # two-tailed
            expl = (
                f"Lookup: P(U ≤ {u_val}) approx. from normal CDF ~ {normal_cdf_for_u_expl:.4f}.\n\n"
                f"For a **two-tailed** test, p (from Z with cont. corr.) "
                f"≈ $2 \\times \\text{{min}}(P(Z \\le Z_{{approx}}), 1-P(Z \\le Z_{{approx}})) \\approx {p_calc_val:.4f}$."
            )
    else:
        expl = "Cannot provide detailed explanation as $\\sigma_U$ is zero or too small."
        
    st.write(expl)

    st.markdown(
        "**APA interpretation (Normal Approximation for U)** \n"
        f"Calculated Mann-Whitney U = {u_val}. With $n_1={n1}, n_2={n2}$, this yields an approximate Z-statistic (with continuity correction) $Z \\approx {z_approx_val:.2f}$, "
        f"and an approximate *$p$* = {p_calc_val:.3f} ({tail}).\n"
        f"The critical Z-value for $\\alpha={alpha:.3f}$ ({tail}) is {crit_display}.\n"
        f"Based on comparison of Z-statistics: H₀ is **{decision}** ({reason_stats}).\n"
        f"Based on comparison of p-value to alpha: H₀ is **{decision}** ({reason_p}).\n"
        f"**APA 7 report (approximate):** A Mann-Whitney U test indicated that the null hypothesis was **{decision}**, "
        f"U = {u_val}, Z $\\approx$ {z_approx_val:.2f}, *p* $\\approx$ {p_calc_val:.3f} ({tail}), at an alpha level of {alpha:.2f}."
    )


def tab_u(): # Original tab_u structure might be different, this is a basic UI for the functions
    st.subheader("Tab 5 • Mann-Whitney U")
    c1, c2 = st.columns(2)
    with c1:
        u_val = st.number_input("U statistic value", min_value=0, value=23, step=1, key="u_val_input_key")
        n1 = st.number_input("n1 (sample 1 size)", min_value=1, value=8, step=1, key="u_n1_input_key")
    with c2:
        n2 = st.number_input("n2 (sample 2 size)", min_value=1, value=10, step=1, key="u_n2_input_key")
        alpha = st.number_input("α for U", value=0.05, step=0.001, min_value=0.0001, max_value=0.5, format="%.4f", key="u_alpha_input_key")
        tail = st.radio("Tail for U", ["one-tailed", "two-tailed"], key="u_tail_input_key")

    if st.button("Update Plot & Table for U", key="u_plot_button_key"):
        try:
            fig_u_dist = plot_u(int(u_val), int(n1), int(n2), float(alpha), tail)
            if fig_u_dist: st.pyplot(fig_u_dist)
        except Exception as e:
            st.error(f"Error generating U-plot: {e}")
        
        try:
            st.write("**U-table (Approx. Lower Critical Values)**")
            ctable_u, cexp_u = st.columns([2,1])
            with ctable_u:
                u_table(int(n1), int(n2), float(alpha), tail)
                # Note: Cumulative note for Z/T might be confusing here as U-tables are different
                st.info("U-table values are approx. lower critical U's. For left-tail, reject H₀ if obs. U ≤ table U.")
            with cexp_u:
                st.subheader("P-value Calculation & APA (Normal Approx.)")
                u_apa(int(u_val), int(n1), int(n2), float(alpha), tail)
        except Exception as e:
            st.error(f"Error displaying U-table or APA: {e}")
            st.exception(e)


def tab_wilcoxon_t(): # Placeholder
    st.subheader("Tab 6 • Wilcoxon Signed-Rank T")
    st.write("Wilcoxon T functionality to be implemented.")

def tab_binomial(): # Placeholder
    st.subheader("Tab 7 • Binomial")
    st.write("Binomial functionality to be implemented.")

# It's crucial how main() is structured if using st.tabs
# If an error occurs in one tab's rendering, it can break all tabs.
def main():
    st.set_page_config(layout="wide", page_title="Statistical Tables Explorer") # Added page title
    st.title("Oli's - Statistical Table Explorer")

    # Using st.tabs as implied by your screenshots showing multiple tabs at the top
    tab_titles = ["t-Dist", "z-Dist", "F-Dist", "Chi-Square", "Mann-Whitney U", "Wilcoxon T", "Binomial"]
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

if __name__ == "__main__": # Standard Python script entry point
    main()
