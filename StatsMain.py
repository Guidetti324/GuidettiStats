###############################################################################
#  PSYC‑250 – Statistical Tables Explorer
#  ---------------------------------------------------------------------------
#  Seven complete tabs:
#       1) t‑Distribution          4) Chi‑Square
#       2) z‑Distribution          5) Mann‑Whitney U
#       3) F‑Distribution          6) Wilcoxon Signed‑Rank T
#       7) Binomial
#
#  NEW FEATURES ADDED FOR (t, z, Mann‑Whitney U, Wilcoxon T, Binomial):
#   1) A "Cumulative Table Note" explaining how to interpret the table for
#      one‑ vs. two‑tailed tests.
#   2) A "P‑Value Calculation Explanation" section next to the table,
#      showing how the table lookup leads to p, depending on one‑ vs. two‑tailed.
#   3) Automatic plot shading based on sign of the test statistic for one‑tailed
#      (negative → left tail, positive → right tail). For two‑tailed, shade both tails.
#
#  F‑Distribution and Chi‑Square remain as before, since they are always one‑tailed.
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
        "one‑tailed tests, use the area directly. For two‑tailed tests, you must "
        "double the area in the tail beyond your observed value (i.e., "
        "p=2×(1−P(Z≤|z|))). The same logic applies for t‑distributions. The table "
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
#                           TAB 1: t‑Distribution
###############################################################################

def plot_t(t_calc, df, alpha, tail):
    """
    Plot the t‑distribution. For one‑tailed, decide left vs. right tail
    based on sign of t_calc. For two‑tailed, shade both tails.
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
                        f"tcrit={crit_pos:.2f}", color="green")
        else:
            # shade left side
            ax.fill_between(xs[xs <= crit_neg], ys[xs <= crit_neg],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_neg, color="green", ls="--")
            place_label(ax, placed_labels, crit_neg, stats.t.pdf(crit_neg, df)+0.02,
                        f"tcrit={crit_neg:.2f}", color="green")
    else:
        # two-tailed: shade both tails
        crit = stats.t.ppf(1 - alpha/2, df)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit], color="red", alpha=0.3)
        ax.fill_between(xs[xs <= -crit], ys[xs <= -crit], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(-crit, color="green", ls="--")
        place_label(ax, placed_labels, crit, stats.t.pdf(crit, df)+0.02,
                    f"+tcrit={crit:.2f}", color="green")
        place_label(ax, placed_labels, -crit, stats.t.pdf(-crit, df)+0.02,
                    f"–tcrit={crit:.2f}", color="green")

    ax.axvline(t_calc, color="blue", ls="--")
    place_label(ax, placed_labels, t_calc, stats.t.pdf(t_calc, df)+0.02,
                f"tcalc={t_calc:.2f}", color="blue")

    ax.set_xlabel("t")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("t‑Distribution")
    fig.tight_layout()
    return fig


def build_t_html(df: int, alpha: float, tail: str) -> str: # user_input_alpha is the 'alpha' parameter
    """
    Single-step highlight for t-table row & column + intersection.
    Table cell values are for fixed alpha levels. Highlighting attempts to match user_input_alpha.
    """
    rows = list(range(max(1, df-5), df+6))
    # These are the fixed alpha levels for which the table columns are generated
    heads = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001)
    ]
    
    mode = "one" if tail.startswith("one") else "two"
    
    col_idx_to_highlight = None
    for i_loop, (m_loop, a_loop) in enumerate(heads, start=1):
        if m_loop == mode and np.isclose(a_loop, alpha): # alpha is user_input_alpha
            col_idx_to_highlight = i_loop
            break

    head_html = "".join(f"<th>{m}_{a}</th>" for m,a in heads)
    body_html = ""
    for r in rows:
        row_cells = f'<td id="t_{r}_0">{r}</td>'
        for i,(m_col,a_col) in enumerate(heads, start=1): # m_col, a_col define the content of the cell
            val = stats.t.ppf(1 - a_col if m_col=="one" else 1 - a_col/2, r)
            row_cells += f'<td id="t_{r}_{i}">{val:.2f}</td>'
        body_html += f"<tr>{row_cells}</tr>"

    table_code = f"<tr><th>df</th>{head_html}</tr>{body_html}"
    html = wrap_table(CSS_BASE, table_code)

    # highlight entire row for df
    for i in range(len(heads)+1):
        html = style_cell(html, f"t_{df}_{i}")
    
    if col_idx_to_highlight is not None:
        # highlight entire column for the matching alpha
        for rr_idx in rows:
            html = style_cell(html, f"t_{rr_idx}_{col_idx_to_highlight}")
        # intersection
        html = style_cell(html, f"t_{df}_{col_idx_to_highlight}", color="blue", px=3)
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
        reject = (abs(t_val) > abs(crit)) if t_val < 0 else (t_val > crit) # Corrected logic for one-tailed
    else:
        p_calc = stats.t.sf(abs(t_val), df) * 2
        crit = stats.t.ppf(1 - alpha/2, df)
        reject = (abs(t_val) > crit)

    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = "because t(calc) exceeded t(crit)"
        reason_p = "because p < α"
    else:
        reason_stats = "because t(calc) did not exceed t(crit)"
        reason_p = "because p ≥ α"

    # "table" cdf value
    cdf_val = stats.t.cdf(t_val, df)

    # Explanation text
    if tail.startswith("one"):
        if t_val >= 0: # Positive t-statistic
            expl = (
                f"Lookup: P(T ≤ {t_val:.2f}) = {cdf_val:.4f}.\n\n"
                f"For a **one‑tailed** test with a positive statistic, "
                f"p = 1 − {cdf_val:.4f} = {(1-cdf_val):.4f}."
            )
        else: # Negative t-statistic
            expl = (
                f"Lookup: P(T ≤ {t_val:.2f}) = {cdf_val:.4f}.\n\n"
                f"For a **one‑tailed** test with a negative statistic, "
                f"p = {cdf_val:.4f} (left tail)."
            )
    else:
        expl = (
            f"Lookup: P(T ≤ {t_val:.2f}) = {cdf_val:.4f}.\n\n"
            f"For a **two‑tailed** test, p = 2 × min({cdf_val:.4f}, "
            f"{1-cdf_val:.4f}) = {2*min(cdf_val,1-cdf_val):.4f}."
        )

    st.write(expl)

    st.markdown(
        "**APA interpretation** \n"
        f"Calculated statistic: *t*({df})={t_val:.2f}, *p*={p_calc:.3f}.  \n"
        f"Critical statistic: t(crit)={crit:.2f}, *p*={alpha:.3f}.  \n"
        f"Comparison of statistics → H0 **{decision}** ({reason_stats}).  \n"
        f"Comparison of *p*-values → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *t*({df})={t_val:.2f}, *p*={p_calc:.3f} "
        f"({tail}). The null hypothesis was **{decision}** at α={alpha:.2f}."
    )


def tab_t():
    st.subheader("Tab 1 • t‑Distribution")

    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("t statistic", value=2.10, key="t_val")
        df = st.number_input("df", min_value=1, value=10, step=1, key="t_df")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="t_alpha", format="%.4f")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="t_tail")

    if st.button("Update Plot", key="t_plot"):
        st.pyplot(plot_t(t_val, df, alpha, tail))

    st.write("**t‑table** (single highlight)")
    # Table & note in left column, explanation in right column
    ctable, cexp = st.columns([2,1])
    with ctable:
        t_table(df, alpha, tail)
        show_cumulative_note()
    with cexp:
        st.subheader("P‑value Calculation Explanation")
        t_apa(t_val, df, alpha, tail)


###############################################################################
#                           TAB 2: z‑Distribution
###############################################################################

def plot_z(z_calc, alpha, tail):
    """
    For one‑tailed, if z_calc>=0, shade the right tail; if negative, shade left tail.
    For two‑tailed, shade both tails.
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
    ax.set_title("z‑Distribution")
    fig.tight_layout()
    return fig


def build_z_html(z_val: float, alpha: float, tail: str) -> str: # alpha, tail not used for z-table cell values but kept for API consistency
    """
    Single-step highlight for z-table. Cell values are standard normal CDF.
    """
    z_val = np.clip(z_val, -3.49, 3.49)
    row = np.floor(z_val*10)/10
    col = round(z_val-row, 2)

    Rows = np.round(np.arange(-3.4, 3.5, 0.1), 1)
    Cols = np.round(np.arange(0, 0.1, 0.01), 2)

    # find nearest col
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

    # highlight row
    for cc in Cols:
        html = style_cell(html, f"z_{row:.1f}_{cc:.2f}")
    html = style_cell(html, f"z_{row:.1f}_0")

    # highlight column
    for rr in show_rows:
        html = style_cell(html, f"z_{rr:.1f}_{col:.2f}")

    # intersection in blue
    html = style_cell(html, f"z_{row:.1f}_{col:.2f}", color="blue", px=3)
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
        p_calc = stats.norm.sf(abs(z_val))
        crit_pos = stats.norm.ppf(1 - alpha)
        crit_neg = -crit_pos
        crit = crit_pos if z_val>=0 else crit_neg # Corrected crit for one-tailed based on z_val sign
        reject = (z_val >= crit_pos) if z_val >= 0 else (z_val <= crit_neg)
    else:
        p_calc = stats.norm.sf(abs(z_val))*2
        crit = stats.norm.ppf(1 - alpha/2)
        reject = (abs(z_val) > crit)


    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = "because z₍calc₎ exceeded z₍crit₎" if not tail.startswith("one") or z_val >=0 else "because z₍calc₎ was beyond z₍crit₎ (left tail)"
        reason_p = "because p < α"
    else:
        reason_stats = "because z₍calc₎ did not exceed z₍crit₎" if not tail.startswith("one") or z_val >=0 else "because z₍calc₎ was not beyond z₍crit₎ (left tail)"
        reason_p = "because p ≥ α"

    # table cdf
    table_val = stats.norm.cdf(z_val)
    # explanation
    if tail.startswith("one"):
        if z_val >= 0:
            expl = (
                f"Lookup: P(Z ≤ {z_val:.2f}) = {table_val:.4f}\n\n"
                f"For a **one‑tailed** test with positive z, p = 1 − {table_val:.4f} = {(1-table_val):.4f}."
            )
        else:
            expl = (
                f"Lookup: P(Z ≤ {z_val:.2f}) = {table_val:.4f}\n\n"
                f"For a **one‑tailed** test with negative z, p = {table_val:.4f}."
            )
    else: # two-tailed
        expl = (
            f"Lookup: P(Z ≤ {z_val:.2f}) = {table_val:.4f}\n\n"
            f"For a **two‑tailed** test, p = 2 × min({table_val:.4f}, "
            f"{1 - table_val:.4f}) = {2*min(table_val, 1-table_val):.4f}."
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


def tab_z():
    st.subheader("Tab 2 • z‑Distribution")

    c1, c2 = st.columns(2)
    with c1:
        z_val = st.number_input("z statistic", value=1.64, key="z_val")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="z_alpha", format="%.4f")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="z_tail")

    if st.button("Update Plot", key="z_plot"):
        st.pyplot(plot_z(z_val, alpha, tail))

    st.write("**z‑table** (single highlight)")
    ctable, cexp = st.columns([2,1])
    with ctable:
        z_table(z_val, alpha, tail) # alpha, tail are passed but not used for z-table cell values
        show_cumulative_note()
    with cexp:
        st.subheader("P‑value Calculation Explanation")
        z_apa(z_val, alpha, tail)


###############################################################################
#                       TAB 3: F‑Distribution (unchanged)
###############################################################################

def plot_f(f_calc, df1, df2, alpha):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    # Adjust x-axis limit dynamically based on critical value and F calculated
    upper_x_limit_ppf = stats.f.ppf(0.999, df1, df2) # Go further for visualization
    upper_x_limit = max(f_calc * 1.2, upper_x_limit_ppf * 1.1, 5) # Ensure a reasonable minimum and include f_calc
    xs = np.linspace(0, upper_x_limit, 400)
    
    ys = stats.f.pdf(xs, df1, df2)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")

    crit = stats.f.ppf(1 - alpha, df1, df2)
    ax.fill_between(xs[xs>=crit], ys[xs>=crit], color="red", alpha=0.3,
                    label="Reject H0")
    ax.axvline(crit, color="green", ls="--")
    ax.axvline(f_calc, color="blue", ls="--")

    # Ensure labels are within plot bounds
    crit_pdf_val = stats.f.pdf(crit, df1, df2) if crit > 0 else 0.01 
    f_calc_pdf_val = stats.f.pdf(f_calc, df1, df2) if f_calc > 0 else 0.01

    place_label(ax, [], crit, crit_pdf_val +0.02,
                f"F₍crit₎={crit:.2f}", color="green")
    place_label(ax, [], f_calc, f_calc_pdf_val +0.02,
                f"F₍calc₎={f_calc:.2f}", color="blue")

    ax.set_xlabel("F")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"F‑Distribution (df1={df1}, df2={df2})")
    # Set x-limits for better visualization, especially with large F or critical values
    ax.set_xlim(0, upper_x_limit)
    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
    fig.tight_layout()
    return fig


def build_f_table(df1: int, df2: int, alpha: float) -> str: # Cell values depend on input alpha
    rows = list(range(max(1,df1-5), df1+6))
    cols = list(range(max(1,df2-5), df2+6))
    
    col_idx_to_highlight = None
    if df2 in cols:
        col_idx_to_highlight = cols.index(df2)+1


    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r_df1 in rows: # r_df1 is the df1 for the current row
        row_html = f'<td id="f_{r_df1}_0">{r_df1}</td>'
        for i_col_idx, c_df2 in enumerate(cols, start=1): # c_df2 is the df2 for the current column
            val = stats.f.ppf(1 - alpha, r_df1, c_df2) # Cell value uses input alpha
            row_html += f'<td id="f_{r_df1}_{i_col_idx}">{val:.2f}</td>'
        body += f"<tr>{row_html}</tr>"

    code = f"<tr><th>df1\\df2</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, code)

    # highlight row for input df1
    for i in range(len(cols)+1):
        html = style_cell(html, f"f_{df1}_{i}")
    
    if col_idx_to_highlight is not None: # If input df2 is in the displayed columns
        # highlight col for input df2
        for rr_df1_val in rows:
            html = style_cell(html, f"f_{rr_df1_val}_{col_idx_to_highlight}")
        # intersection
        html = style_cell(html, f"f_{df1}_{col_idx_to_highlight}", color="blue", px=3)
    return html


def f_table(df1: int, df2: int, alpha: float):
    code = build_f_table(df1, df2, alpha)
    st.markdown(container(code), unsafe_allow_html=True)


def f_apa(f_val: float, df1: int, df2: int, alpha: float):
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


def tab_f():
    st.subheader("Tab 3 • F‑Distribution")

    c1, c2 = st.columns(2)
    with c1:
        f_val = st.number_input("F statistic", value=4.32, key="f_val")
        df1 = st.number_input("df1 (numerator)", min_value=1, value=5, step=1, key="f_df1")
    with c2:
        df2 = st.number_input("df2 (denominator)", min_value=1, value=20, step=1, key="f_df2")
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="f_alpha", format="%.4f")

    if st.button("Update Plot", key="f_plot"):
        st.pyplot(plot_f(f_val, df1, df2, alpha))

    st.write("**F‑table** (always one‑tailed, values based on input α)")
    f_table(df1, df2, alpha)
    f_apa(f_val, df1, df2, alpha)


###############################################################################
#                       TAB 4: Chi-Square (unchanged)
###############################################################################

def plot_chi(chi_calc, df, alpha):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    # Adjust x-axis limit dynamically
    upper_x_limit_ppf = stats.chi2.ppf(0.999, df)
    upper_x_limit = max(chi_calc * 1.2, upper_x_limit_ppf * 1.1, 10) # Ensure reasonable minimum
    xs = np.linspace(0, upper_x_limit, 400)
    
    ys = stats.chi2.pdf(xs, df)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")

    crit = stats.chi2.ppf(1 - alpha, df)
    ax.fill_between(xs[xs>=crit], ys[xs>=crit], color="red", alpha=0.3,
                    label="Reject H0")
    ax.axvline(crit, color="green", ls="--")
    ax.axvline(chi_calc, color="blue", ls="--")
    
    # Ensure labels are within plot bounds
    crit_pdf_val = stats.chi2.pdf(crit, df) if crit > 0 else 0.01
    chi_calc_pdf_val = stats.chi2.pdf(chi_calc, df) if chi_calc > 0 else 0.01

    place_label(ax, [], crit, crit_pdf_val +0.02,
                f"χ²₍crit₎={crit:.2f}", color="green")
    place_label(ax, [], chi_calc, chi_calc_pdf_val +0.02,
                f"χ²₍calc₎={chi_calc:.2f}", color="blue")

    ax.set_xlabel("χ²")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"χ²‑Distribution (df={df})")
    ax.set_xlim(0, upper_x_limit)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig

def build_chi_table(df: int, user_input_alpha: float) -> str: # user_input_alpha is the parameter 'alpha'
    """
    Table cell values are for fixed alpha levels. Highlighting attempts to match user_input_alpha.
    """
    rows = list(range(max(1,df-5), df+6))
    # These are the fixed alpha levels for which the table columns are generated
    fixed_alphas_for_columns = [0.10,0.05,0.01,0.001] 
    
    col_idx_to_highlight = None
    for i_loop, a_col_val in enumerate(fixed_alphas_for_columns, start=1):
        if np.isclose(a_col_val, user_input_alpha):
            col_idx_to_highlight = i_loop
            break

    head = "".join(f"<th>{a}</th>" for a in fixed_alphas_for_columns)
    body = ""
    for r_df in rows: # r_df is the df for the current row
        row_html = f'<td id="chi_{r_df}_0">{r_df}</td>'
        for i_col_idx, a_col_value in enumerate(fixed_alphas_for_columns, start=1): # a_col_value is the alpha for this column
            val = stats.chi2.ppf(1 - a_col_value, r_df) # Cell value uses this column's alpha
            row_html += f'<td id="chi_{r_df}_{i_col_idx}">{val:.2f}</td>'
        body += f"<tr>{row_html}</tr>"

    table_code = f"<tr><th>df\\α</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, table_code)

    # highlight entire row for input df
    for i in range(len(fixed_alphas_for_columns)+1):
        html = style_cell(html, f"chi_{df}_{i}")
    
    if col_idx_to_highlight is not None:
        # highlight entire column for the matching alpha
        for rr_df_val in rows:
            html = style_cell(html, f"chi_{rr_df_val}_{col_idx_to_highlight}")
        # intersection
        html = style_cell(html, f"chi_{df}_{col_idx_to_highlight}", color="blue", px=3)
    return html

def chi_table(df: int, alpha: float):
    code = build_chi_table(df, alpha)
    st.markdown(container(code), unsafe_allow_html=True)

def chi_apa(chi_val: float, df: int, alpha: float):
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


def tab_chi():
    st.subheader("Tab 4 • Chi‑Square")

    c1, c2 = st.columns(2)
    with c1:
        chi_val = st.number_input("χ² statistic", value=7.88, key="chi_val")
        df = st.number_input("df", min_value=1, value=3, step=1, key="chi_df")
    with c2:
        # MODIFIED: Changed from st.selectbox to st.number_input for flexible alpha
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="chi_alpha", format="%.4f")

    if st.button("Update Plot", key="chi_plot"):
        st.pyplot(plot_chi(chi_val, df, alpha))

    st.write("**χ²‑table**")
    chi_table(df, alpha)
    chi_apa(chi_val, df, alpha)


###############################################################################
#                       TAB 5: Mann‑Whitney U (updated)
###############################################################################

def plot_u(u_calc, n1, n2, alpha, tail):
    """
    Normal approx. We'll interpret sign around the midpoint (mu) to decide
    shading for one-tailed. For two-tailed, shade both tails.
    """
    μ = n1*n2/2
    σ = np.sqrt(n1*n2*(n1+n2+1)/12)
    if σ == 0: σ = 1 # Avoid division by zero for plot range if n1 or n2 is too small

    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    # Ensure plot range is sensible even if u_calc is far from mu
    plot_min = min(μ - 4*σ, u_calc - σ if σ > 0 else u_calc -1) 
    plot_max = max(μ + 4*σ, u_calc + σ if σ > 0 else u_calc +1)
    if plot_min >= plot_max: # ensure min < max
        plot_min = u_calc - 2
        plot_max = u_calc + 2


    xs = np.linspace(plot_min, plot_max, 400)
    ys = stats.norm.pdf(xs, μ, σ)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H0")
    
    placed = [] # For place_label

    # Use z-critical value for determining critical U
    # For one-tailed, U_crit is on one side. Ucalc <= Ucrit (left) or Ucalc >= Ucrit (right)
    # For Mann-Whitney, smaller U is typically more significant.
    # So for a one-tailed test where H1 predicts group 1 < group 2, we look at small U.
    # If H1 predicts group 1 > group 2, we look at U' = n1n2 - U, and small U'. This means large U.

    if tail.startswith("one"):
        z_crit_one_tail = stats.norm.ppf(alpha) # for left tail
        # Critical U value is on the side of the distribution indicated by the hypothesis
        # If U_calc is small (less than mean), we test against U_crit_lower
        # If U_calc is large (greater than mean), we test against U_crit_higher
        
        u_crit_lower = μ + z_crit_one_tail * σ # This will be mu - |z|sigma
        u_crit_higher = μ - z_crit_one_tail * σ # This will be mu + |z|sigma
        
        # Decide which tail based on U_calc relative to μ
        if u_calc <= μ: # Test for significantly small U
            crit_val_effective = u_crit_lower
            ax.fill_between(xs[xs<=crit_val_effective], ys[xs<=crit_val_effective], color="red", alpha=0.3,
                            label="Reject H0")
            ax.axvline(crit_val_effective, color="green", ls="--")
            place_label(ax, placed, crit_val_effective, stats.norm.pdf(crit_val_effective, μ, σ)+0.002,
                        f"Ucrit={crit_val_effective:.2f}", color="green")
        else: # Test for significantly large U (small U')
            crit_val_effective = u_crit_higher
            ax.fill_between(xs[xs>=crit_val_effective], ys[xs>=crit_val_effective], color="red", alpha=0.3,
                            label="Reject H0")
            ax.axvline(crit_val_effective, color="green", ls="--")
            place_label(ax, placed, crit_val_effective, stats.norm.pdf(crit_val_effective, μ, σ)+0.002,
                        f"Ucrit={crit_val_effective:.2f}", color="green")
    else: # two-tailed
        z_crit_two_tail = stats.norm.ppf(alpha/2) # for lower tail z-score
        crit_val_lower = μ + z_crit_two_tail * σ  # e.g. mu + (-1.96)*sigma
        crit_val_higher = μ - z_crit_two_tail * σ # e.g. mu - (-1.96)*sigma = mu + 1.96*sigma
        
        ax.fill_between(xs[xs<=crit_val_lower], ys[xs<=crit_val_lower], color="red", alpha=0.3)
        ax.fill_between(xs[xs>=crit_val_higher], ys[xs>=crit_val_higher], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit_val_lower, color="green", ls="--")
        ax.axvline(crit_val_higher, color="green", ls="--")
        place_label(ax, placed, crit_val_lower, stats.norm.pdf(crit_val_lower, μ, σ)+0.002,
                    f"Ucrit={crit_val_lower:.2f}", color="green")
        place_label(ax, placed, crit_val_higher, stats.norm.pdf(crit_val_higher, μ, σ)+0.002,
                    f"Ucrit={crit_val_higher:.2f}", color="green")

    ax.axvline(u_calc, color="blue", ls="--")
    place_label(ax, placed, u_calc, stats.norm.pdf(u_calc, μ, σ)+0.002, # Adjusted y offset for label
                f"Ucalc={u_calc:.0f}", color="blue") # U is integer

    ax.set_xlabel("U")
    ax.set_ylabel("Approx. density")
    ax.legend()
    ax.set_title("Mann‑Whitney U (Normal Approximation)")
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def u_crit_exact(n1:int, n2:int, alpha:float, tail:str)->int: # This seems to be using normal approx in original code
    """Calculates critical U value using normal approximation, consistent with plot"""
    μ = n1*n2/2
    σ = np.sqrt(n1*n2*(n1+n2+1)/12)
    if σ == 0: return μ # Avoid issues if std dev is zero (e.g. very small N)

    if tail.startswith("one"):
        z_crit = stats.norm.ppf(alpha) # Gives negative z for typical alpha < 0.5
        # For U, smaller values are significant. So U_crit = mu + z_crit * sigma
        # If H1 is that group 1 < group 2 (small U), U_calc <= U_crit_lower
        # If H1 is that group 1 > group 2 (large U, small U'), U_calc >= U_crit_higher
        # The u_apa logic will compare u_val against the correct side.
        # This function should return the critical value for the lower tail.
        # The upper critical value is n1n2 - u_crit_lower.
        # Let's return the one that's closer to 0 (more extreme for U)
        u_critical = μ + z_crit * σ
    else: # two-tailed
        z_crit = stats.norm.ppf(alpha/2) # Negative z for lower tail
        u_critical = μ + z_crit * σ # Lower critical value
        # Upper critical value is n1n2 - u_critical_lower
    
    # For table display, usually the smaller U critical value is shown.
    return int(round(u_critical)) # Round to nearest int, U is discrete


def build_u_table(n1:int, n2:int, alpha:float, tail:str)->str: # Cell values depend on input alpha, tail
    rows = list(range(max(2,n1-5), n1+6))
    cols = list(range(max(2,n2-5), n2+6))
    
    col_idx_to_highlight = None
    if n2 in cols:
        col_idx_to_highlight = cols.index(n2)+1

    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r_n1 in rows: # r_n1 is n1 for current row
        row_html = f'<td id="u_{r_n1}_0">{r_n1}</td>'
        for i_col_idx, c_n2 in enumerate(cols, start=1): # c_n2 is n2 for current col
            # Table shows critical value for U (typically lower tail)
            # For one-tailed, it's U_alpha; for two-tailed, it's U_{alpha/2}
            # The u_crit_exact will use the alpha and tail type.
            # For table purpose, usually one-tailed alpha or alpha/2 for two-tailed is used for lookup.
            # Let's assume table shows critical value for specified alpha and tail.
            # The original `u_crit` function had `floor`. Let's maintain `u_crit_exact` as used by APA.
            # For the table, we want to show the critical U. For Mann-Whitney, typically the smaller U values are critical.
            # We use the alpha directly as passed if one-tailed, or alpha/2 if two-tailed, for the one-sided critical value.
            table_alpha_lookup = alpha if tail.startswith("one") else alpha/2
            
            # We need a consistent critical value for the table, usually the lower one for U.
            # stats.mannwhitneyu critical value tables are often for U_lower.
            # Using normal approximation for table values for consistency with plot and APA if exact tables aren't used.
            μ_cell = r_n1 * c_n2 / 2
            σ_cell = np.sqrt(r_n1 * c_n2 * (r_n1 + c_n2 + 1) / 12)
            if σ_cell == 0:
                val = int(round(μ_cell))
            else:
                z_critical_for_table = stats.norm.ppf(table_alpha_lookup) # This gives negative Z
                val = int(round(μ_cell + z_critical_for_table * σ_cell)) # Lower critical U

            row_html += f'<td id="u_{r_n1}_{i_col_idx}">{max(0,val)}</td>' # U cannot be negative
        body += f"<tr>{row_html}</tr>"

    code = f"<tr><th>n1\\n2</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, code)

    # highlight row for input n1
    for i in range(len(cols)+1):
        html = style_cell(html, f"u_{n1}_{i}")
    
    if col_idx_to_highlight is not None: # If input n2 is in displayed columns
        # highlight col for input n2
        for rr_n1_val in rows:
            html = style_cell(html, f"u_{rr_n1_val}_{col_idx_to_highlight}")
        # intersection
        html = style_cell(html, f"u_{n1}_{col_idx_to_highlight}", color="blue", px=3)
    return html


def u_table(n1:int, n2:int, alpha:float, tail:str):
    code = build_u_table(n1,n2,alpha,tail)
    st.markdown(container(code), unsafe_allow_html=True)


def u_apa(u_val: int, n1: int, n2: int, alpha: float, tail: str):
    """
    Show dynamic explanation and final APA lines for Mann‑Whitney U using Normal Approx.
    """
    μ = n1*n2/2
    σ = np.sqrt(n1*n2*(n1+n2+1)/12)

    if σ == 0: # Handle cases with zero standard deviation (e.g., n1 or n2 too small)
        p_calc = 1.0 if u_val != μ else 0.0 # Or some other appropriate p-value logic
        z_calc_for_p = 0
        st.warning("Standard deviation is zero, p-value calculation might be unreliable.")
    else:
        z_calc_for_p = (u_val - μ) / σ
    
    # Calculate p-value
    if tail.startswith("one"):
        # If U is small (u_val < mu), it's left tail: P(Z <= z_calc_for_p)
        # If U is large (u_val > mu), it's right tail: P(Z >= z_calc_for_p) or 1 - P(Z <= z_calc_for_p)
        if u_val <= μ:
            p_calc = stats.norm.cdf(z_calc_for_p)
        else:
            p_calc = stats.norm.sf(z_calc_for_p) # sf = 1 - cdf
    else: # two-tailed
        p_calc = 2 * stats.norm.sf(abs(z_calc_for_p)) # two-tailed p-value from z

    p_calc = np.clip(p_calc, 0, 1) # Ensure p-value is between 0 and 1

    # Determine critical values for U using normal approximation
    if tail.startswith("one"):
        z_crit_one_tail = stats.norm.ppf(alpha) # Negative for left tail
        # U_crit_lower for H1: group1 < group2 (small U values are evidence)
        # U_crit_higher for H1: group1 > group2 (large U values are evidence)
        u_crit_lower = μ + z_crit_one_tail * σ if σ > 0 else μ
        u_crit_higher = μ - z_crit_one_tail * σ if σ > 0 else μ # (mu + |z_crit|*sigma)

        if u_val <= μ: # Expecting small U
            crit_report = u_crit_lower
            reject = u_val <= crit_report 
        else: # Expecting large U
            crit_report = u_crit_higher
            reject = u_val >= crit_report
        
    else: # two-tailed
        z_crit_two_tail = stats.norm.ppf(alpha/2) # Negative Z for lower tail cut-off
        u_crit_lower = μ + z_crit_two_tail * σ if σ > 0 else μ
        u_crit_higher = μ - z_crit_two_tail * σ if σ > 0 else μ
        crit_report = f"{u_crit_lower:.0f} or {u_crit_higher:.0f}" # Report both for two-tailed
        reject = (u_val <= u_crit_lower) or (u_val >= u_crit_higher)

    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = "because U₍calc₎ was in the rejection region"
        reason_p = "because p < α"
    else:
        reason_stats = "because U₍calc₎ was not in the rejection region"
        reason_p = "because p ≥ α"

    # Explanation text
    normal_cdf_val = stats.norm.cdf(z_calc_for_p) if σ > 0 else (0.5 if u_val == μ else (0 if u_val < μ else 1))
    
    if tail.startswith("one"):
        if u_val <= μ: # Testing for significantly small U
            expl = (
                f"Normal approx: z = ({u_val:.0f} - {μ:.2f}) / {σ:.2f} = {z_calc_for_p:.2f}. "
                f"P(Z ≤ {z_calc_for_p:.2f}) = {normal_cdf_val:.4f}.\n\n"
                f"For a **one‑tailed** test (left tail, U ≤ μ), p = {normal_cdf_val:.4f}."
            )
        else: # Testing for significantly large U (U' small)
             expl = (
                f"Normal approx: z = ({u_val:.0f} - {μ:.2f}) / {σ:.2f} = {z_calc_for_p:.2f}. "
                f"P(Z ≥ {z_calc_for_p:.2f}) = {1-normal_cdf_val:.4f}.\n\n"
                f"For a **one‑tailed** test (right tail, U > μ), p = {1-normal_cdf_val:.4f}."
            )
    else: # two-tailed
        expl = (
            f"Normal approx: z = ({u_val:.0f} - {μ:.2f}) / {σ:.2f} = {z_calc_for_p:.2f}. "
            f"P(Z ≤ {abs(z_calc_for_p):.2f}) = {stats.norm.cdf(abs(z_calc_for_p)):.4f} (one tail of |z|).\n\n"
            f"For a **two‑tailed** test, p = 2 × P(Z ≥ |{z_calc_for_p:.2f}|) = {p_calc:.4f}."
        )
    st.write(expl)

    st.markdown(
        "**APA interpretation** \n"
        f"Calculated statistic: *U*={u_val}, *z*<sub>approx</sub>={z_calc_for_p:.2f}, *p*={p_calc:.3f}.  \n"
        f"Critical U region based on α={alpha:.3f}: For one-tailed (lower): U ≤ {u_crit_lower:.0f}; For one-tailed (upper): U ≥ {u_crit_higher:.0f}; For two-tailed: U ≤ {u_crit_lower:.0f} or U ≥ {u_crit_higher:.0f}. \n"
        f"Statistic comparison → H0 **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *U*={u_val}, *p*={p_calc:.3f} "
        f"({tail}, using normal approximation). The null hypothesis was **{decision}** at α={alpha:.2f}."
    )


def tab_u():
    st.subheader("Tab 5 • Mann‑Whitney U")

    c1, c2 = st.columns(2)
    with c1:
        u_val = st.number_input("U statistic", value=23, step=1, key="u_val") # U is integer
        n1 = st.number_input("n₁", min_value=1, value=10, step=1, key="u_n1") # Min_value 1 for formula
    with c2:
        n2 = st.number_input("n₂", min_value=1, value=12, step=1, key="u_n2") # Min_value 1 for formula
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="u_alpha", format="%.4f")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="u_tail")
    
    if n1 < 2 or n2 < 2: # Normal approx less reliable for very small N
        st.warning("Normal approximation for Mann-Whitney U is less reliable for n < 2. Table values might be less accurate.")


    if st.button("Update Plot", key="u_plot"):
        if n1*n2 == 0: # Prevent plot error if one n is 0
             st.error("n1 and n2 must be greater than 0 to plot.")
        else:
             st.pyplot(plot_u(u_val, n1, n2, alpha, tail))


    st.write("**U‑table (critical values based on Normal Approximation)**")
    ctable, cexp = st.columns([2,1])
    with ctable:
        if n1*n2 == 0:
             st.error("n1 and n2 must be greater than 0 for table.")
        else:
            u_table(n1, n2, alpha, tail)
            show_cumulative_note() # Note refers to general interpretation of cumulative tables
    with cexp:
        st.subheader("P‑value Calculation Explanation")
        if n1*n2 == 0:
             st.error("n1 and n2 must be greater than 0 for APA explanation.")
        else:
            u_apa(u_val, n1, n2, alpha, tail)


###############################################################################
#               TAB 6: Wilcoxon Signed‑Rank T (unchanged note/p)
###############################################################################

def w_crit_exact(n: int, alpha: float, tail: str)->int: # Using normal approximation for critical T
    """Calculates critical T value using normal approximation."""
    μ = n*(n+1)/4
    σ = np.sqrt(n*(n+1)*(2*n+1)/24)
    if σ == 0: return int(round(μ))

    alpha_lookup = alpha
    if tail.startswith("two"): # For two-tailed, we need alpha/2 in each tail
        alpha_lookup = alpha/2
    
    # Smaller T values are more significant. We want T_crit such that P(T <= T_crit) = alpha_lookup
    z_crit = stats.norm.ppf(alpha_lookup) # This will be negative for typical alpha
    t_critical = μ + z_crit * σ
    
    return int(round(max(0, t_critical))) # T cannot be negative

def plot_w(t_calc, n, alpha, tail):
    μ = n*(n+1)/4
    σ = np.sqrt(n*(n+1)*(2*n+1)/24)
    if σ == 0: σ = 1 # Avoid division by zero for plot range

    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    
    plot_min = min(μ - 4*σ, t_calc - σ if σ > 0 else t_calc -1)
    plot_max = max(μ + 4*σ, t_calc + σ if σ > 0 else t_calc +1)
    if plot_min >= plot_max:
        plot_min = t_calc - 2
        plot_max = t_calc + 2

    xs = np.linspace(plot_min, plot_max, 400)
    ys = stats.norm.pdf(xs, μ, σ)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")
    
    placed = [] # For place_label

    if tail.startswith("one"):
        # For Wilcoxon T, smaller T values are significant (reject H0 if T_calc <= T_crit)
        # H1: median difference is < 0 (or >0, depending on how differences are calculated)
        # The plot should show the critical region in the lower tail by default for T
        t_crit_lower_one_tail = w_crit_exact(n, alpha, "one‑tailed") # Gets T_alpha
        
        # The direction depends on how T is calculated (sum of positive or negative ranks)
        # Assuming T_calc is the smaller of T+ and T- (conventional for critical tables)
        # Then rejection is always T_calc <= T_crit
        crit_val_effective = t_crit_lower_one_tail
        ax.fill_between(xs[xs<=crit_val_effective], ys[xs<=crit_val_effective], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit_val_effective, color="green", ls="--")
        place_label(ax, placed, crit_val_effective, stats.norm.pdf(crit_val_effective, μ, σ)+0.002,
                        f"Tcrit={crit_val_effective}", color="green")

    else: # two-tailed
        # For two-tailed, T_calc <= T_crit(alpha/2)
        t_crit_lower_two_tail = w_crit_exact(n, alpha, "two‑tailed") # This internally uses alpha/2
        # The upper critical value is not simply mu + |z|sigma for T, as T is sum of ranks.
        # Max T is n(n+1)/2. If T_calc is compared to T_crit_lower.
        # Rejection region is T_calc <= T_crit_lower (alpha/2).
        # Some sources might also define an upper critical T_crit_higher for two-tailed.
        # For simplicity with normal approx of T, usually just T_calc vs T_crit(alpha/2) lower.
        # However, the original code had distinct lower and upper fills.
        # Let's use T_crit(alpha/2) for the lower boundary.
        # Upper boundary would be T_max - T_crit(alpha/2)
        
        crit_val_lower = t_crit_lower_two_tail
        crit_val_higher = n*(n+1)/2 - crit_val_lower # Symmetrical based on total sum of ranks

        ax.fill_between(xs[xs<=crit_val_lower], ys[xs<=crit_val_lower], color="red", alpha=0.3)
        ax.fill_between(xs[xs>=crit_val_higher], ys[xs>=crit_val_higher], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit_val_lower, color="green", ls="--")
        ax.axvline(crit_val_higher, color="green", ls="--")
        place_label(ax, placed, crit_val_lower, stats.norm.pdf(crit_val_lower, μ, σ)+0.002,
                    f"Tcrit_L={crit_val_lower}", color="green")
        place_label(ax, placed, crit_val_higher, stats.norm.pdf(crit_val_higher, μ, σ)+0.002,
                    f"Tcrit_U={crit_val_higher}", color="green")


    ax.axvline(t_calc, color="blue", ls="--")
    place_label(ax, placed, t_calc, stats.norm.pdf(t_calc, μ, σ)+0.002,
                f"Tcalc={t_calc}", color="blue")

    ax.set_xlabel("T")
    ax.set_ylabel("Approx. density")
    ax.legend()
    ax.set_title("Wilcoxon Signed‑Rank T (Normal Approximation)")
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def build_w_html(n:int, user_input_alpha:float, tail:str)->str: # user_input_alpha is 'alpha'
    """
    Table cell values are for fixed alpha levels. Highlighting attempts to match user_input_alpha.
    """
    rows = list(range(max(5,n-5), n+6))
    # These are the fixed alpha levels for which the table columns are generated
    # These alphas are typically used for one-tailed tests in Wilcoxon tables.
    # For two-tailed, the table is often entered with alpha/2.
    # Our w_crit_exact function takes the tail argument, so it handles this.
    fixed_alphas_for_columns = [0.005, 0.01, 0.025, 0.05] # Common Wilcoxon table alphas (one-tailed)
    # Adjusting headers if user provides two-tailed values to match common tables:
    # Example: if user alpha = 0.05 two-tailed, they might look for 0.025 column.

    col_idx_to_highlight = None
    # Highlighting logic needs to be clear. If user enters alpha=0.05 two-tailed,
    # this means 0.025 in each tail. So we should look for a 0.025 column.
    # If user enters alpha=0.05 one-tailed, we look for a 0.05 column.
    alpha_to_find_in_cols = user_input_alpha
    if tail.startswith("two"):
        alpha_to_find_in_cols = user_input_alpha / 2
        
    for i_loop, a_col_val in enumerate(fixed_alphas_for_columns, start=1):
        if np.isclose(a_col_val, alpha_to_find_in_cols):
            col_idx_to_highlight = i_loop
            break
    
    # Displaying table headers as one-tailed alpha levels
    head = "".join(f"<th>α₁={a}</th>" for a in fixed_alphas_for_columns)
    body = ""
    for r_n_val in rows: # r_n_val is N for current row
        row_html = f'<td id="w_{r_n_val}_0">{r_n_val}</td>'
        for i_col_idx, a_col_one_tailed in enumerate(fixed_alphas_for_columns, start=1):
            # Calculate critical T for this column's one-tailed alpha
            # The w_crit_exact will give the one-tailed critical value for a_col_one_tailed
            val = w_crit_exact(r_n_val, a_col_one_tailed, "one‑tailed") 
            row_html += f'<td id="w_{r_n_val}_{i_col_idx}">{val}</td>'
        body += f"<tr>{row_html}</tr>"

    table_code = f"<tr><th>N\\α (one-tailed)</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, table_code)

    # highlight entire row for input N
    for i in range(len(fixed_alphas_for_columns)+1):
        html = style_cell(html, f"w_{n}_{i}")
    
    if col_idx_to_highlight is not None:
        # highlight entire column for the matching alpha
        for rr_n_val_idx in rows:
            html = style_cell(html, f"w_{rr_n_val_idx}_{col_idx_to_highlight}")
        # intersection
        html = style_cell(html, f"w_{n}_{col_idx_to_highlight}", color="blue", px=3)
    return html


def w_table(n:int, alpha:float, tail:str):
    code = build_w_html(n, alpha, tail)
    st.markdown(container(code), unsafe_allow_html=True)


def w_apa(t_val: int, n: int, alpha: float, tail: str):
    """ APA explanation for Wilcoxon T using Normal Approximation """
    μ = n*(n+1)/4
    σ = np.sqrt(n*(n+1)*(2*n+1)/24)

    if σ == 0:
        p_calc = 1.0 if t_val != μ else 0.0
        z_calc_for_p = 0
        st.warning("Standard deviation is zero for Wilcoxon T; p-value may be unreliable.")
    else:
        # Apply continuity correction for normal approximation of discrete distribution
        # If T_calc < mu, use T_calc + 0.5. If T_calc > mu, use T_calc - 0.5
        # For p-value, we want area in tail.
        # If testing T_calc <= T_crit (small T is significant)
        if t_val + 0.5 < μ: # if t_val is clearly less than mean
             z_calc_for_p = (t_val + 0.5 - μ) / σ
        elif t_val - 0.5 > μ: # if t_val is clearly more than mean
             z_calc_for_p = (t_val - 0.5 - μ) / σ
        else: # t_val is close to mean, correction might cross mean
             z_calc_for_p = (t_val - μ) / σ # No correction if ambiguous or exactly mean
    
    # Calculate p-value based on z_calc_for_p
    # Wilcoxon T is typically one-sided (T <= T_crit) or two-sided (T <= T_crit_alpha/2)
    # p_calc from normal approx of T (sum of ranks)
    # If T is small, p = P(sum_ranks <= T_calc)
    
    # P-value calculation using z_calc_for_p (which has continuity correction)
    # The p-value corresponds to how extreme T_calc is.
    # If T_calc represents the smaller sum of ranks, then smaller T_calc is more extreme.
    # P(T <= t_val) using normal approx.
    p_one_sided_from_z = stats.norm.cdf(z_calc_for_p) # Prob of getting a Z as small or smaller

    if tail.startswith("one"):
        # Test is often directional, e.g. H1: median diff < 0 leads to small T.
        # So p-value is P(T <= T_calc)
        p_calc = p_one_sided_from_z 
    else: # two-tailed
        # Need to consider if T_calc could be from upper tail equivalent.
        # Usually T_calc is the smaller of T+ and T-.
        # So, p = 2 * P(T <= T_calc) if using normal approx symmetrically
        p_calc = 2 * min(p_one_sided_from_z, 1 - p_one_sided_from_z) # Symmetrical two-tailed p

    p_calc = np.clip(p_calc, 0, 1)

    # Critical T from normal approximation (this is T_crit for the lower tail)
    crit_t_lower = w_crit_exact(n, alpha, tail) # w_crit_exact handles one/two tail for alpha

    # Decision based on T_calc vs T_crit_lower (since T_calc is usually the smaller sum of ranks)
    reject = t_val <= crit_t_lower

    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = "because T₍calc₎ ≤ T₍crit₎"
        reason_p = "because p < α"
    else:
        reason_stats = "because T₍calc₎ > T₍crit₎"
        reason_p = "because p ≥ α"

    # Explanation
    expl = (
        f"Normal approx. for T (with continuity correction): z ≈ {z_calc_for_p:.2f}.\n"
        f"Using this z, the one-sided p-value P(T ≤ {t_val}) is approx. {p_one_sided_from_z:.4f}.\n"
    )
    if tail.startswith("one"):
        expl += f"For a **one‑tailed** test, p ≈ {p_calc:.4f}."
    else:
        expl += f"For a **two‑tailed** test, p ≈ 2 × min({p_one_sided_from_z:.4f}, {1-p_one_sided_from_z:.4f}) = {p_calc:.4f}."
    st.write(expl)

    st.markdown(
        "**APA interpretation** \n"
        f"Calculated statistic: *T*={t_val}, *z*<sub>approx</sub>={z_calc_for_p:.2f}, *p*={p_calc:.3f}.  \n"
        f"Critical statistic (approx.): T₍crit₎ ≤ {crit_t_lower} for α={alpha:.3f} ({tail}).  \n"
        f"Statistic comparison → H0 **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *T*={t_val}, *p*={p_calc:.3f} ({tail}, using normal approximation with continuity correction). "
        f"The null hypothesis was **{decision}** at α={alpha:.2f}."
    )


def tab_w():
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank T")

    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("T statistic (smaller of T+, T-)", value=15, step=1, key="w_val")
        n = st.number_input("N (non-zero diffs)", min_value=1, value=12, step=1, key="w_n")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="w_alpha", format="%.4f")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="w_tail")
    
    if n < 5 : # Normal approx less reliable for N < 5 or so
        st.warning("Normal approximation for Wilcoxon T is less reliable for N < 5.")


    if st.button("Update Plot", key="w_plot"):
        if n == 0:
            st.error("N must be greater than 0 to plot.")
        else:
            st.pyplot(plot_w(t_val, n, alpha, tail))

    st.write("**T‑table (critical values for one-tailed α, Normal Approx.)**")
    ctable, cexp = st.columns([2,1])
    with ctable:
        if n == 0:
            st.error("N must be greater than 0 for table.")
        else:
            w_table(n, alpha, tail)
            st.info(
            "Note: Wilcoxon T tables often list critical values for one-tailed tests. "
            "For a two-tailed test at significance level α, use the column for α/2 (one-tailed). "
            "Reject H₀ if your calculated T (the smaller of T+ or T-) is ≤ the critical T from the table. "
            "This app's table shows critical T values based on normal approximation for one-tailed α levels listed."
            )
    with cexp:
        st.subheader("P‑value Calculation Explanation")
        if n == 0:
            st.error("N must be greater than 0 for APA explanation.")
        else:
            w_apa(t_val, n, alpha, tail)


###############################################################################
#                           TAB 7: Binomial
###############################################################################

def critical_binom(n: int, p_null: float, alpha: float, tail: str): # p renamed to p_null
    """
    Calculates critical k values for binomial test.
    For one-tailed, returns (k_crit, None) or (None, k_crit).
    For two-tailed, returns (k_lo, k_hi).
    """
    k_lo, k_hi = None, None

    if tail.startswith("one"):
        # Determine direction based on a hypothetical observed k vs expected n*p_null
        # This is tricky without k_observed. Let's assume:
        # Default to finding k_crit for lower tail (P(X <= k_crit) <= alpha)
        # and k_crit for upper tail (P(X >= k_crit) <= alpha)
        # The APA function will then use the appropriate one.
        
        # Lower tail: find largest k_lo such that P(X <= k_lo) <= alpha
        current_sum_p_lower = 0.0
        for k_val_lower in range(n + 1):
            current_sum_p_lower += stats.binom.pmf(k_val_lower, n, p_null)
            if current_sum_p_lower > alpha:
                k_lo = k_val_lower -1 # The previous k was the last one to satisfy
                if k_lo < 0 : k_lo = 0 # if first k already exceeds alpha
                break
        if k_lo is None and current_sum_p_lower <= alpha : k_lo = n # All k satisfy

        # Upper tail: find smallest k_hi such that P(X >= k_hi) <= alpha
        # which is P(X < k_hi) >= 1 - alpha
        current_sum_p_upper = 0.0
        for k_val_upper in range(n, -1, -1): # from n down to 0
            current_sum_p_upper += stats.binom.pmf(k_val_upper, n, p_null)
            if current_sum_p_upper > alpha:
                k_hi = k_val_upper + 1 # The previous k (next one down) was the last to satisfy
                if k_hi > n : k_hi = n
                break
        if k_hi is None and current_sum_p_upper <= alpha : k_hi = 0 # All k (going down) satisfy

        return (k_lo if k_lo is not None else 0, k_hi if k_hi is not None else n) # return tuple of (lower, upper) one-tailed crits

    else: # two-tailed
        # Lower tail for two-tailed (alpha/2)
        target_alpha_half = alpha / 2
        current_sum_p_lower = 0.0
        for k_val_lower in range(n + 1):
            current_sum_p_lower += stats.binom.pmf(k_val_lower, n, p_null)
            if current_sum_p_lower > target_alpha_half:
                k_lo = k_val_lower -1
                if k_lo < 0: k_lo = 0
                break
        if k_lo is None and current_sum_p_lower <= target_alpha_half : k_lo = n


        # Upper tail for two-tailed (alpha/2)
        current_sum_p_upper = 0.0
        for k_val_upper in range(n, -1, -1):
            current_sum_p_upper += stats.binom.pmf(k_val_upper, n, p_null)
            if current_sum_p_upper > target_alpha_half:
                k_hi = k_val_upper + 1
                if k_hi > n: k_hi = n
                break
        if k_hi is None and current_sum_p_upper <= target_alpha_half: k_hi = 0
        
        return (k_lo if k_lo is not None else 0, k_hi if k_hi is not None else n)


def plot_binom(k_obs, n, p_null, alpha, tail): # k renamed to k_obs, p to p_null, added alpha, tail
    xs = np.arange(n+1)
    ys = stats.binom.pmf(xs, n, p_null)
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    ax.bar(xs, ys, color="lightgrey", label=f"P(X=k) for π={p_null}")
    
    # Highlight observed k
    if 0 <= k_obs <= n:
        ax.bar(k_obs, stats.binom.pmf(k_obs,n,p_null), color="blue", label=f"Observed k={k_obs}")

    # Add critical regions
    k_crit_lower, k_crit_upper = critical_binom(n, p_null, alpha, tail)

    crit_label_done = False
    if tail.startswith("one"):
        # For one-tailed, we need to know which direction from k_obs vs n*p_null
        expected_k = n * p_null
        if k_obs <= expected_k: # Assume lower tail test
            if k_crit_lower is not None:
                ax.bar(xs[xs <= k_crit_lower], ys[xs <= k_crit_lower], color="red", alpha=0.6, label=f"Reject H₀ (k≤{k_crit_lower})" if not crit_label_done else "")
                crit_label_done = True
        else: # Assume upper tail test
            if k_crit_upper is not None:
                ax.bar(xs[xs >= k_crit_upper], ys[xs >= k_crit_upper], color="red", alpha=0.6, label=f"Reject H₀ (k≥{k_crit_upper})" if not crit_label_done else "")
                crit_label_done = True
    else: # two-tailed
        if k_crit_lower is not None:
            ax.bar(xs[xs <= k_crit_lower], ys[xs <= k_crit_lower], color="red", alpha=0.6, label=f"Reject H₀ (k≤{k_crit_lower} or k≥{k_crit_upper})" if not crit_label_done else "")
            crit_label_done = True
        if k_crit_upper is not None and k_crit_upper > k_crit_lower: # Avoid double plotting if regions overlap weirdly
             ax.bar(xs[xs >= k_crit_upper], ys[xs >= k_crit_upper], color="red", alpha=0.6, label=f"Reject H₀ (k≤{k_crit_lower} or k≥{k_crit_upper})" if not crit_label_done else "")
             crit_label_done = True


    ax.set_xlabel("k (Number of Successes)")
    ax.set_ylabel("P(X=k)")
    ax.legend()
    ax.set_title(f"Binomial Distribution (n={n}, π₀={p_null}), α={alpha} ({tail})")
    ax.set_xticks(xs) # Ensure all k values are tickable if n is small
    fig.tight_layout()
    return fig


def build_binom_html(k_obs: int, n: int, p_null: float) -> str: # k renamed to k_obs, p to p_null
    """
    Single-step highlight for binomial table near k_obs±5
    """
    k_vals = list(range(max(0,k_obs-5), min(n,k_obs+5)+1))
    head = "<th>P(X=k)</th><th>P(X≤k) (CDF)</th><th>P(X≥k) (SF)</th>"
    body = ""
    for kv in k_vals:
        pmf_val = stats.binom.pmf(kv,n,p_null)
        cdf_val = stats.binom.cdf(kv,n,p_null)
        sf_val = stats.binom.sf(kv-1, n, p_null) if kv > 0 else 1.0 # P(X >= k) = 1 - P(X <= k-1)
        
        row_html = (
            f'<td id="b_{kv}_0">{kv}</td>'
            f'<td id="b_{kv}_1">{pmf_val:.4f}</td>'
            f'<td id="b_{kv}_2">{cdf_val:.4f}</td>'
            f'<td id="b_{kv}_3">{sf_val:.4f}</td>'
        )
        body += f"<tr>{row_html}</tr>"

    table_code = f"<tr><th>k</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, table_code)

    # highlight entire row for k_obs
    if k_obs in k_vals:
        for i in range(4): # 0 to 3 for k, pmf, cdf, sf
            html = style_cell(html, f"b_{k_obs}_{i}")
        # highlight pmf cell in blue
        html = style_cell(html, f"b_{k_obs}_1", color="blue", px=3)
    return html


def binom_table(k_obs: int, n: int, p_null: float): # k renamed to k_obs, p to p_null
    code = build_binom_html(k_obs,n,p_null)
    st.markdown(container(code, height=360), unsafe_allow_html=True) # Adjusted height


def binom_apa(k_obs: int, n: int, p_null: float, alpha: float, tail: str): # k renamed to k_obs, p to p_null
    """
    Show dynamic explanation and final APA lines for the binomial test.
    Uses exact p-value calculation.
    """
    # Calculate exact p-value
    if tail.startswith("one"):
        expected_k = n * p_null
        if k_obs <= expected_k: # Lower tail (observed k is less than or equal to expected)
            p_calc = stats.binom_test(k_obs, n, p_null, alternative='less')
            # p_calc = stats.binom.cdf(k_obs, n, p_null) # Sum P(X<=k_obs)
        else: # Upper tail (observed k is greater than expected)
            p_calc = stats.binom_test(k_obs, n, p_null, alternative='greater')
            # p_calc = stats.binom.sf(k_obs-1, n, p_null) # Sum P(X>=k_obs)
    else: # two-tailed
        p_calc = stats.binom_test(k_obs, n, p_null, alternative='two-sided')

    # Determine critical region
    # For APA, we state if k_obs falls in rejection region based on p_calc vs alpha
    reject = p_calc < alpha 
    
    # For descriptive critical values (k_lo, k_hi for two-tailed, or single k_crit for one-tailed)
    k_crit_lower_desc, k_crit_upper_desc = critical_binom(n, p_null, alpha, tail)


    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = "because the observed k was in the rejection region (p < α)"
        reason_p = "because p < α"
    else:
        reason_stats = "because the observed k was not in the rejection region (p ≥ α)"
        reason_p = "because p ≥ α"

    # Explanation text
    st.write(f"Observed k = {k_obs}, n = {n}, null proportion π₀ = {p_null:.2f}.")
    
    cdf_at_k_obs = stats.binom.cdf(k_obs, n, p_null)
    sf_at_k_obs = stats.binom.sf(k_obs -1, n, p_null) if k_obs > 0 else 1.0

    if tail.startswith("one"):
        expected_k = n * p_null
        if k_obs <= expected_k:
            st.write(
                f"For a **one‑tailed** test (e.g., H₁: π < π₀, observed k ≤ expected k), "
                f"p = P(X ≤ {k_obs}) = {cdf_at_k_obs:.4f} (exact binomial test yields p={p_calc:.4f})."
            )
            crit_region_desc = f"k ≤ {k_crit_lower_desc}"
        else:
            st.write(
                f"For a **one‑tailed** test (e.g., H₁: π > π₀, observed k > expected k), "
                f"p = P(X ≥ {k_obs}) = {sf_at_k_obs:.4f} (exact binomial test yields p={p_calc:.4f})."
            )
            crit_region_desc = f"k ≥ {k_crit_upper_desc}"
    else: # two-tailed
        st.write(
            f"For a **two‑tailed** test (H₁: π ≠ π₀), the exact binomial test sums probabilities in both tails "
            f"that are as or more extreme than observed k={k_obs}. This yields p={p_calc:.4f}."
        )
        crit_region_desc = f"k ≤ {k_crit_lower_desc} or k ≥ {k_crit_upper_desc}"


    st.markdown(
        "**APA interpretation** \n"
        f"Observed successes: k={k_obs}, trials n={n}. Null proportion π₀={p_null:.2f}.  \n"
        f"Exact binomial test: *p*={p_calc:.3f} ({tail}).  \n"
        f"Significance level α={alpha:.3f}. Critical region (approx.): {crit_region_desc}. \n"
        f"Decision based on p-value → H₀ **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** An exact binomial test indicated that the observed number of successes (k={k_obs}, n={n}) "
        f"{'was significantly different from' if reject else 'was not significantly different from'} "
        f"what was expected under the null hypothesis (π₀={p_null:.2f}), *p*={p_calc:.3f} ({tail}). "
        f"The null hypothesis was **{decision}** at α={alpha:.2f}."
    )


def tab_binom():
    st.subheader("Tab 7 • Binomial")

    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("n (trials)", min_value=1, value=20, step=1, key="b_n")
        p_null = st.number_input("π₀ (null proportion)", value=0.50,
                            step=0.01, min_value=0.00, max_value=1.00, key="b_p_null", format="%.2f") # p_null
    with c2:
        k_obs = st.number_input("k (observed successes)", min_value=0, value=10, step=1, key="b_k_obs") # k_obs
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="b_alpha", format="%.4f")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="b_tail")
    
    if k_obs > n:
        st.error("k (observed successes) cannot be greater than n (trials). Setting k = n.")
        k_obs = n


    if st.button("Update Plot", key="b_plot"):
        st.pyplot(plot_binom(k_obs,n,p_null, alpha, tail)) # Pass all args

    st.write("**Binomial table** (shows probabilities around observed k)")
    ctable, cexp = st.columns([2,1])
    with ctable:
        binom_table(k_obs,n,p_null) # Pass k_obs, n, p_null
        st.info(
            "Note: This table shows P(X=k), cumulative P(X≤k), and P(X≥k) for values of k around your observed k. "
            "It helps understand the distribution at π₀. "
            "For hypothesis testing, refer to the p-value calculation explanation."
            )
    with cexp:
        st.subheader("P‑value Calculation Explanation")
        binom_apa(k_obs,n,p_null,alpha,tail) # Pass all args


###############################################################################
#                                   MAIN
###############################################################################

def main():
    st.set_page_config("PSYC250 – Statistical Tables Explorer", layout="wide")
    st.title("Oli's – Statistical Table Explorer")

    tabs = st.tabs([
        "t‑Dist", "z‑Dist", "F‑Dist", "Chi‑Square",
        "Mann–Whitney U", "Wilcoxon T", "Binomial"
    ])

    with tabs[0]:
        tab_t()          # (new features)
    with tabs[1]:
        tab_z()          # (new features)
    with tabs[2]:
        tab_f()          # (no new note/p explanation needed)
    with tabs[3]:
        tab_chi()        # MODIFIED alpha input, table highlighting robustness
    with tabs[4]:
        tab_u()          # (new features) - Note: Mann-Whitney U plot/APA now uses Normal Approx.
    with tabs[5]:
        tab_w()          # MODIFIED table highlighting, plot/APA uses Normal Approx.
    with tabs[6]:
        tab_binom()      # (new features) - Binomial plot/APA updated for clarity


if __name__ == "__main__":
    main()
