###############################################################################
#  PSYC‑250  –  Statistical Tables Explorer  (Streamlit, 12×4‑inch figures)
#  ---------------------------------------------------------------------------
#  Seven complete tabs (t, z, F, Chi‑Sq, Mann‑Whitney U, Wilcoxon T, Binomial)
#
#  Updates:
#   • Removed expanders so tables and plots are always shown (no drop‑downs).
#   • Improved cell highlighting with bolder borders + light background colors.
#   • "Next Step" animation now clearly highlights rows → columns → intersection
#     on successive button clicks (no timer delays).
#   • Ensured scrollable containers remain for large tables.
#
#  To run: copy into app.py, then:
#      streamlit run app.py
###############################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")  # headless backend

# ─────────────────────────────────────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def place_label(ax, placed, x, y, txt, *, color="blue"):
    """
    Place text on a plot, shifting slightly if colliding with previous labels.
    """
    dx = dy = 0.0
    for xx, yy in placed:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            dx += 0.06
            dy += 0.04
    ax.text(x + dx, y + dy, txt, color=color,
            ha="left", va="bottom", fontsize=8, clip_on=True)
    placed.append((x + dx, y + dy))


def style_cell(html: str, cid: str, *, color: str = "red", px: int = 2,
               bg: str = "#ffe5e5") -> str:
    """
    Give one <td id="cid"> a colored border and light background for clarity.
    """
    styled_str = (
        f'id="{cid}" style="border:{px}px solid {color};'
        f' background-color:{bg};"'
    )
    return html.replace(f'id="{cid}"', styled_str, 1)


def wrap_table(css: str, table: str) -> str:
    """
    Wrap a <table> with basic <style>.
    """
    return f"<style>{css}</style><table>{table}</table>"


def container(html: str, *, height: int = 460) -> str:
    """
    Scrollable wrapper for the table so it doesn't expand the entire page.
    """
    return f'<div style="overflow:auto; max-height:{height}px;">{html}</div>'


def multi_step(build_html, frames: int, *, key: str, height: int = 460):
    """
    Display an HTML table with row→column→cell highlighting in steps.
    Each click of "Next Step" increments the highlight phase.
    """
    if key not in st.session_state:
        st.session_state[key] = 0

    step = st.session_state[key]

    # Show current highlight step:
    st.markdown(container(build_html(step), height=height), unsafe_allow_html=True)

    # Step controls
    c1, c2 = st.columns(2)
    if step < frames - 1:
        if c1.button("Next Step", key=f"{key}_next"):
            st.session_state[key] += 1
            st.experimental_rerun()
    else:
        c1.success("All steps complete!")

    if c2.button("Reset Steps", key=f"{key}_reset"):
        st.session_state[key] = 0
        st.experimental_rerun()


CSS_BASE = (
    "table{border-collapse:collapse}"
    "th,td{border:1px solid #000;height:30px;text-align:center;"
    "font-family:sans-serif;font-size:0.9rem}"
    "th{background:#fafafa}"
)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 • t‑Distribution
# ════════════════════════════════════════════════════════════════════════════

def plot_t(t_calc, df, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(-4, 4, 400)
    ys = stats.t.pdf(xs, df)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")
    labels = []

    if tail.startswith("one"):
        crit = stats.t.ppf(1 - alpha, df)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                        color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(crit, color="green", ls="--")
        place_label(ax, labels, crit, stats.t.pdf(crit, df) + 0.02,
                    f"t₍crit₎={crit:.2f}", color="green")
    else:
        crit = stats.t.ppf(1 - alpha/2, df)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit], color="red", alpha=0.30)
        ax.fill_between(xs[xs <= -crit], ys[xs <= -crit], color="red", alpha=0.30,
                        label="Reject H₀")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(-crit, color="green", ls="--")
        place_label(ax, labels, crit, stats.t.pdf(crit, df) + 0.02,
                    f"+t₍crit₎={crit:.2f}", color="green")
        place_label(ax, labels, -crit, stats.t.pdf(-crit, df) + 0.02,
                    f"–t₍crit₎={crit:.2f}", color="green")

    ax.axvline(t_calc, color="blue", ls="--")
    place_label(ax, labels, t_calc, stats.t.pdf(t_calc, df) + 0.02,
                f"t₍calc₎={t_calc:.2f}", color="blue")

    ax.set_xlabel("t")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("t‑Distribution")
    fig.tight_layout()
    return fig


def build_t_html(df: int, alpha: float, tail: str, step: int) -> str:
    """
    Step 0: highlight row
    Step 1: highlight column
    Step 2: highlight intersection
    """
    rows = list(range(max(1, df-5), df+6))
    heads = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001)
    ]
    mode = "one" if tail.startswith("one") else "two"
    # find column that matches (mode, alpha)
    col = next(i for i, (m, a) in enumerate(heads, start=1)
               if m == mode and np.isclose(a, alpha))

    head = "".join(f"<th>{m}_{a}</th>" for m, a in heads)
    # build table body
    body = ""
    for r in rows:
        row_html = f'<td id="t_{r}_0">{r}</td>'
        for i, (m, a) in enumerate(heads, start=1):
            crit_val = stats.t.ppf(1 - a if m == "one" else 1 - a/2, r)
            row_html += f'<td id="t_{r}_{i}">{crit_val:.2f}</td>'
        body += f"<tr>{row_html}</tr>"

    html = wrap_table(CSS_BASE, f"<tr><th>df</th>{head}</tr>{body}")

    # highlight step by step
    if step >= 0:
        # highlight entire row for df
        for i in range(len(heads)+1):
            html = style_cell(html, f"t_{df}_{i}", color="red", px=2)
    if step >= 1:
        # highlight entire column
        for r in rows:
            html = style_cell(html, f"t_{r}_{col}", color="red", px=2)
    if step >= 2:
        # highlight intersection in blue
        html = style_cell(html, f"t_{df}_{col}", color="blue", px=3,
                          bg="#cce5ff")

    return html


def t_table(df: int, alpha: float, tail: str):
    multi_step(
        lambda s: build_t_html(df, alpha, tail, s),
        frames=3, key=f"t_{df}_{alpha}_{tail}"
    )


def t_apa(t_val: float, df: int, alpha: float, tail: str):
    """
    APA-7 style interpretation, including test-stat & p, critical & p,
    decision explanation (both), and final APA sentence.
    """
    if tail.startswith("one"):
        p_calc = stats.t.sf(abs(t_val), df)
        crit = stats.t.ppf(1 - alpha, df)
        reject = (t_val > crit)
    else:
        p_calc = stats.t.sf(abs(t_val), df) * 2
        crit = stats.t.ppf(1 - alpha/2, df)
        reject = (abs(t_val) > crit)

    p_crit = alpha
    decision = "rejected" if reject else "failed to reject"

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *t*({df}) = {t_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"Critical statistic: t₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Comparison of statistics → H₀ **{decision}**.  \n"
        f"Comparison of *p*‑values → H₀ **{decision}**.  \n"
        f"**APA 7 report:** *t*({df}) = {t_val:.2f}, *p* = {p_calc:.3f} "
        f"({tail}). The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )


def tab_t():
    st.subheader("Tab 1 • t‑Distribution")
    c1, c2 = st.columns(2)

    with c1:
        t_val = st.number_input("t statistic", value=2.87, key="t_val")
        df = st.number_input("df", min_value=1, value=55, step=1, key="t_df")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="t_alpha")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="t_tail")

    # Plot
    if st.button("Update Plot", key="t_plot"):
        st.pyplot(plot_t(t_val, df, alpha, tail))

    # Always show table + interpretation (no expander)
    st.write("**Step‑by‑step t‑table**")
    t_table(df, alpha, tail)

    t_apa(t_val, df, alpha, tail)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 • z‑Distribution
# ════════════════════════════════════════════════════════════════════════════

def plot_z(z_calc, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(-4, 4, 400)
    ys = stats.norm.pdf(xs)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")
    labels = []

    if tail.startswith("one"):
        crit = stats.norm.ppf(1 - alpha)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                        color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(crit, color="green", linestyle="--")
        place_label(ax, labels, crit, stats.norm.pdf(crit) + 0.02,
                    f"z₍crit₎={crit:.2f}", color="green")
    else:
        crit = stats.norm.ppf(1 - alpha/2)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit], color="red", alpha=0.30)
        ax.fill_between(xs[xs <= -crit], ys[xs <= -crit],
                        color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(crit, color="green", linestyle="--")
        ax.axvline(-crit, color="green", linestyle="--")
        place_label(ax, labels, crit, stats.norm.pdf(crit) + 0.02,
                    f"+z₍crit₎={crit:.2f}", color="green")
        place_label(ax, labels, -crit, stats.norm.pdf(-crit) + 0.02,
                    f"–z₍crit₎={crit:.2f}", color="green")

    ax.axvline(z_calc, color="blue", linestyle="--")
    place_label(ax, labels, z_calc, stats.norm.pdf(z_calc) + 0.02,
                f"z₍calc₎={z_calc:.2f}", color="blue")

    ax.set_xlabel("z")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("z‑Distribution")
    fig.tight_layout()
    return fig


def build_z_html(z: float, alpha: float, tail: str, step: int) -> str:
    """
    Step 0: highlight row
    Step 1: highlight column
    Step 2: highlight intersection
    Also, ±10 rows around the row that matches z, handle negatives.
    """
    z = np.clip(z, -3.49, 3.49)
    row = np.floor(z * 10) / 10
    col = round(z - row, 2)

    Rows = np.round(np.arange(-3.4, 3.5, 0.1), 1)
    Cols = np.round(np.arange(0, 0.1, 0.01), 2)

    # pick closest col
    col = min(Cols, key=lambda c: abs(c - col))

    # find index of row, then slice ±10
    idx = np.where(Rows == row)[0]
    if len(idx) == 0:
        idx = [len(Rows)//2]  # fallback
    irow = idx[0]
    rows = Rows[max(0, irow-10): irow+11]

    head = "".join(f"<th>{c:.2f}</th>" for c in Cols)
    body = ""
    for r in rows:
        row_html = f'<td id="z_{r:.1f}_0">{r:.1f}</td>'
        for c in Cols:
            val = stats.norm.cdf(r + c)
            row_html += f'<td id="z_{r:.1f}_{c:.2f}">{val:.4f}</td>'
        body += f"<tr>{row_html}</tr>"

    html = wrap_table(CSS_BASE, f"<tr><th>z.x</th>{head}</tr>{body}")

    if step >= 0:
        # highlight entire row
        for c in Cols:
            html = style_cell(html, f"z_{row:.1f}_{c:.2f}")
        html = style_cell(html, f"z_{row:.1f}_0")
    if step >= 1:
        # highlight entire column
        for rr in rows:
            html = style_cell(html, f"z_{rr:.1f}_{col:.2f}")
    if step >= 2:
        # highlight intersection
        html = style_cell(html, f"z_{row:.1f}_{col:.2f}", color="blue", px=3,
                          bg="#cce5ff")

    return html


def z_table(z_val: float, alpha: float, tail: str):
    multi_step(
        lambda s: build_z_html(z_val, alpha, tail, s),
        frames=3, key=f"z_{z_val}_{alpha}_{tail}"
    )


def z_apa(z_val: float, alpha: float, tail: str):
    p_calc = stats.norm.sf(abs(z_val)) * (1 if tail.startswith("one") else 2)
    if tail.startswith("one"):
        crit = stats.norm.ppf(1 - alpha)
        reject = (z_val > crit)
    else:
        crit = stats.norm.ppf(1 - alpha/2)
        reject = (abs(z_val) > crit)

    p_crit = alpha
    decision = "rejected" if reject else "failed to reject"

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *z* = {z_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"Critical statistic: z₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** *z* = {z_val:.2f}, *p* = {p_calc:.3f} "
        f"({tail}). The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )


def tab_z():
    st.subheader("Tab 2 • z‑Distribution")
    c1, c2 = st.columns(2)

    with c1:
        z_val = st.number_input("z statistic", value=1.64, key="z_val")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="z_alpha")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="z_tail")

    if st.button("Update Plot", key="z_plot"):
        st.pyplot(plot_z(z_val, alpha, tail))

    st.write("**Step‑by‑step z‑table**")
    z_table(z_val, alpha, tail)

    z_apa(z_val, alpha, tail)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 • F‑Distribution
# ════════════════════════════════════════════════════════════════════════════

def plot_f(f_calc, df1, df2, alpha):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(0, stats.f.ppf(0.995, df1, df2)*1.1, 400)
    ys = stats.f.pdf(xs, df1, df2)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")

    crit = stats.f.ppf(1 - alpha, df1, df2)
    ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                    color="red", alpha=0.30, label="Reject H₀")

    ax.axvline(crit, color="green", ls="--")
    ax.axvline(f_calc, color="blue", ls="--")
    place_label(ax, [], crit, stats.f.pdf(crit, df1, df2)+.02,
                f"F₍crit₎={crit:.2f}", color="green")
    place_label(ax, [], f_calc, stats.f.pdf(f_calc, df1, df2)+.02,
                f"F₍calc₎={f_calc:.2f}", color="blue")

    ax.set_xlabel("F")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"F‑Distribution (df₁={df1}, df₂={df2})")
    fig.tight_layout()
    return fig


def build_f_html(df1: int, df2: int, alpha: float, step: int) -> str:
    """
    Step 0: highlight row
    Step 1: highlight column
    Step 2: highlight intersection
    """
    rows = list(range(max(1, df1-5), df1+6))
    cols = list(range(max(1, df2-5), df2+6))
    col_idx = cols.index(df2) + 1

    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        row_html = f'<td id="f_{r}_0">{r}</td>'
        for i, c in enumerate(cols, start=1):
            crit_val = stats.f.ppf(1 - alpha, r, c)
            row_html += f'<td id="f_{r}_{i}">{crit_val:.2f}</td>'
        body += f"<tr>{row_html}</tr>"

    html = wrap_table(CSS_BASE, f"<tr><th>df₁＼df₂</th>{head}</tr>{body}")

    if step >= 0:
        for i in range(len(cols)+1):
            html = style_cell(html, f"f_{df1}_{i}")
    if step >= 1:
        for r in rows:
            html = style_cell(html, f"f_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"f_{df1}_{col_idx}", color="blue", px=3,
                          bg="#cce5ff")

    return html


def f_table(df1: int, df2: int, alpha: float):
    multi_step(
        lambda s: build_f_html(df1, df2, alpha, s),
        frames=3, key=f"f_{df1}_{df2}_{alpha}"
    )


def f_apa(f_val: float, df1: int, df2: int, alpha: float):
    p_calc = stats.f.sf(f_val, df1, df2)
    crit = stats.f.ppf(1 - alpha, df1, df2)
    p_crit = alpha
    reject = (f_val > crit)
    decision = "rejected" if reject else "failed to reject"

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *F*({df1}, {df2}) = {f_val:.2f}, "
        f"*p* = {p_calc:.3f}.  \n"
        f"Critical statistic: F₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** *F*({df1}, {df2}) = {f_val:.2f}, "
        f"*p* = {p_calc:.3f}. The null hypothesis was **{decision}** "
        f"at α = {alpha:.2f}."
    )


def tab_f():
    st.subheader("Tab 3 • F‑Distribution")
    c1, c2 = st.columns(2)

    with c1:
        f_val = st.number_input("F statistic", value=4.32, key="f_val")
        df1 = st.number_input("df₁ (numerator)", min_value=1, value=5,
                              step=1, key="f_df1")
    with c2:
        df2 = st.number_input("df₂ (denominator)", min_value=1, value=20,
                              step=1, key="f_df2")
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="f_alpha")

    if st.button("Update Plot", key="f_plot"):
        st.pyplot(plot_f(f_val, df1, df2, alpha))

    st.write("**Step‑by‑step F‑table**")
    f_table(df1, df2, alpha)

    f_apa(f_val, df1, df2, alpha)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 • Chi‑Square
# ════════════════════════════════════════════════════════════════════════════

def plot_chi(chi_calc, df, alpha):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(0, stats.chi2.ppf(0.995, df)*1.1, 400)
    ys = stats.chi2.pdf(xs, df)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")

    crit = stats.chi2.ppf(1 - alpha, df)
    ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                    color="red", alpha=0.30, label="Reject H₀")

    ax.axvline(crit, color="green", ls="--")
    ax.axvline(chi_calc, color="blue", ls="--")
    place_label(ax, [], crit, stats.chi2.pdf(crit, df) + .02,
                f"χ²₍crit₎={crit:.2f}", color="green")
    place_label(ax, [], chi_calc, stats.chi2.pdf(chi_calc, df) + .02,
                f"χ²₍calc₎={chi_calc:.2f}", color="blue")

    ax.set_xlabel("χ²")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"χ²‑Distribution (df = {df})")
    fig.tight_layout()
    return fig


def build_chi_html(df: int, alpha: float, step: int) -> str:
    """
    Step 0: highlight row
    Step 1: highlight column
    Step 2: highlight intersection
    """
    rows = list(range(max(1, df-5), df+6))
    alphas = [0.10, 0.05, 0.01, 0.001]
    col_idx = alphas.index(alpha) + 1

    head = "".join(f"<th>{a}</th>" for a in alphas)
    body = ""
    for r in rows:
        row_html = f'<td id="chi_{r}_0">{r}</td>'
        for i, a in enumerate(alphas, start=1):
            crit_val = stats.chi2.ppf(1 - a, r)
            row_html += f'<td id="chi_{r}_{i}">{crit_val:.2f}</td>'
        body += f"<tr>{row_html}</tr>"

    html = wrap_table(CSS_BASE, f"<tr><th>df＼α</th>{head}</tr>{body}")

    if step >= 0:
        for i in range(len(alphas)+1):
            html = style_cell(html, f"chi_{df}_{i}")
    if step >= 1:
        for r in rows:
            html = style_cell(html, f"chi_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"chi_{df}_{col_idx}", color="blue", px=3,
                          bg="#cce5ff")

    return html


def chi_table(df: int, alpha: float):
    multi_step(
        lambda s: build_chi_html(df, alpha, s),
        frames=3, key=f"chi_{df}_{alpha}"
    )


def chi_apa(chi_val: float, df: int, alpha: float):
    p_calc = stats.chi2.sf(chi_val, df)
    crit = stats.chi2.ppf(1 - alpha, df)
    p_crit = alpha
    reject = (chi_val > crit)
    decision = "rejected" if reject else "failed to reject"

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: χ²({df}) = {chi_val:.2f}, "
        f"*p* = {p_calc:.3f}.  \n"
        f"Critical statistic: χ²₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** χ²({df}) = {chi_val:.2f}, *p* = {p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at α = {alpha:.3f}."
    )


def tab_chi():
    st.subheader("Tab 4 • Chi‑Square (χ²)")
    c1, c2 = st.columns(2)

    with c1:
        chi_val = st.number_input("χ² statistic", value=7.88, key="chi_val")
        df = st.number_input("df", min_value=1, value=3, step=1, key="chi_df")
    with c2:
        alpha = st.selectbox("α", [0.10, 0.05, 0.01, 0.001],
                             index=1, key="chi_alpha")

    if st.button("Update Plot", key="chi_plot"):
        st.pyplot(plot_chi(chi_val, df, alpha))

    st.write("**Step‑by‑step χ²‑table**")
    chi_table(df, alpha)

    chi_apa(chi_val, df, alpha)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 • Mann‑Whitney U
# ════════════════════════════════════════════════════════════════════════════

def u_crit(n1: int, n2: int, alpha: float, tail: str) -> int:
    μ = n1 * n2 / 2
    σ = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if tail.startswith("one"):
        z = stats.norm.ppf(alpha)
    else:
        z = stats.norm.ppf(alpha/2)
    return int(np.floor(μ + z * σ))


def plot_u(u_calc, n1, n2, alpha, tail):
    μ = n1 * n2 / 2
    σ = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(μ - 4*σ, μ + 4*σ, 400)
    ys = stats.norm.pdf(xs, μ, σ)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")

    if tail.startswith("one"):
        crit = u_crit(n1, n2, alpha, tail)
        ax.fill_between(xs[xs <= crit], ys[xs <= crit],
                        color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(crit, color="green", ls="--")
    else:
        crit = u_crit(n1, n2, alpha, tail)
        hi = n1*n2 - crit
        ax.fill_between(xs[xs <= crit], ys[xs <= crit], color="red", alpha=0.30)
        ax.fill_between(xs[xs >= hi], ys[xs >= hi], color="red", alpha=0.30,
                        label="Reject H₀")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(hi, color="green", ls="--")

    ax.axvline(u_calc, color="blue", ls="--")
    place_label(ax, [], u_calc, stats.norm.pdf(u_calc, μ, σ)+.02,
                f"U₍calc₎={u_calc}", color="blue")

    ax.set_xlabel("U")
    ax.set_ylabel("Approx. density")
    ax.legend()
    ax.set_title("Mann‑Whitney U")
    fig.tight_layout()
    return fig


def build_u_html(n1: int, n2: int, alpha: float, tail: str, step: int) -> str:
    """
    Step 0: highlight row
    Step 1: highlight column
    Step 2: highlight intersection
    """
    rows = list(range(max(2, n1-5), n1+6))
    cols = list(range(max(2, n2-5), n2+6))
    col_idx = cols.index(n2) + 1

    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        row_html = f'<td id="u_{r}_0">{r}</td>'
        for i, c in enumerate(cols, start=1):
            val = u_crit(r, c, alpha, tail)
            row_html += f'<td id="u_{r}_{i}">{val}</td>'
        body += f"<tr>{row_html}</tr>"

    html = wrap_table(CSS_BASE, f"<tr><th>n₁＼n₂</th>{head}</tr>{body}")

    if step >= 0:
        for i in range(len(cols) + 1):
            html = style_cell(html, f"u_{n1}_{i}")
    if step >= 1:
        for r in rows:
            html = style_cell(html, f"u_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"u_{n1}_{col_idx}", color="blue", px=3,
                          bg="#cce5ff")

    return html


def u_table(n1: int, n2: int, alpha: float, tail: str):
    multi_step(
        lambda s: build_u_html(n1, n2, alpha, tail, s),
        frames=3, key=f"u_{n1}_{n2}_{alpha}_{tail}"
    )


def u_apa(u_val: int, n1: int, n2: int, alpha: float, tail: str):
    μ = n1 * n2 / 2
    σ = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    if tail.startswith("one"):
        p_calc = stats.norm.cdf((u_val - μ) / σ)
        crit = u_crit(n1, n2, alpha, tail)
        reject = (u_val <= crit)
    else:
        p_calc = stats.norm.sf(abs(u_val - μ) / σ) * 2
        crit = u_crit(n1, n2, alpha, tail)
        reject = (u_val <= crit) or (u_val >= n1*n2 - crit)

    p_crit = alpha
    decision = "rejected" if reject else "failed to reject"
    tail_txt = "one‑tailed" if tail.startswith("one") else "two‑tailed"

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *U* = {u_val}, *p* = {p_calc:.3f}.  \n"
        f"Critical statistic: U₍crit₎ = {crit}, *p* = {p_crit:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** *U* = {u_val}, *p* = {p_calc:.3f} "
        f"({tail_txt}). The null hypothesis was **{decision}** "
        f"at α = {alpha:.2f}."
    )


def tab_u():
    st.subheader("Tab 5 • Mann‑Whitney U")
    c1, c2 = st.columns(2)

    with c1:
        u_val = st.number_input("U statistic", value=23, key="u_val")
        n1 = st.number_input("n₁", min_value=2, value=10, step=1, key="u_n1")
    with c2:
        n2 = st.number_input("n₂", min_value=2, value=12, step=1, key="u_n2")
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="u_alpha")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="u_tail")

    if st.button("Update Plot", key="u_plot"):
        st.pyplot(plot_u(u_val, n1, n2, alpha, tail))

    st.write("**Step‑by‑step U‑table**")
    u_table(n1, n2, alpha, tail)

    u_apa(u_val, n1, n2, alpha, tail)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 • Wilcoxon Signed‑Rank T
# ════════════════════════════════════════════════════════════════════════════

def w_crit(n: int, alpha: float, tail: str) -> int:
    μ = n * (n + 1) / 4
    σ = np.sqrt(n * (n + 1) * (2*n + 1) / 24)
    if tail.startswith("one"):
        z = stats.norm.ppf(alpha)
    else:
        z = stats.norm.ppf(alpha/2)
    return int(np.floor(μ + z * σ))


def plot_w(t_calc, n, alpha, tail):
    μ = n * (n + 1) / 4
    σ = np.sqrt(n * (n + 1) * (2*n + 1) / 24)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(μ - 4*σ, μ + 4*σ, 400)
    ys = stats.norm.pdf(xs, μ, σ)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")

    if tail.startswith("one"):
        crit = w_crit(n, alpha, tail)
        ax.fill_between(xs[xs <= crit], ys[xs <= crit],
                        color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(crit, color="green", ls="--")
    else:
        crit = w_crit(n, alpha, tail)
        hi = n*(n+1)/2 - crit
        ax.fill_between(xs[xs <= crit], ys[xs <= crit], color="red", alpha=0.30)
        ax.fill_between(xs[xs >= hi], ys[xs >= hi], color="red", alpha=0.30,
                        label="Reject H₀")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(hi, color="green", ls="--")

    ax.axvline(t_calc, color="blue", ls="--")
    place_label(ax, [], t_calc, stats.norm.pdf(t_calc, μ, σ)+.02,
                f"T₍calc₎={t_calc}", color="blue")

    ax.set_xlabel("T")
    ax.set_ylabel("Approx. density")
    ax.legend()
    ax.set_title("Wilcoxon Signed‑Rank T")
    fig.tight_layout()
    return fig


def build_w_html(n: int, alpha: float, tail: str, step: int) -> str:
    """
    Step 0: highlight row
    Step 1: highlight column
    Step 2: highlight intersection
    """
    rows = list(range(max(5, n-5), n+6))
    alphas = [0.10, 0.05, 0.01, 0.001]
    col_idx = alphas.index(alpha) + 1

    head = "".join(f"<th>{a}</th>" for a in alphas)
    body = ""
    for r in rows:
        row_html = f'<td id="w_{r}_0">{r}</td>'
        for i, a in enumerate(alphas, start=1):
            val = w_crit(r, a, tail)
            row_html += f'<td id="w_{r}_{i}">{val}</td>'
        body += f"<tr>{row_html}</tr>"

    html = wrap_table(CSS_BASE, f"<tr><th>N＼α</th>{head}</tr>{body}")

    if step >= 0:
        for i in range(len(alphas)+1):
            html = style_cell(html, f"w_{n}_{i}")
    if step >= 1:
        for r in rows:
            html = style_cell(html, f"w_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"w_{n}_{col_idx}", color="blue", px=3,
                          bg="#cce5ff")

    return html


def w_table(n: int, alpha: float, tail: str):
    multi_step(
        lambda s: build_w_html(n, alpha, tail, s),
        frames=3, key=f"w_{n}_{alpha}_{tail}"
    )


def w_apa(t_val: int, n: int, alpha: float, tail: str):
    μ = n * (n + 1) / 4
    σ = np.sqrt(n * (n + 1) * (2*n + 1) / 24)

    if tail.startswith("one"):
        p_calc = stats.norm.cdf((t_val - μ) / σ)
        crit = w_crit(n, alpha, tail)
        reject = (t_val <= crit)
    else:
        p_calc = stats.norm.sf(abs(t_val - μ) / σ) * 2
        crit = w_crit(n, alpha, tail)
        reject = (t_val <= crit) or (t_val >= n*(n+1)/2 - crit)

    p_crit = alpha
    decision = "rejected" if reject else "failed to reject"
    tail_txt = "one‑tailed" if tail.startswith("one") else "two‑tailed"

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *T* = {t_val}, *p* = {p_calc:.3f}.  \n"
        f"Critical statistic: T₍crit₎ = {crit}, *p* = {p_crit:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** *T* = {t_val}, *p* = {p_calc:.3f} "
        f"({tail_txt}). The null hypothesis was **{decision}** "
        f"at α = {alpha:.2f}."
    )


def tab_w():
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank T")
    c1, c2 = st.columns(2)

    with c1:
        t_val = st.number_input("T statistic", value=15, key="w_val")
        n = st.number_input("N (non‑zero diffs)", min_value=5, value=12,
                            step=1, key="w_n")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="w_alpha")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="w_tail")

    if st.button("Update Plot", key="w_plot"):
        st.pyplot(plot_w(t_val, n, alpha, tail))

    st.write("**Step‑by‑step T‑table**")
    w_table(n, alpha, tail)

    w_apa(t_val, n, alpha, tail)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 7 • Binomial
# ════════════════════════════════════════════════════════════════════════════

def critical_binom(n: int, p: float, alpha: float):
    """
    Return (lower, upper) for two‑tailed binomial rejection region.
    """
    # lower bound
    cum = 0.0
    k_lo = 0
    for k in range(n+1):
        cum += stats.binom.pmf(k, n, p)
        if cum >= alpha/2:
            k_lo = k
            break
    # upper bound
    cum = 0.0
    k_hi = n
    for k in range(n, -1, -1):
        cum += stats.binom.pmf(k, n, p)
        if cum >= alpha/2:
            k_hi = k
            break
    return k_lo, k_hi


def plot_binom(k, n, p):
    xs = np.arange(n+1)
    ys = stats.binom.pmf(xs, n, p)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.bar(xs, ys, color="lightgrey")
    ax.bar(k, stats.binom.pmf(k, n, p), color="blue", label=f"k = {k}")

    ax.set_xlabel("k")
    ax.set_ylabel("P(X=k)")
    ax.legend()
    ax.set_title(f"Binomial (n = {n}, p = {p})")
    fig.tight_layout()
    return fig


def build_b_html(k: int, n: int, p: float, step: int) -> str:
    """
    Step 0: highlight entire row
    Step 1: highlight the pmf cell in blue
    (Binomial only uses 2 steps).
    """
    k_vals = list(range(max(0, k-5), min(n, k+5)+1))

    head = "<th>P(X=k)</th><th>P(X≤k)</th><th>P(X≥k)</th>"
    body = ""
    for kv in k_vals:
        pmf_val = stats.binom.pmf(kv, n, p)
        cdf_val = stats.binom.cdf(kv, n, p)
        if kv > 0:
            ccdf_val = 1 - stats.binom.cdf(kv-1, n, p)
        else:
            ccdf_val = 1.0

        row_html = (
            f'<td id="b_{kv}_0">{kv}</td>'
            f'<td id="b_{kv}_1">{pmf_val:.4f}</td>'
            f'<td id="b_{kv}_2">{cdf_val:.4f}</td>'
            f'<td id="b_{kv}_3">{ccdf_val:.4f}</td>'
        )
        body += f"<tr>{row_html}</tr>"

    html = wrap_table(CSS_BASE, f"<tr><th>k</th>{head}</tr>{body}")

    if step >= 0:
        # highlight entire row k
        for i in range(4):
            html = style_cell(html, f"b_{k}_{i}")
    if step >= 1:
        # highlight pmf cell in blue
        html = style_cell(html, f"b_{k}_1", color="blue", px=3, bg="#cce5ff")

    return html


def binom_table(k: int, n: int, p: float):
    multi_step(
        lambda s: build_b_html(k, n, p, s),
        frames=2, key=f"b_{k}_{n}_{p}"
    )


def binom_apa(k: int, n: int, p: float, alpha: float):
    cdf_val = stats.binom.cdf(k, n, p)
    # two-tailed p: double whichever tail is smaller
    p_calc = 2 * min(cdf_val, 1 - cdf_val + stats.binom.pmf(k, n, p))
    p_calc = min(p_calc, 1.0)

    k_lo, k_hi = critical_binom(n, p, alpha)
    p_crit = alpha
    reject = (k <= k_lo) or (k >= k_hi)
    decision = "rejected" if reject else "failed to reject"

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: k = {k}, *p* = {p_calc:.3f}.  \n"
        f"Critical region: k ≤ {k_lo} or k ≥ {k_hi}, *p* = {p_crit:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** Exact binomial test, *p* = {p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )


def tab_binom():
    st.subheader("Tab 7 • Binomial")
    c1, c2 = st.columns(2)

    with c1:
        n = st.number_input("n (trials)", min_value=1, value=20,
                            step=1, key="b_n")
        p = st.number_input("π (null proportion)", value=0.50,
                            step=0.01, min_value=0.01, max_value=0.99, key="b_p")
    with c2:
        k = st.number_input("k (successes)", min_value=0, value=12,
                            step=1, key="b_k")
        alpha = st.number_input("α (two‑tailed)", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="b_alpha")

    if st.button("Update Plot", key="b_plot"):
        st.pyplot(plot_binom(k, n, p))

    st.write("**Quick table (k ±5)**")
    binom_table(k, n, p)

    binom_apa(k, n, p, alpha)

# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config("PSYC250 – Statistical Tables Explorer", layout="wide")
    st.title("PSYC250 – Statistical Tables Explorer (12×4 figures)")

    tabs = st.tabs([
        "t‑Dist", "z‑Dist", "F‑Dist", "Chi‑Square",
        "Mann–Whitney U", "Wilcoxon T", "Binomial"
    ])

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
        tab_w()
    with tabs[6]:
        tab_binom()


if __name__ == "__main__":
    main()
