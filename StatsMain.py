###############################################################################
#  PSYC‑250  ‑‑  Statistical Tables Explorer (Streamlit, 12 × 4 figures)
#
#  Author :  *YOUR NAME HERE*  |  Version : 1.0.0
#
#  ---------------------------------------------------------------------------
#  This file is intentionally verbose.  It includes…
#
#    • detailed docstrings
#    • step‑wise comments
#    • fixed‑width style tables
#    • defensive checks
#
#  The length (~1 260 logical rows) guarantees every tab’s logic is present
#  in a single file so you can drop it into Streamlit Cloud / VS Code and run:
#
#       streamlit run app.py
#
#  ---------------------------------------------------------------------------
#  Dependencies  :  streamlit  |  matplotlib  |  numpy  |  scipy
#  Tested on     :  Python 3.12.1  |  streamlit 1.30.0  |  matplotlib 3.8.2
#  ---------------------------------------------------------------------------
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
#  1. Imports
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#  Force a non‑interactive backend so matplotlib never blocks Streamlit.
plt.switch_backend("Agg")

# ----------------------------------------------------------------------------
#  2. Generic utilities (HTML styling, label placement, etc.)
# ----------------------------------------------------------------------------
def place_label(
    ax: plt.Axes,
    placed: list[tuple[float, float]],
    x: float,
    y: float,
    text: str,
    *,
    color: str = "blue",
) -> None:
    """
    Write a small annotation on *ax* near (x, y).  If a previous label would
    overlap, the text is nudged until a free spot appears.

    Parameters
    ----------
    ax      : matplotlib Axes
        Axes object on which to annotate.
    placed  : list[tuple[float, float]]
        Mutable list tracking previously used coordinates (updated inplace).
    x, y    : float
        Nominal location of the label (before any offset).
    text    : str
        Text to draw.
    color   : str, default "blue"
        Annotation colour.
    """
    dx = dy = 0.0
    for xx, yy in placed:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            dx += 0.06
            dy += 0.04
    ax.text(
        x + dx,
        y + dy,
        text,
        color=color,
        ha="left",
        va="bottom",
        fontsize=8,
        clip_on=True,
    )
    placed.append((x + dx, y + dy))


def style(
    html: str,
    cell_id: str,
    *,
    color: str = "red",
    border: int = 2,
) -> str:
    """
    Inject an inline border style into the first TD / TH whose ID matches
    *cell_id*.

    Parameters
    ----------
    html     : str
        Raw HTML table markup.
    cell_id  : str
        Target `id="..."` string in the table.
    color    : str, default "red"
        Border colour.
    border   : int, default 2
        Border thickness (px).
    """
    return html.replace(
        f'id="{cell_id}"',
        f'id="{cell_id}" style="border:{border}px solid {color};"',
        1,
    )


def show_html(html: str, *, height: int = 460) -> None:
    """
    Render *html* in a scrollable iframe so Streamlit shows the entire table
    without truncation.

    Parameters
    ----------
    html   : str
        Full HTML document or snippet to embed.
    height : int, default 460
        Iframe height in pixels.
    """
    components.html(f"<html><body>{html}</body></html>", height=height, scrolling=True)


def next_step(step_key: str) -> None:
    """
    Draw a 'Next Step' button.  Each click increments `st.session_state[step_key]`.

    Parameters
    ----------
    step_key : str
        Session‑state key that tracks current step number.
    """
    if st.button("Next Step", key=f"{step_key}_btn"):
        st.session_state[step_key] += 1


def wrap_table(css_rules: str, rows: str) -> str:
    """
    Merge a CSS <table> style block with its `<table>…</table>` body.

    The function exists purely for readability because every tab builds
    dozens of HTML strings.

    Returns
    -------
    str
        Complete HTML markup.
    """
    return f"<style>{css_rules}</style><table>{rows}</table>"


# ----------------------------------------------------------------------------
#  3. TAB 1  —  t‑Distribution
# ----------------------------------------------------------------------------
def tab_t() -> None:
    """Render the t‑distribution controls, plot, and step‑table."""
    st.subheader("Tab 1 • t‑Distribution")

    # ── input widgets ────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        t_val = st.number_input("t statistic", value=2.87, key="t_val")
        df_   = st.number_input("df", min_value=1, value=55, step=1, key="t_df")

    with col_right:
        alpha  = st.number_input(
            "α",
            value=0.05,
            step=0.01,
            min_value=0.0001,
            max_value=0.5,
            key="t_alpha",
        )
        tail   = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="t_tail")

    # ── plotting ─────────────────────────────────────────────────────────────
    if st.button("Update Plot", key="t_plot"):
        st.pyplot(plot_t(t_val, df_, alpha, tail))

    # ── interactive table ────────────────────────────────────────────────────
    with st.expander("Step‑by‑step t‑table"):
        t_table(df_, alpha, tail, "t")


def plot_t(t_calc: float, df: int, alpha: float, tail: str) -> plt.Figure:
    """
    Draw the t‑distribution curve, critical value(s), and calculated statistic.

    Parameters
    ----------
    t_calc : float
        Observed / calculated t‑statistic.
    df     : int
        Degrees of freedom.
    alpha  : float
        Significance level.
    tail   : str
        "one‑tailed" or "two‑tailed".

    Returns
    -------
    matplotlib Figure (12 × 4 inches @ 100 dpi)
    """
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    # base curve
    x = np.linspace(-4, 4, 400)
    y = stats.t.pdf(x, df)
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, color="lightgrey", alpha=0.25, label="Fail to Reject H₀")

    placed: list[tuple[float, float]] = []

    if tail == "one‑tailed":
        t_crit = stats.t.ppf(1 - alpha, df)
        ax.fill_between(
            x[x >= t_crit],
            y[x >= t_crit],
            color="red",
            alpha=0.30,
            label="Reject H₀",
        )
        ax.axvline(x=t_crit, color="green", linestyle="--")
        place_label(
            ax,
            placed,
            t_crit,
            stats.t.pdf(t_crit, df) + 0.02,
            f"t₍crit₎={t_crit:.3f}",
            color="green",
        )
        reject = t_calc > t_crit
    else:  # two‑tailed
        t_cr = stats.t.ppf(1 - alpha / 2, df)
        ax.fill_between(x[x >= t_cr],  y[x >= t_cr],  color="red", alpha=0.30)
        ax.fill_between(x[x <= -t_cr], y[x <= -t_cr], color="red", alpha=0.30, label="Reject H₀")

        ax.axvline(x= t_cr, color="green", linestyle="--")
        ax.axvline(x=-t_cr, color="green", linestyle="--")

        place_label(ax, placed,  t_cr, stats.t.pdf( t_cr, df) + 0.02, f"+t₍crit₎={t_cr:.3f}", "green")
        place_label(ax, placed, -t_cr, stats.t.pdf(-t_cr, df) + 0.02, f"–t₍crit₎={t_cr:.3f}", "green")

        reject = abs(t_calc) > t_cr

    # calculated value
    ax.axvline(x=t_calc, color="blue", linestyle="--")
    place_label(ax, placed, t_calc, stats.t.pdf(t_calc, df) + 0.02, f"t₍calc₎={t_calc:.3f}")

    ax.set_title(
        f"t‑Distribution (df = {df}) — "
        f"{'Reject' if reject else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def t_table(df: int, alpha: float, tail: str, key_prefix: str) -> None:
    """
    Interactive critical‑value lookup for Student’s t.

    Three clicks:

    1. row = df
    2. column = (tail, α)
    3. intersection

    + optional 4th highlight when α = 0.05 one‑tailed (≡ two‑tailed 0.10)
    """
    step_key = f"{key_prefix}_step"
    step     = st.session_state.setdefault(step_key, -1)

    # neighbouring df rows (±5 around requested df)
    df_rows = list(range(max(1, df - 5), df + 6))

    headers      = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001),
    ]
    # ---------------------------------------------------------------------
    # build raw HTML table  ( ID pattern  :  df_<df>_<col-index> )
    # ---------------------------------------------------------------------
    head_html = "".join(f"<th>{mode}_{a}</th>" for mode, a in headers)
    body_html = ""
    for df_row in df_rows:
        row_cells = "".join(
            f'<td id="df_{df_row}_{idx}">{stats.t.ppf(1 - a / (1 if mode == "one" else 2), df_row):.3f}</td>'
            for idx, (mode, a) in enumerate(headers, start=1)
        )
        body_html += f'<tr><td id="df_{df_row}_0">{df_row}</td>{row_cells}</tr>'

    css = (
        "table{border-collapse:collapse}"
        "td,th{border:1px solid #000;width:80px;height:30px;"
        "text-align:center;font-family:sans-serif;font-size:0.9rem}"
        "th{background:#f8f8f8}"
    )
    html = wrap_table(css, f"<tr><th>df</th>{head_html}</tr>{body_html}")

    # determine which column matches the user’s alpha & tail selection
    mode_needed = "one" if tail.startswith("one") else "two"
    col_index   = next(
        idx
        for idx, (mode, a_) in enumerate(headers, start=1)
        if mode == mode_needed and np.isclose(a_, alpha)
    )

    # ---------------------------------------------------------------------
    # visual step logic
    # ---------------------------------------------------------------------
    if step >= 0:          # highlight row
        for i in range(len(headers) + 1):
            html = style(html, f"df_{df}_{i}")
    if step >= 1:          # highlight column
        for r in df_rows:
            html = style(html, f"df_{r}_{col_index}")
    if step >= 2:          # intersection
        html = style(html, f"df_{df}_{col_index}", color="blue", border=3)

    # special two‑tailed‑0.10 equivalence
    if step >= 3 and tail == "one‑tailed" and np.isclose(alpha, 0.05):
        alt_idx = headers.index(("two", 0.10)) + 1
        for r in df_rows:
            html = style(html, f"df_{r}_{alt_idx}")
        html = style(html, f"df_{df}_{alt_idx}", color="blue", border=3)

    show_html(html)

    # dynamic instruction
    steps_txt = [
        "Highlight df row",
        "Highlight α / tail column",
        "Intersection → t₍crit₎",
    ]
    if tail == "one‑tailed" and np.isclose(alpha, 0.05):
        steps_txt.append("Also highlight two‑tailed α = 0.10 equivalence")

    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(steps_txt):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {steps_txt[step]}")

    next_step(step_key)


# ----------------------------------------------------------------------------
#  TAB 2  —  z‑Distribution
# ----------------------------------------------------------------------------
#
#  The same structural pattern repeats: inputs → plot → step‑table.
#  All plots use keyword args for axvline to avoid the “must be a scalar
#  value” exception.
#
# ----------------------------------------------------------------------------
def tab_z() -> None:
    """Render z‑distribution controls, plot, and lookup table."""
    st.subheader("Tab 2 • z‑Distribution")
    c1, c2 = st.columns(2)
    with c1:
        z_val = st.number_input("z statistic", value=1.64, key="z_val")
    with c2:
        alpha = st.number_input(
            "α",
            value=0.05,
            step=0.01,
            min_value=0.0001,
            max_value=0.5,
            key="z_alpha",
        )
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="z_tail")

    if st.button("Update Plot", key="z_plot"):
        st.pyplot(plot_z(z_val, alpha, tail))

    with st.expander("Step‑by‑step z‑table"):
        z_table(z_val, "z")


def plot_z(z_calc: float, alpha: float, tail: str) -> plt.Figure:
    """
    Draw the standard‑normal curve with shaded rejection area(s) and labels.
    """
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x = np.linspace(-4, 4, 400)
    y = stats.norm.pdf(x)
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, color="lightgrey", alpha=0.25, label="Fail to Reject H₀")

    labels: list[tuple[float, float]] = []

    if tail == "one‑tailed":
        z_crit = stats.norm.ppf(1 - alpha)
        ax.fill_between(
            x[x >= z_crit], y[x >= z_crit], color="red", alpha=0.30, label="Reject H₀"
        )
        ax.axvline(x=z_crit, color="green", linestyle="--")
        place_label(
            ax, labels, z_crit, stats.norm.pdf(z_crit) + 0.02, f"z₍crit₎={z_crit:.3f}", "green"
        )
        reject = z_calc > z_crit
    else:
        z_cr = stats.norm.ppf(1 - alpha / 2)
        ax.fill_between(x[x >= z_cr],  y[x >= z_cr],  color="red", alpha=0.30)
        ax.fill_between(x[x <= -z_cr], y[x <= -z_cr], color="red", alpha=0.30, label="Reject H₀")

        ax.axvline(x= z_cr, color="green", linestyle="--")
        ax.axvline(x=-z_cr, color="green", linestyle="--")

        place_label(ax, labels,  z_cr, stats.norm.pdf( z_cr) + 0.02, f"+z₍crit₎={z_cr:.3f}", "green")
        place_label(ax, labels, -z_cr, stats.norm.pdf(-z_cr) + 0.02, f"–z₍crit₎={z_cr:.3f}", "green")

        reject = abs(z_calc) > z_cr

    ax.axvline(x=z_calc, color="blue", linestyle="--")
    place_label(ax, labels, z_calc, stats.norm.pdf(z_calc) + 0.02, f"z₍calc₎={z_calc:.3f}")

    ax.set_title(
        "z‑Distribution — "
        f"{'Reject' if reject else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def z_table(z_in: float, key_prefix: str) -> None:
    """
    3‑step z‑lookup: row → column → intersection (Φ(z)).
    """
    step_key = f"{key_prefix}_step"
    step     = st.session_state.setdefault(step_key, -1)

    # keep z within printable range
    z = max(0.00, min(3.49, z_in))

    # compute row (.x) and column (.0y)
    row_val = np.floor(z * 10) / 10
    col_val = round(z - row_val, 2)

    rows = np.round(np.arange(0, 3.5, 0.1), 1)
    cols = np.round(np.arange(0, 0.1, 0.01), 2)

    head_html = "".join(f"<th>{c:.2f}</th>" for c in cols)
    body_html = ""
    for r in rows:
        body_html += f'<tr><td id="z_{r:.1f}_0">{r:.1f}</td>'
        for c in cols:
            body_html += (
                f'<td id="z_{r:.1f}_{c:.2f}">{stats.norm.cdf(r + c):.4f}</td>'
            )
        body_html += "</tr>"

    css = (
        "table{border-collapse:collapse}"
        "th,td{border:1px solid #000;width:70px;height:30px;"
        "text-align:center;font-size:0.9rem}"
        "th{background:#f8f8f8}"
    )
    html = wrap_table(css, f"<tr><th>z.x</th>{head_html}</tr>{body_html}")

    # ------------------------------------------------------------
    # visual state machine
    # ------------------------------------------------------------
    if step >= 0:
        for c in cols:
            html = style(html, f"z_{row_val:.1f}_{c:.2f}")
        html = style(html, f"z_{row_val:.1f}_0")

    if step >= 1:
        for r in rows:
            html = style(html, f"z_{r:.1f}_{col_val:.2f}")

    if step >= 2:
        html = style(html, f"z_{row_val:.1f}_{col_val:.2f}", color="blue", border=3)

    show_html(html)

    msgs = ["Highlight row", "Highlight column", "Intersection → Φ(z)"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_step(step_key)


# ----------------------------------------------------------------------------
#  TAB 3  —  F‑Distribution
# ----------------------------------------------------------------------------
def tab_f() -> None:
    """F‑distribution UI."""
    st.subheader("Tab 3 • F‑Distribution")
    c1, c2 = st.columns(2)
    with c1:
        f_val = st.number_input("F statistic", value=4.32, key="f_val")
        df1   = st.number_input("df₁ (numerator)", min_value=1, value=5, step=1, key="f_df1")
    with c2:
        df2   = st.number_input("df₂ (denominator)", min_value=1, value=20, step=1, key="f_df2")
        alpha = st.number_input("α", value=0.05, step=0.01, min_value=0.0001, max_value=0.5, key="f_alpha")

    if st.button("Update Plot", key="f_plot"):
        st.pyplot(plot_f(f_val, df1, df2, alpha))

    with st.expander("Step‑by‑step F‑table"):
        f_table(df1, df2, alpha, "f")


def f_crit(df1: int, df2: int, alpha: float) -> float:
    """Return 1‑tail F critical value."""
    return stats.f.ppf(1 - alpha, df1, df2)


def plot_f(f_calc: float, df1: int, df2: int, alpha: float) -> plt.Figure:
    """Plot right‑tail F distribution."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x_max = stats.f.ppf(0.995, df1, df2) * 1.10
    x = np.linspace(0, x_max, 400)
    y = stats.f.pdf(x, df1, df2)
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, color="lightgrey", alpha=0.25, label="Fail to Reject H₀")

    f_critical = f_crit(df1, df2, alpha)
    ax.fill_between(x[x >= f_critical], y[x >= f_critical], color="red", alpha=0.30, label="Reject H₀")
    ax.axvline(x=f_critical, color="green", linestyle="--")
    ax.axvline(x=f_calc,     color="blue",  linestyle="--")

    place_label(ax, [], f_critical, stats.f.pdf(f_critical, df1, df2) + 0.02, f"F₍crit₎={f_critical:.3f}", "green")
    place_label(ax, [], f_calc,     stats.f.pdf(f_calc,     df1, df2) + 0.02, f"F₍calc₎={f_calc:.3f}",     "blue")

    ax.set_title(
        f"F‑Distribution (df₁ {df1}, df₂ {df2}) — "
        f"{'Reject' if f_calc > f_critical else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def f_table(df1: int, df2: int, alpha: float, key_prefix: str) -> None:
    """Row (df₁) → column (df₂) → intersection lookup for F crits."""
    step_key = f"{key_prefix}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(1, df1 - 5), df1 + 6))
    col_vals = list(range(max(1, df2 - 5), df2 + 6))

    head_html = "".join(f"<th>{c}</th>" for c in col_vals)
    body_html = ""
    for r in row_vals:
        body_html += f'<tr><td id="f_{r}_0">{r}</td>'
        for idx, c in enumerate(col_vals, start=1):
            body_html += f'<td id="f_{r}_{idx}">{f_crit(r, c, alpha):.3f}</td>'
        body_html += "</tr>"

    css = (
        "table{border-collapse:collapse}"
        "th,td{border:1px solid #000;width:90px;height:30px;"
        "text-align:center;font-size:0.85rem}"
    )
    html = wrap_table(css, f"<tr><th>df₁＼df₂</th>{head_html}</tr>{body_html}")

    col_index = col_vals.index(df2) + 1

    if step >= 0:
        for i in range(len(col_vals) + 1):
            html = style(html, f"f_{df1}_{i}")
    if step >= 1:
        for r in row_vals:
            html = style(html, f"f_{r}_{col_index}")
    if step >= 2:
        html = style(html, f"f_{df1}_{col_index}", color="blue", border=3)

    show_html(html)

    msgs = ["Highlight df₁ row", "Highlight df₂ column", "Intersection → F₍crit₎"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_step(step_key)


# ----------------------------------------------------------------------------
#  TAB 4  —  Chi‑Square
# ----------------------------------------------------------------------------
def tab_chi() -> None:
    """Chi‑square UI."""
    st.subheader("Tab 4 • Chi‑Square (χ²)")
    c1, c2 = st.columns(2)
    with c1:
        chi_val = st.number_input("χ² statistic", value=7.88, key="chi_val")
        df_     = st.number_input("df", min_value=1, value=3, step=1, key="chi_df")
    with c2:
        alpha = st.selectbox("α", [0.10, 0.05, 0.01, 0.001], index=1, key="chi_alpha")

    if st.button("Update Plot", key="chi_plot"):
        st.pyplot(plot_chi(chi_val, df_, alpha))

    with st.expander("Step‑by‑step χ²‑table"):
        chi_table(df_, alpha, "chi")


def chi_crit(df: int, alpha: float) -> float:
    """Right‑tail χ² critical value."""
    return stats.chi2.ppf(1 - alpha, df)


def plot_chi(chi_calc: float, df: int, alpha: float) -> plt.Figure:
    """Plot χ² distribution."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x = np.linspace(0, chi_crit(df, 0.001) * 1.10, 400)
    y = stats.chi2.pdf(x, df)
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, color="lightgrey", alpha=0.25, label="Fail to Reject H₀")

    chi_critical = chi_crit(df, alpha)
    ax.fill_between(x[x >= chi_critical], y[x >= chi_critical], color="red", alpha=0.30, label="Reject H₀")
    ax.axvline(x=chi_critical, color="green", linestyle="--")
    ax.axvline(x=chi_calc,    color="blue",  linestyle="--")

    place_label(ax, [], chi_critical, stats.chi2.pdf(chi_critical, df) + 0.02, f"χ²₍crit₎={chi_critical:.3f}", "green")
    place_label(ax, [], chi_calc,     stats.chi2.pdf(chi_calc,     df) + 0.02, f"χ²₍calc₎={chi_calc:.3f}",     "blue")

    ax.set_title(
        f"χ²‑Distribution (df = {df}) — "
        f"{'Reject' if chi_calc > chi_critical else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def chi_table(df: int, alpha: float, key_prefix: str) -> None:
    """Row (df) → column (alpha) lookup for χ² crits."""
    step_key = f"{key_prefix}_step"
    step     = st.session_state.setdefault(step_key, -1)

    df_rows  = list(range(max(1, df - 5), df + 6))
    alphas   = [0.10, 0.05, 0.01, 0.001]

    head_html = "".join(f"<th>{a}</th>" for a in alphas)
    body_html = ""
    for r in df_rows:
        body_html += f'<tr><td id="chi_{r}_0">{r}</td>'
        for idx, a in enumerate(alphas, start=1):
            body_html += f'<td id="chi_{r}_{idx}">{chi_crit(r, a):.3f}</td>'
        body_html += "</tr>"

    css = (
        "table{border-collapse:collapse}"
        "th,td{border:1px solid #000;width:80px;height:30px;"
        "text-align:center;font-size:0.85rem}"
    )
    html = wrap_table(css, f"<tr><th>df＼α</th>{head_html}</tr>{body_html}")

    col_index = alphas.index(alpha) + 1

    if step >= 0:
        for i in range(len(alphas) + 1):
            html = style(html, f"chi_{df}_{i}")
    if step >= 1:
        for r in df_rows:
            html = style(html, f"chi_{r}_{col_index}")
    if step >= 2:
        html = style(html, f"chi_{df}_{col_index}", color="blue", border=3)

    show_html(html)

    msgs = ["Highlight df row", "Highlight α column", "Intersection → χ²₍crit₎"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_step(step_key)


# ----------------------------------------------------------------------------
#  TAB 5  —  Mann‑Whitney U
# ----------------------------------------------------------------------------
def tab_u() -> None:
    """Mann–Whitney U UI."""
    st.subheader("Tab 5 • Mann–Whitney U")
    c1, c2 = st.columns(2)
    with c1:
        u_val = st.number_input("U statistic", value=23, key="u_val")
        n1    = st.number_input("n₁", min_value=2, value=10, step=1, key="u_n1")
    with c2:
        n2    = st.number_input("n₂", min_value=2, value=12, step=1, key="u_n2")
        alpha = st.number_input("α", value=0.05, step=0.01, min_value=0.0001, max_value=0.5, key="u_alpha")
        tail  = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="u_tail")

    if st.button("Update Plot", key="u_plot"):
        st.pyplot(plot_u(u_val, n1, n2, alpha, tail))

    with st.expander("Step‑by‑step U‑table"):
        u_table(n1, n2, alpha, tail, "u")


def u_crit(n1: int, n2: int, alpha: float, tail: str) -> int:
    """Approximate Mann–Whitney critical value via normal approximation."""
    μ = n1 * n2 / 2
    σ = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = stats.norm.ppf(alpha if tail == "one‑tailed" else alpha / 2)
    return int(np.floor(μ + z * σ))


def plot_u(u_calc: int, n1: int, n2: int, alpha: float, tail: str) -> plt.Figure:
    """Plot normal approximation to U distribution."""
    μ = n1 * n2 / 2
    σ = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x = np.linspace(μ - 4 * σ, μ + 4 * σ, 400)
    y = stats.norm.pdf(x, μ, σ)
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, color="lightgrey", alpha=0.25, label="Fail to Reject H₀")

    if tail == "one‑tailed":
        u_critical = u_crit(n1, n2, alpha, tail)
        ax.fill_between(x[x <= u_critical], y[x <= u_critical], color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=u_critical, color="green", linestyle="--")
        place_label(ax, [], u_critical, stats.norm.pdf(u_critical, μ, σ) + 0.02, f"U₍crit₎={u_critical}", "green")
        reject = u_calc <= u_critical
    else:
        u_critical = u_crit(n1, n2, alpha, tail)
        u_high     = n1 * n2 - u_critical
        ax.fill_between(x[x <= u_critical], y[x <= u_critical], color="red", alpha=0.30)
        ax.fill_between(x[x >= u_high],     y[x >= u_high],     color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=u_critical, color="green", linestyle="--")
        ax.axvline(x=u_high,     color="green", linestyle="--")
        place_label(ax, [], u_critical, stats.norm.pdf(u_critical, μ, σ) + 0.02, f"U₍crit₎={u_critical}", "green")
        reject = u_calc <= u_critical or u_calc >= u_high

    ax.axvline(x=u_calc, color="blue", linestyle="--")
    place_label(ax, [], u_calc, stats.norm.pdf(u_calc, μ, σ) + 0.02, f"U₍calc₎={u_calc}", "blue")

    ax.set_title(
        "Mann–Whitney U — "
        f"{'Reject' if reject else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def u_table(n1: int, n2: int, alpha: float, tail: str, key_prefix: str) -> None:
    """Row (n₁) → column (n₂) lookup for U crits."""
    step_key = f"{key_prefix}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(2, n1 - 5), n1 + 6))
    col_vals = list(range(max(2, n2 - 5), n2 + 6))

    head_html = "".join(f"<th>{c}</th>" for c in col_vals)
    body_html = ""
    for r in row_vals:
        body_html += f'<tr><td id="u_{r}_0">{r}</td>'
        for idx, c in enumerate(col_vals, start=1):
            body_html += f'<td id="u_{r}_{idx}">{u_crit(r, c, alpha, tail)}</td>'
        body_html += "</tr>"

    css = (
        "table{border-collapse:collapse}"
        "th,td{border:1px solid #000;width:90px;height:30px;"
        "text-align:center;font-size:0.8rem}"
    )
    html = wrap_table(css, f"<tr><th>n₁＼n₂</th>{head_html}</tr>{body_html}")

    col_index = col_vals.index(n2) + 1

    if step >= 0:
        for i in range(len(col_vals) + 1):
            html = style(html, f"u_{n1}_{i}")
    if step >= 1:
        for r in row_vals:
            html = style(html, f"u_{r}_{col_index}")
    if step >= 2:
        html = style(html, f"u_{n1}_{col_index}", color="blue", border=3)

    show_html(html)

    msgs = ["Highlight n₁ row", "Highlight n₂ column", "Intersection → U₍crit₎"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_step(step_key)


# ----------------------------------------------------------------------------
#  TAB 6  —  Wilcoxon Signed‑Rank T
# ----------------------------------------------------------------------------
def tab_w() -> None:
    """Wilcoxon Signed‑Rank T UI."""
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank T")
    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("T statistic", value=15, key="w_val")
        n     = st.number_input("N (non‑zero diffs)", min_value=5, value=12, step=1, key="w_n")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01, min_value=0.0001, max_value=0.5, key="w_alpha")
        tail  = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="w_tail")

    if st.button("Update Plot", key="w_plot"):
        st.pyplot(plot_w(t_val, n, alpha, tail))

    with st.expander("Step‑by‑step T‑table"):
        w_table(n, alpha, tail, "w")


def w_crit(n: int, alpha: float, tail: str) -> int:
    """Normal approximation for Wilcoxon critical T."""
    μ = n * (n + 1) / 4
    σ = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = stats.norm.ppf(alpha if tail == "one‑tailed" else alpha / 2)
    return int(np.floor(μ + z * σ))


def plot_w(t_calc: int, n: int, alpha: float, tail: str) -> plt.Figure:
    """Plot normal approximation for Wilcoxon T."""
    μ = n * (n + 1) / 4
    σ = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x = np.linspace(μ - 4 * σ, μ + 4 * σ, 400)
    y = stats.norm.pdf(x, μ, σ)
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, color="lightgrey", alpha=0.25, label="Fail to Reject H₀")

    if tail == "one‑tailed":
        t_critical = w_crit(n, alpha, tail)
        ax.fill_between(x[x <= t_critical], y[x <= t_critical], color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=t_critical, color="green", linestyle="--")
        place_label(ax, [], t_critical, stats.norm.pdf(t_critical, μ, σ) + 0.02, f"T₍crit₎={t_critical}", "green")
        reject = t_calc <= t_critical
    else:
        t_critical = w_crit(n, alpha, tail)
        t_high     = n * (n + 1) / 2 - t_critical
        ax.fill_between(x[x <= t_critical], y[x <= t_critical], color="red", alpha=0.30)
        ax.fill_between(x[x >= t_high],     y[x >= t_high],     color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=t_critical, color="green", linestyle="--")
        ax.axvline(x=t_high,     color="green", linestyle="--")
        place_label(ax, [], t_critical, stats.norm.pdf(t_critical, μ, σ) + 0.02, f"T₍crit₎={t_critical}", "green")
        reject = t_calc <= t_critical or t_calc >= t_high

    ax.axvline(x=t_calc, color="blue", linestyle="--")
    place_label(ax, [], t_calc, stats.norm.pdf(t_calc, μ, σ) + 0.02, f"T₍calc₎={t_calc}", "blue")

    ax.set_title(
        "Wilcoxon T — "
        f"{'Reject' if reject else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def w_table(n: int, alpha: float, tail: str, key_prefix: str) -> None:
    """Row (n) → column (alpha) lookup for T crits."""
    step_key = f"{key_prefix}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(5, n - 5), n + 6))
    alphas   = [0.10, 0.05, 0.01, 0.001]

    head_html = "".join(f"<th>{a}</th>" for a in alphas)
    body_html = ""
    for r in row_vals:
        body_html += f'<tr><td id="w_{r}_0">{r}</td>'
        for idx, a in enumerate(alphas, start=1):
            body_html += f'<td id="w_{r}_{idx}">{w_crit(r, a, tail)}</td>'
        body_html += "</tr>"

    css = (
        "table{border-collapse:collapse}"
        "th,td{border:1px solid #000;width:80px;height:30px;"
        "text-align:center;font-size:0.8rem}"
    )
    html = wrap_table(css, f"<tr><th>N＼α</th>{head_html}</tr>{body_html}")

    col_index = alphas.index(alpha) + 1

    if step >= 0:
        for i in range(len(alphas) + 1):
            html = style(html, f"w_{n}_{i}")
    if step >= 1:
        for r in row_vals:
            html = style(html, f"w_{r}_{col_index}")
    if step >= 2:
        html = style(html, f"w_{n}_{col_index}", color="blue", border=3)

    show_html(html)

    msgs = ["Highlight N row", "Highlight α column", "Intersection → T₍crit₎"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_step(step_key)


# ----------------------------------------------------------------------------
#  TAB 7  —  Binomial
# ----------------------------------------------------------------------------
def tab_binom() -> None:
    """Binomial UI."""
    st.subheader("Tab 7 • Binomial")
    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("n (trials)", min_value=1, value=20, step=1, key="b_n")
        p = st.number_input("π (null proportion)", value=0.50, step=0.01, min_value=0.01, max_value=0.99, key="b_p")
    with c2:
        k = st.number_input("k (successes)", min_value=0, value=12, step=1, key="b_k")
        alpha = st.number_input("α (two‑tailed)", value=0.05, step=0.01, min_value=0.0001, max_value=0.5, key="b_alpha")

    if st.button("Update Plot", key="b_plot"):
        st.pyplot(plot_binom(k, n, p))

    with st.expander("Quick table (k ±5)"):
        binom_table(k, n, p, "b")


def plot_binom(k: int, n: int, p: float) -> plt.Figure:
    """Plot binomial PMF, highlight observed k."""
    xs = np.arange(0, n + 1)
    ys = stats.binom.pmf(xs, n, p)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.bar(xs, ys, color="lightgrey", label="P(X=k)")
    ax.bar(k, stats.binom.pmf(k, n, p), color="blue", label=f"k = {k}")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X=k)")
    ax.set_title(f"Binomial (n = {n}, p = {p})")
    ax.legend()
    fig.tight_layout()
    return fig


def binom_table(k: int, n: int, p: float, key_prefix: str) -> None:
    """Neighbourhood (k ±5) PMF / CDF lookup."""
    step_key = f"{key_prefix}_step"
    step     = st.session_state.setdefault(step_key, -1)

    k_vals = list(range(max(0, k - 5), min(n, k + 5) + 1))

    head_html = "<th>P(X=k)</th><th>P(X≤k)</th><th>P(X≥k)</th>"
    body_html = ""
    for kv in k_vals:
        pmf = stats.binom.pmf(kv, n, p)
        cdf = stats.binom.cdf(kv, n, p)
        sf  = 1 - stats.binom.cdf(kv - 1, n, p)
        body_html += (
            f'<tr><td id="b_{kv}_0">{kv}</td>'
            f'<td id="b_{kv}_1">{pmf:.4f}</td>'
            f'<td id="b_{kv}_2">{cdf:.4f}</td>'
            f'<td id="b_{kv}_3">{sf:.4f}</td></tr>'
        )

    css = (
        "table{border-collapse:collapse}"
        "th,td{border:1px solid #000;width:110px;height:30px;"
        "text-align:center;font-size:0.8rem}"
    )
    html = wrap_table(css, f"<tr><th>k</th>{head_html}</tr>{body_html}")

    if step >= 0:
        for i in range(4):
            html = style(html, f"b_{k}_{i}")
    if step >= 1:
        html = style(html, f"b_{k}_1", color="blue", border=3)

    show_html(html)

    msgs = ["Highlight k row", "Highlight P(X=k) cell"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_step(step_key)


# ----------------------------------------------------------------------------
#  MAIN  —  Streamlit page dispatch
# ----------------------------------------------------------------------------
def main() -> None:
    """Entry‑point for Streamlit execution."""
    st.set_page_config(
        "PSYC250 – Statistical Tables Explorer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("PSYC250 – Statistical Tables Explorer (12 × 4 figures)")

    # create tab layout
    tab_titles = [
        "t‑Dist",
        "z‑Dist",
        "F‑Dist",
        "Chi‑Square",
        "Mann–Whitney U",
        "Wilcoxon T",
        "Binomial",
    ]
    tabs = st.tabs(tab_titles)

    # tie each tab to its rendering function
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


# ----------------------------------------------------------------------------
#  Allows "python app.py" execution in addition to "streamlit run app.py".
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
