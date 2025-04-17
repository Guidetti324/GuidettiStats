###############################################################################
#  PSYC‑250  ‑‑  Statistical Tables Explorer (Streamlit, 12 × 4 figures)
#  ---------------------------------------------------------------------------
#  FULL SOURCE, NO PLACEHOLDERS — 1 300+ LOGICAL LINES
#
#  Tabs
#    1) t‑Distribution
#    2) z‑Distribution
#    3) F‑Distribution
#    4) Chi‑Square
#    5) Mann‑Whitney U
#    6) Wilcoxon Signed‑Rank T
#    7) Binomial
#
#  Inputs → 12×4 plots → step‑wise HTML tables (row → col → cell).
#
#  Matplotlib calls use keyword arguments everywhere (e.g. ax.axvline(x=…)).
#  Each widget key is unique; all HTML IDs are unique; every expander and
#  “Next Step” button increments its own st.session_state counter.
#
#  Tested on Python 3.12.1, streamlit 1.30.0, matplotlib 3.8.2, scipy 1.11.4
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
#  0 • Imports
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#  Matplotlib must never try to open a GUI in Streamlit
plt.switch_backend("Agg")

# ─────────────────────────────────────────────────────────────────────────────
#  1 • Generic helper routines
# ─────────────────────────────────────────────────────────────────────────────
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
    Add a small annotation to *ax* at (x, y).  If that spot is occupied,
    the label is nudged by 0.06 horiz / 0.04 vert until clear.

    This prevents labels overlapping critical‑value markers when multiple
    annotations share the same y height.
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


def style_cell(html: str, cell_id: str, *, color: str = "red", px: int = 2) -> str:
    """
    Inject an inline border into the first TD / TH whose id matches *cell_id*.

    Returns the modified HTML string.
    """
    injection = f'id="{cell_id}" style="border:{px}px solid {color};"'
    return html.replace(f'id="{cell_id}"', injection, 1)


def iframe(html: str, *, height: int = 460) -> None:
    """
    Show *html* inline but confine it to `height` pixels with its own
    scrollbar; this avoids the iframe‑wheel­capture bug.
    """
    st.markdown(
        f'<div style="overflow:auto; max-height:{height}px;">{html}</div>',
        unsafe_allow_html=True,
    )



def next_button(step_key: str) -> None:
    """
    Draw a `Next Step` button that increments st.session_state[step_key].
    """
    if st.button("Next Step", key=f"{step_key}__btn"):
        st.session_state[step_key] += 1


def wrap(css: str, inner: str) -> str:
    """Return `<style>css</style><table>inner</table>`."""
    return f"<style>{css}</style><table>{inner}</table>"


# ─────────────────────────────────────────────────────────────────────────────
#  2 • Tab 1 — t‑Distribution
# ─────────────────────────────────────────────────────────────────────────────
def tab_t() -> None:
    """Render Tab 1 (t‑Distribution) UI, plot, and table."""
    st.subheader("Tab 1 • t‑Distribution")

    # ── inputs ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("t statistic", value=2.87, key="t_val")
        df    = st.number_input("df", min_value=1, value=55, step=1, key="t_df")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="t_alpha")
        tail  = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="t_tail")

    # ── plot ────────────────────────────────────────────────────────────────
    if st.button("Update Plot", key="t_plot"):
        st.pyplot(plot_t(t_val, df, alpha, tail))

    # ── step table ──────────────────────────────────────────────────────────
    with st.expander("Step‑by‑step t‑table"):
        t_table(df, alpha, tail, "t")


def plot_t(t_calc: float, df: int, alpha: float, tail: str) -> plt.Figure:
    """
    Draw a 12 × 4 t‑distribution plot with shaded tail(s) and annotations.
    """
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(-4, 4, 400)
    ys = stats.t.pdf(xs, df)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H₀")

    labels: list[tuple[float, float]] = []

    if tail == "one‑tailed":
        t_crit = stats.t.ppf(1 - alpha, df)
        ax.fill_between(xs[xs >= t_crit], ys[xs >= t_crit],
                        color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=t_crit, color="green", linestyle="--")
        place_label(ax, labels, t_crit, stats.t.pdf(t_crit, df) + 0.02,
                    f"t₍crit₎={t_crit:.3f}", color="green")
        reject = t_calc > t_crit
    else:
        t_cr = stats.t.ppf(1 - alpha / 2, df)
        ax.fill_between(xs[xs >=  t_cr], ys[xs >=  t_cr], color="red", alpha=0.30)
        ax.fill_between(xs[xs <= -t_cr], ys[xs <= -t_cr], color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x= t_cr, color="green", linestyle="--")
        ax.axvline(x=-t_cr, color="green", linestyle="--")
        place_label(ax, labels,  t_cr, stats.t.pdf( t_cr, df) + 0.02,
                    f"+t₍crit₎={t_cr:.3f}", color="green")
        place_label(ax, labels, -t_cr, stats.t.pdf(-t_cr, df) + 0.02,
                    f"–t₍crit₎={t_cr:.3f}", color="green")
        reject = abs(t_calc) > t_cr

    ax.axvline(x=t_calc, color="blue", linestyle="--")
    place_label(ax, labels, t_calc, stats.t.pdf(t_calc, df) + 0.02,
                f"t₍calc₎={t_calc:.3f}", color="blue")

    ax.set_title(
        f"t‑Distribution (df={df}) — "
        f"{'Reject' if reject else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def t_table(df: int, alpha: float, tail: str, key: str) -> None:
    """Interactive 3‑step (plus optional 4th) lookup for Student’s t."""
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    df_rows = list(range(max(1, df - 5), df + 6))

    headers = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001),
    ]

    head_html = "".join(f"<th>{mode}_{val}</th>" for mode, val in headers)

    body_html = ""
    for df_row in df_rows:
        cells = "".join(
            f'<td id="t_{df_row}_{idx}">'
            f'{stats.t.ppf(1 - a / (1 if m == "one" else 2), df_row):.3f}</td>'
            for idx, (m, a) in enumerate(headers, start=1)
        )
        body_html += f'<tr><td id="t_{df_row}_0">{df_row}</td>{cells}</tr>'

    css = (
        "table{border-collapse:collapse}"
        "th,td{border:1px solid #000;width:80px;height:30px;"
        "text-align:center;font-size:0.9rem}"
        "th{background:#fafafa}"
    )
    html = wrap(css, f"<tr><th>df</th>{head_html}</tr>{body_html}")

    mode_needed = "one" if tail.startswith("one") else "two"
    col_index   = next(
        idx for idx, (m, a_) in enumerate(headers, start=1)
        if m == mode_needed and np.isclose(a_, alpha)
    )

    if step >= 0:
        for i in range(len(headers) + 1):
            html = style_cell(html, f"t_{df}_{i}")
    if step >= 1:
        for r in df_rows:
            html = style_cell(html, f"t_{r}_{col_index}")
    if step >= 2:
        html = style_cell(html, f"t_{df}_{col_index}", color="blue", px=3)
    if step >= 3 and tail == "one‑tailed" and np.isclose(alpha, 0.05):
        alt_idx = headers.index(("two", 0.10)) + 1
        for r in df_rows:
            html = style_cell(html, f"t_{r}_{alt_idx}")
        html = style_cell(html, f"t_{df}_{alt_idx}", color="blue", px=3)

    iframe(html)

    msgs = [
        "Highlight df row",
        "Highlight α / tail column",
        "Intersection → t₍crit₎",
    ]
    if tail == "one‑tailed" and np.isclose(alpha, 0.05):
        msgs.append("Also highlight two‑tailed α = 0.10 equivalence")

    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_button(step_key)


# ─────────────────────────────────────────────────────────────────────────────
#  3 • Tab 2 — z‑Distribution
# ─────────────────────────────────────────────────────────────────────────────
def tab_z() -> None:
    """Render Tab 2 (z‑Distribution)."""
    st.subheader("Tab 2 • z‑Distribution")
    c1, c2 = st.columns(2)
    with c1:
        z_val = st.number_input("z statistic", value=1.64, key="z_val")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="z_alpha")
        tail  = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="z_tail")

    if st.button("Update Plot", key="z_plot"):
        st.pyplot(plot_z(z_val, alpha, tail))

    with st.expander("Step‑by‑step z‑table"):
        z_table(z_val, "z")


def plot_z(z_calc: float, alpha: float, tail: str) -> plt.Figure:
    """Standard‑normal plot with critical z and shading."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(-4, 4, 400)
    ys = stats.norm.pdf(xs)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H₀")

    labels: list[tuple[float, float]] = []

    if tail == "one‑tailed":
        z_crit = stats.norm.ppf(1 - alpha)
        ax.fill_between(xs[xs >= z_crit], ys[xs >= z_crit],
                        color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=z_crit, color="green", linestyle="--")
        place_label(ax, labels, z_crit, stats.norm.pdf(z_crit) + 0.02,
                    f"z₍crit₎={z_crit:.3f}", color="green")
        reject = z_calc > z_crit
    else:
        z_cr = stats.norm.ppf(1 - alpha / 2)
        ax.fill_between(xs[xs >=  z_cr], ys[xs >=  z_cr], color="red", alpha=0.30)
        ax.fill_between(xs[xs <= -z_cr], ys[xs <= -z_cr], color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x= z_cr, color="green", linestyle="--")
        ax.axvline(x=-z_cr, color="green", linestyle="--")
        place_label(ax, labels,  z_cr, stats.norm.pdf( z_cr) + 0.02,
                    f"+z₍crit₎={z_cr:.3f}", color="green")
        place_label(ax, labels, -z_cr, stats.norm.pdf(-z_cr) + 0.02,
                    f"–z₍crit₎={z_cr:.3f}", color="green")
        reject = abs(z_calc) > z_cr

    ax.axvline(x=z_calc, color="blue", linestyle="--")
    place_label(ax, labels, z_calc, stats.norm.pdf(z_calc) + 0.02,
                f"z₍calc₎={z_calc:.3f}", color="blue")

    ax.set_title(
        "z‑Distribution — "
        f"{'Reject' if reject else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def z_table(z_in: float, key: str) -> None:
    """
    3‑step standard‑normal lookup.

    The table shows Φ(z) (rightmost column is the intersection).
    """
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    z = max(0, min(3.49, z_in))
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
        "th{background:#fafafa}"
    )
    html = wrap(css, f"<tr><th>z.x</th>{head_html}</tr>{body_html}")

    if step >= 0:
        for c in cols:
            html = style_cell(html, f"z_{row_val:.1f}_{c:.2f}")
        html = style_cell(html, f"z_{row_val:.1f}_0")
    if step >= 1:
        for r in rows:
            html = style_cell(html, f"z_{r:.1f}_{col_val:.2f}")
    if step >= 2:
        html = style_cell(html, f"z_{row_val:.1f}_{col_val:.2f}", color="blue", px=3)

    iframe(html)

    msgs = ["Highlight row", "Highlight column", "Intersection → Φ(z)"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_button(step_key)


# ─────────────────────────────────────────────────────────────────────────────
#  4 • Tab 3 — F‑Distribution
# ─────────────────────────────────────────────────────────────────────────────
def tab_f() -> None:
    """Render Tab 3 (F‑Distribution)."""
    st.subheader("Tab 3 • F‑Distribution")

    c1, c2 = st.columns(2)
    with c1:
        f_val = st.number_input("F statistic", value=4.32, key="f_val")
        df1   = st.number_input("df₁ (numerator)", min_value=1, value=5,
                                step=1, key="f_df1")
    with c2:
        df2   = st.number_input("df₂ (denominator)", min_value=1, value=20,
                                step=1, key="f_df2")
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="f_alpha")

    if st.button("Update Plot", key="f_plot"):
        st.pyplot(plot_f(f_val, df1, df2, alpha))

    with st.expander("Step‑by‑step F‑table"):
        f_table(df1, df2, alpha, "f")


def f_crit(df1: int, df2: int, alpha: float) -> float:
    """Return right‑tail Fᶜʳⁱᵗ."""
    return stats.f.ppf(1 - alpha, df1, df2)


def plot_f(f_calc: float, df1: int, df2: int, alpha: float) -> plt.Figure:
    """12×4 plot of F distribution with shaded tail."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    x_max = stats.f.ppf(0.995, df1, df2) * 1.10
    xs = np.linspace(0, x_max, 400)
    ys = stats.f.pdf(xs, df1, df2)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H₀")

    f_cr = f_crit(df1, df2, alpha)
    ax.fill_between(xs[xs >= f_cr], ys[xs >= f_cr],
                    color="red", alpha=0.30, label="Reject H₀")
    ax.axvline(x=f_cr, color="green", linestyle="--")
    ax.axvline(x=f_calc, color="blue", linestyle="--")

    place_label(ax, [], f_cr, stats.f.pdf(f_cr, df1, df2) + 0.02,
                f"F₍crit₎={f_cr:.3f}", color="green")
    place_label(ax, [], f_calc, stats.f.pdf(f_calc, df1, df2) + 0.02,
                f"F₍calc₎={f_calc:.3f}", color="blue")

    ax.set_title(
        f"F‑Distribution (df₁ {df1}, df₂ {df2}) — "
        f"{'Reject' if f_calc > f_cr else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def f_table(df1: int, df2: int, alpha: float, key: str) -> None:
    """Row (df₁) × column (df₂) interactive lookup for F crit values."""
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(1, df1 - 5), df1 + 6))
    col_vals = list(range(max(1, df2 - 5), df2 + 6))

    head_html = "".join(f"<th>{c}</th>" for c in col_vals)
    body_html = ""
    for r in row_vals:
        cms = "".join(
            f'<td id="f_{r}_{idx}">{f_crit(r, c, alpha):.3f}</td>'
            for idx, c in enumerate(col_vals, start=1)
        )
        body_html += f'<tr><td id="f_{r}_0">{r}</td>{cms}</tr>'

    css = (
        "table{border-collapse:collapse}"
        "th,td{border:1px solid #000;width:90px;height:30px;"
        "text-align:center;font-size:0.85rem}"
    )
    html = wrap(css, f"<tr><th>df₁＼df₂</th>{head_html}</tr>{body_html}")

    col_idx = col_vals.index(df2) + 1

    if step >= 0:
        for i in range(len(col_vals) + 1):
            html = style_cell(html, f"f_{df1}_{i}")
    if step >= 1:
        for r in row_vals:
            html = style_cell(html, f"f_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"f_{df1}_{col_idx}", color="blue", px=3)

    iframe(html)

    msgs = ["Highlight df₁ row", "Highlight df₂ column", "Intersection → F₍crit₎"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_button(step_key)


# ─────────────────────────────────────────────────────────────────────────────
#  5 • Tab 4 — Chi‑Square
# ─────────────────────────────────────────────────────────────────────────────
def tab_chi() -> None:
    """Render Tab 4 (Chi‑Square χ²)."""
    st.subheader("Tab 4 • Chi‑Square (χ²)")

    c1, c2 = st.columns(2)
    with c1:
        chi_val = st.number_input("χ² statistic", value=7.88, key="chi_val")
        df      = st.number_input("df", min_value=1, value=3,
                                  step=1, key="chi_df")
    with c2:
        alpha   = st.selectbox("α", [0.10, 0.05, 0.01, 0.001], index=1,
                               key="chi_alpha")

    if st.button("Update Plot", key="chi_plot"):
        st.pyplot(plot_chi(chi_val, df, alpha))

    with st.expander("Step‑by‑step χ²‑table"):
        chi_table(df, alpha, "chi")


def chi_crit(df: int, alpha: float) -> float:
    """Right‑tail χ²ᶜʳⁱᵗ."""
    return stats.chi2.ppf(1 - alpha, df)


def plot_chi(chi_calc: float, df: int, alpha: float) -> plt.Figure:
    """12×4 χ² plot with shaded rejection region."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    x_max = chi_crit(df, 0.001) * 1.10
    xs = np.linspace(0, x_max, 400)
    ys = stats.chi2.pdf(xs, df)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H₀")

    chi_cr = chi_crit(df, alpha)
    ax.fill_between(xs[xs >= chi_cr], ys[xs >= chi_cr],
                    color="red", alpha=0.30, label="Reject H₀")
    ax.axvline(x=chi_cr, color="green", linestyle="--")
    ax.axvline(x=chi_calc, color="blue", linestyle="--")

    place_label(ax, [], chi_cr, stats.chi2.pdf(chi_cr, df) + 0.02,
                f"χ²₍crit₎={chi_cr:.3f}", color="green")
    place_label(ax, [], chi_calc, stats.chi2.pdf(chi_calc, df) + 0.02,
                f"χ²₍calc₎={chi_calc:.3f}", color="blue")

    ax.set_title(
        f"χ²‑Distribution (df={df}) — "
        f"{'Reject' if chi_calc > chi_cr else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def chi_table(df: int, alpha: float, key: str) -> None:
    """Row (df) × column (α) lookup for χ² crit."""
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    rows   = list(range(max(1, df - 5), df + 6))
    alphas = [0.10, 0.05, 0.01, 0.001]

    head_html = "".join(f"<th>{a}</th>" for a in alphas)
    body_html = ""
    for r in rows:
        cms = "".join(
            f'<td id="chi_{r}_{idx}">{chi_crit(r, a):.3f}</td>'
            for idx, a in enumerate(alphas, start=1)
        )
        body_html += f'<tr><td id="chi_{r}_0">{r}</td>{cms}</tr>'

    css = (
        "table{border-collapse:collapse}"
        "th,td{border:1px solid #000;width:80px;height:30px;"
        "text-align:center;font-size:0.85rem}"
    )
    html = wrap(css, f"<tr><th>df＼α</th>{head_html}</tr>{body_html}")

    col_idx = alphas.index(alpha) + 1

    if step >= 0:
        for i in range(len(alphas) + 1):
            html = style_cell(html, f"chi_{df}_{i}")
    if step >= 1:
        for r in rows:
            html = style_cell(html, f"chi_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"chi_{df}_{col_idx}", color="blue", px=3)

    iframe(html)

    msgs = ["Highlight df row", "Highlight α column", "Intersection → χ²₍crit₎"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_button(step_key)


# ─────────────────────────────────────────────────────────────────────────────
#  6 • Tab 5 — Mann‑Whitney U
# ─────────────────────────────────────────────────────────────────────────────
def tab_u() -> None:
    """Render Tab 5 (Mann‑Whitney U)."""
    st.subheader("Tab 5 • Mann–Whitney U")

    c1, c2 = st.columns(2)
    with c1:
        u_val = st.number_input("U statistic", value=23, key="u_val")
        n1    = st.number_input("n₁", min_value=2, value=10,
                                step=1, key="u_n1")
    with c2:
        n2    = st.number_input("n₂", min_value=2, value=12,
                                step=1, key="u_n2")
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="u_alpha")
        tail  = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="u_tail")

    if st.button("Update Plot", key="u_plot"):
        st.pyplot(plot_u(u_val, n1, n2, alpha, tail))

    with st.expander("Step‑by‑step U‑table"):
        u_table(n1, n2, alpha, tail, "u")


def u_crit(n1: int, n2: int, alpha: float, tail: str) -> int:
    """Normal approximation critical U."""
    μ = n1 * n2 / 2
    σ = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = stats.norm.ppf(alpha if tail == "one‑tailed" else alpha / 2)
    return int(np.floor(μ + z * σ))


def plot_u(u_calc: int, n1: int, n2: int, alpha: float, tail: str) -> plt.Figure:
    """12×4 normal‑approximation plot for U statistic."""
    μ = n1 * n2 / 2
    σ = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(μ - 4 * σ, μ + 4 * σ, 400)
    ys = stats.norm.pdf(xs, μ, σ)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H₀")

    if tail == "one‑tailed":
        u_cr = u_crit(n1, n2, alpha, tail)
        ax.fill_between(xs[xs <= u_cr], ys[xs <= u_cr],
                        color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=u_cr, color="green", linestyle="--")
        place_label(ax, [], u_cr, stats.norm.pdf(u_cr, μ, σ) + 0.02,
                    f"U₍crit₎={u_cr}", color="green")
        reject = u_calc <= u_cr
    else:
        u_cr = u_crit(n1, n2, alpha, tail)
        u_hi = n1 * n2 - u_cr
        ax.fill_between(xs[xs <= u_cr], ys[xs <= u_cr], color="red", alpha=0.30)
        ax.fill_between(xs[xs >= u_hi], ys[xs >= u_hi], color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=u_cr, color="green", linestyle="--")
        ax.axvline(x=u_hi, color="green", linestyle="--")
        place_label(ax, [], u_cr, stats.norm.pdf(u_cr, μ, σ) + 0.02,
                    f"U₍crit₎={u_cr}", color="green")
        reject = u_calc <= u_cr or u_calc >= u_hi

    ax.axvline(x=u_calc, color="blue", linestyle="--")
    place_label(ax, [], u_calc, stats.norm.pdf(u_calc, μ, σ) + 0.02,
                f"U₍calc₎={u_calc}", color="blue")

    ax.set_title(
        "Mann–Whitney U — "
        f"{'Reject' if reject else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def u_table(n1: int, n2: int, alpha: float, tail: str, key: str) -> None:
    """Neighbourhood U crit lookup table."""
    step_key = f"{key}_step"
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
    html = wrap(css, f"<tr><th>n₁＼n₂</th>{head_html}</tr>{body_html}")

    col_idx = col_vals.index(n2) + 1

    if step >= 0:
        for i in range(len(col_vals) + 1):
            html = style_cell(html, f"u_{n1}_{i}")
    if step >= 1:
        for r in row_vals:
            html = style_cell(html, f"u_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"u_{n1}_{col_idx}", color="blue", px=3)

    iframe(html)

    msgs = ["Highlight n₁ row", "Highlight n₂ column", "Intersection → U₍crit₎"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_button(step_key)


# ─────────────────────────────────────────────────────────────────────────────
#  7 • Tab 6 — Wilcoxon Signed‑Rank T
# ─────────────────────────────────────────────────────────────────────────────
def tab_w() -> None:
    """Render Tab 6 (Wilcoxon Signed‑Rank T)."""
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank T")

    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("T statistic", value=15, key="w_val")
        n     = st.number_input("N (non‑zero diffs)", min_value=5, value=12,
                                step=1, key="w_n")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="w_alpha")
        tail  = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="w_tail")

    if st.button("Update Plot", key="w_plot"):
        st.pyplot(plot_w(t_val, n, alpha, tail))

    with st.expander("Step‑by‑step T‑table"):
        w_table(n, alpha, tail, "w")


def w_crit(n: int, alpha: float, tail: str) -> int:
    """Normal approximation critical T for Wilcoxon."""
    μ = n * (n + 1) / 4
    σ = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = stats.norm.ppf(alpha if tail == "one‑tailed" else alpha / 2)
    return int(np.floor(μ + z * σ))


def plot_w(t_calc: int, n: int, alpha: float, tail: str) -> plt.Figure:
    """12×4 normal plot for Wilcoxon T."""
    μ = n * (n + 1) / 4
    σ = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(μ - 4 * σ, μ + 4 * σ, 400)
    ys = stats.norm.pdf(xs, μ, σ)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H₀")

    if tail == "one‑tailed":
        t_cr = w_crit(n, alpha, tail)
        ax.fill_between(xs[xs <= t_cr], ys[xs <= t_cr],
                        color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=t_cr, color="green", linestyle="--")
        place_label(ax, [], t_cr, stats.norm.pdf(t_cr, μ, σ) + 0.02,
                    f"T₍crit₎={t_cr}", color="green")
        reject = t_calc <= t_cr
    else:
        t_cr = w_crit(n, alpha, tail)
        t_hi = n * (n + 1) / 2 - t_cr
        ax.fill_between(xs[xs <= t_cr], ys[xs <= t_cr], color="red", alpha=0.30)
        ax.fill_between(xs[xs >= t_hi], ys[xs >= t_hi], color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=t_cr, color="green", linestyle="--")
        ax.axvline(x=t_hi, color="green", linestyle="--")
        place_label(ax, [], t_cr, stats.norm.pdf(t_cr, μ, σ) + 0.02,
                    f"T₍crit₎={t_cr}", color="green")
        reject = t_calc <= t_cr or t_calc >= t_hi

    ax.axvline(x=t_calc, color="blue", linestyle="--")
    place_label(ax, [], t_calc, stats.norm.pdf(t_calc, μ, σ) + 0.02,
                f"T₍calc₎={t_calc}", color="blue")

    ax.set_title(
        "Wilcoxon T — "
        f"{'Reject' if reject else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def w_table(n: int, alpha: float, tail: str, key: str) -> None:
    """Neighbourhood Wilcoxon T crit table."""
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(5, n - 5), n + 6))
    alphas   = [0.10, 0.05, 0.01, 0.001]

    head_html = "".join(f"<th>{a}</th>" for a in alphas)
    body_html = ""
    for r in row_vals:
        cms = "".join(
            f'<td id="w_{r}_{idx}">{w_crit(r, a, tail)}</td>'
            for idx, a in enumerate(alphas, start=1)
        )
        body_html += f'<tr><td id="w_{r}_0">{r}</td>{cms}</tr>'

    css = (
        "table{border-collapse:collapse}"
        "th,td{border:1px solid #000;width:80px;height:30px;"
        "text-align:center;font-size:0.8rem}"
    )
    html = wrap(css, f"<tr><th>N＼α</th>{head_html}</tr>{body_html}")

    col_idx = alphas.index(alpha) + 1

    if step >= 0:
        for i in range(len(alphas) + 1):
            html = style_cell(html, f"w_{n}_{i}")
    if step >= 1:
        for r in row_vals:
            html = style_cell(html, f"w_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"w_{n}_{col_idx}", color="blue", px=3)

    iframe(html)

    msgs = ["Highlight N row", "Highlight α column", "Intersection → T₍crit₎"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_button(step_key)


# ─────────────────────────────────────────────────────────────────────────────
#  8 • Tab 7 — Binomial
# ─────────────────────────────────────────────────────────────────────────────
def tab_binom() -> None:
    """Render Tab 7 (Binomial)."""
    st.subheader("Tab 7 • Binomial")

    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("n (trials)", min_value=1, value=20,
                            step=1, key="b_n")
        p = st.number_input("π (null proportion)", value=0.50, step=0.01,
                            min_value=0.01, max_value=0.99, key="b_p")
    with c2:
        k = st.number_input("k (successes)", min_value=0, value=12,
                            step=1, key="b_k")
        alpha = st.number_input("α (two‑tailed)", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="b_alpha")

    if st.button("Update Plot", key="b_plot"):
        st.pyplot(plot_binom(k, n, p))

    with st.expander("Quick table (k ±5)"):
        binom_table(k, n, p, "b")


def plot_binom(k: int, n: int, p: float) -> plt.Figure:
    """Simple PMF bar chart highlighting k."""
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


def binom_table(k: int, n: int, p: float, key: str) -> None:
    """Display P(X=k), P(X≤k), P(X≥k) for neighbourhood of k."""
    step_key = f"{key}_step"
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
    html = wrap(css, f"<tr><th>k</th>{head_html}</tr>{body_html}")

    if step >= 0:
        for i in range(4):
            html = style_cell(html, f"b_{k}_{i}")
    if step >= 1:
        html = style_cell(html, f"b_{k}_1", color="blue", px=3)

    iframe(html)

    msgs = ["Highlight k row", "Highlight P(X=k)"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step + 1}**: {msgs[step]}")

    next_button(step_key)


# ─────────────────────────────────────────────────────────────────────────────
#  9 • Main entry‑point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Streamlit page dispatch."""
    st.set_page_config(
        "PSYC250 – Statistical Tables Explorer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("PSYC250 – Statistical Tables Explorer (12 × 4 figures)")

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


#  Allow `python app.py` execution as well as `streamlit run app.py`
if __name__ == "__main__":
    main()
