###############################################################################
#  PSYC‑250  ‑‑  Statistical Tables Explorer  (Streamlit, 12 × 4 figures)
#  ---------------------------------------------------------------------------
#  Seven tabs:
#     1) t‑Distribution
#     2) z‑Distribution
#     3) F‑Distribution
#     4) Chi‑Square
#     5) Mann‑Whitney U
#     6) Wilcoxon Signed‑Rank T
#     7) Binomial
#
#  Each tab provides:
#     • Input widgets
#     • 12 × 4 Matplotlib plot
#     • Step‑wise HTML lookup table (row → col → cell)
#     • APA‑7 narrative generated from the current inputs
#
#  Dependencies tested with:
#     Python 3.12.1 • streamlit 1.30.0 • matplotlib 3.8.2 • scipy 1.11.4
###############################################################################

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#  Keep Matplotlib head‑less inside Streamlit
plt.switch_backend("Agg")

# ─────────────────────────────────────────────────────────────────────────────
#  Generic helper routines
# ─────────────────────────────────────────────────────────────────────────────
def place_label(ax: plt.Axes,
                placed: list[tuple[float, float]],
                x: float,
                y: float,
                text: str,
                *,
                color: str = "blue") -> None:
    """
    Add a small annotation to *ax* at (x, y).  If that spot is occupied,
    the label is nudged by 0.06 horiz / 0.04 vert until clear.

    This prevents labels overlapping critical‑value markers.
    """
    dx = dy = 0.0
    for xx, yy in placed:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            dx += 0.06
            dy += 0.04
    ax.text(x + dx, y + dy, text,
            color=color, ha="left", va="bottom", fontsize=8, clip_on=True)
    placed.append((x + dx, y + dy))


def style_cell(html_in: str, cell_id: str, *, color: str = "red", px: int = 2) -> str:
    """Inject an inline border into the first TD/TH whose id == *cell_id*."""
    return html_in.replace(f'id="{cell_id}"',
                           f'id="{cell_id}" style="border:{px}px solid {color};"', 1)


def iframe(html: str, *, height: int = 460) -> None:
    """
    Show *html* inline and confine it to *height* px with its own scrollbar.
    This avoids the iframe‑wheel capture issue so the whole page still scrolls.
    """
    st.markdown(
        f'<div style="overflow:auto; max-height:{height}px;">{html}</div>',
        unsafe_allow_html=True,
    )


def next_button(step_key: str) -> None:
    """Draw a “Next Step” button that increments st.session_state[step_key]."""
    if st.button("Next Step", key=f"{step_key}__btn"):
        st.session_state[step_key] += 1


def wrap(css: str, inner: str) -> str:
    """Return `<style>css</style><table>inner</table>`."""
    return f"<style>{css}</style><table>{inner}</table>"


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 1 • t‑Distribution
# ─────────────────────────────────────────────────────────────────────────────
def tab_t() -> None:
    st.subheader("Tab 1 • t‑Distribution")

    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("t statistic", value=2.87, key="t_val")
        df    = st.number_input("df", min_value=1, value=55, step=1, key="t_df")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="t_alpha")
        tail  = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="t_tail")

    if st.button("Update Plot", key="t_plot"):
        st.pyplot(plot_t(t_val, df, alpha, tail))

    with st.expander("Step‑by‑step t‑table"):
        t_table(df, alpha, tail, "t")


def plot_t(t_calc: float, df: int, alpha: float, tail: str) -> plt.Figure:
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
        ax.axvline(x=crit, color="green", linestyle="--")
        place_label(ax, labels, crit, stats.t.pdf(crit, df)+.02,
                    f"t₍crit₎={crit:.3f}", color="green")
        reject = t_calc > crit
    else:
        crit = stats.t.ppf(1 - alpha/2, df)
        ax.fill_between(xs[xs >=  crit], ys[xs >=  crit], color="red", alpha=0.30)
        ax.fill_between(xs[xs <= -crit], ys[xs <= -crit], color="red", alpha=0.30,
                        label="Reject H₀")
        ax.axvline(x= crit, color="green", linestyle="--")
        ax.axvline(x=-crit, color="green", linestyle="--")
        place_label(ax, labels,  crit, stats.t.pdf( crit, df)+.02,
                    f"+t₍crit₎={crit:.3f}", color="green")
        place_label(ax, labels, -crit, stats.t.pdf(-crit, df)+.02,
                    f"–t₍crit₎={crit:.3f}", color="green")
        reject = abs(t_calc) > crit

    ax.axvline(x=t_calc, color="blue", linestyle="--")
    place_label(ax, labels, t_calc, stats.t.pdf(t_calc, df)+.02,
                f"t₍calc₎={t_calc:.3f}", color="blue")

    ax.set_title(f"t‑Distribution (df = {df}) — "
                 f"{'Reject' if reject else 'Fail to Reject'} H₀")
    ax.legend()
    fig.tight_layout()
    return fig


def t_table(df: int, alpha: float, tail: str, key: str) -> None:
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    rows = list(range(max(1, df-5), df+6))
    headers = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001),
    ]

    header_html = "".join(f"<th>{m}_{a}</th>" for m, a in headers)
    body_html   = ""
    for r in rows:
        body_html += '<tr>'
        body_html += f'<td id="t_{r}_0">{r}</td>'
        for idx, (m, a) in enumerate(headers, start=1):
            tcrit = stats.t.ppf(1 - a/(1 if m == "one" else 2), r)
            body_html += f'<td id="t_{r}_{idx}">{tcrit:.3f}</td>'
        body_html += '</tr>'

    css = ("table{border-collapse:collapse}"
           "th,td{border:1px solid #000;width:80px;height:30px;"
           "text-align:center;font-size:0.9rem;font-family:sans-serif}"
           "th{background:#fafafa}")
    html = wrap(css, f"<tr><th>df</th>{header_html}</tr>{body_html}")

    mode_needed = "one" if tail.startswith("one") else "two"
    col_idx = next(
        idx for idx, (m, a) in enumerate(headers, start=1)
        if m == mode_needed and np.isclose(a, alpha)
    )

    #  step highlights
    if step >= 0:
        for i in range(len(headers)+1):
            html = style_cell(html, f"t_{df}_{i}")
    if step >= 1:
        for r in rows:
            html = style_cell(html, f"t_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"t_{df}_{col_idx}", color="blue", px=3)
    if step >= 3 and tail.startswith("one") and np.isclose(alpha, 0.05):
        alt_idx = headers.index(("two", 0.10)) + 1
        for r in rows:
            html = style_cell(html, f"t_{r}_{alt_idx}")
        html = style_cell(html, f"t_{df}_{alt_idx}", color="blue", px=3)

    iframe(html)

    msgs = [
        "Highlight df row",
        "Highlight α / tail column",
        "Intersection → t₍crit₎",
    ]
    if tail.startswith("one") and np.isclose(alpha, 0.05):
        msgs.append("Also highlight two‑tailed α = 0.10 equivalence")

    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step+1}**: {msgs[step]}")

    next_button(step_key)

    # ── APA‑style narrative ────────────────────────────────────────────────
    t_val = st.session_state["t_val"]
    if tail.startswith("one"):
        p_val = stats.t.sf(abs(t_val), df)
    else:
        p_val = stats.t.sf(abs(t_val), df) * 2
    decision = "rejected" if p_val < alpha else "failed to reject"
    tail_txt = "one‑tailed" if tail.startswith("one") else "two‑tailed"
    st.markdown(
        f"**APA‑style interpretation**  \n"
        f"*t*({df}) = {t_val:.2f}, *p* = {p_val:.3f} "
        f"({tail_txt}). The null hypothesis was **{decision}** at "
        f"α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 2 • z‑Distribution
# ─────────────────────────────────────────────────────────────────────────────
def tab_z() -> None:
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
        z_table(z_val, "z", alpha, tail)


def plot_z(z_calc: float, alpha: float, tail: str) -> plt.Figure:
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
        ax.axvline(x=crit, color="green", linestyle="--")
        place_label(ax, labels, crit, stats.norm.pdf(crit)+.02,
                    f"z₍crit₎={crit:.3f}", color="green")
        reject = z_calc > crit
    else:
        crit = stats.norm.ppf(1 - alpha / 2)
        ax.fill_between(xs[xs >=  crit], ys[xs >=  crit], color="red", alpha=0.30)
        ax.fill_between(xs[xs <= -crit], ys[xs <= -crit], color="red", alpha=0.30,
                        label="Reject H₀")
        ax.axvline(x= crit, color="green", linestyle="--")
        ax.axvline(x=-crit, color="green", linestyle="--")
        place_label(ax, labels,  crit, stats.norm.pdf( crit)+.02,
                    f"+z₍crit₎={crit:.3f}", color="green")
        place_label(ax, labels, -crit, stats.norm.pdf(-crit)+.02,
                    f"–z₍crit₎={crit:.3f}", color="green")
        reject = abs(z_calc) > crit

    ax.axvline(x=z_calc, color="blue", linestyle="--")
    place_label(ax, labels, z_calc, stats.norm.pdf(z_calc)+.02,
                f"z₍calc₎={z_calc:.3f}", color="blue")

    ax.set_title("z‑Distribution — "
                 f"{'Reject' if reject else 'Fail to Reject'} H₀")
    ax.legend()
    fig.tight_layout()
    return fig


def z_table(z_in: float, key: str, alpha: float, tail: str) -> None:
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    z     = max(0, min(3.49, z_in))
    row_v = np.floor(z * 10) / 10
    col_v = round(z - row_v, 2)

    rows = np.round(np.arange(0, 3.5, 0.1), 1)
    cols = np.round(np.arange(0, 0.1, 0.01), 2)

    header_html = "".join(f"<th>{c:.2f}</th>" for c in cols)
    body_html   = ""
    for r in rows:
        body_html += f'<tr><td id="z_{r:.1f}_0">{r:.1f}</td>'
        for c in cols:
            body_html += (
                f'<td id="z_{r:.1f}_{c:.2f}">{stats.norm.cdf(r+c):.4f}</td>'
            )
        body_html += '</tr>'

    css = ("table{border-collapse:collapse}"
           "th,td{border:1px solid #000;width:70px;height:30px;"
           "text-align:center;font-size:0.9rem;font-family:sans-serif}"
           "th{background:#fafafa}")
    html = wrap(css, f"<tr><th>z.x</th>{header_html}</tr>{body_html}")

    if step >= 0:
        for c in cols:
            html = style_cell(html, f"z_{row_v:.1f}_{c:.2f}")
        html = style_cell(html, f"z_{row_v:.1f}_0")
    if step >= 1:
        for r in rows:
            html = style_cell(html, f"z_{r:.1f}_{col_v:.2f}")
    if step >= 2:
        html = style_cell(html, f"z_{row_v:.1f}_{col_v:.2f}",
                          color="blue", px=3)

    iframe(html)

    msgs = ["Highlight row", "Highlight column", "Intersection → Φ(z)"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step+1}**: {msgs[step]}")

    next_button(step_key)

    #  APA narrative
    z_val = st.session_state["z_val"]
    p_val = stats.norm.sf(abs(z_val)) * (1 if tail.startswith("one") else 2)
    decision = "rejected" if p_val < alpha else "failed to reject"
    tail_txt = "one‑tailed" if tail.startswith("one") else "two‑tailed"
    st.markdown(
        f"**APA‑style interpretation**  \n"
        f"*z* = {z_val:.2f}, *p* = {p_val:.3f} ({tail_txt}). "
        f"The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 3 • F‑Distribution
# ─────────────────────────────────────────────────────────────────────────────
def tab_f() -> None:
    st.subheader("Tab 3 • F‑Distribution")

    c1, c2 = st.columns(2)
    with c1:
        f_val = st.number_input("F statistic", value=4.32, key="f_val")
        df1   = st.number_input("df₁ (numerator)",
                                min_value=1, value=5, step=1, key="f_df1")
    with c2:
        df2   = st.number_input("df₂ (denominator)",
                                min_value=1, value=20, step=1, key="f_df2")
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="f_alpha")

    if st.button("Update Plot", key="f_plot"):
        st.pyplot(plot_f(f_val, df1, df2, alpha))

    with st.expander("Step‑by‑step F‑table"):
        f_table(df1, df2, alpha, "f", f_val)


def f_crit(df1: int, df2: int, alpha: float) -> float:
    return stats.f.ppf(1 - alpha, df1, df2)


def plot_f(f_calc: float, df1: int, df2: int, alpha: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x_max = stats.f.ppf(0.995, df1, df2) * 1.10
    xs = np.linspace(0, x_max, 400)
    ys = stats.f.pdf(xs, df1, df2)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")

    crit = f_crit(df1, df2, alpha)
    ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                    color="red", alpha=0.30, label="Reject H₀")
    ax.axvline(x=crit, color="green", linestyle="--")
    ax.axvline(x=f_calc, color="blue", linestyle="--")

    place_label(ax, [], crit, stats.f.pdf(crit, df1, df2)+.02,
                f"F₍crit₎={crit:.3f}", color="green")
    place_label(ax, [], f_calc, stats.f.pdf(f_calc, df1, df2)+.02,
                f"F₍calc₎={f_calc:.3f}", color="blue")

    ax.set_title(
        f"F‑Distribution (df₁ {df1}, df₂ {df2}) — "
        f"{'Reject' if f_calc > crit else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def f_table(df1: int, df2: int, alpha: float, key: str, f_val: float) -> None:
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(1, df1-5), df1+6))
    col_vals = list(range(max(1, df2-5), df2+6))

    head = "".join(f"<th>{c}</th>" for c in col_vals)
    body = ""
    for r in row_vals:
        body += f'<tr><td id="f_{r}_0">{r}</td>'
        for idx, c in enumerate(col_vals, start=1):
            body += f'<td id="f_{r}_{idx}">{f_crit(r, c, alpha):.3f}</td>'
        body += "</tr>"

    css = ("table{border-collapse:collapse}"
           "th,td{border:1px solid #000;width:90px;height:30px;"
           "text-align:center;font-size:0.85rem}")
    html = wrap(css, f"<tr><th>df₁＼df₂</th>{head}</tr>{body}")

    col_idx = col_vals.index(df2) + 1

    if step >= 0:
        for i in range(len(col_vals)+1):
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
        st.write(f"**Step {step+1}**: {msgs[step]}")

    next_button(step_key)

    p_val   = stats.f.sf(f_val, df1, df2)
    decision = "rejected" if p_val < alpha else "failed to reject"
    st.markdown(
        f"**APA‑style interpretation**  \n"
        f"*F*({df1}, {df2}) = {f_val:.2f}, *p* = {p_val:.3f}. "
        f"The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 4 • Chi‑Square
# ─────────────────────────────────────────────────────────────────────────────
def tab_chi() -> None:
    st.subheader("Tab 4 • Chi‑Square (χ²)")

    c1, c2 = st.columns(2)
    with c1:
        chi_val = st.number_input("χ² statistic", value=7.88, key="chi_val")
        df      = st.number_input("df", min_value=1, value=3,
                                  step=1, key="chi_df")
    with c2:
        alpha   = st.selectbox("α", [0.10, 0.05, 0.01, 0.001],
                               index=1, key="chi_alpha")

    if st.button("Update Plot", key="chi_plot"):
        st.pyplot(plot_chi(chi_val, df, alpha))

    with st.expander("Step‑by‑step χ²‑table"):
        chi_table(df, alpha, "chi", chi_val)


def chi_crit(df: int, alpha: float) -> float:
    return stats.chi2.ppf(1 - alpha, df)


def plot_chi(chi_calc: float, df: int, alpha: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x_max = chi_crit(df, 0.001) * 1.10
    xs = np.linspace(0, x_max, 400)
    ys = stats.chi2.pdf(xs, df)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")

    crit = chi_crit(df, alpha)
    ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                    color="red", alpha=0.30, label="Reject H₀")
    ax.axvline(x=crit, color="green", linestyle="--")
    ax.axvline(x=chi_calc, color="blue", linestyle="--")

    place_label(ax, [], crit, stats.chi2.pdf(crit, df)+.02,
                f"χ²₍crit₎={crit:.3f}", color="green")
    place_label(ax, [], chi_calc, stats.chi2.pdf(chi_calc, df)+.02,
                f"χ²₍calc₎={chi_calc:.3f}", color="blue")

    ax.set_title(
        f"χ²‑Distribution (df = {df}) — "
        f"{'Reject' if chi_calc > crit else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def chi_table(df: int, alpha: float, key: str, chi_val: float) -> None:
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(1, df-5), df+6))
    alphas   = [0.10, 0.05, 0.01, 0.001]

    head = "".join(f"<th>{a}</th>" for a in alphas)
    body = ""
    for r in row_vals:
        body += f'<tr><td id="chi_{r}_0">{r}</td>'
        for idx, a in enumerate(alphas, start=1):
            body += f'<td id="chi_{r}_{idx}">{chi_crit(r, a):.3f}</td>'
        body += "</tr>"

    css = ("table{border-collapse:collapse}"
           "th,td{border:1px solid #000;width:80px;height:30px;"
           "text-align:center;font-size:0.85rem}")
    html = wrap(css, f"<tr><th>df＼α</th>{head}</tr>{body}")

    col_idx = alphas.index(alpha) + 1

    if step >= 0:
        for i in range(len(alphas)+1):
            html = style_cell(html, f"chi_{df}_{i}")
    if step >= 1:
        for r in row_vals:
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
        st.write(f"**Step {step+1}**: {msgs[step]}")

    next_button(step_key)

    p_val   = stats.chi2.sf(chi_val, df)
    decision = "rejected" if p_val < alpha else "failed to reject"
    st.markdown(
        f"**APA‑style interpretation**  \n"
        f"χ²({df}) = {chi_val:.2f}, *p* = {p_val:.3f}. "
        f"The null hypothesis was **{decision}** at α = {alpha:.3f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 5 • Mann‑Whitney U
# ─────────────────────────────────────────────────────────────────────────────
def tab_u() -> None:
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
        u_table(n1, n2, alpha, tail, "u", u_val)


def u_crit(n1: int, n2: int, alpha: float, tail: str) -> int:
    μ = n1 * n2 / 2
    σ = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = stats.norm.ppf(alpha if tail.startswith("one") else alpha / 2)
    return int(np.floor(μ + z * σ))


def plot_u(u_calc: int, n1: int, n2: int, alpha: float, tail: str) -> plt.Figure:
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
        ax.axvline(x=crit, color="green", linestyle="--")
        place_label(ax, [], crit, stats.norm.pdf(crit, μ, σ)+.02,
                    f"U₍crit₎={crit}", color="green")
        reject = u_calc <= crit
    else:
        crit = u_crit(n1, n2, alpha, tail)
        hi   = n1 * n2 - crit
        ax.fill_between(xs[xs <= crit], ys[xs <= crit], color="red", alpha=0.30)
        ax.fill_between(xs[xs >= hi  ], ys[xs >= hi  ], color="red", alpha=0.30,
                        label="Reject H₀")
        ax.axvline(x=crit, color="green", linestyle="--")
        ax.axvline(x=hi,   color="green", linestyle="--")
        place_label(ax, [], crit, stats.norm.pdf(crit, μ, σ)+.02,
                    f"U₍crit₎={crit}", color="green")
        reject = u_calc <= crit or u_calc >= hi

    ax.axvline(x=u_calc, color="blue", linestyle="--")
    place_label(ax, [], u_calc, stats.norm.pdf(u_calc, μ, σ)+.02,
                f"U₍calc₎={u_calc}", color="blue")

    ax.set_title(
        "Mann–Whitney U — "
        f"{'Reject' if reject else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def u_table(n1: int, n2: int, alpha: float, tail: str,
            key: str, u_val: int) -> None:
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(2, n1-5), n1+6))
    col_vals = list(range(max(2, n2-5), n2+6))

    head = "".join(f"<th>{c}</th>" for c in col_vals)
    body = ""
    for r in row_vals:
        body += f'<tr><td id="u_{r}_0">{r}</td>'
        for idx, c in enumerate(col_vals, start=1):
            body += f'<td id="u_{r}_{idx}">{u_crit(r, c, alpha, tail)}</td>'
        body += "</tr>"

    css = ("table{border-collapse:collapse}"
           "th,td{border:1px solid #000;width:90px;height:30px;"
           "text-align:center;font-size:0.8rem}")
    html = wrap(css, f"<tr><th>n₁＼n₂</th>{head}</tr>{body}")

    col_idx = col_vals.index(n2) + 1

    if step >= 0:
        for i in range(len(col_vals)+1):
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
        st.write(f"**Step {step+1}**: {msgs[step]}")

    next_button(step_key)

    μ = n1 * n2 / 2
    σ = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if tail.startswith("one"):
        p_val = stats.norm.cdf((u_val - μ) / σ)
    else:
        p_val = stats.norm.sf(abs(u_val - μ) / σ) * 2
    decision = "rejected" if p_val < alpha else "failed to reject"
    tail_txt = "one‑tailed" if tail.startswith("one") else "two‑tailed"
    st.markdown(
        f"**APA‑style interpretation**  \n"
        f"*U* = {u_val}, *p* = {p_val:.3f} ({tail_txt}). "
        f"The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 6 • Wilcoxon Signed‑Rank T
# ─────────────────────────────────────────────────────────────────────────────
def tab_w() -> None:
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank T")

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
        w_table(n, alpha, tail, "w", t_val)


def w_crit(n: int, alpha: float, tail: str) -> int:
    μ = n * (n + 1) / 4
    σ = np.sqrt(n * (n + 1) * (2*n + 1) / 24)
    z = stats.norm.ppf(alpha if tail.startswith("one") else alpha / 2)
    return int(np.floor(μ + z * σ))


def plot_w(t_calc: int, n: int, alpha: float, tail: str) -> plt.Figure:
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
        ax.axvline(x=crit, color="green", linestyle="--")
        place_label(ax, [], crit, stats.norm.pdf(crit, μ, σ)+.02,
                    f"T₍crit₎={crit}", color="green")
        reject = t_calc <= crit
    else:
        crit = w_crit(n, alpha, tail)
        hi   = n * (n + 1) / 2 - crit
        ax.fill_between(xs[xs <= crit], ys[xs <= crit], color="red", alpha=0.30)
        ax.fill_between(xs[xs >= hi  ], ys[xs >= hi  ], color="red", alpha=0.30,
                        label="Reject H₀")
        ax.axvline(x=crit, color="green", linestyle="--")
        ax.axvline(x=hi,   color="green", linestyle="--")
        place_label(ax, [], crit, stats.norm.pdf(crit, μ, σ)+.02,
                    f"T₍crit₎={crit}", color="green")
        reject = t_calc <= crit or t_calc >= hi

    ax.axvline(x=t_calc, color="blue", linestyle="--")
    place_label(ax, [], t_calc, stats.norm.pdf(t_calc, μ, σ)+.02,
                f"T₍calc₎={t_calc}", color="blue")

    ax.set_title(
        "Wilcoxon T — "
        f"{'Reject' if reject else 'Fail to Reject'} H₀"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def w_table(n: int, alpha: float, tail: str,
            key: str, t_val: int) -> None:
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(5, n-5), n+6))
    alphas   = [0.10, 0.05, 0.01, 0.001]

    head = "".join(f"<th>{a}</th>" for a in alphas)
    body = ""
    for r in row_vals:
        body += f'<tr><td id="w_{r}_0">{r}</td>'
        for idx, a in enumerate(alphas, start=1):
            body += f'<td id="w_{r}_{idx}">{w_crit(r, a, tail)}</td>'
        body += "</tr>"

    css = ("table{border-collapse:collapse}"
           "th,td{border:1px solid #000;width:80px;height:30px;"
           "text-align:center;font-size:0.8rem}")
    html = wrap(css, f"<tr><th>N＼α</th>{head}</tr>{body}")

    col_idx = alphas.index(alpha) + 1

    if step >= 0:
        for i in range(len(alphas)+1):
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
        st.write(f"**Step {step+1}**: {msgs[step]}")

    next_button(step_key)

    μ = n * (n + 1) / 4
    σ = np.sqrt(n * (n + 1) * (2*n + 1) / 24)
    if tail.startswith("one"):
        p_val = stats.norm.cdf((t_val - μ) / σ)
    else:
        p_val = stats.norm.sf(abs(t_val - μ) / σ) * 2

    decision = "rejected" if p_val < alpha else "failed to reject"
    tail_txt = "one‑tailed" if tail.startswith("one") else "two‑tailed"
    st.markdown(
        f"**APA‑style interpretation**  \n"
        f"*T* = {t_val}, *p* = {p_val:.3f} ({tail_txt}). "
        f"The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 7 • Binomial
# ─────────────────────────────────────────────────────────────────────────────
def tab_binom() -> None:
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

    with st.expander("Quick table (k ±5)"):
        binom_table(k, n, p, "b", alpha)


def plot_binom(k: int, n: int, p: float) -> plt.Figure:
    xs = np.arange(0, n+1)
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


def binom_table(k: int, n: int, p: float,
                key: str, alpha: float) -> None:
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    k_vals = list(range(max(0, k-5), min(n, k+5)+1))

    header = "<th>P(X=k)</th><th>P(X≤k)</th><th>P(X≥k)</th>"
    body   = ""
    for kv in k_vals:
        pmf = stats.binom.pmf(kv, n, p)
        cdf = stats.binom.cdf(kv, n, p)
        sf  = 1 - stats.binom.cdf(kv-1, n, p)
        body += (
            f'<tr><td id="b_{kv}_0">{kv}</td>'
            f'<td id="b_{kv}_1">{pmf:.4f}</td>'
            f'<td id="b_{kv}_2">{cdf:.4f}</td>'
            f'<td id="b_{kv}_3">{sf :.4f}</td></tr>'
        )

    css = ("table{border-collapse:collapse}"
           "th,td{border:1px solid #000;width:110px;height:30px;"
           "text-align:center;font-size:0.8rem}")
    html = wrap(css, f"<tr><th>k</th>{header}</tr>{body}")

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
        st.write(f"**Step {step+1}**: {msgs[step]}")

    next_button(step_key)

    #  Two‑tailed exact binomial p‑value (double the smaller tail)
    cdf = stats.binom.cdf(k, n, p)
    p_two = 2 * min(cdf, 1-cdf + stats.binom.pmf(k, n, p))
    p_two = min(p_two, 1.0)  # cap at 1

    decision = "rejected" if p_two < alpha else "failed to reject"
    st.markdown(
        f"**APA‑style interpretation**  \n"
        f"Exact binomial test, *p* = {p_two:.3f}. "
        f"The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
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

    with tabs[0]: tab_t()
    with tabs[1]: tab_z()
    with tabs[2]: tab_f()
    with tabs[3]: tab_chi()
    with tabs[4]: tab_u()
    with tabs[5]: tab_w()
    with tabs[6]: tab_binom()


if __name__ == "__main__":
    main()
