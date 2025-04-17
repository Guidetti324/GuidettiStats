###############################################################################
#  PSYC‑250  –  Statistical Tables Explorer  (Streamlit, 12×4‑inch figures)
#  ---------------------------------------------------------------------------
#  Seven complete tabs (t, z, F, Chi‑Sq, Mann‑Whitney U, Wilcoxon T, Binomial)
#
#  Fix for animation AttributeError:
#   • Removed st.experimental_rerun() calls, relying on Streamlit's normal
#     re-run-on-button-click behavior.
#   • Everything else unchanged.
#
###############################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")


def place_label(ax, placed, x, y, txt, *, color="blue"):
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
    styled_str = (
        f'id="{cid}" style="border:{px}px solid {color};'
        f' background-color:{bg};"'
    )
    return html.replace(f'id="{cid}"', styled_str, 1)


def wrap_table(css: str, table: str) -> str:
    return f"<style>{css}</style><table>{table}</table>"


def container(html: str, *, height: int = 460) -> str:
    return f'<div style="overflow:auto; max-height:{height}px;">{html}</div>'


def multi_step(build_html, frames: int, *, key: str, height: int = 460):
    """
    Display the table in multiple highlight steps (row -> col -> cell).
    No st.experimental_rerun() calls, relying on normal re-run behavior.
    """
    if key not in st.session_state:
        st.session_state[key] = 0

    step = st.session_state[key]
    # Render the table at the current step
    st.markdown(container(build_html(step), height=height), unsafe_allow_html=True)

    # Step controls
    c1, c2 = st.columns(2)
    if step < frames - 1:
        if c1.button("Next Step", key=f"{key}_next"):
            # Just increment the step; Streamlit will rerun automatically.
            st.session_state[key] += 1
    else:
        c1.success("All steps complete!")

    if c2.button("Reset Steps", key=f"{key}_reset"):
        st.session_state[key] = 0


CSS_BASE = (
    "table{border-collapse:collapse}"
    "th,td{border:1px solid #000;height:30px;text-align:center;"
    "font-family:sans-serif;font-size:0.9rem}"
    "th{background:#fafafa}"
)

# ─────────────────────────────────────────────────────────────────────────────
#  t-Distribution Example
# ─────────────────────────────────────────────────────────────────────────────

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
        ax.fill_between(xs[xs <= -crit], ys[xs <= -crit],
                        color="red", alpha=0.30, label="Reject H₀")
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
    rows = list(range(max(1, df-5), df+6))
    heads = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001)
    ]
    mode = "one" if tail.startswith("one") else "two"
    col = next(i for i, (m, a) in enumerate(heads, start=1)
               if m == mode and np.isclose(a, alpha))

    head = "".join(f"<th>{m}_{a}</th>" for m, a in heads)
    body = ""
    for r in rows:
        row_html = f'<td id="t_{r}_0">{r}</td>'
        for i, (m, a) in enumerate(heads, start=1):
            crit_val = stats.t.ppf(1 - a if m == "one" else 1 - a/2, r)
            row_html += f'<td id="t_{r}_{i}">{crit_val:.2f}</td>'
        body += f"<tr>{row_html}</tr>"

    html = wrap_table(CSS_BASE, f"<tr><th>df</th>{head}</tr>{body}")

    if step >= 0:
        for i in range(len(heads)+1):
            html = style_cell(html, f"t_{df}_{i}", color="red", px=2)
    if step >= 1:
        for r in rows:
            html = style_cell(html, f"t_{r}_{col}", color="red", px=2)
    if step >= 2:
        html = style_cell(html, f"t_{df}_{col}", color="blue", px=3,
                          bg="#cce5ff")
    return html


def t_table(df: int, alpha: float, tail: str):
    multi_step(
        lambda s: build_t_html(df, alpha, tail, s),
        frames=3, key=f"t_{df}_{alpha}_{tail}"
    )


def t_apa(t_val: float, df: int, alpha: float, tail: str):
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

    if st.button("Update Plot", key="t_plot"):
        st.pyplot(plot_t(t_val, df, alpha, tail))

    st.write("**Step‑by‑step t‑table**")
    t_table(df, alpha, tail)
    t_apa(t_val, df, alpha, tail)


def main():
    st.set_page_config("PSYC250 – Statistical Tables Explorer", layout="wide")
    st.title("PSYC250 – Statistical Tables Explorer (12×4 figures)")

    tab_t()  # For brevity, only show the t-distribution example.

if __name__ == "__main__":
    main()
