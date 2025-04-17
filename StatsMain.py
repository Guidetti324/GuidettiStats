import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")  # ensure non‑interactive backend

###############################################################################
#                               COMMON HELPERS
###############################################################################

def place_label(ax, seen, x, y, txt, colour="blue"):
    """Write a label, nudging if it would overlap previous labels."""
    dx = dy = 0.0
    for xx, yy in seen:
        if abs(x - xx) < .15 and abs(y - yy) < .05:
            dx += .06
            dy += .04
    ax.text(x + dx, y + dy, txt, colour, ha="left", va="bottom", fontsize=8)
    seen.append((x + dx, y + dy))

def style_cell(html, cid, colour="red", px=2):
    return html.replace(f'id="{cid}"',
                        f'id="{cid}" style="border:{px}px solid {colour};"', 1)

def render_html(html, height=450):
    """Show raw HTML inside Streamlit (scrollable)."""
    components.html(f"<html><body>{html}</body></html>",
                    height=height, scrolling=True)

def bump_step(key):
    if st.button("Next Step", key=key + "_btn"):
        st.session_state[key] += 1

def table_wrapper(css, body):
    return (f"<style>{css}</style>"
            f"<table>{body}</table>")

###############################################################################
#                               t‑DISTRIBUTION
###############################################################################

def tab_t():
    st.subheader("Tab 1 • t‑Distribution")
    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("t statistic", value=2.87, key="t_val")
        df_val = st.number_input("df", min_value=1, value=55, step=1, key="t_df")
    with c2:
        alpha = st.number_input("α", min_value=0.0001, max_value=0.5,
                                value=0.05, step=0.01, key="t_a")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="t_tail")

    if st.button("Update Plot", key="t_up"):
        st.pyplot(plot_t(t_val, df_val, alpha, tail))

    with st.expander("Step‑by‑step t‑table"):
        t_lookup(df_val, alpha, tail, "t")

def plot_t(t, df, a, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x = np.linspace(-4, 4, 400)
    ax.plot(x, stats.t.pdf(x, df), 'k')
    ax.fill_between(x, stats.t.pdf(x, df), color="lightgrey", alpha=.2,
                    label="Fail to Reject H₀")
    lbl = []
    if tail == "one‑tailed":
        crit = stats.t.ppf(1 - a, df)
        ax.fill_between(x[x >= crit], stats.t.pdf(x[x >= crit], df),
                        color="red", alpha=.3, label="Reject H₀")
        ax.axvline(crit, color="green", ls="--")
        place_label(ax, lbl, crit, stats.t.pdf(crit, df) + .02,
                    f"t₍crit₎={crit:.3f}", "green")
        reject = t > crit
    else:
        crit = stats.t.ppf(1 - a / 2, df)
        ax.fill_between(x[x >=  crit], stats.t.pdf(x[x >=  crit], df),
                        color="red", alpha=.3)
        ax.fill_between(x[x <= -crit], stats.t.pdf(x[x <= -crit], df),
                        color="red", alpha=.3, label="Reject H₀")
        ax.axvline( crit, color="green", ls="--")
        ax.axvline(-crit, color="green", ls="--")
        place_label(ax, lbl,  crit, stats.t.pdf( crit, df) + .02,
                    f"+t₍crit₎={crit:.3f}", "green")
        place_label(ax, lbl, -crit, stats.t.pdf(-crit, df) + .02,
                    f"–t₍crit₎={crit:.3f}", "green")
        reject = abs(t) > crit
    ax.axvline(t, color="blue", ls="--")
    place_label(ax, lbl, t, stats.t.pdf(t, df) + .02,
                f"t₍calc₎={t:.3f}", "blue")
    ax.set_title(f"t‑Distribution (df={df}) – "
                 f"{'Reject' if reject else 'Fail to Reject'} H₀")
    ax.legend(); fig.tight_layout(); return fig

def t_lookup(df, a, tail, key):
    step = st.session_state.setdefault(key + "_step", 0)
    dfs = list(range(max(1, df - 5), df + 6))
    heads = [("one", .10), ("one", .05), ("one", .01), ("one", .001),
             ("two", .10), ("two", .05), ("two", .01), ("two", .001)]

    def crit(d, mode, al):
        return stats.t.ppf(1 - al / (1 if mode == "one" else 2), d)

    header = "".join(f"<th>{m}_{al}</th>" for m, al in heads)
    rows = ""
    for d in dfs:
        cells = "".join(f'<td id="df_{d}_{i}">{crit(d, m, al):.3f}</td>'
                        for i, (m, al) in enumerate(heads, 1))
        rows += f'<tr><td id="df_{d}_0">{d}</td>{cells}</tr>'
    css = ("table{border-collapse:collapse}"
           "td,th{border:1px solid#000;width:80px;height:30px;text-align:center"
           ";font-family:sans-serif;font-size:.9rem}")
    table = table_wrapper(css,
                          f"<tr><th>df</th>{header}</tr>{rows}")

    mode = "one" if tail.startswith("one") else "two"
    col = [i for i, (m, al) in enumerate(heads, 1)
           if m == mode and abs(al - a) < 1e-12][0]

    if step >= 0:
        for i in range(len(heads) + 1):
            table = style_cell(table, f"df_{df}_{i}")
    if step >= 1:
        for d in dfs:
            table = style_cell(table, f"df_{d}_{col}")
    if step >= 2:
        table = style_cell(table, f"df_{df}_{col}", "blue", 3)
    render_html(table)
    msgs = ["Highlight df row", "Highlight α/tail column",
            "Intersection → t₍crit₎"]
    st.write(f"**Step {step+1}**: {msgs[step]}" if step < 3
             else "All steps complete!")
    bump_step(key + "_step")

###############################################################################
#                               z‑DISTRIBUTION
###############################################################################

def tab_z():
    st.subheader("Tab 2 • z‑Distribution")
    c1, c2 = st.columns(2)
    with c1:
        z = st.number_input("z statistic", value=1.64, key="z_val")
    with c2:
        a = st.number_input("α", min_value=0.0001, max_value=0.5,
                            value=0.05, step=0.01, key="z_a")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="z_tail")
    if st.button("Update Plot", key="z_up"):
        st.pyplot(plot_z(z, a, tail))
    with st.expander("Step‑by‑step z‑table"):
        z_lookup(z, "z")

def plot_z(z, a, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x = np.linspace(-4, 4, 400)
    ax.plot(x, stats.norm.pdf(x), 'k')
    ax.fill_between(x, stats.norm.pdf(x), color="lightgrey", alpha=.2,
                    label="Fail to Reject H₀")
    lbl = []
    if tail == "one‑tailed":
        crit = stats.norm.ppf(1 - a)
        ax.fill_between(x[x >= crit], stats.norm.pdf(x[x >= crit]),
                        color="red", alpha=.3, label="Reject H₀")
        ax.axvline(crit, color="green", ls="--")
        place_label(ax, lbl, crit, stats.norm.pdf(crit) + .02,
                    f"z₍crit₎={crit:.3f}", "green")
        reject = z > crit
    else:
        crit = stats.norm.ppf(1 - a / 2)
        ax.fill_between(x[x >=  crit], stats.norm.pdf(x[x >=  crit]),
                        color="red", alpha=.3)
        ax.fill_between(x[x <= -crit], stats.norm.pdf(x[x <= -crit]),
                        color="red", alpha=.3, label="Reject H₀")
        ax.axvline( crit, color="green", ls="--")
        ax.axvline(-crit, color="green", ls="--")
        place_label(ax, lbl,  crit, stats.norm.pdf( crit) + .02,
                    f"+z₍crit₎={crit:.3f}", "green")
        place_label(ax, lbl, -crit, stats.norm.pdf(-crit) + .02,
                    f"–z₍crit₎={crit:.3f}", "green")
        reject = abs(z) > crit
    ax.axvline(z, color="blue", ls="--")
    place_label(ax, lbl, z, stats.norm.pdf(z) + .02,
                f"z₍calc₎={z:.3f}", "blue")
    ax.set_title(f"z‑Distribution – "
                 f"{'Reject' if reject else 'Fail to Reject'} H₀")
    ax.legend(); fig.tight_layout(); return fig

def z_lookup(z, key):
    step = st.session_state.setdefault(key + "_step", 0)
    z = max(0, min(3.49, z))
    row = np.floor(z * 10) / 10
    col = round(z - row, 2)
    rows = np.round(np.arange(0, 3.5, .1), 1)
    cols = np.round(np.arange(0, .1, .01), 2)
    r_idx = np.where(rows == row)[0][0]
    sub = rows[max(0, r_idx - 10):min(len(rows), r_idx + 11)]

    head = "".join(f"<th>{c:.2f}</th>" for c in cols)
    body = ""
    for r in sub:
        body += f'<tr><td id="z_{r:.1f}_0">{r:.1f}</td>'
        for c in cols:
            body += (f'<td id="z_{r:.1f}_{c:.2f}">'
                     f'{stats.norm.cdf(r + c):.4f}</td>')
        body += "</tr>"
    css = ("table{border-collapse:collapse}"
           "td,th{border:1px solid#000;width:70px;height:30px;text-align:center"
           ";font-family:sans-serif;font-size:.9rem}")
    table = table_wrapper(css,
                          f"<tr><th>z.x</th>{head}</tr>{body}")

    if step >= 0:
        for c in cols:
            table = style_cell(table, f"z_{row:.1f}_{c:.2f}")
        table = style_cell(table, f"z_{row:.1f}_0")
    if step >= 1:
        for r in sub:
            table = style_cell(table, f"z_{r:.1f}_{col:.2f}")
    if step >= 2:
        table = style_cell(table, f"z_{row:.1f}_{col:.2f}", "blue", 3)
    render_html(table)
    msgs = ["Locate row", "Locate column", "Intersection → Φ(z)"]
    st.write(f"**Step {step+1}**: {msgs[step]}" if step < 3
             else "All steps complete!")
    bump_step(key + "_step")

###############################################################################
#                               F‑DISTRIBUTION
###############################################################################

def f_crit(df1, df2, a):
    return stats.f.ppf(1 - a, df1, df2)

def tab_f():
    st.subheader("Tab 3 • F‑Distribution")
    c1, c2 = st.columns(2)
    with c1:
        f = st.number_input("F statistic", value=4.32, key="f_val")
        df1 = st.number_input("df₁ (numerator)", min_value=1,
                              value=5, step=1, key="f_df1")
    with c2:
        df2 = st.number_input("df₂ (denominator)", min_value=1,
                              value=20, step=1, key="f_df2")
        a   = st.number_input("α", min_value=0.0001, max_value=0.5,
                              value=0.05, step=0.01, key="f_a")
    if st.button("Update Plot", key="f_up"):
        st.pyplot(plot_f(f, df1, df2, a))
    with st.expander("Step‑by‑step F‑table"):
        f_lookup(df1, df2, a, "f")

def plot_f(f, df1, df2, a):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x = np.linspace(0, stats.f.ppf(.995, df1, df2) * 1.1, 400)
    ax.plot(x, stats.f.pdf(x, df1, df2), 'k')
    ax.fill_between(x, stats.f.pdf(x, df1, df2), color="lightgrey", alpha=.2,
                    label="Fail to Reject H₀")
    crit = f_crit(df1, df2, a)
    ax.fill_between(x[x >= crit], stats.f.pdf(x[x >= crit], df1, df2),
                    color="red", alpha=.3, label="Reject H₀")
    ax.axvline(crit, color="green", ls="--")
    ax.axvline(f, color="blue", ls="--")
    place_label(ax, [], crit, stats.f.pdf(crit, df1, df2) + .02,
                f"F₍crit₎={crit:.3f}", "green")
    place_label(ax, [], f, stats.f.pdf(f, df1, df2) + .02,
                f"F₍calc₎={f:.3f}", "blue")
    ax.set_title(f"F‑Distribution (df₁={df1}, df₂={df2}) – "
                 f"{'Reject' if f > crit else 'Fail to Reject'} H₀")
    ax.legend(); fig.tight_layout(); return fig

def f_lookup(df1, df2, a, key):
    step = st.session_state.setdefault(key + "_step", 0)
    rows = list(range(max(1, df1 - 5), df1 + 6))
    cols = list(range(max(1, df2 - 5), df2 + 6))
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        body += f'<tr><td id="f_{r}_0">{r}</td>'
        for idx, c in enumerate(cols, 1):
            body += (f'<td id="f_{r}_{idx}">'
                     f'{f_crit(r, c, a):.3f}</td>')
        body += "</tr>"
    css = ("table{border-collapse:collapse}"
           "td,th{border:1px solid#000;width:90px;height:30px;text-align:center"
           ";font-family:sans-serif;font-size:.85rem}")
    table = table_wrapper(css,
                          f"<tr><th>df₁＼df₂</th>{head}</tr>{body}")

    col_idx = cols.index(df2) + 1
    if step >= 0:
        for i in range(len(cols) + 1):
            table = style_cell(table, f"f_{df1}_{i}")
    if step >= 1:
        for r in rows:
            table = style_cell(table, f"f_{r}_{col_idx}")
    if step >= 2:
        table = style_cell(table, f"f_{df1}_{col_idx}", "blue", 3)
    render_html(table)
    msgs = ["Highlight df₁ row", "Highlight df₂ column",
            "Intersection → F₍crit₎"]
    st.write(f"**Step {step+1}**: {msgs[step]}" if step < 3
             else "All steps complete!")
    bump_step(key + "_step")

###############################################################################
#                               χ²‑DISTRIBUTION
###############################################################################

def chi_crit(df, a):
    return stats.chi2.ppf(1 - a, df)

def tab_chi():
    st.subheader("Tab 4 • Chi‑Square (χ²)")
    c1, c2 = st.columns(2)
    with c1:
        chi = st.number_input("χ² statistic", value=7.88, key="c_val")
        df  = st.number_input("df", min_value=1, value=3, step=1, key="c_df")
    with c2:
        a   = st.selectbox("α", [.10, .05, .01, .001], index=1, key="c_a")
    if st.button("Update Plot", key="c_up"):
        st.pyplot(plot_chi(chi, df, a))
    with st.expander("Step‑by‑step χ²‑table"):
        chi_lookup(df, a, "c")

def plot_chi(chi, df, a):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x = np.linspace(0, chi_crit(df, .001) * 1.1, 400)
    ax.plot(x, stats.chi2.pdf(x, df), 'k')
    ax.fill_between(x, stats.chi2.pdf(x, df), color="lightgrey", alpha=.2,
                    label="Fail to Reject H₀")
    crit = chi_crit(df, a)
    ax.fill_between(x[x >= crit], stats.chi2.pdf(x[x >= crit], df),
                    color="red", alpha=.3, label="Reject H₀")
    ax.axvline(crit, color="green", ls="--")
    ax.axvline(chi, color="blue", ls="--")
    place_label(ax, [], crit, stats.chi2.pdf(crit, df) + .02,
                f"χ²₍crit₎={crit:.3f}", "green")
    place_label(ax, [], chi, stats.chi2.pdf(chi, df) + .02,
                f"χ²₍calc₎={chi:.3f}", "blue")
    ax.set_title(f"χ²‑Distribution (df={df}) – "
                 f"{'Reject' if chi > crit else 'Fail to Reject'} H₀")
    ax.legend(); fig.tight_layout(); return fig

def chi_lookup(df, a, key):
    step = st.session_state.setdefault(key + "_step", 0)
    dfs = list(range(max(1, df - 5), df + 6))
    alphas = [.10, .05, .01, .001]
    head = "".join(f"<th>{al}</th>" for al in alphas)
    body = ""
    for d in dfs:
        body += f'<tr><td id="chi_{d}_0">{d}</td>'
        for idx, al in enumerate(alphas, 1):
            body += f'<td id="chi_{d}_{idx}">{chi_crit(d, al):.3f}</td>'
        body += "</tr>"
    css = ("table{border-collapse:collapse}"
           "td,th{border:1px solid#000;width:80px;height:30px;text-align:center"
           ";font-family:sans-serif;font-size:.85rem}")
    table = table_wrapper(css,
                          f"<tr><th>df＼α</th>{head}</tr>{body}")

    col_idx = alphas.index(a) + 1
    if step >= 0:
        for i in range(len(alphas) + 1):
            table = style_cell(table, f"chi_{df}_{i}")
    if step >= 1:
        for d in dfs:
            table = style_cell(table, f"chi_{d}_{col_idx}")
    if step >= 2:
        table = style_cell(table, f"chi_{df}_{col_idx}", "blue", 3)
    render_html(table)
    msgs = ["Highlight df row", "Highlight α column",
            "Intersection → χ²₍crit₎"]
    st.write(f"**Step {step+1}**: {msgs[step]}" if step < 3
             else "All steps complete!")
    bump_step(key + "_step")

###############################################################################
#                               Mann‑Whitney U
###############################################################################

def u_crit(n1, n2, a, tail):
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = stats.norm.ppf(a if tail == "one‑tailed" else a / 2)
    return int(np.floor(mu + z * sigma))

def tab_u():
    st.subheader("Tab 5 • Mann–Whitney U")
    c1, c2 = st.columns(2)
    with c1:
        u   = st.number_input("U statistic", value=23, key="u_val")
        n1  = st.number_input("n₁", min_value=2, value=10, step=1, key="u_n1")
    with c2:
        n2  = st.number_input("n₂", min_value=2, value=12, step=1, key="u_n2")
        a   = st.number_input("α", min_value=0.0001, max_value=0.5,
                              value=0.05, step=0.01, key="u_a")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="u_tail")
    if st.button("Update Plot", key="u_up"):
        st.pyplot(plot_u(u, n1, n2, a, tail))
    with st.expander("Step‑by‑step U‑table"):
        u_lookup(n1, n2, a, tail, "u")

def plot_u(u, n1, n2, a, tail):
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k')
    ax.fill_between(x, stats.norm.pdf(x, mu, sigma),
                    color="lightgrey", alpha=.2, label="Fail to Reject H₀")
    if tail == "one‑tailed":
        crit = u_crit(n1, n2, a, tail)
        ax.fill_between(x[x <= crit], stats.norm.pdf(x[x <= crit], mu, sigma),
                        color="red", alpha=.3, label="Reject H₀")
        ax.axvline(crit, color="green", ls="--")
        place_label(ax, [], crit,
                    stats.norm.pdf(crit, mu, sigma) + .02,
                    f"U₍crit₎={crit}", "green")
        reject = u <= crit
    else:
        crit = u_crit(n1, n2, a, tail)
        hi = n1 * n2 - crit
        ax.fill_between(x[x <= crit], stats.norm.pdf(x[x <= crit], mu, sigma),
                        color="red", alpha=.3)
        ax.fill_between(x[x >= hi], stats.norm.pdf(x[x >= hi], mu, sigma),
                        color="red", alpha=.3, label="Reject H₀")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(hi, color="green", ls="--")
        place_label(ax, [], crit,
                    stats.norm.pdf(crit, mu, sigma) + .02,
                    f"U₍crit₎={crit}", "green")
        reject = u <= crit or u >= hi
    ax.axvline(u, color="blue", ls="--")
    place_label(ax, [], u, stats.norm.pdf(u, mu, sigma) + .02,
                f"U₍calc₎={u}", "blue")
    ax.set_title(f"Mann–Whitney U – "
                 f"{'Reject' if reject else 'Fail to Reject'} H₀")
    ax.legend(); fig.tight_layout(); return fig

def u_lookup(n1, n2, a, tail, key):
    step = st.session_state.setdefault(key + "_step", 0)
    rows = list(range(max(2, n1 - 5), n1 + 6))
    cols = list(range(max(2, n2 - 5), n2 + 6))
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        body += f'<tr><td id="u_{r}_0">{r}</td>'
        for idx, c in enumerate(cols, 1):
            body += (f'<td id="u_{r}_{idx}">'
                     f'{u_crit(r, c, a, tail)}</td>')
        body += "</tr>"
    css = ("table{border-collapse:collapse}"
           "td,th{border:1px solid#000;width:90px;height:30px;text-align:center"
           ";font-family:sans-serif;font-size:.8rem}")
    table = table_wrapper(css,
                          f"<tr><th>n₁＼n₂</th>{head}</tr>{body}")

    col_idx = cols.index(n2) + 1
    if step >= 0:
        for i in range(len(cols) + 1):
            table = style_cell(table, f"u_{n1}_{i}")
    if step >= 1:
        for r in rows:
            table = style_cell(table, f"u_{r}_{col_idx}")
    if step >= 2:
        table = style_cell(table, f"u_{n1}_{col_idx}", "blue", 3)
    render_html(table)
    msgs = ["Highlight n₁ row", "Highlight n₂ column",
            "Intersection → U₍crit₎"]
    st.write(f"**Step {step+1}**: {msgs[step]}" if step < 3
             else "All steps complete!")
    bump_step(key + "_step")

###############################################################################
#                          Wilcoxon Signed‑Rank T
###############################################################################

def w_crit(n, a, tail):
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = stats.norm.ppf(a if tail == "one‑tailed" else a / 2)
    return int(np.floor(mu + z * sigma))

def tab_w():
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank T")
    c1, c2 = st.columns(2)
    with c1:
        t = st.number_input("T statistic", value=15, key="w_val")
        n = st.number_input("N (non‑zero diffs)", min_value=5,
                            value=12, step=1, key="w_n")
    with c2:
        a   = st.number_input("α", min_value=0.0001, max_value=0.5,
                              value=0.05, step=0.01, key="w_a")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="w_tail")
    if st.button("Update Plot", key="w_up"):
        st.pyplot(plot_w(t, n, a, tail))
    with st.expander("Step‑by‑step T‑table"):
        w_lookup(n, a, tail, "w")

def plot_w(t, n, a, tail):
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k')
    ax.fill_between(x, stats.norm.pdf(x, mu, sigma),
                    color="lightgrey", alpha=.2, label="Fail to Reject H₀")
    if tail == "one‑tailed":
        crit = w_crit(n, a, tail)
        ax.fill_between(x[x <= crit], stats.norm.pdf(x[x <= crit], mu, sigma),
                        color="red", alpha=.3, label="Reject H₀")
        ax.axvline(crit, color="green", ls="--")
        place_label(ax, [], crit,
                    stats.norm.pdf(crit, mu, sigma) + .02,
                    f"T₍crit₎={crit}", "green")
        reject = t <= crit
    else:
        crit = w_crit(n, a, tail)
        hi = n * (n + 1) / 2 - crit
        ax.fill_between(x[x <= crit], stats.norm.pdf(x[x <= crit], mu, sigma),
                        color="red", alpha=.3)
        ax.fill_between(x[x >= hi], stats.norm.pdf(x[x >= hi], mu, sigma),
                        color="red", alpha=.3, label="Reject H₀")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(hi, color="green", ls="--")
        place_label(ax, [], crit,
                    stats.norm.pdf(crit, mu, sigma) + .02,
                    f"T₍crit₎={crit}", "green")
        reject = t <= crit or t >= hi
    ax.axvline(t, color="blue", ls="--")
    place_label(ax, [], t, stats.norm.pdf(t, mu, sigma) + .02,
                f"T₍calc₎={t}", "blue")
    ax.set_title(f"Wilcoxon T – "
                 f"{'Reject' if reject else 'Fail to Reject'} H₀")
    ax.legend(); fig.tight_layout(); return fig

def w_lookup(n, a, tail, key):
    step = st.session_state.setdefault(key + "_step", 0)
    rows = list(range(max(5, n - 5), n + 6))
    alphas = [.10, .05, .01, .001]
    head = "".join(f"<th>{al}</th>" for al in alphas)
    body = ""
    for r in rows:
        body += f'<tr><td id="w_{r}_0">{r}</td>'
        for idx, al in enumerate(alphas, 1):
            body += f'<td id="w_{r}_{idx}">{w_crit(r, al, tail)}</td>'
        body += "</tr>"
    css = ("table{border-collapse:collapse}"
           "td,th{border:1px solid#000;width:80px;height:30px;text-align:center"
           ";font-family:sans-serif;font-size:.8rem}")
    table = table_wrapper(css,
                          f"<tr><th>N＼α</th>{head}</tr>{body}")

    col_idx = alphas.index(a) + 1
    if step >= 0:
        for i in range(len(alphas) + 1):
            table = style_cell(table, f"w_{n}_{i}")
    if step >= 1:
        for r in rows:
            table = style_cell(table, f"w_{r}_{col_idx}")
    if step >= 2:
        table = style_cell(table, f"w_{n}_{col_idx}", "blue", 3)
    render_html(table)
    msgs = ["Highlight N row", "Highlight α column",
            "Intersection → T₍crit₎"]
    st.write(f"**Step {step+1}**: {msgs[step]}" if step < 3
             else "All steps complete!")
    bump_step(key + "_step")

###############################################################################
#                               BINOMIAL
###############################################################################

def tab_binom():
    st.subheader("Tab 7 • Binomial")
    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("n (trials)", min_value=1, value=20, step=1, key="b_n")
        p = st.number_input("π (null proportion)", min_value=0.01, max_value=0.99,
                            value=0.50, step=0.01, key="b_p")
    with c2:
        k = st.number_input("k (successes)", min_value=0, value=12, step=1, key="b_k")
        a = st.number_input("α (two‑tailed)", min_value=0.0001, max_value=0.5,
                            value=0.05, step=0.01, key="b_a")
    if st.button("Update Plot", key="b_up"):
        st.pyplot(plot_binom(k, n, p))
    with st.expander("Quick table (k ±5)"):
        binom_table(k, n, p, "b")

def plot_binom(k, n, p):
    x = np.arange(0, n + 1)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.bar(x, stats.binom.pmf(x, n, p), color="lightgrey")
    ax.bar(k, stats.binom.pmf(k, n, p), color="blue")
    ax.set_title(f"Binomial (n={n}, p={p}) – k={k}")
    ax.set_xlabel("k"); ax.set_ylabel("P(X=k)"); fig.tight_layout(); return fig

def binom_table(k, n, p, key):
    step = st.session_state.setdefault(key + "_step", 0)
    ks = list(range(max(0, k - 5), min(n, k + 5) + 1))
    head = "<th>P(X=k)</th><th>P(X≤k)</th><th>P(X≥k)</th>"
    body = ""
    for k_ in ks:
        pmf = stats.binom.pmf(k_, n, p)
        cdf = stats.binom.cdf(k_, n, p)
        sf  = 1 - stats.binom.cdf(k_ - 1, n, p)
        body += (f'<tr><td id="b_{k_}_0">{k_}</td>'
                 f'<td id="b_{k_}_1">{pmf:.4f}</td>'
                 f'<td id="b_{k_}_2">{cdf:.4f}</td>'
                 f'<td id="b_{k_}_3">{sf:.4f}</td></tr>')
    css = ("table{border-collapse:collapse}"
           "td,th{border:1px solid#000;width:120px;height:30px;text-align:center"
           ";font-family:sans-serif;font-size:.8rem}")
    table = table_wrapper(css,
                          f"<tr><th>k</th>{head}</tr>{body}")

    if step >= 0:
        for i in range(4):
            table = style_cell(table, f"b_{k}_{i}")
    if step >= 1:
        table = style_cell(table, f"b_{k}_1", "blue", 3)
    render_html(table)
    msgs = ["Highlight k row", "Highlight P(X=k)"]
    st.write(f"**Step {step+1}**: {msgs[step]}" if step < 2
             else "All steps complete!")
    bump_step(key + "_step")

###############################################################################
#                                    MAIN
###############################################################################

def main():
    st.set_page_config("PSYC250 – Statistical Tables Explorer", layout="wide")
    st.title("PSYC250 – Statistical Tables Explorer (12 × 4 figures)")
    tabs = st.tabs(["t‑Dist", "z‑Dist", "F‑Dist", "Chi‑Square",
                    "Mann–Whitney U", "Wilcoxon T", "Binomial"])
    with tabs[0]: tab_t()
    with tabs[1]: tab_z()
    with tabs[2]: tab_f()
    with tabs[3]: tab_chi()
    with tabs[4]: tab_u()
    with tabs[5]: tab_w()
    with tabs[6]: tab_binom()

if __name__ == "__main__":
    main()
