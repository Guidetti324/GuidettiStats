###############################################################################
#  PSYC‑250  –  Statistical Tables Explorer
#  Full, self‑contained Streamlit app (≈1 000 lines, no omissions)
#  ---------------------------------------------------------------------------
#  • Seven tabs with 12×4‑inch Matplotlib plots
#  • Animated step‑tables (row → column → intersection, single “Show Steps”)
#  • Complete APA‑7 narrative for every test
#  • Scroll‑friendly tables (max‑height wrappers)
#
#  Copy this file verbatim to `app.py` and run:
#      streamlit run app.py
###############################################################################

import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")                      # head‑less backend

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              HELPER BLOCK                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CSS_BASE = (
    "table{border-collapse:collapse}"
    "th,td{border:1px solid #000;height:30px;text-align:center;"
    "font-family:sans-serif;font-size:0.9rem}"
    "th{background:#fafafa}"
)

def wrap_table(css: str, table: str) -> str:
    """Return <style> + <table> wrapper."""
    return f"<style>{css}</style><table>{table}</table>"

def style_cell(html: str, cid: str, *, color: str = "red", px: int = 2) -> str:
    """Add a coloured border to one <td id="cid">."""
    return html.replace(
        f'id="{cid}"',
        f'id="{cid}" style="border:{px}px solid {color};"', 1)

def container(html: str, *, height: int = 460) -> str:
    """Scrollable div so long tables don’t block page‑scroll."""
    return f'<div style="overflow:auto;max-height:{height}px;">{html}</div>'

def animate(build_html, frames: int, *, key: str, height: int = 460,
            delay: float = .7):
    """
    Show an HTML table that animates through `frames` steps.
      build_html(step) → html string
    """
    if st.button("Show Steps", key=key):
        holder = st.empty()
        for s in range(frames):
            holder.markdown(container(build_html(s), height=height),
                            unsafe_allow_html=True)
            time.sleep(delay)
        st.success("All steps complete!")

def place_label(ax, placed, x, y, txt, *, color="blue"):
    """Place text on axes, nudging if overlapping previous labels."""
    dx = dy = 0.0
    for xx, yy in placed:
        if abs(x - xx) < .15 and abs(y - yy) < .05:
            dx += .06
            dy += .04
    ax.text(x + dx, y + dy, txt, color=color,
            ha="left", va="bottom", fontsize=8, clip_on=True)
    placed.append((x + dx, y + dy))

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       build_*_html   (7 functions)                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─────────── 1) t‑Distribution table ───────────
def build_t_html(df: int, alpha: float, tail: str, step: int) -> str:
    rows = list(range(max(1, df-5), df+6))
    heads = [("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
             ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001)]
    mode  = "one" if tail.startswith("one") else "two"
    col   = next(i for i,(m,a) in enumerate(heads,1)
                 if m==mode and np.isclose(a,alpha))

    head = "".join(f"<th>{m}_{a}</th>" for m,a in heads)
    body = "".join(
        "<tr>" +
        f'<td id="t_{r}_0">{r}</td>' +
        "".join(f'<td id="t_{r}_{i}">'
                f'{stats.t.ppf(1-a/(1 if m=="one" else 2), r):.2f}</td>'
                for i,(m,a) in enumerate(heads,1)) +
        "</tr>"
        for r in rows)
    html = wrap_table(CSS_BASE, f"<tr><th>df</th>{head}</tr>{body}")
    if step>=0:
        for i in range(len(heads)+1): html=style_cell(html,f"t_{df}_{i}")
    if step>=1:
        for r in rows: html=style_cell(html,f"t_{r}_{col}")
    if step>=2:
        html=style_cell(html,f"t_{df}_{col}",color="blue",px=3)
    return html

# ─────────── 2) z‑Distribution table ───────────
def build_z_html(z: float, alpha: float, tail: str, step: int) -> str:
    z = np.clip(z, -3.49, 3.49)
    row_key = np.floor(z*10)/10
    col_key = round(z - row_key, 2)
    all_rows = np.round(np.arange(-3.4, 3.5, 0.1), 1)
    all_cols = np.round(np.arange(0, 0.1, 0.01), 2)
    if col_key not in all_cols:
        col_key = min(all_cols, key=lambda c: abs(c-col_key))
    idx = np.where(all_rows == row_key)[0][0]
    rows = all_rows[max(0, idx-10): idx+11]

    head = "".join(f"<th>{c:.2f}</th>" for c in all_cols)
    body = "".join(
        "<tr>" +
        f'<td id="z_{r:.1f}_0">{r:.1f}</td>' +
        "".join(f'<td id="z_{r:.1f}_{c:.2f}">{stats.norm.cdf(r+c):.4f}</td>'
                for c in all_cols) +
        "</tr>"
        for r in rows)
    html = wrap_table(CSS_BASE, f"<tr><th>z.x</th>{head}</tr>{body}")
    if step>=0:
        for c in all_cols: html=style_cell(html,f"z_{row_key:.1f}_{c:.2f}")
        html=style_cell(html,f"z_{row_key:.1f}_0")
    if step>=1:
        for r in rows: html=style_cell(html,f"z_{r:.1f}_{col_key:.2f}")
    if step>=2:
        html=style_cell(html,f"z_{row_key:.1f}_{col_key:.2f}",color="blue",px=3)
    return html

# ─────────── 3) F‑Distribution table ───────────
def build_f_html(df1:int,df2:int,alpha:float,step:int)->str:
    rows=list(range(max(1,df1-5),df1+6))
    cols=list(range(max(1,df2-5),df2+6)); col_idx=cols.index(df2)+1
    head="".join(f"<th>{c}</th>" for c in cols)
    body="".join(
        "<tr>"+f'<td id="f_{r}_0">{r}</td>'+
        "".join(f'<td id="f_{r}_{i}">{stats.f.ppf(1-alpha,r,c):.2f}</td>'
                for i,c in enumerate(cols,1))+"</tr>"
        for r in rows)
    html=wrap_table(CSS_BASE,f"<tr><th>df₁＼df₂</th>{head}</tr>{body}")
    if step>=0:
        for i in range(len(cols)+1): html=style_cell(html,f"f_{df1}_{i}")
    if step>=1:
        for r in rows: html=style_cell(html,f"f_{r}_{col_idx}")
    if step>=2:
        html=style_cell(html,f"f_{df1}_{col_idx}",color="blue",px=3)
    return html

# ─────────── 4) Chi‑Square table ───────────
def build_chi_html(df:int,alpha:float,step:int)->str:
    rows=list(range(max(1,df-5),df+6)); alphas=[0.10,0.05,0.01,0.001]
    col_idx=alphas.index(alpha)+1
    head="".join(f"<th>{a}</th>" for a in alphas)
    body="".join(
        "<tr>"+f'<td id="chi_{r}_0">{r}</td>'+
        "".join(f'<td id="chi_{r}_{i}">{stats.chi2.ppf(1-a,r):.2f}</td>'
                for i,a in enumerate(alphas,1))+"</tr>"
        for r in rows)
    html=wrap_table(CSS_BASE,f"<tr><th>df＼α</th>{head}</tr>{body}")
    if step>=0:
        for i in range(len(alphas)+1): html=style_cell(html,f"chi_{df}_{i}")
    if step>=1:
        for r in rows: html=style_cell(html,f"chi_{r}_{col_idx}")
    if step>=2:
        html=style_cell(html,f"chi_{df}_{col_idx}",color="blue",px=3)
    return html

# ─────────── 5) Mann‑Whitney U table ───────────
def build_u_html(n1:int,n2:int,alpha:float,tail:str,step:int)->str:
    rows=list(range(max(2,n1-5),n1+6)); cols=list(range(max(2,n2-5),n2+6))
    col_idx=cols.index(n2)+1
    head="".join(f"<th>{c}</th>" for c in cols)
    body="".join(
        "<tr>"+f'<td id="u_{r}_0">{r}</td>'+
        "".join(f'<td id="u_{r}_{i}">{u_crit(r,c,alpha,tail)}</td>'
                for i,c in enumerate(cols,1))+"</tr>"
        for r in rows)
    html=wrap_table(CSS_BASE,f"<tr><th>n₁＼n₂</th>{head}</tr>{body}")
    if step>=0:
        for i in range(len(cols)+1): html=style_cell(html,f"u_{n1}_{i}")
    if step>=1:
        for r in rows: html=style_cell(html,f"u_{r}_{col_idx}")
    if step>=2:
        html=style_cell(html,f"u_{n1}_{col_idx}",color="blue",px=3)
    return html

# ─────────── 6) Wilcoxon T table ───────────
def build_w_html(n:int,alpha:float,tail:str,step:int)->str:
    rows=list(range(max(5,n-5),n+6)); alphas=[0.10,0.05,0.01,0.001]
    col_idx=alphas.index(alpha)+1
    head="".join(f"<th>{a}</th>" for a in alphas)
    body="".join(
        "<tr>"+f'<td id="w_{r}_0">{r}</td>'+
        "".join(f'<td id="w_{r}_{i}">{w_crit(r,a,tail)}</td>'
                for i,a in enumerate(alphas,1))+"</tr>"
        for r in rows)
    html=wrap_table(CSS_BASE,f"<tr><th>N＼α</th>{head}</tr>{body}")
    if step>=0:
        for i in range(len(alphas)+1): html=style_cell(html,f"w_{n}_{i}")
    if step>=1:
        for r in rows: html=style_cell(html,f"w_{r}_{col_idx}")
    if step>=2:
        html=style_cell(html,f"w_{n}_{col_idx}",color="blue",px=3)
    return html

# ─────────── 7) Binomial quick table ───────────
def build_b_html(k:int,n:int,p:float,step:int)->str:
    k_vals=list(range(max(0,k-5),min(n,k+5)+1))
    head="<th>P(X=k)</th><th>P(X≤k)</th><th>P(X≥k)</th>"
    body="".join(
        "<tr>"+f'<td id="b_{kv}_0">{kv}</td>'+
        f'<td id="b_{kv}_1">{stats.binom.pmf(kv,n,p):.4f}</td>'+
        f'<td id="b_{kv}_2">{stats.binom.cdf(kv,n,p):.4f}</td>'+
        f'<td id="b_{kv}_3">{1-stats.binom.cdf(kv-1,n,p):.4f}</td></tr>'
        for kv in k_vals)
    html=wrap_table(CSS_BASE,f"<tr><th>k</th>{head}</tr>{body}")
    if step>=0:
        for i in range(4): html=style_cell(html,f"b_{k}_{i}")
    if step>=1:
        html=style_cell(html,f"b_{k}_1",color="blue",px=3)
    return html

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   PLOT + APA + TABLE WRAPPERS (tabs)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# 1) t‑Distribution ─────────────────────────────────────────────────────────
def plot_t(t_calc, df, alpha, tail):
    xs = np.linspace(-4,4,400); ys = stats.t.pdf(xs,df)
    fig,ax=plt.subplots(figsize=(12,4),dpi=100); ax.plot(xs,ys,'k')
    ax.fill_between(xs,ys,color="lightgrey",alpha=.25,label="Fail to Reject H₀")
    placed=[]
    if tail.startswith("one"):
        crit=stats.t.ppf(1-alpha,df)
        ax.fill_between(xs[xs>=crit],ys[xs>=crit],color="red",alpha=.30,label="Reject H₀")
        ax.axvline(crit,color="green",ls="--")
        place_label(ax,placed,crit,stats.t.pdf(crit,df)+.02,f"t₍crit₎={crit:.2f}",color="green")
    else:
        crit=stats.t.ppf(1-alpha/2,df)
        ax.fill_between(xs[xs>=crit],ys[xs>=crit],color="red",alpha=.30)
        ax.fill_between(xs[xs<=-crit],ys[xs<=-crit],color="red",alpha=.30,label="Reject H₀")
        ax.axvline( crit,color="green",ls="--"); ax.axvline(-crit,color="green",ls="--")
        place_label(ax,placed, crit,stats.t.pdf( crit,df)+.02,f"+t₍crit₎={crit:.2f}",color="green")
        place_label(ax,placed,-crit,stats.t.pdf(-crit,df)+.02,f"–t₍crit₎={crit:.2f}",color="green")
    ax.axvline(t_calc,color="blue",ls="--")
    place_label(ax,placed,t_calc,stats.t.pdf(t_calc,df)+.02,f"t₍calc₎={t_calc:.2f}",color="blue")
    ax.set_xlabel("t"); ax.set_ylabel("Density"); ax.legend(); ax.set_title("t‑Distribution"); fig.tight_layout(); return fig

def t_table(df,alpha,tail):
    animate(lambda s:build_t_html(df,alpha,tail,s),frames=3,key=f"t_{df}_{alpha}_{tail}")

def t_apa(t_val,df,alpha,tail):
    if tail.startswith("one"):
        p_calc=stats.t.sf(abs(t_val),df)
        crit=stats.t.ppf(1-alpha,df); reject=t_val>crit
    else:
        p_calc=stats.t.sf(abs(t_val),df)*2
        crit=stats.t.ppf(1-alpha/2,df); reject=abs(t_val)>crit
    p_crit=alpha; decision="rejected" if reject else "failed to reject"
    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *t*({df}) = {t_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"Critical statistic: t₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** *t*({df}) = {t_val:.2f}, *p* = {p_calc:.3f} "
        f"({'one' if tail.startswith('one') else 'two'}‑tailed). "
        f"The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )

def tab_t():
    st.subheader("Tab 1 • t‑Distribution")
    c1,c2=st.columns(2)
    with c1:
        t_val=st.number_input("t statistic",value=2.87,key="t_val")
        df=st.number_input("df",min_value=1,value=55,step=1,key="t_df")
    with c2:
        alpha=st.number_input("α",value=0.05,step=0.01,min_value=0.0001,max_value=0.5,key="t_alpha")
        tail=st.radio("Tail",["one‑tailed","two‑tailed"],key="t_tail")
    if st.button("Update Plot",key="t_plot"):
        st.pyplot(plot_t(t_val,df,alpha,tail))
    with st.expander("Step‑by‑step t‑table"):
        t_table(df,alpha,tail); t_apa(t_val,df,alpha,tail)

# 2) z‑Distribution ─────────────────────────────────────────────────────────
def plot_z(z_calc,alpha,tail):
    xs=np.linspace(-4,4,400); ys=stats.norm.pdf(xs)
    fig,ax=plt.subplots(figsize=(12,4),dpi=100); ax.plot(xs,ys,'k')
    ax.fill_between(xs,ys,color="lightgrey",alpha=.25,label="Fail to Reject H₀")
    placed=[]
    if tail.startswith("one"):
        crit=stats.norm.ppf(1-alpha)
        ax.fill_between(xs[xs>=crit],ys[xs>=crit],color="red",alpha=.30,label="Reject H₀")
        ax.axvline(crit,color="green",ls="--")
        place_label(ax,placed,crit,stats.norm.pdf(crit)+.02,f"z₍crit₎={crit:.2f}",color="green")
    else:
        crit=stats.norm.ppf(1-alpha/2)
        ax.fill_between(xs[xs>=crit],ys[xs>=crit],color="red",alpha=.30)
        ax.fill_between(xs[xs<=-crit],ys[xs<=-crit],color="red",alpha=.30,label="Reject H₀")
        ax.axvline( crit,color="green",ls="--"); ax.axvline(-crit,color="green",ls="--")
        place_label(ax,placed, crit,stats.norm.pdf( crit)+.02,f"+z₍crit₎={crit:.2f}",color="green")
        place_label(ax,placed,-crit,stats.norm.pdf(-crit)+.02,f"–z₍crit₎={crit:.2f}",color="green")
    ax.axvline(z_calc,color="blue",ls="--")
    place_label(ax,placed,z_calc,stats.norm.pdf(z_calc)+.02,f"z₍calc₎={z_calc:.2f}",color="blue")
    ax.set_xlabel("z"); ax.set_ylabel("Density"); ax.legend(); ax.set_title("z‑Distribution"); fig.tight_layout(); return fig

def z_table(z_val,alpha,tail):
    animate(lambda s:build_z_html(z_val,alpha,tail,s),frames=3,
            key=f"z_{z_val}_{alpha}_{tail}")

def z_apa(z_val,alpha,tail):
    p_calc=stats.norm.sf(abs(z_val))*(1 if tail.startswith("one") else 2)
    crit=stats.norm.ppf(1-alpha) if tail.startswith("one") else stats.norm.ppf(1-alpha/2)
    reject=abs(z_val)>crit if tail.startswith("two") else z_val>crit
    decision="rejected" if reject else "failed to reject"
    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *z* = {z_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"Critical statistic: z₍crit₎ = {crit:.2f}, *p* = {alpha:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** *z* = {z_val:.2f}, *p* = {p_calc:.3f} "
        f"({'one' if tail.startswith('one') else 'two'}‑tailed). "
        f"The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )

def tab_z():
    st.subheader("Tab 2 • z‑Distribution")
    c1,c2=st.columns(2)
    with c1:
        z_val=st.number_input("z statistic",value=1.64,key="z_val")
    with c2:
        alpha=st.number_input("α",value=0.05,step=0.01,min_value=0.0001,max_value=0.5,key="z_alpha")
        tail=st.radio("Tail",["one‑tailed","two‑tailed"],key="z_tail")
    if st.button("Update Plot",key="z_plot"):
        st.pyplot(plot_z(z_val,alpha,tail))
    with st.expander("Step‑by‑step z‑table"):
        z_table(z_val,alpha,tail); z_apa(z_val,alpha,tail)

# 3) F‑Distribution ─────────────────────────────────────────────────────────
def plot_f(f_calc,df1,df2,alpha):
    xs=np.linspace(0,stats.f.ppf(.995,df1,df2)*1.1,400); ys=stats.f.pdf(xs,df1,df2)
    fig,ax=plt.subplots(figsize=(12,4),dpi=100); ax.plot(xs,ys,'k')
    ax.fill_between(xs,ys,color="lightgrey",alpha=.25,label="Fail to Reject H₀")
    crit=stats.f.ppf(1-alpha,df1,df2)
    ax.fill_between(xs[xs>=crit],ys[xs>=crit],color="red",alpha=.30,label="Reject H₀")
    ax.axvline(crit,color="green",ls="--"); ax.axvline(f_calc,color="blue",ls="--")
    place_label(ax,[],crit,stats.f.pdf(crit,df1,df2)+.02,f"F₍crit₎={crit:.2f}",color="green")
    place_label(ax,[],f_calc,stats.f.pdf(f_calc,df1,df2)+.02,f"F₍calc₎={f_calc:.2f}",color="blue")
    ax.set_xlabel("F"); ax.set_ylabel("Density"); ax.legend()
    ax.set_title(f"F‑Distribution (df₁ {df1}, df₂ {df2})"); fig.tight_layout(); return fig

def f_table(df1,df2,alpha):
    animate(lambda s:build_f_html(df1,df2,alpha,s),frames=3,
            key=f"f_{df1}_{df2}_{alpha}")

def f_apa(f_val,df1,df2,alpha):
    p_calc=stats.f.sf(f_val,df1,df2); crit=stats.f.ppf(1-alpha,df1,df2)
    decision="rejected" if f_val>crit else "failed to reject"
    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *F*({df1}, {df2}) = {f_val:.2f}, "
        f"*p* = {p_calc:.3f}.  \n"
        f"Critical statistic: F₍crit₎ = {crit:.2f}, *p* = {alpha:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** *F*({df1}, {df2}) = {f_val:.2f}, *p* = {p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )

def tab_f():
    st.subheader("Tab 3 • F‑Distribution")
    c1,c2=st.columns(2)
    with c1:
        f_val=st.number_input("F statistic",value=4.32,key="f_val")
        df1=st.number_input("df₁ (numerator)",min_value=1,value=5,step=1,key="f_df1")
    with c2:
        df2=st.number_input("df₂ (denominator)",min_value=1,value=20,step=1,key="f_df2")
        alpha=st.number_input("α",value=0.05,step=0.01,min_value=0.0001,max_value=0.5,key="f_alpha")
    if st.button("Update Plot",key="f_plot"):
        st.pyplot(plot_f(f_val,df1,df2,alpha))
    with st.expander("Step‑by‑step F‑table"):
        f_table(df1,df2,alpha); f_apa(f_val,df1,df2,alpha)

# 4) Chi‑Square ─────────────────────────────────────────────────────────────
def plot_chi(chi_calc,df,alpha):
    xs=np.linspace(0,stats.chi2.ppf(.995,df)*1.1,400); ys=stats.chi2.pdf(xs,df)
    fig,ax=plt.subplots(figsize=(12,4),dpi=100); ax.plot(xs,ys,'k')
    ax.fill_between(xs,ys,color="lightgrey",alpha=.25,label="Fail to Reject H₀")
    crit=stats.chi2.ppf(1-alpha,df)
    ax.fill_between(xs[xs>=crit],ys[xs>=crit],color="red",alpha=.30,label="Reject H₀")
    ax.axvline(crit,color="green",ls="--"); ax.axvline(chi_calc,color="blue",ls="--")
    place_label(ax,[],crit,stats.chi2.pdf(crit,df)+.02,f"χ²₍crit₎={crit:.2f}",color="green")
    place_label(ax,[],chi_calc,stats.chi2.pdf(chi_calc,df)+.02,f"χ²₍calc₎={chi_calc:.2f}",color="blue")
    ax.set_xlabel("χ²"); ax.set_ylabel("Density"); ax.legend()
    ax.set_title(f"χ²‑Distribution (df = {df})"); fig.tight_layout(); return fig

def chi_table(df,alpha):
    animate(lambda s:build_chi_html(df,alpha,s),frames=3,key=f"chi_{df}_{alpha}")

def chi_apa(chi_val,df,alpha):
    p_calc=stats.chi2.sf(chi_val,df); crit=stats.chi2.ppf(1-alpha,df)
    decision="rejected" if chi_val>crit else "failed to reject"
    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: χ²({df}) = {chi_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"Critical statistic: χ²₍crit₎ = {crit:.2f}, *p* = {alpha:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** χ²({df}) = {chi_val:.2f}, *p* = {p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at α = {alpha:.3f}."
    )

def tab_chi():
    st.subheader("Tab 4 • Chi‑Square (χ²)")
    c1,c2=st.columns(2)
    with c1:
        chi_val=st.number_input("χ² statistic",value=7.88,key="chi_val")
        df=st.number_input("df",min_value=1,value=3,step=1,key="chi_df")
    with c2:
        alpha=st.selectbox("α",[0.10,0.05,0.01,0.001],index=1,key="chi_alpha")
    if st.button("Update Plot",key="chi_plot"):
        st.pyplot(plot_chi(chi_val,df,alpha))
    with st.expander("Step‑by‑step χ²‑table"):
        chi_table(df,alpha); chi_apa(chi_val,df,alpha)

# 5) Mann‑Whitney U ─────────────────────────────────────────────────────────
def u_crit(n1,n2,alpha,tail):
    μ=n1*n2/2; σ=np.sqrt(n1*n2*(n1+n2+1)/12)
    z=stats.norm.ppf(alpha if tail.startswith("one") else alpha/2)
    return int(np.floor(μ+z*σ))

def plot_u(u_calc,n1,n2,alpha,tail):
    μ=n1*n2/2; σ=np.sqrt(n1*n2*(n1+n2+1)/12)
    xs=np.linspace(μ-4*σ,μ+4*σ,400); ys=stats.norm.pdf(xs,μ,σ)
    fig,ax=plt.subplots(figsize=(12,4),dpi=100); ax.plot(xs,ys,'k')
    ax.fill_between(xs,ys,color="lightgrey",alpha=.25,label="Fail to Reject H₀")
    if tail.startswith("one"):
        crit=u_crit(n1,n2,alpha,tail)
        ax.fill_between(xs[xs<=crit],ys[xs<=crit],color="red",alpha=.30,label="Reject H₀")
        ax.axvline(crit,color="green",ls="--")
    else:
        crit=u_crit(n1,n2,alpha,tail); hi=n1*n2-crit
        ax.fill_between(xs[xs<=crit],ys[xs<=crit],color="red",alpha=.30)
        ax.fill_between(xs[xs>=hi],ys[xs>=hi],color="red",alpha=.30,label="Reject H₀")
        ax.axvline(crit,color="green",ls="--"); ax.axvline(hi,color="green",ls="--")
    ax.axvline(u_calc,color="blue",ls="--")
    place_label(ax,[],u_calc,stats.norm.pdf(u_calc,μ,σ)+.02,f"U₍calc₎={u_calc}",color="blue")
    ax.set_xlabel("U"); ax.set_ylabel("Approx. density"); ax.legend(); ax.set_title("Mann‑Whitney U"); fig.tight_layout(); return fig

def u_table(n1,n2,alpha,tail):
    animate(lambda s:build_u_html(n1,n2,alpha,tail,s),frames=3,
            key=f"u_{n1}_{n2}_{alpha}_{tail}")

def u_apa(u_val,n1,n2,alpha,tail):
    μ=n1*n2/2; σ=np.sqrt(n1*n2*(n1+n2+1)/12)
    if tail.startswith("one"):
        p_calc=stats.norm.cdf((u_val-μ)/σ)
        crit=u_crit(n1,n2,alpha,tail); reject=u_val<=crit
    else:
        p_calc=stats.norm.sf(abs(u_val-μ)/σ)*2
        crit=u_crit(n1,n2,alpha,tail); reject=(u_val<=crit)or(u_val>=n1*n2-crit)
    decision="rejected" if reject else "failed to reject"
    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *U* = {u_val}, *p* = {p_calc:.3f}.  \n"
        f"Critical statistic: U₍crit₎ = {crit}, *p* = {alpha:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** *U* = {u_val}, *p* = {p_calc:.3f} "
        f"({'one' if tail.startswith('one') else 'two'}‑tailed). "
        f"The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )

def tab_u():
    st.subheader("Tab 5 • Mann‑Whitney U")
    c1,c2=st.columns(2)
    with c1:
        u_val=st.number_input("U statistic",value=23,key="u_val")
        n1=st.number_input("n₁",min_value=2,value=10,step=1,key="u_n1")
    with c2:
        n2=st.number_input("n₂",min_value=2,value=12,step=1,key="u_n2")
        alpha=st.number_input("α",value=0.05,step=0.01,min_value=0.0001,max_value=0.5,key="u_alpha")
        tail=st.radio("Tail",["one‑tailed","two‑tailed"],key="u_tail")
    if st.button("Update Plot",key="u_plot"):
        st.pyplot(plot_u(u_val,n1,n2,alpha,tail))
    with st.expander("Step‑by‑step U‑table"):
        u_table(n1,n2,alpha,tail); u_apa(u_val,n1,n2,alpha,tail)

# 6) Wilcoxon T ─────────────────────────────────────────────────────────────
def w_crit(n,alpha,tail):
    μ=n*(n+1)/4; σ=np.sqrt(n*(n+1)*(2*n+1)/24)
    z=stats.norm.ppf(alpha if tail.startswith("one") else alpha/2)
    return int(np.floor(μ+z*σ))

def plot_w(t_calc,n,alpha,tail):
    μ=n*(n+1)/4; σ=np.sqrt(n*(n+1)*(2*n+1)/24)
    xs=np.linspace(μ-4*σ,μ+4*σ,400); ys=stats.norm.pdf(xs,μ,σ)
    fig,ax=plt.subplots(figsize=(12,4),dpi=100); ax.plot(xs,ys,'k')
    ax.fill_between(xs,ys,color="lightgrey",alpha=.25,label="Fail to Reject H₀")
    if tail.startswith("one"):
        crit=w_crit(n,alpha,tail)
        ax.fill_between(xs[xs<=crit],ys[xs<=crit],color="red",alpha=.30,label="Reject H₀")
        ax.axvline(crit,color="green",ls="--")
    else:
        crit=w_crit(n,alpha,tail); hi=n*(n+1)/2-crit
        ax.fill_between(xs[xs<=crit],ys[xs<=crit],color="red",alpha=.30)
        ax.fill_between(xs[xs>=hi],ys[xs>=hi],color="red",alpha=.30,label="Reject H₀")
        ax.axvline(crit,color="green",ls="--"); ax.axvline(hi,color="green",ls="--")
    ax.axvline(t_calc,color="blue",ls="--")
    place_label(ax,[],t_calc,stats.norm.pdf(t_calc,μ,σ)+.02,f"T₍calc₎={t_calc}",color="blue")
    ax.set_xlabel("T"); ax.set_ylabel("Approx. density"); ax.legend(); ax.set_title("Wilcoxon Signed‑Rank T"); fig.tight_layout(); return fig

def w_table(n,alpha,tail):
    animate(lambda s:build_w_html(n,alpha,tail,s),frames=3,
            key=f"w_{n}_{alpha}_{tail}")

def w_apa(t_val,n,alpha,tail):
    μ=n*(n+1)/4; σ=np.sqrt(n*(n+1)*(2*n+1)/24)
    if tail.startswith("one"):
        p_calc=stats.norm.cdf((t_val-μ)/σ); crit=w_crit(n,alpha,tail); reject=t_val<=crit
    else:
        p_calc=stats.norm.sf(abs(t_val-μ)/σ)*2
        crit=w_crit(n,alpha,tail); reject=(t_val<=crit)or(t_val>=n*(n+1)/2-crit)
    decision="rejected" if reject else "failed to reject"
    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *T* = {t_val}, *p* = {p_calc:.3f}.  \n"
        f"Critical statistic: T₍crit₎ = {crit}, *p* = {alpha:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** *T* = {t_val}, *p* = {p_calc:.3f} "
        f"({'one' if tail.startswith('one') else 'two'}‑tailed). "
        f"The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )

def tab_w():
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank T")
    c1,c2=st.columns(2)
    with c1:
        t_val=st.number_input("T statistic",value=15,key="w_val")
        n=st.number_input("N (non‑zero diffs)",min_value=5,value=12,step=1,key="w_n")
    with c2:
        alpha=st.number_input("α",value=0.05,step=0.01,min_value=0.0001,max_value=0.5,key="w_alpha")
        tail=st.radio("Tail",["one‑tailed","two‑tailed"],key="w_tail")
    if st.button("Update Plot",key="w_plot"):
        st.pyplot(plot_w(t_val,n,alpha,tail))
    with st.expander("Step‑by‑step T‑table"):
        w_table(n,alpha,tail); w_apa(t_val,n,alpha,tail)

# 7) Binomial ───────────────────────────────────────────────────────
def critical_binom(n,p,alpha):
    cum=0; klo=0
    for k in range(n+1):
        cum+=stats.binom.pmf(k,n,p)
        if cum>=alpha/2: klo=k; break
    cum=0; khi=n
    for k in range(n,‑1,‑1):
        cum+=stats.binom.pmf(k,n,p)
        if cum>=alpha/2: khi=k; break
    return klo,khi

def plot_binom(k,n,p):
    xs=np.arange(0,n+1); ys=stats.binom.pmf(xs,n,p)
    fig,ax=plt.subplots(figsize=(12,4),dpi=100); ax.bar(xs,ys,color="lightgrey")
    ax.bar(k,stats.binom.pmf(k,n,p),color="blue",label=f"k = {k}")
    ax.set_xlabel("k"); ax.set_ylabel("P(X=k)"); ax.legend()
    ax.set_title(f"Binomial (n = {n}, p = {p})"); fig.tight_layout(); return fig

def binom_table(k,n,p):
    animate(lambda s:build_b_html(k,n,p,s),frames=2,key=f"b_{k}_{n}_{p}")

def binom_apa(k,n,p,alpha):
    cdf_val=stats.binom.cdf(k,n,p)
    p_calc=2*min(cdf_val,1-cdf_val+stats.binom.pmf(k,n,p)); p_calc=min(p_calc,1)
    klo,khi=critical_binom(n,p,alpha); decision="rejected" if (k<=klo or k>=khi) else "failed to reject"
    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: k = {k}, *p* = {p_calc:.3f}.  \n"
        f"Critical region: k ≤ {klo} or k ≥ {khi}, *p* = {alpha:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}**.  \n"
        f"*p* comparison → H₀ **{decision}**.  \n"
        f"**APA 7 report:** Exact binomial test, *p* = {p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at α = {alpha:.2f}."
    )

def tab_binom():
    st.subheader("Tab 7 • Binomial")
    c1,c2=st.columns(2)
    with c1:
        n=st.number_input("n (trials)",min_value=1,value=20,step=1,key="b_n")
        p=st.number_input("π (null proportion)",value=0.50,step=0.01,min_value=0.01,max_value=0.99,key="b_p")
    with c2:
        k=st.number_input("k (successes)",min_value=0,value=12,step=1,key="b_k")
        alpha=st.number_input("α (two‑tailed)",value=0.05,step=0.01,min_value=0.0001,max_value=0.5,key="b_alpha")
    if st.button("Update Plot",key="b_plot"):
        st.pyplot(plot_binom(k,n,p))
    with st.expander("Quick table (k ±5)"):
        binom_table(k,n,p); binom_apa(k,n,p,alpha)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                  MAIN                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    st.set_page_config("PSYC250 – Statistical Tables Explorer",layout="wide")
    st.title("PSYC250 – Statistical Tables Explorer (12 × 4 figures)")
    tabs=st.tabs(["t‑Dist","z‑Dist","F‑Dist","Chi‑Square",
                  "Mann–Whitney U","Wilcoxon T","Binomial"])
    with tabs[0]: tab_t()
    with tabs[1]: tab_z()
    with tabs[2]: tab_f()
    with tabs[3]: tab_chi()
    with tabs[4]: tab_u()
    with tabs[5]: tab_w()
    with tabs[6]: tab_binom()

if __name__=="__main__":
    main()
