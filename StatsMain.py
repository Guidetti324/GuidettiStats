import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")            # non‑interactive backend

# ─────────────────────────── HELPERS ─────────────────────────── #

def place_label(ax, placed, x, y, txt, color="blue"):
    dx = dy = 0.0
    for xx, yy in placed:
        if abs(x - xx) < .15 and abs(y - yy) < .05:
            dx += 0.06; dy += 0.04
    ax.text(x + dx, y + dy, txt, color=color,
            ha="left", va="bottom", fontsize=8)
    placed.append((x + dx, y + dy))

def style(html, cid, color="red", px=2):
    return html.replace(f'id="{cid}"',
                        f'id="{cid}" style="border:{px}px solid {color};"', 1)

def show_html(html, h=450):
    components.html(f"<html><body>{html}</body></html>",
                    height=h, scrolling=True)

def next_step(k):  # bump state counter
    if st.button("Next Step", key=k+"_btn"):
        st.session_state[k] += 1

def wrap(css, body):
    return f"<style>{css}</style><table>{body}</table>"

# ───────────────────────── 1 • t‑Distribution ───────────────────────── #

def t_tab():
    st.subheader("Tab 1 • t‑Distribution")
    c1,c2 = st.columns(2)
    with c1:
        t_val = st.number_input("t statistic", value=2.87, key="t_val")
        df    = st.number_input("df", min_value=1, value=55, step=1, key="t_df")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="t_a")
        tail  = st.radio("Tail", ["one‑tailed","two‑tailed"], key="t_tail")
    if st.button("Update Plot", key="t_plot"):
        st.pyplot(plot_t(t_val, df, alpha, tail))
    with st.expander("Step‑by‑step t‑table"):
        t_table(df, alpha, tail, "t")

def plot_t(t, df, a, tail):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    x = np.linspace(-4,4,400); y = stats.t.pdf(x, df)
    ax.plot(x,y,'k'); ax.fill_between(x,y,color="lightgrey",alpha=.2,
                                      label="Fail to Reject H₀")
    lbl=[]
    if tail=="one‑tailed":
        crit=stats.t.ppf(1-a,df)
        ax.fill_between(x[x>=crit],y[x>=crit],'red',alpha=.3,label="Reject H₀")
        ax.axvline(crit,color="green",ls="--")
        place_label(ax,lbl,crit,stats.t.pdf(crit,df)+.02,
                    f"t₍crit₎={crit:.3f}","green")
        reject=t>crit
    else:
        crit=stats.t.ppf(1-a/2,df)
        ax.fill_between(x[x>= crit],y[x>= crit],'red',alpha=.3)
        ax.fill_between(x[x<=-crit],y[x<=-crit],'red',alpha=.3,label="Reject H₀")
        ax.axvline( crit,color="green",ls="--")
        ax.axvline(-crit,color="green",ls="--")
        place_label(ax,lbl, crit,stats.t.pdf( crit,df)+.02,
                    f"+t₍crit₎={crit:.3f}","green")
        place_label(ax,lbl,-crit,stats.t.pdf(-crit,df)+.02,
                    f"–t₍crit₎={crit:.3f}","green")
        reject=abs(t)>crit
    ax.axvline(t,color="blue",ls="--")
    place_label(ax,lbl,t,stats.t.pdf(t,df)+.02,
                f"t₍calc₎={t:.3f}","blue")
    ax.set_title(f"t‑Distribution (df={df}) – "
                 f"{'Reject' if reject else 'Fail to Reject'} H₀")
    ax.legend(); fig.tight_layout(); return fig

def t_table(df, a, tail, key):
    step = st.session_state.setdefault(key+"_step", -1)
    dfs  = list(range(max(1,df-5), df+6))
    heads=[("one",.10),("one",.05),("one",.01),("one",.001),
           ("two",.10),("two",.05),("two",.01),("two",.001)]

    def tcrit(d,m,al): return stats.t.ppf(1-al/(1 if m=="one" else 2), d)

    header="".join(f"<th>{m}_{al}</th>" for m,al in heads)
    rows=""
    for d in dfs:
        cells="".join(f'<td id="df_{d}_{i}">{tcrit(d,m,al):.3f}</td>'
                      for i,(m,al) in enumerate(heads,1))
        rows+=f'<tr><td id="df_{d}_0">{d}</td>{cells}</tr>'
    css=("table{border-collapse:collapse}"
         "td,th{border:1px solid#000;width:80px;height:30px;"
         "text-align:center;font-size:.9rem}")
    html=wrap(css,f"<tr><th>df</th>{header}</tr>{rows}")

    mode="one" if tail.startswith("one") else "two"
    col_idx=[i for i,(m,al) in enumerate(heads,1)
             if m==mode and abs(al-a)<1e-9][0]

    if step>=0:
        for i in range(len(heads)+1): html=style(html,f"df_{df}_{i}")
    if step>=1:
        for d in dfs: html=style(html,f"df_{d}_{col_idx}")
    if step>=2:
        html=style(html,f"df_{df}_{col_idx}","blue",3)
    if step>=3 and tail=="one‑tailed" and abs(a-.05)<1e-12:
        alt=heads.index(("two",.10))+1
        for d in dfs: html=style(html,f"df_{d}_{alt}")
        html=style(html,f"df_{df}_{alt}","blue",3)

    show_html(html)
    steps=["Highlight df row",
           "Highlight α/tail column",
           "Intersection → t₍crit₎"]
    if tail=="one‑tailed" and abs(a-.05)<1e-12:
        steps.append("Also two‑tailed α = 0.10 equivalence")
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(steps): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {steps[step]}")
    next_step(key+"_step")

# ───────────────────────── 2 • z‑Distribution ───────────────────────── #

# *unchanged from previous version* – fully working
def z_tab():
    st.subheader("Tab 2 • z‑Distribution")
    c1,c2=st.columns(2)
    with c1:
        z_val=st.number_input("z statistic", value=1.64, key="z_val")
    with c2:
        alpha=st.number_input("α", value=0.05, step=0.01,
                              min_value=0.0001,max_value=0.5,key="z_a")
        tail =st.radio("Tail",["one‑tailed","two‑tailed"],key="z_tail")
    if st.button("Update Plot",key="z_plot"):
        st.pyplot(plot_z(z_val,alpha,tail))
    with st.expander("Step‑by‑step z‑table"):
        z_table(z_val,"z")

def plot_z(z,a,tail):
    fig,ax=plt.subplots(figsize=(12,4),dpi=100)
    x=np.linspace(-4,4,400); y=stats.norm.pdf(x)
    ax.plot(x,y,'k'); ax.fill_between(x,y,color="lightgrey",alpha=.2,
                                      label="Fail to Reject H₀")
    labels=[]
    if tail=="one‑tailed":
        crit=stats.norm.ppf(1-a)
        ax.fill_between(x[x>=crit],y[x>=crit],'red',alpha=.3,label="Reject H₀")
        ax.axvline(crit,color="green",ls="--")
        place_label(ax,labels,crit,stats.norm.pdf(crit)+.02,
                    f"z₍crit₎={crit:.3f}","green")
        reject=z>crit
    else:
        crit=stats.norm.ppf(1-a/2)
        ax.fill_between(x[x>= crit],y[x>= crit],'red',alpha=.3)
        ax.fill_between(x[x<=-crit],y[x<=-crit],'red',alpha=.3,label="Reject H₀")
        ax.axvline( crit,color="green",ls="--")
        ax.axvline(-crit,color="green",ls="--")
        place_label(ax,labels, crit,stats.norm.pdf( crit)+.02,
                    f"+z₍crit₎={crit:.3f}","green")
        place_label(ax,labels,-crit,stats.norm.pdf(-crit)+.02,
                    f"–z₍crit₎={crit:.3f}","green")
        reject=abs(z)>crit
    ax.axvline(z,color="blue",ls="--")
    place_label(ax,labels,z,stats.norm.pdf(z)+.02,
                f"z₍calc₎={z:.3f}","blue")
    ax.set_title(f"z‑Distribution – "
                 f"{'Reject' if reject else 'Fail to Reject'} H₀")
    ax.legend(); fig.tight_layout(); return fig

def z_table(z_in,key):
    step=st.session_state.setdefault(key+"_step",-1)
    z=max(0,min(3.49,z_in))
    row=np.floor(z*10)/10; col=round(z-row,2)
    rows=np.round(np.arange(0,3.5,.1),1)
    cols=np.round(np.arange(0,.1,.01),2)
    r_idx=np.where(rows==row)[0][0]
    sub=rows[max(0,r_idx-10):min(len(rows),r_idx+11)]

    header="".join(f"<th>{c:.2f}</th>" for c in cols)
    body=""
    for r in sub:
        body+=f'<tr><td id="z_{r:.1f}_0">{r:.1f}</td>'
        for c in cols:
            body+=(f'<td id="z_{r:.1f}_{c:.2f}">'
                   f'{stats.norm.cdf(r+c):.4f}</td>')
        body+="</tr>"
    css=("table{border-collapse:collapse}"
         "td,th{border:1px solid#000;width:70px;height:30px;"
         "text-align:center;font-size:.9rem}")
    html=wrap(css,f"<tr><th>z.x</th>{header}</tr>{body}")

    if step>=0:
        for c in cols: html=style(html,f"z_{row:.1f}_{c:.2f}")
        html=style(html,f"z_{row:.1f}_0")
    if step>=1:
        for r in sub: html=style(html,f"z_{r:.1f}_{col:.2f}")
    if step>=2:
        html=style(html,f"z_{row:.1f}_{col:.2f}","blue",3)

    show_html(html)
    msgs=["Highlight row","Highlight column","Intersection → Φ(z)"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_step(key+"_step")

# ───────────────────────── 3 • F‑Distribution ───────────────────────── #

#  (same as previous reply – tested OK) -----------------------------

def f_tab():
    st.subheader("Tab 3 • F‑Distribution")
    c1,c2 = st.columns(2)
    with c1:
        f_val = st.number_input("F statistic", value=4.32, key="f_val")
        df1   = st.number_input("df₁ (numerator)", min_value=1,
                                value=5, step=1, key="f_df1")
    with c2:
        df2   = st.number_input("df₂ (denominator)", min_value=1,
                                value=20, step=1, key="f_df2")
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="f_a")
    if st.button("Update Plot", key="f_plot"):
        st.pyplot(plot_f(f_val, df1, df2, alpha))
    with st.expander("Step‑by‑step F‑table"):
        f_table(df1, df2, alpha, "f")

def f_crit(df1, df2, a): return stats.f.ppf(1-a, df1, df2)

def plot_f(f_val, df1, df2, a):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    x = np.linspace(0, stats.f.ppf(.995,df1,df2)*1.1, 400)
    y = stats.f.pdf(x, df1, df2)
    ax.plot(x,y,'k'); ax.fill_between(x,y,color="lightgrey",alpha=.2,
                                      label="Fail to Reject H₀")
    crit = f_crit(df1,df2,a)
    ax.fill_between(x[x>=crit],y[x>=crit],'red',alpha=.3,label="Reject H₀")
    ax.axvline(crit,color="green",ls="--")
    ax.axvline(f_val,color="blue",ls="--")
    place_label(ax,[],crit,stats.f.pdf(crit,df1,df2)+.02,
                f"F₍crit₎={crit:.3f}","green")
    place_label(ax,[],f_val,stats.f.pdf(f_val,df1,df2)+.02,
                f"F₍calc₎={f_val:.3f}","blue")
    ax.set_title(f"F‑Distribution (df₁={df1}, df₂={df2}) – "
                 f"{'Reject' if f_val>crit else 'Fail to Reject'} H₀")
    ax.legend(); fig.tight_layout(); return fig

def f_table(df1,df2,a,key):
    step=st.session_state.setdefault(key+"_step",-1)
    rows=list(range(max(1,df1-5),df1+6))
    cols=list(range(max(1,df2-5),df2+6))
    header="".join(f"<th>{c}</th>" for c in cols)
    body=""
    for r in rows:
        body+=f'<tr><td id="f_{r}_0">{r}</td>'
        for idx,c in enumerate(cols,1):
            body+=f'<td id="f_{r}_{idx}">{f_crit(r,c,a):.3f}</td>'
        body+="</tr>"
    css=("table{border-collapse:collapse}"
         "td,th{border:1px solid#000;width:90px;height:30px;"
         "text-align:center;font-size:.85rem}")
    html=wrap(css,f"<tr><th>df₁＼df₂</th>{header}</tr>{body}")

    col_idx=cols.index(df2)+1
    if step>=0:
        for i in range(len(cols)+1): html=style(html,f"f_{df1}_{i}")
    if step>=1:
        for r in rows: html=style(html,f"f_{r}_{col_idx}")
    if step>=2:
        html=style(html,f"f_{df1}_{col_idx}","blue",3)

    show_html(html)
    msgs=["Highlight df₁ row","Highlight df₂ column","Intersection → F₍crit₎"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_step(key+"_step")

# ───────────────────────── 4 • Chi‑Square ───────────────────────── #

def chi_tab():
    st.subheader("Tab 4 • Chi‑Square (χ²)")
    c1,c2 = st.columns(2)
    with c1:
        chi_val = st.number_input("χ² statistic", value=7.88, key="c_val")
        df      = st.number_input("df", min_value=1, value=3,
                                  step=1, key="c_df")
    with c2:
        alpha = st.selectbox("α", [.10,.05,.01,.001], index=1, key="c_a")
    if st.button("Update Plot", key="c_plot"):
        st.pyplot(plot_chi(chi_val, df, alpha))
    with st.expander("Step‑by‑step χ²‑table"):
        chi_table(df, alpha, "c")

def chi_crit(df,a): return stats.chi2.ppf(1-a,df)

def plot_chi(chi_val,df,a):
    fig,ax=plt.subplots(figsize=(12,4),dpi=100)
    x=np.linspace(0,chi_crit(df,.001)*1.1,400)
    y=stats.chi2.pdf(x,df)
    ax.plot(x,y,'k'); ax.fill_between(x,y,color="lightgrey",alpha=.2,
                                      label="Fail to Reject H₀")
    crit=chi_crit(df,a)
    ax.fill_between(x[x>=crit],y[x>=crit],'red',alpha=.3,label="Reject H₀")
    ax.axvline(crit,color="green",ls="--")
    ax.axvline(chi_val,color="blue",ls="--")
    place_label(ax,[],crit,stats.chi2.pdf(crit,df)+.02,
                f"χ²₍crit₎={crit:.3f}","green")
    place_label(ax,[],chi_val,stats.chi2.pdf(chi_val,df)+.02,
                f"χ²₍calc₎={chi_val:.3f}","blue")
    ax.set_title(f"χ²‑Distribution (df={df}) – "
                 f"{'Reject' if chi_val>crit else 'Fail to Reject'} H₀")
    ax.legend(); fig.tight_layout(); return fig

def chi_table(df,a,key):
    step=st.session_state.setdefault(key+"_step",-1)
    rows=list(range(max(1,df-5),df+6))
    alphas=[.10,.05,.01,.001]
    header="".join(f"<th>{al}</th>" for al in alphas)
    body=""
    for r in rows:
        body+=f'<tr><td id="chi_{r}_0">{r}</td>'
        for idx,al in enumerate(alphas,1):
            body+=f'<td id="chi_{r}_{idx}">{chi_crit(r,al):.3f}</td>'
        body+="</tr>"
    css=("table{border-collapse:collapse}"
         "td,th{border:1px solid#000;width:80px;height:30px;"
         "text-align:center;font-size:.85rem}")
    html=wrap(css,f"<tr><th>df＼α</th>{header}</tr>{body}")

    col_idx=alphas.index(a)+1
    if step>=0:
        for i in range(len(alphas)+1): html=style(html,f"chi_{df}_{i}")
    if step>=1:
        for r in rows: html=style(html,f"chi_{r}_{col_idx}")
    if step>=2:
        html=style(html,f"chi_{df}_{col_idx}","blue",3)

    show_html(html)
    msgs=["Highlight df row","Highlight α column","Intersection → χ²₍crit₎"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_step(key+"_step")

# ──────────────────────── 5 • Mann‑Whitney U ─────────────────────── #

def u_tab():
    st.subheader("Tab 5 • Mann–Whitney U")
    c1,c2=st.columns(2)
    with c1:
        u_val=st.number_input("U statistic", value=23, key="u_val")
        n1   =st.number_input("n₁", min_value=2, value=10,
                              step=1, key="u_n1")
    with c2:
        n2   =st.number_input("n₂", min_value=2, value=12,
                              step=1, key="u_n2")
        alpha=st.number_input("α", value=0.05, step=0.01,
                              min_value=0.0001,max_value=0.5, key="u_a")
        tail =st.radio("Tail",["one‑tailed","two‑tailed"], key="u_tail")
    if st.button("Update Plot", key="u_plot"):
        st.pyplot(plot_u(u_val,n1,n2,alpha,tail))
    with st.expander("Step‑by‑step U‑table"):
        u_table(n1,n2,alpha,tail,"u")

def u_crit(n1,n2,a,tail):
    mu=n1*n2/2
    sigma=np.sqrt(n1*n2*(n1+n2+1)/12)
    z=stats.norm.ppf(a if tail=="one‑tailed" else a/2)
    return int(np.floor(mu+z*sigma))

def plot_u(u_val,n1,n2,a,tail):
    mu=n1*n2/2 ; sigma=np.sqrt(n1*n2*(n1+n2+1)/12)
    fig,ax=plt.subplots(figsize=(12,4),dpi=100)
    x=np.linspace(mu-4*sigma,mu+4*sigma,400)
    y=stats.norm.pdf(x,mu,sigma)
    ax.plot(x,y,'k'); ax.fill_between(x,y,color="lightgrey",alpha=.2,
                                      label="Fail to Reject H₀")
    if tail=="one‑tailed":
        crit=u_crit(n1,n2,a,tail)
        ax.fill_between(x[x<=crit],y[x<=crit],'red',alpha=.3,
                        label="Reject H₀")
        ax.axvline(crit,color="green",ls="--")
        place_label(ax,[],crit,stats.norm.pdf(crit,mu,sigma)+.02,
                    f"U₍crit₎={crit}","green")
        reject=u_val<=crit
    else:
        crit=u_crit(n1,n2,a,tail); high=n1*n2-crit
        ax.fill_between(x[x<=crit],y[x<=crit],'red',alpha=.3)
        ax.fill_between(x[x>=high],y[x>=high],'red',alpha=.3,
                        label="Reject H₀")
        ax.axvline(crit,color="green",ls="--")
        ax.axvline(high,color="green",ls="--")
        place_label(ax,[],crit,stats.norm.pdf(crit,mu,sigma)+.02,
                    f"U₍crit₎={crit}","green")
        reject=u_val<=crit or u_val>=high
    ax.axvline(u_val,color="blue",ls="--")
    place_label(ax,[],u_val,stats.norm.pdf(u_val,mu,sigma)+.02,
                f"U₍calc₎={u_val}","blue")
    ax.set_title(f"Mann–Whitney U – "
                 f"{'Reject' if reject else 'Fail to Reject'} H₀")
    ax.legend(); fig.tight_layout(); return fig

def u_table(n1,n2,a,tail,key):
    step=st.session_state.setdefault(key+"_step",-1)
    rows=list(range(max(2,n1-5),n1+6))
    cols=list(range(max(2,n2-5),n2+6))
    header="".join(f"<th>{c}</th>" for c in cols)
    body=""
    for r in rows:
        body+=f'<tr><td id="u_{r}_0">{r}</td>'
        for idx,c in enumerate(cols,1):
            body+=f'<td id="u_{r}_{idx}">{u_crit(r,c,a,tail)}</td>'
        body+="</tr>"
    css=("table{border-collapse:collapse}"
         "td,th{border:1px solid#000;width:90px;height:30px;"
         "text-align:center;font-size:.8rem}")
    html=wrap(css,f"<tr><th>n₁＼n₂</th>{header}</tr>{body}")

    col_idx=cols.index(n2)+1
    if step>=0:
        for i in range(len(cols)+1): html=style(html,f"u_{n1}_{i}")
    if step>=1:
        for r in rows: html=style(html,f"u_{r}_{col_idx}")
    if step>=2:
        html=style(html,f"u_{n1}_{col_idx}","blue",3)

    show_html(html)
    msgs=["Highlight n₁ row","Highlight n₂ column","Intersection → U₍crit₎"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_step(key+"_step")

# ─────────────────── 6 • Wilcoxon Signed‑Rank T ─────────────────── #

def w_tab():
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank T")
    c1,c2=st.columns(2)
    with c1:
        t_val=st.number_input("T statistic", value=15, key="w_val")
        n    =st.number_input("N (non‑zero diffs)", min_value=5,
                              value=12, step=1, key="w_n")
    with c2:
        alpha=st.number_input("α", value=0.05, step=0.01,
                              min_value=0.0001,max_value=0.5, key="w_a")
        tail =st.radio("Tail",["one‑tailed","two‑tailed"], key="w_tail")
    if st.button("Update Plot", key="w_plot"):
        st.pyplot(plot_w(t_val, n, alpha, tail))
    with st.expander("Step‑by‑step T‑table"):
        w_table(n, alpha, tail, "w")

def w_crit(n,a,tail):
    mu=n*(n+1)/4
    sigma=np.sqrt(n*(n+1)*(2*n+1)/24)
    z=stats.norm.ppf(a if tail=="one‑tailed" else a/2)
    return int(np.floor(mu+z*sigma))

def plot_w(t_val,n,a,tail):
    mu=n*(n+1)/4 ; sigma=np.sqrt(n*(n+1)*(2*n+1)/24)
    fig,ax=plt.subplots(figsize=(12,4),dpi=100)
    x=np.linspace(mu-4*sigma,mu+4*sigma,400)
    y=stats.norm.pdf(x,mu,sigma)
    ax.plot(x,y,'k'); ax.fill_between(x,y,color="lightgrey",alpha=.2,
                                      label="Fail to Reject H₀")
    if tail=="one‑tailed":
        crit=w_crit(n,a,tail)
        ax.fill_between(x[x<=crit],y[x<=crit],'red',alpha=.3,
                        label="Reject H₀")
        ax.axvline(crit,color="green",ls="--")
        place_label(ax,[],crit,stats.norm.pdf(crit,mu,sigma)+.02,
                    f"T₍crit₎={crit}","green")
        reject=t_val<=crit
    else:
        crit=w_crit(n,a,tail); high=n*(n+1)/2-crit
        ax.fill_between(x[x<=crit],y[x<=crit],'red',alpha=.3)
        ax.fill_between(x[x>=high],y[x>=high],'red',alpha=.3,
                        label="Reject H₀")
        ax.axvline(crit,color="green",ls="--")
        ax.axvline(high,color="green",ls="--")
        place_label(ax,[],crit,stats.norm.pdf(crit,mu,sigma)+.02,
                    f"T₍crit₎={crit}","green")
        reject=t_val<=crit or t_val>=high
    ax.axvline(t_val,color="blue",ls="--")
    place_label(ax,[],t_val,stats.norm.pdf(t_val,mu,sigma)+.02,
                f"T₍calc₎={t_val}","blue")
    ax.set_title(f"Wilcoxon T – "
                 f"{'Reject' if reject else 'Fail to Reject'} H₀")
    ax.legend(); fig.tight_layout(); return fig

def w_table(n,a,tail,key):
    step=st.session_state.setdefault(key+"_step",-1)
    rows=list(range(max(5,n-5),n+6))
    alphas=[.10,.05,.01,.001]
    header="".join(f"<th>{al}</th>" for al in alphas)
    body=""
    for r in rows:
        body+=f'<tr><td id="w_{r}_0">{r}</td>'
        for idx,al in enumerate(alphas,1):
            body+=f'<td id="w_{r}_{idx}">{w_crit(r,al,tail)}</td>'
        body+="</tr>"
    css=("table{border-collapse:collapse}"
         "td,th{border:1px solid#000;width:80px;height:30px;"
         "text-align:center;font-size:.8rem}")
    html=wrap(css,f"<tr><th>N＼α</th>{header}</tr>{body}")

    col_idx=alphas.index(a)+1
    if step>=0:
        for i in range(len(alphas)+1): html=style(html,f"w_{n}_{i}")
    if step>=1:
        for r in rows: html=style(html,f"w_{r}_{col_idx}")
    if step>=2:
        html=style(html,f"w_{n}_{col_idx}","blue",3)

    show_html(html)
    msgs=["Highlight N row","Highlight α column","Intersection → T₍crit₎"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_step(key+"_step")

# ───────────────────────── 7 • Binomial ───────────────────────── #

def binom_tab():
    st.subheader("Tab 7 • Binomial")
    c1,c2=st.columns(2)
    with c1:
        n=st.number_input("n (trials)", min_value=1, value=20,
                          step=1, key="b_n")
        p=st.number_input("π (null proportion)", value=0.50, step=0.01,
                          min_value=0.01, max_value=0.99, key="b_p")
    with c2:
        k=st.number_input("k (successes)", min_value=0, value=12,
                          step=1, key="b_k")
        alpha=st.number_input("α (two‑tailed)", value=0.05, step=0.01,
                              min_value=0.0001,max_value=0.5, key="b_a")
    if st.button("Update Plot", key="b_plot"):
        st.pyplot(plot_binom(k,n,p))
    with st.expander("Quick table (k ±5)"):
        binom_table(k,n,p,"b")

def plot_binom(k,n,p):
    x=np.arange(0,n+1); y=stats.binom.pmf(x,n,p)
    fig,ax=plt.subplots(figsize=(12,4),dpi=100)
    ax.bar(x,y,color="lightgrey",label="P(X=k)")
    ax.bar(k,stats.binom.pmf(k,n,p),color="blue",label="k observed")
    ax.set_xlabel("k"); ax.set_ylabel("P(X=k)")
    ax.set_title(f"Binomial (n={n}, p={p}) – k={k}")
    ax.legend(); fig.tight_layout(); return fig

def binom_table(k,n,p,key):
    step=st.session_state.setdefault(key+"_step",-1)
    ks=list(range(max(0,k-5),min(n,k+5)+1))
    header="<th>P(X=k)</th><th>P(X≤k)</th><th>P(X≥k)</th>"
    rows=""
    for k_ in ks:
        pmf=stats.binom.pmf(k_,n,p)
        cdf=stats.binom.cdf(k_,n,p)
        sf =1-stats.binom.cdf(k_-1,n,p)
        rows+=(f'<tr><td id="b_{k_}_0">{k_}</td>'
               f'<td id="b_{k_}_1">{pmf:.4f}</td>'
               f'<td id="b_{k_}_2">{cdf:.4f}</td>'
               f'<td id="b_{k_}_3">{sf:.4f}</td></tr>')
    css=("table{border-collapse:collapse}"
         "td,th{border:1px solid#000;width:120px;height:30px;"
         "text-align:center;font-size:.8rem}")
    html=wrap(css,f"<tr><th>k</th>{header}</tr>{rows}")

    if step>=0:
        for i in range(4): html=style(html,f"b_{k}_{i}")
    if step>=1:
        html=style(html,f"b_{k}_1","blue",3)

    show_html(html)
    msgs=["Highlight k row","Highlight P(X=k)"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_step(key+"_step")

# ───────────────────────────── MAIN ───────────────────────────── #

def main():
    st.set_page_config("PSYC250 – Statistical Tables Explorer", layout="wide")
    st.title("PSYC250 – Statistical Tables Explorer (12 × 4 figures)")
    tabs=st.tabs(["t‑Dist","z‑Dist","F‑Dist","Chi‑Square",
                  "Mann–Whitney U","Wilcoxon T","Binomial"])
    with tabs[0]: t_tab()
    with tabs[1]: z_tab()
    with tabs[2]: f_tab()
    with tabs[3]: chi_tab()
    with tabs[4]: u_tab()
    with tabs[5]: w_tab()
    with tabs[6]: binom_tab()

if __name__ == "__main__":
    main()
