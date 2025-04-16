###############################################################################
# StatsMain.py - Single-file Streamlit app for PSYC250, with NO DUPLICATE KEYS
###############################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")

###############################################################################
# UTILITY: place_label
###############################################################################

def place_label(ax, label_positions, x, y, text, color='blue'):
    offset_x = 0.0
    offset_y = 0.02
    for (xx, yy) in label_positions:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            offset_x += 0.06
            offset_y += 0.04
    final_x = x + offset_x
    final_y = y + offset_y
    ax.text(final_x, final_y, text, color=color, ha="left", va="bottom", fontsize=8)
    label_positions.append((final_x, final_y))

###############################################################################
# 1) T-DISTRIBUTION TAB
###############################################################################

def show_t_distribution_tab():
    st.subheader("Tab 1: t-Distribution")

    col1, col2 = st.columns(2)
    with col1:
        t_val = st.number_input(
            "Enter t statistic (Tab1)",
            value=2.87,
            key="tab1_tstat"
        )
        df_val = st.number_input(
            "df (Tab1)",
            value=55,
            key="tab1_df"
        )
    with col2:
        alpha_val = st.number_input(
            "Alpha for t-dist (Tab1)",
            value=0.05,
            step=0.01,
            key="tab1_alpha"
        )
        tail_type = st.radio(
            "Tail Type (Tab1 t-dist)",
            ["one-tailed", "two-tailed"],
            key="tab1_tail"
        )

    if st.button("Update Plot (Tab1)", key="tab1_update"):
        fig = plot_t_distribution(t_val, df_val, alpha_val, tail_type)
        st.pyplot(fig)

    with st.expander("Show t-Table Lookup (±5) (Tab1)"):
        st.write("Pretend we highlight ±5 around df etc.")

def plot_t_distribution(t_val, df, alpha, tail_s):
    fig, ax = plt.subplots(figsize=(5,3))
    label_positions=[]
    x = np.linspace(-4,4,400)
    y = stats.t.pdf(x, df)
    ax.plot(x, y, color='black')
    ax.fill_between(x, y, color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    def labelme(xx, yy, txt, c='green'):
        place_label(ax, label_positions, xx, yy, txt, c)

    if tail_s=="one-tailed":
        t_crit= stats.t.ppf(1 - alpha, df)
        note=""
        if abs(alpha-0.05)<1e-9:
            note="(equiv two-tailed α=0.10)"
        region= x[x>= t_crit]
        ax.fill_between(region, y[x>= t_crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit, color='green', linestyle='--')
        labelme(t_crit, stats.t.pdf(t_crit, df)+0.02, f"t_crit={t_crit:.2f} {note}", 'green')
        sig= (t_val> t_crit)
        final_crit= t_crit
    else:
        t_crit_r= stats.t.ppf(1- alpha/2, df)
        t_crit_l= stats.t.ppf(alpha/2, df)
        ax.fill_between(x[x>= t_crit_r], y[x>= t_crit_r], color='red', alpha=0.3)
        ax.fill_between(x[x<= t_crit_l], y[x<= t_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit_r, color='green', linestyle='--')
        ax.axvline(t_crit_l, color='green', linestyle='--')
        labelme(t_crit_r, stats.t.pdf(t_crit_r, df)+0.02, f"+t_crit={t_crit_r:.2f}", 'green')
        labelme(t_crit_l, stats.t.pdf(t_crit_l, df)+0.02, f"-t_crit={t_crit_l:.2f}", 'green')
        sig= (abs(t_val)> abs(t_crit_r))
        final_crit= abs(t_crit_r)

    ax.axvline(t_val, color='blue', linestyle='--')
    place_label(ax, label_positions, t_val, stats.t.pdf(t_val, df)+0.02,
                f"t_calc={t_val:.2f}", 'blue')

    if sig:
        msg= f"t={t_val:.2f} > {final_crit:.2f} => Reject H₀"
    else:
        msg= f"t={t_val:.2f} ≤ {final_crit:.2f} => Fail to Reject H₀"

    ax.set_title(f"t-Distribution (df={df})\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig

###############################################################################
# 2) Z-DISTRIBUTION TAB
###############################################################################

def show_z_distribution_tab():
    st.subheader("Tab 2: z-Distribution")

    col1, col2 = st.columns(2)
    with col1:
        z_val= st.number_input("z statistic (Tab2)", value=1.64, key="tab2_zstat")
    with col2:
        alpha_val= st.number_input("Alpha for z-dist (Tab2)", value=0.05, step=0.01, key="tab2_alpha")
        tail_type= st.radio("Tail Type (Tab2 z-dist)", ["one-tailed", "two-tailed"], key="tab2_tail")

    if st.button("Update Plot (Tab2)", key="tab2_update"):
        fig= plot_z_distribution(z_val, alpha_val, tail_type)
        st.pyplot(fig)

    with st.expander("Show z-Table Lookup ±10 (Tab2)"):
        st.write("Pretend partial z-table ±10...")

def plot_z_distribution(z_val, alpha, tail_s):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2,label="Fail to Reject H₀")

    def labelme(xx,yy,txt,c='blue'):
        place_label(ax, label_positions, xx, yy, txt, c)

    if tail_s=="one-tailed":
        z_crit= stats.norm.ppf(1- alpha)
        ax.fill_between(x[x>=z_crit], y[x>=z_crit], color='red', alpha=0.3,label="Reject H₀")
        ax.axvline(z_crit, color='green', linestyle='--')
        labelme(z_crit, stats.norm.pdf(z_crit), f"z_crit={z_crit:.2f}", 'green')
        sig= (z_val> z_crit)
        final_crit= z_crit
    else:
        z_crit_r= stats.norm.ppf(1- alpha/2)
        z_crit_l= -z_crit_r
        ax.fill_between(x[x>= z_crit_r], y[x>= z_crit_r], color='red', alpha=0.3)
        ax.fill_between(x[x<= z_crit_l], y[x<= z_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(z_crit_r, color='green', linestyle='--')
        ax.axvline(z_crit_l, color='green', linestyle='--')
        labelme(z_crit_r, stats.norm.pdf(z_crit_r), f"+z_crit={z_crit_r:.2f}", 'green')
        labelme(z_crit_l, stats.norm.pdf(z_crit_l), f"-z_crit={z_crit_r:.2f}", 'green')
        sig= (abs(z_val)> z_crit_r)
        final_crit= z_crit_r

    ax.axvline(z_val, color='blue', linestyle='--')
    labelme(z_val, stats.norm.pdf(z_val), f"z_calc={z_val:.2f}", 'blue')

    msg= (f"z={z_val:.2f} > {final_crit:.2f} => Reject H₀"
          if sig else f"z={z_val:.2f} ≤ {final_crit:.2f} => Fail to Reject H₀")

    ax.set_title(f"Z-Distribution\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig

###############################################################################
# 3) F-DISTRIBUTION TAB
###############################################################################

def show_f_distribution_tab():
    st.subheader("Tab 3: F-Distribution (One-tailed)")

    c1, c2 = st.columns(2)
    with c1:
        f_val= st.number_input("F statistic (Tab3)", value=3.49, key="tab3_fval")
        df1= st.number_input("df1 (Tab3 F-dist)", value=3, key="tab3_df1")
        df2= st.number_input("df2 (Tab3 F-dist)", value=12, key="tab3_df2")
    with c2:
        alpha= st.number_input("Alpha (Tab3 F-dist)", value=0.05, step=0.01, key="tab3_alpha")

    if st.button("Update Plot (Tab3)", key="tab3_update"):
        fig= plot_f_distribution(f_val, df1, df2, alpha)
        st.pyplot(fig)

    with st.expander("Show F Table Lookup ±5 (Tab3)"):
        st.write("Pretend highlight row df1, col df2, etc.")

def plot_f_distribution(f_val, df1, df2, alpha):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    x= np.linspace(0,5,500)
    y= stats.f.pdf(x, df1, df2)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey',alpha=0.2,label="Fail to Reject H₀")

    f_crit= stats.f.ppf(1-alpha, df1, df2)
    ax.fill_between(x[x>= f_crit], y[x>= f_crit], color='red', alpha=0.3, label="Reject H₀")
    ax.axvline(f_crit, color='green', linestyle='--')
    place_label(ax, label_positions, f_crit, stats.f.pdf(f_crit,df1,df2)+0.02,
                f"F_crit={f_crit:.2f}", 'green')

    ax.axvline(f_val, color='blue', linestyle='--')
    place_label(ax, label_positions, f_val, stats.f.pdf(f_val,df1,df2)+0.02,
                f"F_calc={f_val:.2f}", 'blue')

    sig= (f_val> f_crit)
    msg= (f"F={f_val:.2f} > {f_crit:.2f} => Reject H₀"
          if sig else
          f"F={f_val:.2f} ≤ {f_crit:.2f} => Fail to Reject H₀")

    ax.set_title(f"F-Distribution (df1={df1}, df2={df2})\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig

###############################################################################
# 4) CHI-SQUARE TAB
###############################################################################

def show_chi_square_tab():
    st.subheader("Tab 4: Chi-Square (One-tailed)")

    c1, c2 = st.columns(2)
    with c1:
        chi_val= st.number_input("Chi-square stat (Tab4)", value=10.5, key="tab4_chi")
        df_val= st.number_input("df (Tab4 Chi-dist)", value=12, key="tab4_chi_df")
    with c2:
        alpha= st.number_input("Alpha (Tab4 Chi-dist)", value=0.05, step=0.01, key="tab4_chi_alpha")

    if st.button("Update Plot (Tab4)", key="tab4_update"):
        fig= plot_chi_square(chi_val, df_val, alpha)
        st.pyplot(fig)

    with st.expander("Show Chi-Square Table Lookup (Tab4)"):
        st.write("Pretend ±5 around df highlight")

def plot_chi_square(chi_val, df, alpha):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    x_max= max(30, df*2)
    x= np.linspace(0,x_max,400)
    y= stats.chi2.pdf(x, df)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey',alpha=0.2,label="Fail to Reject H₀")

    chi_crit= stats.chi2.ppf(1-alpha, df)
    ax.fill_between(x[x>= chi_crit], y[x>= chi_crit], color='red', alpha=0.3, label="Reject H₀")
    ax.axvline(chi_crit, color='green', linestyle='--')
    place_label(ax, label_positions, chi_crit, stats.chi2.pdf(chi_crit,df)+0.02,
                f"chi2_crit={chi_crit:.2f}", 'green')

    ax.axvline(chi_val, color='blue', linestyle='--')
    place_label(ax, label_positions, chi_val, stats.chi2.pdf(chi_val,df)+0.02,
                f"chi2_calc={chi_val:.2f}", 'blue')

    sig= (chi_val> chi_crit)
    msg= (f"χ²={chi_val:.2f} > {chi_crit:.2f} => Reject H₀"
          if sig else
          f"χ²={chi_val:.2f} ≤ {chi_crit:.2f} => Fail to Reject H₀")

    ax.set_title(f"Chi-Square (df={df})\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig

###############################################################################
# 5) MANN–WHITNEY U TAB
###############################################################################

def show_mann_whitney_tab():
    st.subheader("Tab 5: Mann–Whitney U")

    c1, c2 = st.columns(2)
    with c1:
        U_val= st.number_input("U statistic (Tab5)", value=5, key="tab5_U")
        n1= st.number_input("n1 (Tab5 MW)", value=5, key="tab5_n1")
        n2= st.number_input("n2 (Tab5 MW)", value=6, key="tab5_n2")
    with c2:
        alpha= st.number_input("Alpha (Tab5 MW)", value=0.05, step=0.01, key="tab5_alpha")
        tail_s= st.radio("Tail Type (Tab5 MW)", ["one-tailed","two-tailed"], key="tab5_tail")

    if st.button("Update Plot (Tab5)", key="tab5_update"):
        fig= plot_mann_whitney(U_val,n1,n2,alpha,tail_s)
        st.pyplot(fig)

    with st.expander("Show Mann–Whitney Table Lookup (Tab5)"):
        st.write("±5 approach demonstration")

def plot_mann_whitney(U_val,n1,n2,alpha,tail_s):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    meanU= n1*n2/2
    sdU= np.sqrt(n1*n2*(n1+n2+1)/12)
    z_val= (U_val- meanU)/sdU

    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey',alpha=0.2,label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        zCrit= stats.norm.ppf(1- alpha)
        ax.fill_between(x[x>=zCrit], y[x>=zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit), f"zCrit={zCrit:.2f}", 'green')
        sig= (z_val> zCrit)
    else:
        zCrit= stats.norm.ppf(1- alpha/2)
        ax.fill_between(x[x>=zCrit], y[x>=zCrit], color='red', alpha=0.3)
        ax.fill_between(x[x<=-zCrit], y[x<=-zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        ax.axvline(-zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit), f"zCritR={zCrit:.2f}", 'green')
        place_label(ax, label_positions, -zCrit, stats.norm.pdf(-zCrit), f"zCritL={zCrit:.2f}", 'green')
        sig= (abs(z_val)> zCrit)

    ax.axvline(z_val, color='blue', linestyle='--')
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val), f"z_calc={z_val:.2f}", 'blue')

    msg= (f"U={U_val}, z={z_val:.2f} => Reject H₀" if sig else
          f"U={U_val}, z={z_val:.2f} => Fail to Reject H₀")

    ax.set_title(f"Mann–Whitney: n1={n1}, n2={n2}\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig

###############################################################################
# 6) WILCOXON SIGNED-RANK TAB
###############################################################################

def show_wilcoxon_tab():
    st.subheader("Tab 6: Wilcoxon Signed-Rank")

    c1, c2 = st.columns(2)
    with c1:
        T_val= st.number_input("T statistic (Tab6 Wilcoxon)", value=5, key="tab6_T")
        N_val= st.number_input("N (non-zero diffs) (Tab6)", value=6, key="tab6_N")
    with c2:
        alpha= st.number_input("Alpha (Tab6 Wilcoxon)", value=0.05, step=0.01, key="tab6_alpha")
        tail_s= st.radio("Tail Type (Tab6 Wilcoxon)", ["one-tailed","two-tailed"], key="tab6_tail")

    if st.button("Update Plot (Tab6)", key="tab6_update"):
        fig= plot_wilcoxon(T_val, N_val, alpha, tail_s)
        st.pyplot(fig)

    with st.expander("Show Wilcoxon Table Lookup (Tab6)"):
        st.write("Minimal ±5 approach")

def plot_wilcoxon(T_val, N, alpha, tail_s):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    meanT= N*(N+1)/4
    sdT= np.sqrt(N*(N+1)*(2*N+1)/24)
    z_val= (T_val- meanT)/ sdT

    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2,label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        zCrit= stats.norm.ppf(1- alpha)
        ax.fill_between(x[x>=zCrit], y[x>=zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit), f"zCrit={zCrit:.2f}", 'green')
        sig= (z_val> zCrit)
    else:
        zCrit= stats.norm.ppf(1- alpha/2)
        zCR= zCrit
        zCL= -zCrit
        ax.fill_between(x[x>= zCR], y[x>= zCR], color='red', alpha=0.3)
        ax.fill_between(x[x<= zCL], y[x<= zCL], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCR, color='green', linestyle='--')
        ax.axvline(zCL, color='green', linestyle='--')
        place_label(ax, label_positions, zCR, stats.norm.pdf(zCR), f"+zCrit={zCR:.2f}", 'green')
        place_label(ax, label_positions, zCL, stats.norm.pdf(zCL), f"-zCrit={zCR:.2f}", 'green')
        sig= (abs(z_val)> zCrit)

    ax.axvline(z_val, color='blue', linestyle='--')
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val), f"z_calc={z_val:.2f}", 'blue')

    msg= (f"T={T_val}, z={z_val:.2f} => Reject H₀" if sig
          else f"T={T_val}, z={z_val:.2f} => Fail to Reject H₀")

    ax.set_title(f"Wilcoxon (N={N})\n{msg}")
    ax.legend()
    fig.tight_layout()
    return fig

###############################################################################
# 7) BINOMIAL TAB
###############################################################################

def show_binomial_tab():
    st.subheader("Tab 7: Binomial")

    c1, c2 = st.columns(2)
    with c1:
        n= st.number_input("n (Tab7 Binomial)", value=10, key="tab7_n")
        x= st.number_input("x (successes) (Tab7)", value=3, key="tab7_x")
    with c2:
        p= st.number_input("p (Tab7 Binomial)", value=0.5, step=0.01, key="tab7_p")
        alpha= st.number_input("Alpha (Tab7 Binomial)", value=0.05, step=0.01, key="tab7_alpha")
        tail_s= st.radio("Tail Type (Tab7 Binomial)", ["one-tailed","two-tailed"], key="tab7_tail")

    if st.button("Update Plot (Tab7)", key="tab7_update"):
        fig= plot_binomial(n,x,p,alpha,tail_s)
        st.pyplot(fig)

    with st.expander("Show Binomial Table Lookup (Tab7 ±5)"):
        st.write("Minimal demonstration of row => col => intersection")

def plot_binomial(n,x,p,alpha,tail_s):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    k_vals= np.arange(n+1)
    pmf_vals= stats.binom.pmf(k_vals,n,p)
    bars= ax.bar(k_vals, pmf_vals, color='lightgrey', edgecolor='black')

    mean_= n*p
    if tail_s=="one-tailed":
        if x<= mean_:
            p_val= stats.binom.cdf(x,n,p)
        else:
            p_val= 1- stats.binom.cdf(x-1,n,p)
    else:
        if x<= mean_:
            p_val= stats.binom.cdf(x,n,p)*2
        else:
            p_val= (1- stats.binom.cdf(x-1,n,p))*2
        p_val= min(1.0, p_val)

    sig= (p_val< alpha)
    msg= f"p-value={p_val:.4f} => " + ("Reject H₀" if sig else "Fail to Reject H₀")

    # entire region shading
    if tail_s=="one-tailed":
        if x< mean_:
            for i in range(0, x+1):
                bars[i].set_color('red')
        else:
            for i in range(x, n+1):
                bars[i].set_color('red')
    else:
        if x<= mean_:
            for i in range(0, x+1):
                bars[i].set_color('red')
            hi_start= max(0, n-x)
            for i in range(hi_start, n+1):
                bars[i].set_color('red')
        else:
            for i in range(x, n+1):
                bars[i].set_color('red')
            for i in range(0, (n-x)+1):
                bars[i].set_color('red')

    if 0<= x<= n:
        bars[x].set_color('blue')

    import matplotlib.patches as mpatches
    legend_patches= [
        mpatches.Patch(facecolor='lightgrey', edgecolor='black', label='Fail to Reject H₀'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='Reject H₀ region'),
        mpatches.Patch(facecolor='blue', edgecolor='black', label='Observed x')
    ]
    ax.legend(handles=legend_patches, loc='best')

    ax.set_title(f"Binomial(n={n}, p={p:.2f})\n{msg}")
    ax.set_xlabel("x (successes)")
    ax.set_ylabel("PMF")
    fig.tight_layout()
    return fig

###############################################################################
# MAIN
###############################################################################

def main():
    st.set_page_config(page_title="PSYC250 - Streamlit Stats Explorer (No More Duplicates)", layout="wide")
    st.title("PSYC250 - Statistical Tables Explorer (No Duplicate IDs)")

    tab_labels = [
        "t-Distribution",
        "z-Distribution",
        "F-Distribution",
        "Chi-Square",
        "Mann–Whitney U",
        "Wilcoxon Signed-Rank",
        "Binomial"
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        show_t_distribution_tab()
    with tabs[1]:
        show_z_distribution_tab()
    with tabs[2]:
        show_f_distribution_tab()
    with tabs[3]:
        show_chi_square_tab()
    with tabs[4]:
        show_mann_whitney_tab()
    with tabs[5]:
        show_wilcoxon_tab()
    with tabs[6]:
        show_binomial_tab()

if __name__=="__main__":
    main()
