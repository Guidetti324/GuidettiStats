import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")

###############################################################################
#                            HELPER FUNCTIONS
###############################################################################

def place_label(ax, lbl_pos, x, y, text, colour='blue'):
    dx, dy = 0.0, 0.02
    for (xx, yy) in lbl_pos:
        if abs(x-xx) < .15 and abs(y-yy) < .05:
            dx += .06; dy += .04
    ax.text(x+dx, y+dy, text, colour, ha='left', va='bottom', fontsize=8)
    lbl_pos.append((x+dx, y+dy))

def highlight(html, cell_id, colour="red", px=2):
    return html.replace(f'id="{cell_id}"',
                        f'id="{cell_id}" style="border:{px}px solid {colour};"', 1)

def html_table(body, h=450):
    components.html(f"<html><body>{body}</body></html>", height=h, scrolling=True)

def next_step(key):
    if st.button("Next Step", key=key+"_btn"):
        st.session_state[key] += 1

###############################################################################
#                               T‑DISTRIBUTION
###############################################################################

def tab_t():
    st.subheader("Tab 1 · t‑Distribution")
    c1, c2 = st.columns(2)
    with c1:
        t = st.number_input("t statistic", 2.87, key="t_val")
        df = st.number_input("df", 55, min_value=1, key="t_df")
    with c2:
        α = st.number_input("α", .05, .01, key="t_alpha")
        tail = st.radio("Tail type", ["one‑tailed", "two‑tailed"], key="t_tail")

    if st.button("Update Plot", key="t_plot"):
        st.pyplot(plot_t(t, df, α, tail))

    with st.expander("Show t‑Table Lookup (±5 df)"):
        lookup_t(df, α, tail, "t")

def plot_t(t, df, α, tail):
    fig,
