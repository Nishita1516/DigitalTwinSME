import streamlit as st
import os

def load_css():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, "ui.css")

    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def page_title():
    st.markdown('<div class="main-title">Digital Twin Dashboard</div>', unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_header(title):
    st.markdown(
        f'<div class="section-header">{title}</div>',
        unsafe_allow_html=True
    )
