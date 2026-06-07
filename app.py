import streamlit as st
from page.single_stock_pred import single_stock_page
from page.portfolio_pred import portfolio_pred_page

st.set_page_config(
    page_title="Stock & Portfolio Prediction",
    layout="wide",
    page_icon="logo.png"
)
# Streamlit App Title
st.markdown("<h1 style='text-align: center;'>Portfolio Analyzer and Stock Prediction Dashboard</h1>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(" ", ["Single Stock Prediction", "Portfolio Analysis"],
                        captions=[
                                "Predict behaviour for one stock.",
                                "Add your portfolio and predict behaviour for entire portfolio.",
                            ])

if page == "Single Stock Prediction":
    single_stock_page()
elif page == "Portfolio Analysis":
    portfolio_pred_page()
