import streamlit as st
import database as db
import time

st.set_page_config(page_title="Login - Devanagari Scribe", page_icon="🔐", layout="centered")
db.init_db()

# --- CSS BLOCK  ---


# --- STATE MANAGEMENT ---
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None

# --- LOGIN PAGE UI ---
st.markdown("<h2 style='text-align: center;'>Access Your Account</h2>", unsafe_allow_html=True)
st.divider()

# Using columns to center the login form content
login_col, center_col, spacer_col = st.columns([1, 2, 1])

with center_col:
    action = st.radio("Choose Action", ["Login", "Sign Up"], horizontal=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if action == "Login":
        if st.button("Login", use_container_width=True, type="primary"):
            if username and password:
                user_id = db.verify_user(username, password)
                if user_id:
                    st.session_state['user_id'] = user_id
                    st.session_state['username'] = username
                    st.success("Logged in successfully! Redirecting...")
                    time.sleep(1)
                    st.switch_page("Home.py")
                else:
                    st.error("Invalid username or password.")
            else:
                st.warning("Please enter your username and password.")

    else:  # Sign Up
        if st.button("Create Account", use_container_width=True, type="primary"):
            if username and password:
                if db.create_user(username, password):
                    st.success("Account created! Please switch to Login.")
                else:
                    st.error("Username already exists.")
            else:
                st.warning("Please fill in all fields.")

    st.divider()
    if st.button("← Back to Home"):
        st.switch_page("Home.py")