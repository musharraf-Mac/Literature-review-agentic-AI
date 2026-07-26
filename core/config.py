import os

def get_secret(key_name):
    try:
        import streamlit as st
        return st.secrets[key_name]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(key_name)

GROQ_API = get_secret("GROQ_API")
OPEN_ROUTER_API = get_secret("OPEN_ROUTER_API")