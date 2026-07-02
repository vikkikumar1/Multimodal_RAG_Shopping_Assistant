import streamlit as st

def display_product_card(product):
    st.image(product.get('image_url', 'https://via.placeholder.com/150'), width=150)
    st.write(f"**{product.get('title')}**")
    st.write(f"Price: ${product.get('price')}")