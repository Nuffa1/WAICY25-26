import streamlit as st
import database as db
import os
import time
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="My Collection", page_icon="📚", layout="wide")
db.init_db()


# --- HELPER: IMAGE TO BASE64 ---
def get_img_as_base64(file_path):
    """Reads an image file and converts it to a base64 string for HTML rendering."""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# --- CSS: GALLERY STYLING ---
st.markdown("""
<style>
    /* CARD CONTAINER */
    /* This styles the Streamlit column that acts as the card */
    div[data-testid="column"] > div > div > div > div {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    /* CUSTOM IMAGE BOX CLASS */
    /* We will use this in the HTML string below */
    .gallery-image-box {
        width: 100%;
        height: 150px;          /* STRICT FIXED HEIGHT */
        border-radius: 5px;
        display: flex;
        align-items: center;    /* Vertically Center */
        justify-content: center;/* Horizontally Center */
        overflow: hidden;
        margin-bottom: 10px;
    }

    .gallery-image-box img {
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;    /* Ensure image never stretches */
    }
</style>
""", unsafe_allow_html=True)

# --- AUTH CHECK ---
if 'user_id' not in st.session_state or st.session_state['user_id'] is None:
    st.warning("Please login to view your gallery.")
    if st.button("Go to Login"):
        st.switch_page("pages/Login.py")
    st.stop()

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>My Sketchbook️</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Your collection of generated Devanagari calligraphy.</p>",
            unsafe_allow_html=True)
st.divider()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Gallery Options")
    search_query = st.text_input("Search text...", placeholder="Filter by word...")
    sort_order = st.radio("Sort by:", ["Newest First", "Oldest First"])

    st.divider()
    if st.button("← Back to Generator"):
        st.switch_page("Home.py")

# --- DATA FETCHING ---
images = db.get_user_images(st.session_state['user_id'])

# --- FILTERING LOGIC ---
if search_query:
    images = [img for img in images if search_query.lower() in img[1].lower()]

if sort_order == "Oldest First":
    images.reverse()

# --- GALLERY RENDER ---
if not images:
    st.info("You haven't saved any sentences yet. Go to the Home page to start creating!")
else:
    cols = st.columns(3)

    for idx, (img_id, text_content, filename, timestamp) in enumerate(images):
        col_index = idx % 3

        with cols[col_index]:
            with st.container():
                # 1. Render Image via HTML (Base64)
                img_path = os.path.join("saved_images", filename)

                if os.path.exists(img_path):
                    # Convert to base64 to embed in HTML
                    b64_img = get_img_as_base64(img_path)

                    # Render strict HTML container
                    # This guarantees the height is exactly 150px
                    html_code = f"""
                    <div class="gallery-image-box">
                        <img src="data:image/png;base64,{b64_img}">
                    </div>
                    """
                    st.markdown(html_code, unsafe_allow_html=True)
                else:
                    st.markdown('<div class="gallery-image-box" style="color:#e74c3c">Image Missing</div>',
                                unsafe_allow_html=True)

                # 2. Display Info
                st.markdown(f"**{text_content}**")
                st.caption(f"{timestamp[:10]}")

                # 3. Action Buttons
                b_col1, b_col2 = st.columns([1, 1])

                with b_col1:
                    if os.path.exists(img_path):
                        # We still need to open the file for the download button logic
                        with open(img_path, "rb") as file:
                            st.download_button(
                                label="Download",
                                data=file,
                                file_name=f"devanagari_{text_content[:10]}.png",
                                mime="image/png",
                                key=f"dl_{img_id}"
                            )

                with b_col2:
                    if st.button("Delete", key=f"del_{img_id}", type="secondary"):
                        if db.delete_image(img_id):
                            st.toast("Image deleted.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to delete.")

            st.write("")  # Spacer