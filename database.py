import sqlite3
import hashlib
import os
import shutil
from datetime import datetime

DB_NAME = "user_data.db"
IMG_FOLDER = "saved_images"

if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)


def init_db():
    """Initialize the database with users and saved_sentences tables."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE, 
                  password TEXT)''')
    # Create Saved Sentences Table
    c.execute('''CREATE TABLE IF NOT EXISTS saved_sentences 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  user_id INTEGER, 
                  text_content TEXT, 
                  filename TEXT, 
                  timestamp DATETIME,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def create_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username exists
    finally:
        conn.close()


def verify_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed = hash_password(password)
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, hashed))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None


def save_generated_image(user_id, text, temp_image_path):
    """Moves the temp image to permanent storage and logs in DB."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{user_id}_{timestamp}.png"
    destination = os.path.join(IMG_FOLDER, filename)

    # Copy the file to the saved_images folder
    shutil.copy(temp_image_path, destination)

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO saved_sentences (user_id, text_content, filename, timestamp) VALUES (?, ?, ?, ?)",
              (user_id, text, filename, datetime.now()))
    conn.commit()
    conn.close()
    return True

def get_user_images(user_id):
    """Fetch all saved images for a specific user, ordered by newest first."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "SELECT id, text_content, filename, timestamp FROM saved_sentences WHERE user_id=? ORDER BY timestamp DESC",
        (user_id,))
    data = c.fetchall()
    conn.close()
    return data


def delete_image(image_id):
    """Deletes a record from the DB and removes the file from storage."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # 1. Get filename to delete from disk
    c.execute("SELECT filename FROM saved_sentences WHERE id=?", (image_id,))
    result = c.fetchone()

    if result:
        filename = result[0]
        file_path = os.path.join(IMG_FOLDER, filename)

        # Delete from Disk
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file: {e}")

        # Delete from DB
        c.execute("DELETE FROM saved_sentences WHERE id=?", (image_id,))
        conn.commit()
        conn.close()
        return True

    conn.close()
    return False