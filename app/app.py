import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import psycopg2  # For PostgreSQL database connection
from datetime import datetime  # To log current timestamp

# DB Connection function
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "digit_logs"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "Roro")
    )


# Page title
st.title("PyTorch MNIST Digit Recognizer")
st.write('Draw a digit (0-9) below and click Predict.')

# Load model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device('cpu')))
model.eval()

# Split into two columns
canvas_col, result_col = st.columns([1, 1])

with canvas_col:
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    # Prediction button 
    predict_button = st.button("Predict", key="predict_button")

# Initialize session state to keep prediction
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None

with result_col:
    if predict_button and canvas_result.image_data is not None:
        # Prepare the image
        image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
        image = image.resize((28, 28))
        image = ImageOps.invert(image)

        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        input_tensor = transform(image).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, 1).item()
            conf = torch.softmax(output, dim=1)[0][pred].item()

        st.session_state.prediction = pred
        st.session_state.confidence = conf

    # Display prediction if exists
    if st.session_state.prediction is not None:
        # Display result
        arrow = "↑" if st.session_state.confidence > 0.7 else "↓"  # Up arrow if confidence is high

        st.markdown(
            f"""
            <div style='line-height:1.2; font-size:16px; margin-bottom:20px'>
                <strong>Prediction</strong><br>
                <span style='font-size:30px'>{st.session_state.prediction}</span><br>
                <span style='font-size:22px; color: green'>{arrow} {st.session_state.confidence * 100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("feedback_form"):
            true_label = st.number_input("Enter True Label:", min_value=0, max_value=9, step=1, key="true_label_input")
            submit_feedback = st.form_submit_button("Submit Feedback")
        
            if submit_feedback:                    
                # Log prediction to PostgreSQL database
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO "Prediction_History" ("Timestamp", "Pred", "True", "Conf")
                        VALUES (%s, %s, %s, %s)
                    """, (datetime.now(), st.session_state.prediction, true_label, st.session_state.confidence))
                    conn.commit()
                    cur.close()
                    conn.close()
                    st.success("✔️ Logged to database.")
                    
                except Exception as e:
                    st.error(f"Database Error: {e}")

# Display prediction history table
st.markdown("### Prediction History")
try:
    conn = get_db_connection()
    df = pd.read_sql('SELECT * FROM "Prediction_History" ORDER BY "Timestamp" DESC', conn)
    conn.close()

    # Modify the format of the Conf column as a percentage like 20%
    if 'Conf' in df.columns:
        df['Conf'] = df['Conf'].apply(lambda x: f"{x * 100:.1f}%")

    df_display = df.head(10) # Show only the first 10 rows
    st.table(df_display) # st.dataframe(df_display, use_container_width=True)    

except Exception as e:
    st.error(f"Error fetching data: {e}")
