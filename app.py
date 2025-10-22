import gradio as gr
import joblib
import pandas as pd
import numpy as np

# --- 1. Load All Saved Artifacts ---
# These .pkl files MUST be in the same directory as this app.py
try:
    model = joblib.load('forest_cover_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pt = joblib.load('power_transformer.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    print("Error: Model or preprocessor files not found.")
    print("Please run the last cell of your 'Forest Cover.ipynb' to save the .pkl files.")
    # You might want to exit or raise an error here in a real app
    exit()


# --- 2. Define Column Lists (from the notebook) ---

# (From Cell 8)
continuous_cols = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]

# (From Cell 10's output) - Columns that were transformed
skewed_cols = [
    'Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Horizontal_Distance_To_Fire_Points'
]

# (Implicit from notebook) - All the one-hot encoded columns
wilderness_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]
soil_cols = [f'Soil_Type{i}' for i in range(1, 41)]

# This is the final, full list of 54 features in the correct order
all_feature_cols = continuous_cols + wilderness_cols + soil_cols

# --- 3. Define Human-Readable Labels ---
# These are the original class names for the target variable
label_map = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# --- 4. The Prediction Function ---
def predict_cover_type(
    # 10 continuous features
    elevation, aspect, slope, 
    h_dist_hydro, v_dist_hydro, h_dist_road,
    hillshade_9, hillshade_noon, hillshade_3,
    h_dist_fire,
    
    # 2 categorical features that we will one-hot encode
    wilderness_area_num, 
    soil_type_num
):
    """
    Takes 12 user inputs, preprocesses them, and returns a prediction.
    """
    
    # --- a. Create the input dictionary ---
    input_data = {
        'Elevation': elevation,
        'Aspect': aspect,
        'Slope': slope,
        'Horizontal_Distance_To_Hydrology': h_dist_hydro,
        'Vertical_Distance_To_Hydrology': v_dist_hydro,
        'Horizontal_Distance_To_Roadways': h_dist_road,
        'Hillshade_9am': hillshade_9,
        'Hillshade_Noon': hillshade_noon,
        'Hillshade_3pm': hillshade_3,
        'Horizontal_Distance_To_Fire_Points': h_dist_fire
    }
    
    # --- b. One-Hot Encode Categorical Inputs ---
    # Initialize all 44 binary columns to 0
    for col in wilderness_cols + soil_cols:
        input_data[col] = 0
        
    # Set the one selected wilderness area to 1
    if 1 <= wilderness_area_num <= 4:
        input_data[f'Wilderness_Area{wilderness_area_num}'] = 1
        
    # Set the one selected soil type to 1
    if 1 <= soil_type_num <= 40:
        input_data[f'Soil_Type{soil_type_num}'] = 1

    # --- c. Create DataFrame in the correct order ---
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    # Ensure columns are in the *exact* order the model was trained on
    input_df = input_df[all_feature_cols]

    # --- d. Apply Preprocessing Steps (from notebook) ---
    
    # Step 1: Apply PowerTransformer (from Cell 10)
    # We use .transform() here, NOT .fit_transform()
    input_df[skewed_cols] = pt.transform(input_df[skewed_cols])
    
    # NOTE: Skipping Outlier Clipping (from Cell 12)
    # Your notebook applied outlier clipping in a way that is not
    # reproducible in a live app (the quantile bounds were not saved).
    # This app will work, but predictions might slightly differ
    # from your notebook's results because this step is omitted.
    # The StandardScaler will help mitigate this.
    
    # Step 2: Apply StandardScaler (from Cell 16)
    # We use .transform() here, NOT .fit_transform()
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])

    # --- e. Make Prediction ---
    try:
        # Predict the ENCODED label (e.g., 0-6)
        prediction_encoded = model.predict(input_df)
        
        # --- f. Decode Prediction ---
        # Inverse transform to get the ORIGINAL label (e.g., 1-7)
        prediction_original = le.inverse_transform(prediction_encoded)
        
        # Get the class number (e.g., 5)
        class_number = prediction_original[0]
        
        # Get the human-readable name (e.g., "Aspen")
        class_name = label_map.get(class_number, "Unknown")
        
        return f"{class_name} (Class {class_number})"
        
    except Exception as e:
        return f"Error during prediction: {e}"

# --- 5. Define Gradio Inputs ---
# We use the first row of data from your notebook (Cell 1) as default values
inputs = [
    # Continuous Features
    gr.Number(label="Elevation", value=2596),
    gr.Number(label="Aspect (0-360 degrees)", value=51),
    gr.Number(label="Slope (0-90 degrees)", value=3),
    gr.Number(label="Horizontal Distance to Hydrology", value=258),
    gr.Number(label="Vertical Distance to Hydrology", value=0),
    gr.Number(label="Horizontal Distance to Roadways", value=510),
    gr.Number(label="Hillshade (9am, 0-255)", value=221),
    gr.Number(label="Hillshade (Noon, 0-255)", value=232),
    gr.Number(label="Hillshade (3pm, 0-255)", value=148),
    gr.Number(label="Horizontal Distance to Fire Points", value=6279),
    
    # Categorical Features
    gr.Dropdown(
        label="Wilderness Area", 
        choices=[1, 2, 3, 4], 
        value=1
    ),
    gr.Dropdown(
        label="Soil Type", 
        choices=list(range(1, 41)), 
        value=29
    )
]

# Define the output component
outputs = gr.Label(label="Predicted Cover Type")

# --- 6. Create and Launch the Interface ---
title = "Forest Cover Type Prediction"
description = (
    "Predict the forest cover type based on cartographic data. "
    "This app uses the RandomForest model trained in the 'Forest Cover.ipynb' notebook."
)

# This is the first row of data from your notebook, which is Cover_Type 5 (Aspen)
example_row = [2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279, 1, 29]
# --- NEW: Define the article text ---
# This will be displayed at the bottom of the app
forest_types_article = """
<div style='margin-top: 20px;'>
    <h3>Types of Forests (Classes)</h3>
    <p>The model will predict one of the following 7 cover types:</p>
    <ul>
        <li><b>Class 1:</b> Spruce/Fir</li>
        <li><b>Class 2:</b> Lodgepole Pine</li>
        <li><b>Class 3:</b> Ponderosa Pine</li>
        <li><b>Class 4:</b> Cottonwood/Willow</li>
        <li><b>Class 5:</b> Aspen</li>
        <li><b>Class 6:</b> Douglas-fir</li>
        <li><b>Class 7:</b> Krummholz</li>
    </ul>
</div>
"""
iface = gr.Interface(
    fn=predict_cover_type,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    examples=[example_row],
    article=forest_types_article,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
