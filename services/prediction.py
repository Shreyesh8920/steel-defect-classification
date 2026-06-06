import joblib
import pandas as pd

artifacts = joblib.load("models/steel_defect_artifacts.pkl")

model = artifacts["model"]
steel_encoder = artifacts["steel_encoder"]
defect_encoder = artifacts["defect_encoder"]
feature_order = artifacts["feature_order"]


def predict_defect(data):

    steel_encoded = steel_encoder.transform(
        [data["Steel_Type"]]
    )[0]

    input_df = pd.DataFrame([{
        "Pixels_Areas": data["Pixels_Areas"],
        "X_Perimeter": data["X_Perimeter"],
        "Y_Perimeter": data["Y_Perimeter"],
        "Steel_Plate_Thickness": data["Steel_Plate_Thickness"],
        "Length_of_Conveyer": data["Length_of_Conveyer"],
        "Maximum_of_Luminosity": data["Maximum_of_Luminosity"],
        "Minimum_of_Luminosity": data["Minimum_of_Luminosity"],
        "Sum_of_Luminosity": data["Sum_of_Luminosity"],
        "X_Maximum": data["X_Maximum"],
        "Y_Maximum": data["Y_Maximum"],
        "Steel_Type": steel_encoded
    }])

    input_df = input_df[feature_order]

    prediction = model.predict(input_df)[0]

    defect_name = defect_encoder.inverse_transform(
        [prediction]
    )[0]

    return defect_name