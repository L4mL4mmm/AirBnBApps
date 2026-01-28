from flask import Flask, request, render_template
from src.Airbnb.pipelines.Prediction_Pipeline import CustomData, PredictPipeline
from src.Airbnb.database import Database
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
db = Database()

try:
    listings_path = os.path.join("Artifacts", "New_Airbnb_Data.csv")
    if os.path.exists(listings_path):
        df_listings = pd.read_csv(listings_path)
        df_listings['price'] = pd.to_numeric(df_listings['price'], errors='coerce')
    else:
        df_listings = None
        print("Warning: New_Airbnb_Data.csv not found.")
except Exception as e:
    df_listings = None
    print(f"Error loading listings data: {e}")

def get_similar_listings(city, room_type):
    if df_listings is None:
        return []
    
    try:
        subset = df_listings[
            (df_listings['city'] == city) & 
            (df_listings['room_type'] == room_type)
        ]
        
        if subset.empty:
            return []
        
        n = min(5, len(subset))
        samples = subset.sample(n)
        
        results = []
        for _, row in samples.iterrows():
            name = row.get('name', 'N/A')
            if pd.isna(name): name = "Apartment"
            
            neigh = row.get('neighbourhood', 'N/A')
            if pd.isna(neigh) or neigh == "Neighborhood highlights":
                neigh = f"{city} Area"
                
            results.append({
                'name': name,
                'neighbourhood': neigh,
                'price': row.get('price', 'N/A'),
                'rating': row.get('review_scores_rating', 'N/A')
            })
            
        return results
    except Exception as e:
        print(f"Error getting similar listings: {e}")
        return []

@app.route("/", methods=["GET", "POST"])
def home():
    history = db.get_history()
    similar_listings = []
    
    if request.method == "POST":
        try:
            def map_tf(val):
                return 't' if val == '1' else 'f'
            
            def map_bool_str(val):
                return 'True' if val == '1' else 'False'

            data = CustomData(
                property_type=request.form.get("property_type"),
                room_type=request.form.get("room_type"),
                amenities=int(request.form.get("amenities")),
                accommodates=int(request.form.get("accommodates")),
                bathrooms=float(request.form.get("bathrooms")),
                bed_type=request.form.get("bed_type"),
                cancellation_policy=request.form.get("cancellation_policy"),
                cleaning_fee=map_bool_str(request.form.get("cleaning_fee")),
                city=request.form.get("city"),
                host_has_profile_pic=map_tf(request.form.get("host_has_profile_pic")),
                host_identity_verified=map_tf(request.form.get("host_identity_verified")),
                host_response_rate=int(request.form.get("host_response_rate")),
                instant_bookable=map_tf(request.form.get("instant_bookable")),
                latitude=float(request.form.get("latitude")),
                longitude=float(request.form.get("longitude")),
                number_of_reviews=int(request.form.get("number_of_reviews")),
                review_scores_rating=int(request.form.get("review_scores_rating")),
                bedrooms=int(request.form.get("bedrooms")),
                beds=int(request.form.get("beds"))
            )

            final_data = data.get_data_as_dataframe()

            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_data)
            
            result = round(np.exp(pred[0]), 2)
            
            similar_listings = get_similar_listings(
                request.form.get("city"), 
                request.form.get("room_type")
            )
            
            db.insert_prediction(
                city=request.form.get("city"), 
                property_type=request.form.get("property_type"), 
                room_type=request.form.get("room_type"), 
                accommodates=int(request.form.get("accommodates")), 
                price=result
            )
            
            history = db.get_history()
            
            return render_template("index.html", 
                                   final_result=result, 
                                   history=history, 
                                   form_data=request.form,
                                   similar_listings=similar_listings)
            
        except Exception as e:
            error_message = f"Error during prediction: {str(e)}"
            return render_template("index.html", 
                                   error_message=error_message, 
                                   history=history, 
                                   form_data=request.form)

    else:
        return render_template("index.html", history=history, form_data={})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
