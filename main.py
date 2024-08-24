from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import googlemaps
from shapely import wkt
from shapely.geometry import Polygon

df = pd.read_csv('taxi_zone_geo.csv')

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load the model and preprocessor
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

# Initialize the Google Maps client
gmaps = googlemaps.Client(key='AIzaSyAA5cM-Rm3-BHKv3K6358sqnhn9w-c6C5Y')

class Item(BaseModel):
    vendorid: int
    passenger_count: int
    pulocationid: int
    dolocationid: int
    payment_type: int
    day: int

@app.post("/receive_data")
async def receive_data(item: Item):

    def calculate_distance_miles(wkt_string1, wkt_string2):
        try:
            # Convert WKT strings to Shapely Polygon objects
            polygon1 = wkt.loads(wkt_string1)
            polygon2 = wkt.loads(wkt_string2)

            # Compute centroids for both polygons
            centroid1 = polygon1.centroid
            centroid2 = polygon2.centroid

            # Get the driving distance between the centroids
            result = gmaps.distance_matrix(
                (centroid1.y, centroid1.x),
                (centroid2.y, centroid2.x),
                mode="driving"  # Use "driving" for taxi-like routing
            )

            # Extract distance in meters
            distance_meters = result['rows'][0]['elements'][0]['distance']['value']

            # Convert distance to kilometers and then to miles
            distance_kilometers = distance_meters / 1000
            distance_miles = distance_kilometers * 0.621371

            return distance_miles, distance_kilometers

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calculating distance: {e}")

    def calculate_duration_seconds(wkt_string1, wkt_string2):
        try:
            # Convert WKT strings to Shapely Polygon objects
            polygon1 = wkt.loads(wkt_string1)
            polygon2 = wkt.loads(wkt_string2)

            # Compute centroids for both polygons
            centroid1 = polygon1.centroid
            centroid2 = polygon2.centroid

            # Get the duration between the centroids
            result = gmaps.directions(
                (centroid1.y, centroid1.x),
                (centroid2.y, centroid2.x),
                mode="driving"  # Change to "walking", "bicycling", or "transit" if needed
            )

            # Extract duration in seconds
            duration_seconds = result[0]['legs'][0]['duration']['value']

            return duration_seconds, result[0]['overview_polyline']['points']

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calculating duration: {e}")

    try:
        # Get WKT strings from DataFrame
        wkt_string1 = df.loc[df['zone_id'] == item.pulocationid, 'zone_geom'].values[0]
        wkt_string2 = df.loc[df['zone_id'] == item.dolocationid, 'zone_geom'].values[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="PU or DO Location ID not found")

    # Calculate distance and duration
    distance_miles, distance_kilometers = calculate_distance_miles(wkt_string1, wkt_string2)
    duration_seconds, route_polyline = calculate_duration_seconds(wkt_string1, wkt_string2)

    # Prepare the data for prediction
    new_data = pd.DataFrame([[
        item.vendorid,
        item.passenger_count,
        distance_miles,
        item.pulocationid,
        item.dolocationid,
        item.payment_type,
        item.day,
        duration_seconds
    ]], columns=[
        'vendorid', 'passenger_count', 'trip_distance', 'pulocationid',
        'dolocationid', 'payment_type', 'day', 'duration_seconds'
    ], dtype='object')

    # Transform the data using the preprocessor
    transformed_data = preprocessor.transform(new_data)

    # Predict using the model
    prediction = model.predict(transformed_data)

    # Convert prediction to a Python native type
    prediction_value = prediction[0] if isinstance(prediction[0], (int, float)) else float(prediction[0])

    # Convert duration from seconds to minutes
    duration_minutes = duration_seconds / 60

    # Generate the Google Maps URL for the route
    route_url = f"https://www.google.com/maps/dir/?api=1&origin={wkt.loads(wkt_string1).centroid.y},{wkt.loads(wkt_string1).centroid.x}&destination={wkt.loads(wkt_string2).centroid.y},{wkt.loads(wkt_string2).centroid.x}&travelmode=driving"

    # Return the prediction along with distance in km, duration in min, and route UR
    return {
        "prediction": prediction_value,
        "distance_kilometers": distance_kilometers,
        "duration_minutes": duration_minutes,
        "route_url": route_url
    }