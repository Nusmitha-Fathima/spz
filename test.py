from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests

app = Flask(__name__)
linear_model = joblib.load('Future_Earnings_model.joblib')

def get_data_from_api():
    api_url = "http://192.168.1.18:9000/admin_app/weekly_income/"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['end_date'] = pd.to_datetime(df['end_date'])
        df.sort_values(by=['turf_id', 'end_date'], inplace=True)
        df['income_LastWeek'] = df.groupby('turf_id')['income'].shift(1)
        df['income_2Weeksback'] = df.groupby('turf_id')['income'].shift(2)
        df['income_3Weeksback'] = df.groupby('turf_id')['income'].shift(3)
        df.fillna(950, inplace=True)
        result_df = df[['turf_id', 'end_date', 'income_LastWeek', 'income_2Weeksback', 'income_3Weeksback', 'income']]
        return result_df
    else:
        return None

@app.route('/income', methods=['GET'])
def income():
    turf_id = 2  # Replace with the desired turf_id

    # Fetch data from the API and process into result_df
    result_df = get_data_from_api()

    if result_df is not None:
        # Filter data for the specified turf_id
        turf_data = result_df[result_df['turf_id'] == turf_id]

        if not turf_data.empty:
            # Get the most recent row for the specified turf_id
            recent_row = turf_data.iloc[-1]

            # Prepare the input features for prediction
            input_features = [
                recent_row['income_LastWeek'],
                recent_row['income_2Weeksback'],
                recent_row['income_3Weeksback']
            ]

            # Make a prediction using the linear model
            predicted_income = linear_model.predict([input_features])[0]

            # Round the predicted income to the nearest integer
            rounded_predicted_income = round(predicted_income)

            # You can return the rounded prediction as JSON or in any format you prefer
            return jsonify({"current_week_income": rounded_predicted_income})
        else:
            return jsonify({"error": "No data found for the specified turf_id"})
    else:
        return jsonify({"error": "Error fetching data from the API"})

if __name__ == '__main__':
    app.run(debug=True)
