from flask import Flask, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
import json



app = Flask(__name__)
model = joblib.load('fives_model.joblib')


def get_probdata_from_api():
    api_url = "https://c2e5-116-68-110-250.ngrok-free.app/admin_app/match_rating_view/"
    response = requests.get(api_url)

    if response.status_code == 200:
        try:
            data = response.json()
            messages = data['message']
            df = pd.DataFrame(messages)
            flattened_players = []

            for entry in messages:
                players = entry['players']
                player_details = {}
                for i, player in enumerate(players, start=1):
                    player_details[f"player{i}_id"] = player[f'player{i}_id']
                    player_details[f"player{i}_skill"] = player[f'player{i}_skill']
                flattened_players.append(player_details)

            df_players = pd.DataFrame(flattened_players)

            df_combined = pd.concat([pd.DataFrame(messages), df_players], axis=1)
            df_combined.drop('players', axis=1, inplace=True)
            # print(df_combined)

            player_levels = {}
            for index, row in df_combined.iterrows():
                for i in range(1, 6):
                    player_id = row[f'player{i}_id']
                    player_level = row[f'player{i}_skill']
                    player_levels[player_id] = player_level

            unique_players = set().union(*[set(df_combined[f'player{i}_id']) for i in range(1, 6)])

            player_counts = {player: 0 for player in unique_players}
            player_win_counts = {player: 0 for player in unique_players}

            # Iterate through the data to calculate counts and win counts
            for i in range(len(df_combined)):
                row = df_combined.iloc[i]
                for player in unique_players:
                    if player in row.values:
                        player_counts[player] += 1
                        if player in row.values and row['result'].lower() == 'win':
                            player_win_counts[player] += 1

            # Create DataFrames for player counts and win counts
            player_counts_df = pd.DataFrame(list(player_counts.items()), columns=['Player', 'Count'])
            player_win_counts_df = pd.DataFrame(list(player_win_counts.items()), columns=['Player', 'Win_Count'])

            # Merge DataFrames and calculate win ratio
            player_df = pd.merge(player_counts_df, player_win_counts_df, on='Player', how='outer')
            player_df['win_ratio'] = (player_df['Win_Count'] / player_df['Count']).round(2)

            # Extract player levels and add to the DataFrame
            player_df['Player_Level'] = [player_levels.get(player, 0) for player in player_df['Player']]
            # print(player_df)

            Team_id = set(df_combined['team_id'])

            team_counts = {team: 0 for team in Team_id}
            team_win_counts = {team: 0 for team in Team_id}

            for i in range(len(df_combined)):
                row = df_combined.iloc[i]
                team = row['team_id']
                team_counts[team] += 1

                if row['result'].lower() == 'win':
                    team_win_counts[team] += 1

            team_counts_df = pd.DataFrame(list(team_counts.items()), columns=['Team_ID', 'Count'])
            team_win_counts_df = pd.DataFrame(list(team_win_counts.items()), columns=['Team_ID', 'win_Count'])
            teams_df = pd.merge(team_counts_df, team_win_counts_df, on='Team_ID', how='left')
            teams_df['win_Count'].fillna(0, inplace=True)
            teams_df['win_ratio'] = (teams_df['win_Count'] / teams_df['Count']).round(2)
            # print(teams_df)

            return df_combined, player_df, teams_df
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    else:
        print(f"API request failed with status code: {response.status_code}")
    player_df.to_csv('player_df.csv', index=False)

    return None, None, None





@app.route('/probabilityfives', methods=['POST', 'GET'])
def probabilityfives():
    df_combined, player_df, teams_df = get_probdata_from_api()

    if df_combined is None or player_df is None or teams_df is None:
        return jsonify({"error": "Failed to retrieve data from the API"}), 500

    dic = [{'team1': '4', 'players': [1, 9, 3, 4,5]},
           {'team2': '5', 'players': [11, 12, 13, 14,15]}]

    list_ = []

    for team_info in dic:
        # Extract the team_id from the dictionary
        team_id_key = 'team1' if 'team1' in team_info else 'team2'
        team_id_value = team_info[team_id_key]

        try:
            team_id = int(team_id_value)
        except ValueError:
            print(f"Invalid team_id value: {team_id_value}. It should be an integer.")
            continue  # Skip this iteration if team_id is not an integer

        matching_rows = teams_df[teams_df['Team_ID'] == team_id]

        print(f"Team ID: {team_id}, Matching Rows:\n{matching_rows}")

        if not matching_rows.empty:
            team_win_ratio = matching_rows['win_ratio'].values[0]
            print(team_win_ratio)
            list_.append(team_win_ratio)
        else:
            print(f"No matching rows found for Team ID: {team_id}")
            list_.append(0)  # You can adjust this default value as needed

        for player in team_info['players']:
            try:
                player_str = str(player)
            except ValueError:
                print(f"Invalid player ID value: {player}. It should be convertible to a string.")
                continue  # Skip this iteration if player ID is not convertible to a string

            matching_players = player_df[player_df['Player'] == player_str]

            print(f"Player: {player_str}, Matching Players:\n{matching_players}")

            if not matching_players.empty:
                player_level = matching_players['Player_Level'].values[0]
                if player_level.lower() == 'beginner':
                    list_.append(0)
                elif player_level.lower() == 'intermediate':
                    list_.append(1)
                else:
                    list_.append(2)

                player_win_ratio = matching_players['win_ratio'].values[0]
                list_.append(player_win_ratio)

            else:
                print(f"No matching players found for Player: {player_str}")
                list_.append(0)
                list_.append(0)  # Assuming win ratio should be 0 when no matching players are found

    try:
        input_data = np.array(list_).reshape(1, -1)
        probability = model.predict_proba(input_data)
        team_a_probability = round(probability[0, 1] * 100, 2)
        team_b_probability = round(probability[0, 2] * 100, 2)
        draw_probability = round(probability[0, 0] * 100, 2)

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Set default probabilities in case of an exception
        team_a_probability = 50.0
        team_b_probability = 50.0
        draw_probability = 0.0

    response = {
        "probability": {
            "Team 1": f"{team_a_probability:.2f}%",
            "Team 2": f"{team_b_probability:.2f}%",
            "Draw": f"{draw_probability:.2f}%"
        }
    }
    return jsonify(response)






@app.route('/playerdata', methods=['GET'])
def player_data_api():
    df_combined, player_df, teams_df = get_probdata_from_api()

    if player_df is None:
        return jsonify({"error": "Failed to retrieve player data"}), 500

    try:
        # Convert the DataFrame to a JSON response
        player_data_json = player_df.to_json(orient='records')

        return player_data_json
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500




########################################################################################################



linear_model = joblib.load('Future_Earnings_model.joblib')

def get_data_from_api():
    api_url = "https://c2e5-116-68-110-250.ngrok-free.app/admin_app/weekly_income"
    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            messages = data['message']
            df = pd.DataFrame(messages)
            df['end_date'] = pd.to_datetime(df['end_date'])

            df.sort_values(by=['turf__id', 'end_date'], inplace=True)

            df['income_LastWeek'] = df.groupby('turf__id')['total_income'].shift(1)
            df['income_2Weeksback'] = df.groupby('turf__id')['total_income'].shift(2)
            df['income_3Weeksback'] = df.groupby('turf__id')['total_income'].shift(3)

            df.fillna(950, inplace=True)

            result_df = df[['turf__id', 'end_date', 'income_LastWeek', 'income_2Weeksback', 'income_3Weeksback', 'total_income']]

            return result_df
        else:
            print(f"API request failed with status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")

    # Return default values in case of an exception
    return None

@app.route('/income', methods=['GET', 'POST'])
def income():
    result_df = get_data_from_api()

    if result_df is not None:
        predictions = []

        for turf__id in result_df['turf__id'].unique():
            turf_data = result_df[result_df['turf__id'] == turf__id]

            if not turf_data.empty:
                recent_row = turf_data.iloc[-1]

                try:
                    input_features = [
                        recent_row['income_LastWeek'],
                        recent_row['income_2Weeksback'],
                        recent_row['income_3Weeksback']
                    ]

                    predicted_income = linear_model.predict([input_features])[0]
                    rounded_predicted_income = round(predicted_income)

                except Exception as e:
                    print("There was an error during income prediction.")
                    # Set a default value for prediction
                    rounded_predicted_income = 2000  # Set your desired default value

                predictions.append({"turf__id": turf__id, "predicted_income": rounded_predicted_income})
            else:
                predictions.append({"turf__id": turf__id, "error": "No data found for the specified turf_id"})

        predictions_df = pd.DataFrame(predictions)

        return predictions_df.to_json(orient='records')
    else:
        print("Error fetching data from the API")
        return jsonify({"error": "Error fetching data from the API"})








###########################################################################################################


pipeline = joblib.load('Future_Bookings_model.joblib')

def get_booking_data_from_api():
    api_url = "https://c2e5-116-68-110-250.ngrok-free.app/admin_app/weekly_income"
    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            messages = data['message']
            df = pd.DataFrame(messages)
            df['end_date'] = pd.to_datetime(df['end_date'])

            df.sort_values(by=['turf__id', 'end_date'], inplace=True)

            df['booking_LastWeek'] = df.groupby('turf__id')['total_booking'].shift(1)
            df['booking_2Weeksback'] = df.groupby('turf__id')['total_booking'].shift(2)
            df['booking_3Weeksback'] = df.groupby('turf__id')['total_booking'].shift(3)

            df.fillna(2, inplace=True)

            book_df = df[['turf__id', 'end_date', 'booking_LastWeek', 'booking_2Weeksback', 'booking_3Weeksback', 'total_booking']]

            return book_df
        else:
            print(f"API request failed with status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")

    # Return default values in case of an exception
    return None

@app.route('/booking', methods=['GET', 'POST'])
def booking():
    book_df = get_booking_data_from_api()

    if book_df is not None:
        predictions = []

        for turf__id in book_df['turf__id'].unique():
            turf_data = book_df[book_df['turf__id'] == turf__id]

            if not turf_data.empty:
                recent_row = turf_data.iloc[-1]

                try:
                    input_features = [
                        recent_row['booking_LastWeek'],
                        recent_row['booking_2Weeksback'],
                        recent_row['booking_3Weeksback']
                    ]

                    predicted_booking = pipeline.predict([input_features])[0]
                    rounded_predicted_booking = round(predicted_booking)

                except Exception as e:
                    print("There was an error occured so except value is predicting.")
                    # Set a default value for prediction
                    rounded_predicted_booking = 50  # Set your desired default value

                predictions.append({"turf_id": turf__id, "predicted_bookings": rounded_predicted_booking})
            else:
                predictions.append({"turf_id": turf__id, "error": "No data found for the specified turf__id"})

        predictions_df = pd.DataFrame(predictions)

        return predictions_df.to_json(orient='records')
    else:
        print("Error fetching data from the API")
        return jsonify({"error": "Error fetching data from the API"})



if __name__ == '__main__':
    app.run(debug=True)
