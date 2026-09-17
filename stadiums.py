# World Cup 2026 stadium data
# altitude in metres, avg June/July temp in °C

STADIUMS = {
    'MetLife Stadium':        {'city': 'New York',      'country': 'United States', 'altitude': 7,    'avg_temp': 28, 'humidity': 'high'},
    'AT&T Stadium':           {'city': 'Dallas',        'country': 'United States', 'altitude': 143,  'avg_temp': 35, 'humidity': 'high'},
    'Arrowhead Stadium':      {'city': 'Kansas City',   'country': 'United States', 'altitude': 266,  'avg_temp': 32, 'humidity': 'high'},
    'NRG Stadium':            {'city': 'Houston',       'country': 'United States', 'altitude': 15,   'avg_temp': 34, 'humidity': 'very high'},
    'Mercedes-Benz Stadium':  {'city': 'Atlanta',       'country': 'United States', 'altitude': 320,  'avg_temp': 31, 'humidity': 'high'},
    'Hard Rock Stadium':      {'city': 'Miami',         'country': 'United States', 'altitude': 3,    'avg_temp': 32, 'humidity': 'very high'},
    'Lincoln Financial Field':{'city': 'Philadelphia',  'country': 'United States', 'altitude': 12,   'avg_temp': 30, 'humidity': 'moderate'},
    'Lumen Field':            {'city': 'Seattle',       'country': 'United States', 'altitude': 5,    'avg_temp': 22, 'humidity': 'low'},
    'Levi\'s Stadium':        {'city': 'San Francisco', 'country': 'United States', 'altitude': 3,    'avg_temp': 24, 'humidity': 'low'},
    'SoFi Stadium':           {'city': 'Los Angeles',   'country': 'United States', 'altitude': 30,   'avg_temp': 26, 'humidity': 'low'},
    'Gillette Stadium':       {'city': 'Boston',        'country': 'United States', 'altitude': 89,   'avg_temp': 27, 'humidity': 'moderate'},
    'BMO Field':              {'city': 'Toronto',       'country': 'Canada',        'altitude': 76,   'avg_temp': 25, 'humidity': 'moderate'},
    'BC Place':               {'city': 'Vancouver',     'country': 'Canada',        'altitude': 2,    'avg_temp': 21, 'humidity': 'low'},
    'Estadio Azteca':         {'city': 'Mexico City',   'country': 'Mexico',        'altitude': 2240, 'avg_temp': 22, 'humidity': 'low'},
    'Estadio Akron':          {'city': 'Guadalajara',   'country': 'Mexico',        'altitude': 1566, 'avg_temp': 26, 'humidity': 'moderate'},
    'Estadio BBVA':           {'city': 'Monterrey',     'country': 'Mexico',        'altitude': 540,  'avg_temp': 33, 'humidity': 'moderate'},
}

# Teams accustomed to high altitude (home leagues/stadiums above 1000m)
HIGH_ALTITUDE_TEAMS = ['Mexico', 'Colombia', 'Ecuador']

# Teams from hot climates (comfortable in 30°C+ conditions)
HOT_CLIMATE_TEAMS = ['Egypt', 'Saudi Arabia', 'Qatar', 'Iraq', 'Jordan',
                     'Algeria', 'Tunisia', 'Morocco', 'Senegal', 'Ivory Coast',
                     'Ghana', 'DR Congo', 'Cape Verde', 'Haiti', 'Paraguay',
                     'Brazil', 'Colombia']

# Host nations
HOST_NATIONS = ['United States', 'Canada', 'Mexico']

def get_condition_adjustment(team, opponent, stadium_name):
    """
    Returns an Elo adjustment (in Elo points) for a team based on
    stadium conditions. Positive = advantage, negative = disadvantage.

    Based on published research:
    - Home advantage in international football: ~50-100 Elo points
    - High altitude advantage for acclimatised teams: ~30-50 points
    - Heat advantage for hot climate teams: ~20-30 points
    """
    stadium = STADIUMS.get(stadium_name)
    if stadium is None:
        return 0

    adjustment = 0

    # --- Host nation advantage ---
    if team in HOST_NATIONS and stadium['country'] == team_to_country(team):
        adjustment += 80   # playing in their own country

    # --- Altitude effects ---
    if stadium['altitude'] > 1500:
        if team in HIGH_ALTITUDE_TEAMS:
            adjustment += 40   # acclimatised team at altitude
        else:
            adjustment -= 25   # sea-level team struggling at altitude

    # --- Heat effects ---
    if stadium['avg_temp'] >= 32:
        if team in HOT_CLIMATE_TEAMS:
            adjustment += 25   # comfortable in the heat
        else:
            adjustment -= 15   # European teams suffer in heat

    return adjustment

def team_to_country(team):
    """Map team name to country for host check"""
    return team  # team names match country names in our data

def adjusted_win_prob(team_a, team_b, stadium_name, elo_dict):
    """Win probability with stadium condition adjustments applied"""
    elo_a = elo_dict.get(team_a, 1500)
    elo_b = elo_dict.get(team_b, 1500)

    # Apply condition adjustments
    elo_a += get_condition_adjustment(team_a, team_b, stadium_name)
    elo_b += get_condition_adjustment(team_b, team_a, stadium_name)

    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))