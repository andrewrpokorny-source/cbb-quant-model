"""MLB ballpark run factors (2024-2025 averages from Baseball Reference)."""

# Park factor > 1.0 = hitter-friendly, < 1.0 = pitcher-friendly.
# Source: Baseball Reference 2024-2025 multi-year park factors.
BALLPARK_FACTORS = {
    "American Family Field": 1.05,
    "Angel Stadium": 0.97,
    "Busch Stadium": 0.97,
    "Chase Field": 1.06,
    "Citi Field": 0.93,
    "Citizens Bank Park": 1.07,
    "Comerica Park": 0.98,
    "Coors Field": 1.18,
    "Daikin Park": 0.96,
    "Dodger Stadium": 0.97,
    "Fenway Park": 1.08,
    "Globe Life Field": 1.02,
    "Great American Ball Park": 1.10,
    "Kauffman Stadium": 0.97,
    "Nationals Park": 1.01,
    "Oracle Park": 0.92,
    "Oriole Park at Camden Yards": 1.01,
    "PNC Park": 0.92,
    "Petco Park": 0.93,
    "Progressive Field": 0.97,
    "Rate Field": 1.02,
    "Rogers Centre": 1.04,
    "Sutter Health Park": 1.05,
    "T-Mobile Park": 0.92,
    "Target Field": 1.01,
    "Truist Park": 1.00,
    "Wrigley Field": 1.06,
    "Yankee Stadium": 1.07,
    "loanDepot park": 0.91,
}

# GPS coordinates for weather lookups (lat, lon)
STADIUM_COORDINATES = {
    "American Family Field": (43.028, -87.971),
    "Angel Stadium": (33.800, -117.883),
    "Busch Stadium": (38.623, -90.193),
    "Chase Field": (33.445, -112.067),
    "Citi Field": (40.757, -73.846),
    "Citizens Bank Park": (39.906, -75.167),
    "Comerica Park": (42.339, -83.049),
    "Coors Field": (39.756, -104.994),
    "Daikin Park": (29.757, -95.355),
    "Dodger Stadium": (34.074, -118.240),
    "Fenway Park": (42.346, -71.097),
    "Globe Life Field": (32.747, -97.084),
    "Great American Ball Park": (39.097, -84.508),
    "Kauffman Stadium": (39.052, -94.481),
    "Nationals Park": (38.873, -77.007),
    "Oracle Park": (37.778, -122.389),
    "Oriole Park at Camden Yards": (39.284, -76.622),
    "PNC Park": (40.447, -80.006),
    "Petco Park": (32.707, -117.157),
    "Progressive Field": (41.496, -81.685),
    "Rate Field": (41.830, -87.634),
    "Rogers Centre": (43.641, -79.389),
    "Sutter Health Park": (38.580, -121.510),
    "T-Mobile Park": (47.591, -122.332),
    "Target Field": (44.982, -93.278),
    "Truist Park": (33.891, -84.468),
    "Wrigley Field": (41.948, -87.656),
    "Yankee Stadium": (40.829, -73.926),
    "loanDepot park": (25.778, -80.220),
}

# Indoor/retractable-roof stadiums (weather defaults to neutral)
INDOOR_STADIUMS = {
    "American Family Field",
    "Chase Field",
    "Daikin Park",
    "Globe Life Field",
    "Rogers Centre",
    "T-Mobile Park",
    "loanDepot park",
}


def get_park_factor(venue_name):
    """Return park run factor for a venue, defaulting to 1.0 for unknowns."""
    return BALLPARK_FACTORS.get(venue_name, 1.0)


def is_outdoor_stadium(venue_name):
    """Return True if the stadium is outdoor (weather-affected)."""
    return venue_name not in INDOOR_STADIUMS
