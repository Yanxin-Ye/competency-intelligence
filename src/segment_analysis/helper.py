def get_generation(year):
    if year >= 2013:
        return "Gen Alpha"
    elif year >= 1997:
        return "Gen Z"
    elif year >= 1981:
        return "Millennial"
    elif year >= 1965:
        return "Gen X"
    elif year >= 1946:
        return "Boomer"
    else:
        return "Unknown"
