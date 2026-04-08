def validate_input(data):
    if data["Age"] < 18 or data["Age"] > 100:
        return False, "Invalid Age"
    if data["Total Spend"] < 0:
        return False, "Invalid Spend"
    return True, "Valid"