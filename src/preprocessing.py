def create_features(df):
    df["Spend_per_Tenure"] = df["Total Spend"] / (df["Tenure"] + 1)
    df["Usage_Intensity"] = df["Usage Frequency"] / (df["Tenure"] + 1)
    return df