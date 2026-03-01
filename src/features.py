def select_features(df):

    # Criar target binário
    threshold = df["Total Profit"].median()
    df["High_Profit"] = (df["Total Profit"] >= threshold).astype(int)

    target = "High_Profit"

    # Converter Order_Ship_Days para inteiro
    df["Order_Ship_Days"] = (
        df["Order_Ship_Days"]
        .str.replace(" days", "", regex=False)
        .astype(int)
    )

    numeric_features = [
        "Order_Ship_Days"
    ]

    categorical_features = [
        "Region",
        "Country",
        "Item Type",
        "Sales Channel",
        "Order Priority"
    ]

    X = df[numeric_features + categorical_features]
    y = df[target]

    return X, y, numeric_features, categorical_features