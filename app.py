import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="Spotify Music Analytics",
    layout="wide"
)

st.title("Spotify Music Analytics")
st.write("Explore Spotify songs and predict track popularity.")

uploaded_file = st.file_uploader(
    "Upload Spotify CSV file",
    type="csv"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.drop_duplicates().copy()

    st.subheader("Data Preview")
    st.write(f"Number of tracks: {len(df):,}")
    st.dataframe(df.head(10), hide_index=True)

    if "popularity" not in df.columns:
        st.error("The CSV must contain a popularity column.")
        st.stop()

    st.subheader("Popularity Distribution")
    fig = px.histogram(
        df,
        x="popularity",
        nbins=20,
        title="Distribution of Spotify Track Popularity",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Audio Features Explorer")

    default_features = [
        "danceability", "energy", "valence", "tempo", "loudness"
    ]
    available_features = [
        col for col in default_features if col in df.columns
    ]

    if available_features:
        x_col = st.selectbox(
            "Choose an audio feature",
            available_features
        )

        plot_data = df.sample(
            n=min(5000, len(df)), random_state=42
        )

        fig2 = px.scatter(
            plot_data,
            x=x_col,
            y="popularity",
            color="energy" if "energy" in df.columns else None,
            opacity=0.6,
            title=f"{x_col} vs. Popularity",
            template="plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Popularity Prediction")

    feature_cols = ["danceability", "energy"]
    target_col = "popularity"

    model_data = df[feature_cols + [target_col]].dropna()

    X = model_data[feature_cols]
    y = model_data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    col1, col2 = st.columns(2)
    col1.metric("Test R²", f"{r2_score(y_test, y_pred):.3f}")
    col2.metric("Test MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")

    st.write("Enter two audio features to estimate track popularity.")

    input1 = st.number_input(
        "Danceability (0-1)",
        min_value=0.0,
        max_value=1.0,
        value=0.57
    )

    input2 = st.number_input(
        "Energy (0-1)",
        min_value=0.0,
        max_value=1.0,
        value=0.64
    )

    if st.button("Predict Popularity"):
        input_df = pd.DataFrame({
            "danceability": [input1],
            "energy": [input2]
        })

        prediction = model.predict(input_df)[0]
        prediction = max(0, min(100, prediction))

        st.metric("Predicted Popularity", f"{prediction:.1f} / 100")

    st.caption(
        "This model predicts a track's popularity score (0-100) using "
        "danceability and energy. Predictions may be inaccurate, so treat "
        "them as estimates rather than actual popularity scores."
    )

else:
    st.info("Upload your Spotify CSV file to get started.")
