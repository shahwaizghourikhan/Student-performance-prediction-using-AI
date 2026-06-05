import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)



try:
    df = pd.read_csv("student_performance_dataset.csv")

    required_columns = [
        "Student_Name",
        "Study_Hours",
        "Attendance",
        "Assignment_Marks",
        "Quiz_Marks",
        "Final_Grade"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing columns in CSV: {', '.join(missing_columns)}"
        )
        st.stop()

except FileNotFoundError:
    st.error(
        "student_performance_dataset.csv not found in project folder."
    )
    st.stop()

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Marks",
        "Quiz_Marks"
    ]
]

y = df["Final_Grade"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)



st.title("🎓 AI-Based Student Performance Prediction System")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["Prediction", "Dataset", "Analysis"]
)



with tab1:

    st.subheader("Select Student")

    name = st.selectbox(
        "Student Name",
        df["Student_Name"].tolist()
    )

    student_record = df[
        df["Student_Name"] == name
    ].iloc[0]

    st.write("### Student Record")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Study Hours",
            int(student_record["Study_Hours"])
        )

        st.metric(
            "Attendance %",
            int(student_record["Attendance"])
        )

    with col2:
        st.metric(
            "Assignment Marks",
            int(student_record["Assignment_Marks"])
        )

        st.metric(
            "Quiz Marks",
            int(student_record["Quiz_Marks"])
        )

    st.metric(
        "Actual Final Grade",
        round(float(student_record["Final_Grade"]), 2)
    )

    if st.button("Predict Student Performance"):

        prediction = model.predict([[
            student_record["Study_Hours"],
            student_record["Attendance"],
            student_record["Assignment_Marks"],
            student_record["Quiz_Marks"]
        ]])[0]

        prediction = round(prediction, 2)

        if prediction >= 90:
            category = "Excellent"
        elif prediction >= 80:
            category = "Good"
        elif prediction >= 70:
            category = "Average"
        else:
            category = "Needs Improvement"

        st.success(
            f"Predicted Grade: {prediction}%"
        )

        st.info(
            f"Performance Category: {category}"
        )

        if prediction < 50 or student_record["Attendance"] < 60:
            st.warning(
                "⚠ Student is at Risk of Poor Performance"
            )


with tab2:

    st.subheader("Student Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )



with tab3:

    st.subheader("Dataset Analysis")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "Average Grade",
            round(df["Final_Grade"].mean(), 2)
        )

    with metric_col2:
        st.metric(
            "Average Attendance",
            round(df["Attendance"].mean(), 2)
        )

    with metric_col3:
        st.metric(
            "Model Accuracy (R²)",
            f"{accuracy:.2%}"
        )

    st.markdown("---")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df["Study_Hours"],
        df["Final_Grade"]
    )

    ax.set_title(
        "Study Hours vs Final Grade"
    )

    ax.set_xlabel(
        "Study Hours"
    )

    ax.set_ylabel(
        "Final Grade"
    )

    st.pyplot(fig)

st.markdown("---")
st.caption(
    "AI Lab Project - Student Performance Prediction System"
)