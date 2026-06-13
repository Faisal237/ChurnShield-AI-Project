import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:5000"
API_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

col1, col2 = st.columns([1, 6])  # adjust ratio as needed

with col1:
    st.image("ChurnShield Logo.png", width=1000)

with col2:
    st.header("ChurnShield AI")
st.header("An AI-Based Model For Predicting Customer Churn And Recommending Retention Strategies")

# ---------------- INPUT FIELDS ----------------
age = st.number_input("Age", min_value=1, value=100)
last_login_days = st.number_input("Last Login Days", min_value=0, value=9999)
monthly_fee = st.number_input(
    "Monthly Fee",
    min_value=0.0,
    value=699.0,
    step=50.0,
)
number_of_profiles = st.number_input("Number of Profiles", min_value=1, value=99)
avg_watch_time_per_day = st.number_input(
    "Average Watch Time Per Day",
    min_value=0.0,
    value=23.99,
    step=0.10,
)

# ---------------- PREDICT ----------------
if st.button("Predict Customer Churn"):

    payload = {
        "age": age,
        "last_login_days": last_login_days,
        "monthly_fee": monthly_fee,
        "number_of_profiles": number_of_profiles,
        "avg_watch_time_per_day": avg_watch_time_per_day
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        churn_prob = float(result.get("Churn Probability", 0))
        prediction = result.get("Prediction", "")

        # ---------------- DISPLAY RESULTS ----------------
        st.metric("📊 Churn Probability", f"{churn_prob:.2%}")

        if str(prediction).lower() == "churn":
            st.error("Customer is likely to CHURN")
        else:
            st.success("Customer will stay; NO CHURN")

        # ---------------- RETENTION STRATEGY ----------------
        if churn_prob >= 0.90:
            retention_strategy = (
                "🚨 Critical Risk: Contact the customer immediately, "
                "offer a premium retention package or significant discount, "
                "and assign a dedicated support representative."
            )
        elif churn_prob >= 0.75:
            retention_strategy = (
                "🔴 High Risk: Offer personalized discounts, free premium features, "
                "or a subscription renewal incentive."
            )
        elif churn_prob >= 0.50:
            retention_strategy = (
                "🟠 Moderate Risk: Send targeted recommendations, "
                "engagement campaigns, and loyalty rewards."
            )
        elif churn_prob >= 0.25:
            retention_strategy = (
                "🟡 Low Risk: Encourage continued engagement through "
                "regular communications and feature updates."
            )
        else:
            retention_strategy = (
                "🟢 Very Low Risk: Continue the current customer experience "
                "and monitor behavior periodically."
            )

        st.subheader("Recommended Retention Strategy")
        st.info(retention_strategy)

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Flask API. Make sure app.py is running.")

    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Ensure the API is running and reachable.")

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request error: {e}")

    except ValueError:
        st.error("❌ Unable to process JSON response from the API.")

    except Exception as e:
        st.error(f"❌ Error: {e}")