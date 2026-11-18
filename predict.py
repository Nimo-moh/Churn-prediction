import sys
import joblib
import pandas as pd

THRESHOLD = 0.7

def main(csv_path, out_path="predictions.csv"):
    model = joblib.load("churn_model.joblib")
    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    ids = df["customerID"] if "customerID" in df else pd.Series(range(len(df)))
    X = df.drop(columns=[c for c in ["customerID", "Churn"] if c in df])

    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({"customerID": ids, "churn_prob": proba.round(3)})
    out["segment"] = pd.cut(proba, [0, 0.4, THRESHOLD, 1.0],
                            labels=["Low", "Medium", "High"], include_lowest=True)
    out.sort_values("churn_prob", ascending=False).to_csv(out_path, index=False)
    print(out["segment"].value_counts())

if __name__ == "__main__":
    main(*sys.argv[1:])
