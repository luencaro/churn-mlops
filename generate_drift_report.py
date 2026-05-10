import joblib
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def main():
    print("Loading datasets...")
    # Load the datasets
    splits = joblib.load('data/train_test_splits.joblib')
    X_train, X_test, y_train, y_test = splits

    print("Generating Evidently Data Drift Report...")
    # Initialize and configure the report
    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    # Calculate drift metrics
    drift_report.run(reference_data=X_train, current_data=X_test)
    
    # Save the report as an HTML file
    report_path = "drift_report.html"
    drift_report.save_html(report_path)
    print(f"Drift report generated successfully: {report_path}")

if __name__ == "__main__":
    main()
