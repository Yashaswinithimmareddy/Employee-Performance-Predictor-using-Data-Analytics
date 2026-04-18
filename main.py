from src.data_generation import generate_hr_data
from src.model_pipeline import run_pipeline
import os

def main():
    print("=====================================================")
    print(" Employee Performance Predictor using Data Analytics ")
    print("=====================================================\n")

    # Step 1: Generate Data
    print("--- Phase 1: Data Generation ---")
    data_path = "data/synthetic_hr_data.csv"
    generate_hr_data(num_records=1000, output_path=data_path)
    print("Phase 1 completed.\n")

    # Step 2: Run Pipeline (EDA, Preprocessing, Modeling, Evaluation)
    print("--- Phase 2: ML Pipeline & Evaluation ---")
    run_pipeline(data_path=data_path, outputs_dir="outputs", images_dir="images")
    print("Phase 2 completed.\n")

    print("""
=====================================================
 Process Finished Successfully. 
 
 View the generated outputs in:
 - 'data/': Contains the synthetic dataset.
 - 'images/': Contains EDA plots, Confusion Matrix, etc.
 - 'outputs/': Contains 'metrics.txt' with model stats.
=====================================================
""")

if __name__ == "__main__":
    # Ensure correct working directory context
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
