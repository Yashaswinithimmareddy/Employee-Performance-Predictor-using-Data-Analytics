import pandas as pd
import numpy as np
import os

# Set a random seed for reproducibility
np.random.seed(42)

def generate_hr_data(num_records=1000, output_path="data/synthetic_hr_data.csv"):
    """
    Generates synthetic Employee Performance data simulating a real HR database.
    """
    print(f"Generating {num_records} synthetic employee records...")

    # Define parameters
    departments = ['IT', 'HR', 'Sales', 'Marketing', 'Finance']
    education_levels = ['Bachelors', 'Masters', 'PhD']
    
    # Generate basic attributes
    employee_ids = np.arange(1001, 1001 + num_records)
    age = np.random.randint(22, 60, size=num_records)
    
    # Experience is correlated with age
    experience = np.maximum(0, age - 22 + np.random.randint(-2, 3, size=num_records))
    
    department = np.random.choice(departments, size=num_records)
    education = np.random.choice(education_levels, size=num_records, p=[0.6, 0.3, 0.1])
    
    # Generate Training Hours (IT/Finance might have slightly more on average)
    training_hours = np.random.randint(10, 100, size=num_records)
    
    # Generate Salary based on experience and department
    base_salary = 40000
    salary = base_salary + (experience * 3000) + np.random.randint(5000, 15000, size=num_records)
    # Adjust for dept
    dept_salary_mult = {'IT': 1.2, 'Finance': 1.15, 'Marketing': 1.05, 'Sales': 1.1, 'HR': 1.0}
    salary = [int(s * dept_salary_mult[d]) for s, d in zip(salary, department)]
    
    # Calculate a hidden performance continuous score based on features
    # Higher training, optimal experience lead to better performance
    base_perf = 50
    perf_score_continuous = (
        base_perf + 
        (training_hours * 0.3) + 
        (experience * 1.5) - 
        (np.abs(age - 35) * 0.2) + # penalty for age just to add some non-linearity
        np.random.normal(0, 10, size=num_records) # noise
    )
    
    # Categorize Performance
    # low (< 33rd percentile), medium (33-66), high (> 66)
    p33 = np.percentile(perf_score_continuous, 33)
    p66 = np.percentile(perf_score_continuous, 66)
    
    performance_rating = []
    for score in perf_score_continuous:
        if score < p33:
            performance_rating.append('Low')
        elif score < p66:
            performance_rating.append('Medium')
        else:
            performance_rating.append('High')
            
    # Combine into DataFrame
    df = pd.DataFrame({
        'Employee_ID': employee_ids,
        'Age': age,
        'Experience_Years': experience,
        'Department': department,
        'Education_Level': education,
        'Training_Hours': training_hours,
        'Salary': salary,
        'Performance_Rating': performance_rating
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully generated and saved to '{output_path}'.")
    return df

if __name__ == "__main__":
    generate_hr_data()
