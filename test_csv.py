import pandas as pd

# Create sample CSV data
sample_data = {
    'developer_id': [72103],
    'YoE': [8],
    'phase': ['2. Phase 2'],
    'resume_plain': ['Gajendra Nath Chaturvedi - Full Stack JavaScript Developer - Contact: 91-9899746885 gmchaturvedi@gmail.com - H-109,Govindpuram, Ghazianbad,201013 - linkedin.com/in/gmchaturvedi']
}

# Create DataFrame and save to CSV
df = pd.DataFrame(sample_data)
df.to_csv('dev_data.csv', index=False)

print("Sample CSV file created successfully!")
print("Data:")
print(df) 