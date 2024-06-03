import pandas as pd

# Step 1: Read the Excel file
file_path = 'Copper_Set.xlsx'  # replace with your file path
sheet_name = 'Result 1'  # replace with your sheet name if different

# Load the Excel file
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Step 2: Extract the specific column
column_name = 'status'  # replace with your column name
column_data = df[column_name]

# Step 3: Tokenize the text
words = column_data.str.split()

# Flatten the list of lists and convert to a single list of words
all_words = [word for sublist in words.dropna() for word in sublist]

# Step 4: Find unique words
unique_words = set(all_words)

# Print or return the unique words
print(unique_words)
