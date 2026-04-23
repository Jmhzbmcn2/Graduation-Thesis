import pandas as pd

# Path to your thesis testcase
file_path = r"/home/Graduation-Thesis/testcase/eval_ragas_focused.xlsx"

# Load the specific sheet named "summary"
# Note: sheet_name is case-sensitive!
df_summary = pd.read_excel(file_path, sheet_name="Summary")

# Print the entire dataframe
print(df_summary)