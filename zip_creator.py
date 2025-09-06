import zipfile
import os

# Path to your Python file
python_file = r"C:\MLProjects\churn.py"
zip_file = r"C:\MLProjects\churn_project.zip"

# Create the zip file
with zipfile.ZipFile(zip_file, 'w') as zipf:
    zipf.write(python_file, os.path.basename(python_file))

print(f"Zip file created successfully: {zip_file}")
