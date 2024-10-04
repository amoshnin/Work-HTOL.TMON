import os

# Get the name of the current script
current_script = os.path.basename(__file__)

# Iterate over files in the current directory
path = "./program/code/dashboard"
for filename in os.listdir(path):
    print(filename)
    # Check if it's a .py file and not the current script
    if filename.endswith('.py') and filename != current_script:
        # Construct new filename with .txt extension
        new_filename = os.path.splitext(filename)[0] + '.txt'

        # Read the content of the .py file
        with open(os.path.join(path, filename), 'r') as py_file:
            content = py_file.read()

        # Write the content to the new .txt file
        with open(os.path.join(path, new_filename), 'w') as txt_file:
            txt_file.write(content)

        print(f"Converted {filename} to {new_filename}")