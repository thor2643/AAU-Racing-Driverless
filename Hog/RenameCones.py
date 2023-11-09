import os
import re

def rename_files_in_folder(folder_path, prefix):
    # Initialize the maximum index
    max_index = -1

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            # Use regular expressions to check if the filename matches the expected pattern
            match = re.match(f"{prefix}(\\d+)", filename)
            if match:
                index = int(match.group(1))
                max_index = max(max_index, index)

    # Iterate through the files in the folder and rename only those that don't match the expected pattern
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            # Use regular expressions to check if the filename matches the expected pattern
            match = re.match(f"{prefix}(\\d+)", filename)
            if not match:
                # Check the file extension
                current_extension = os.path.splitext(filename)[1]
                if current_extension != ".jpg":
                    continue

                # Generate the new filename with the next available index and the ".jpg" extension
                max_index += 1
                new_filename = f"{prefix}{max_index}.jpg"

                # Rename the file
                os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
                print(f"Renamed: {filename} -> {new_filename}")

def main():
    # Prompt for the prefix and folder path
    prefix = input("Enter the prefix: ")
    folder_path = input("Enter the folder path: ")
    # Call the function to rename files
    rename_files_in_folder(folder_path, prefix)

if __name__ == "__main__":
    main()
