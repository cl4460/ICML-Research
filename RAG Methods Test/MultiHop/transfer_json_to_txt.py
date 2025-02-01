import json

# Specify the path to your JSON file
json_file_path = "/Users/chengze/Desktop/corpus.json"

# Specify the output text file
output_file_path = "/Users/chengze/Desktop/processed_corpus.txt"

try:
    # Load the JSON file
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    
    # Process the data
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for item in data:
            title = item.get("title", "No title available")
            body = item.get("body", "No body available")
            output_file.write(f"title:\n{title}\n\nbody:\n{body}\n\n")
    
    print(f"Processing complete. Output saved to {output_file_path}.")
except Exception as e:
    print(f"An error occurred: {e}")
