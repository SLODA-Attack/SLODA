import json

def convert_jsonl_to_json(jsonl_filename, json_filename):
    # Read the jsonl file and parse each line
    with open(jsonl_filename, 'r') as jsonl_file:
        jsonl_data = [json.loads(line) for line in jsonl_file]

    # Convert the jsonl data to the desired json format
    json_data = []
    current_idx = 0
    for item in jsonl_data:
        json_item = {
            "instruction": "Detect whether the following code contains vulnerabilities.",
            "input": item["func"],
            "output": str(item["target"]),
            # "idx": item["idx"]
            "idx": current_idx
        }
        json_data.append(json_item)
        current_idx += 1

    # Write the json data to the specified json file
    with open(json_filename, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


# Call the function with the names of the input and output files
convert_jsonl_to_json('valid.jsonl', 'valid.json')
# convert_jsonl_to_json('test.jsonl', 'test.json')