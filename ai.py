import json

# Load configuration
def load_config():
    with open("go.json", "r") as file:
        config = json.load(file)
    return config

# Update configuration
def update_config(key, value):
    config = load_config()

    config[key] = value

    with open("go.json", "w") as file:
        json.dump(config, file, indent=4)

    print(f"{key} updated successfully!")

# Test load
config = load_config()
print("Current Config:")
print(config)

# Test update
update_config("max_tokens", 2000)

# Verify update
print("\nUpdated Config:")
print(load_config())