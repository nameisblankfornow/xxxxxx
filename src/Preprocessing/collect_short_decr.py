import json



with open("chemprot_mappings.json", "r") as f:
    mappings = json.load(f)

short_descr = {}
for pred, descriptors in mappings.items():
    short_descr[pred] = descriptors[-1]

with open("chemprot_short_description.json", "w") as f:
    json.dump(short_descr, f, indent=2)