import csv
import json

if __name__ == "__main__":
    rawFile = open("data/0_2018_Apr_09_2139_quit.csv", "r")
    reader = csv.DictReader(rawFile, delimiter="#")
    for row in reader:
        data = row["trial_data"]
        if data != "--":
            transformed_data = json.loads(data)
            print(transformed_data)
