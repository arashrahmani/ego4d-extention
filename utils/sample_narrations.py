# Stage 1 of preprocessing
import json
import random
import os

ego4d_path = "/home/rider/Downloads/ego4d_combined/ego4d_data"
narration_path = os.path.join(ego4d_path, "v1/annotations/narration.json")

with open(narration_path, 'r', encoding='utf-8') as f:
    ego4d_narrations = json.load(f)  # Load JSON data

N = 5 # Number of samples
sampled_dataset = {}
for video_uid in ego4d_narrations.keys():
    if 'narration_pass_2' in ego4d_narrations[video_uid]:
        sampled_dataset[video_uid] = []

        # for each narration pass we have narrations and summarizations fields
        clips = ego4d_narrations[video_uid]['narration_pass_2']["narrations"]

        # Sample N consecutive narrations (If there was less than N samples, choose all of them)
        if len(clips) <= N:
            sampled = clips
        else:
            start_idx = random.randint(0, len(clips) - N) # Because we want consecutive narrations
            sampled = clips[start_idx:start_idx + N]       
        for sample in sampled:
            sample.pop("_unmapped_timestamp_sec")
            sampled_dataset[video_uid].append(sample)

with open("sampled_narrations.json", "w") as json_file:
    json.dump(sampled_dataset, json_file, indent=4)