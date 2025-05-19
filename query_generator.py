from google import genai
from google.genai.types import HttpOptions
import json
import random
import os
import timeit
import random
import string
import argparse

def split_dict(d, n):
    """Split a dictionary `d` into `n` chunks."""
    items = list(d.items())
    k, m = divmod(len(items), n)
    return [dict(items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]) for i in range(n)]

os.environ["GOOGLE_CLOUD_PROJECT"] = "evident-cosine-343512"
os.environ["GOOGLE_CLOUD_LOCATION"]="global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"]="True"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/rider/Research/EgocentricVision/extention/evident-cosine-343512-bf74e79cb36e.json"

ego4d_path = "/home/rider/Downloads/ego4d_combined/ego4d_data"
narration_path = "utils/sampled_narrations.json"
metadata_path = os.path.join(ego4d_path, "ego4d.json")

with open(narration_path, 'r', encoding='utf-8') as f:
    ego4d_narrations = json.load(f)  # Load JSON data
with open(metadata_path, 'r', encoding='utf-8') as f:
    ego4d_metadata = json.load(f)  # Load JSON data

def generate_content(data_block_indx) -> str:

    narration_parts = split_dict(ego4d_narrations, 5)
    data_part = narration_parts[data_block_indx]

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    # Step 1: Send the guide as the context (only once)
    guide_text = open("llm_prompt.txt", "r", encoding="utf-8").read()
    
    chat_session = client.chats.create(model="gemini-2.0-flash-001") # another available model is gemini-2.5-flash-preview-04-17

    # Teach the annotation guide to LLM
    chat_session.send_message(f"This is a guide you should keep in mind and work based on what it wants: \n{guide_text}")

    response_text = ""
    partition_size = len(data_part.keys())
    count = 0
    for video_id in data_part:
        for clip_indx, clip in enumerate(data_part[video_id]):
            response_text = ""
            start = timeit.default_timer()
            for chunk in chat_session.send_message(clip["narration_text"]):
                response_text += chunk.text
            clip["nlq_query"] = response_text
            print("text:", clip["narration_text"])
            print("query:", response_text)
            stop = timeit.default_timer()
            print('Time: ', stop - start)
            print("-----------------------------------------------")
        count += 1
        print(str(count)+"/"+str(partition_size))
    with open(f"queries_part{data_block_indx}.json", "w") as json_file:
        json.dump(data_part, json_file, indent=4)
    return response_text

if __name__ == "__main__":

    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("data_part_index", help="The index of a specific data block.", type=int)

    args = parser.parse_args()
    generate_content(args.data_part_index)