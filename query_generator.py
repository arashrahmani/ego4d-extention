from google import genai
from google.genai.types import HttpOptions, ModelContent, Part, UserContent
import json
import os
import timeit
import argparse

def split_dict(d, n):
    """Split a dictionary `d` into `n` chunks."""
    items = list(d.items())
    k, m = divmod(len(items), n)
    return [dict(items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]) for i in range(n)]

# Read the config file
with open("config.json", 'r', encoding='utf-8') as f:
    conf = json.load(f)  # Load JSON data

# Set the environment variables required to request to genai API
os.environ["GOOGLE_CLOUD_PROJECT"] = conf["GOOGLE_CLOUD_PROJECT"]
os.environ["GOOGLE_CLOUD_LOCATION"] = conf["GOOGLE_CLOUD_LOCATION"]
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = conf["GOOGLE_GENAI_USE_VERTEXAI"]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = conf["GOOGLE_APPLICATION_CREDENTIALS"]

narration_path = conf["narration_path"]

with open(narration_path, 'r', encoding='utf-8') as f:
    ego4d_narrations = json.load(f)  # Load JSON data

def generate_content(data_block_indx) -> str:

    narration_parts = split_dict(ego4d_narrations, 5)
    data_part = narration_parts[data_block_indx]

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    # Step 1: Send the guide as the context (only once)
    guide_text = open("llm_prompt.txt", "r", encoding="utf-8").read()
    
    init_guide = f"This is a guide you should keep in mind and work based on what it wants: \n{guide_text}"
    chat_session = client.chats.create(
        # Teach the annotation guide to LLM
        model="gemini-2.0-flash-001",
        history=[
            UserContent(parts=[Part(text=init_guide)]),
            ModelContent(
                parts=[Part(text="It seems it's for ego4d dataset. yes. Sure! I will send the json format you want for each narration you send.")],
            ),
        ],
    )

    partition_size = len(data_part.keys())
    count = 0
    for video_id in data_part:
        for clip_indx, clip in enumerate(data_part[video_id]):
            response = ""
            start = timeit.default_timer()
            response = chat_session.send_message(clip["narration_text"]).text
            clip["nlq_query"] = response
            print("text:", clip["narration_text"])
            print("query:", response)
            stop = timeit.default_timer()
            print('Time: ', stop - start)
            print("-----------------------------------------------")
        count += 1
        print(str(count)+"/"+str(partition_size))
    with open(f"queries_part{data_block_indx}.json", "w") as json_file:
        json.dump(data_part, json_file, indent=4)
    return response

if __name__ == "__main__":

    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("data_part_index", help="The index of a specific data block.", type=int)

    args = parser.parse_args()
    generate_content(args.data_part_index)