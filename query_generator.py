from google import genai
from google.genai.types import HttpOptions
import json
import random
import os
import timeit
import random
import string
import argparse

def fast_random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Example usage
s = fast_random_string(16)
print(s)

os.environ["GOOGLE_CLOUD_PROJECT"] = "evident-cosine-343512"
os.environ["GOOGLE_CLOUD_LOCATION"]="global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"]="True"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/rider/Research/EgocentricVision/extention/evident-cosine-343512-bf74e79cb36e.json"

ego4d_path = "/home/rider/Downloads/ego4d_combined/ego4d_data"
narration_path = os.path.join(ego4d_path, "v1/annotations/narration.json")
metadata_path = os.path.join(ego4d_path, "ego4d.json")

with open(narration_path, 'r', encoding='utf-8') as f:
    ego4d_narrations = json.load(f)  # Load JSON data
with open(metadata_path, 'r', encoding='utf-8') as f:
    ego4d_metadata = json.load(f)  # Load JSON data


def generate_content() -> str:
    # [START googlegenaisdk_textgen_chat_stream_with_txt]
    from google import genai
    from google.genai.types import HttpOptions

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    # Step 1: Send the guide as the context (only once)
    guide_text = open("llm_prompt.txt", "r", encoding="utf-8").read()
    
    chat_session = client.chats.create(model="gemini-2.5-flash-preview-04-17") # another available model is gemini-2.0-flash-001

    # Use a system message or initial user message to upload guide
    chat_session.send_message(f"This is a guide you should keep in mind and work based on what it wants: \n{guide_text}")

    response_text = ""
    for i in range(1000):
        start = timeit.default_timer()
        s = fast_random_string(16)

        for chunk in chat_session.send_message_stream(f"The person #C is laughing. {s}"):
            print(chunk.text, end="")
            response_text += chunk.text
        stop = timeit.default_timer()

        print('Time: ', stop - start)  
    # for chunk in chat_session.send_message_stream("The person D is picking a card."):
    #     print(chunk.text, end="")
    #     response_text += chunk.text
    # Example response:
    # The
    #  sky appears blue due to a phenomenon called **Rayleigh scattering**. Here's
    #  a breakdown of why:
    # ...
    # [END googlegenaisdk_textgen_chat_stream_with_txt]
    return response_text


if __name__ == "__main__":

    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("start_index", help="Start index for the desired narrations.", type=int)
    parser.add_argument("end_index", help="End index for the desired narrations.", type=int)

    args = parser.parse_args()
    print("start index", args.start_index)
    print("end index", args.end_index)

    generate_content()









# count =0
# # print(type(data))
# # print(data.keys())
# # sampled_clips_per_video = {}
# final_dataset = {}
# final_dataset['version'] = "1"
# final_dataset['date'] = "260517"
# final_dataset['description	'] = "Synthetic NLQ annotation using Gemini LLM"
# final_dataset['videos'] = []
# final_dataset['manifest']= "s3://ego4d-consortium-sharing/public/v1/full_scale/manifest.csv"
# N = 3  # number of clips to sample per video


# for video_uid in ego4d_narrations.keys():
#     # print("-----------------------")
#     # if video_uid == 'd250521e-5197-44aa-8baa-2f42b24444d2':
#     #     print(video_uid)
#     # print("---------------------data--------------------------------------------")

#     print(" >>>>>>>>>>>>>>>>>>>>>>> summaries <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#     if 'narration_pass_2' in ego4d_narrations[video_uid]:
#         video = {}
#         clip = {}
#          # for each narration pass we have narrations and summarizations fields
#         clips = ego4d_narrations[video_uid]['narration_pass_2']["narrations"]
#         # print("clips", clips)
#         # break
#         video["video_uid"] = video_uid
#         video["clips"] = []
#         video["split"] = "train"

#         # Sample N consecutive narrations (If there was less than N samples, choose all of them)
#         if len(clips) <= N:
#             sampled = clips
#         else:
#             start_idx = random.randint(0, len(clips) - N) # Because we want consecutive narrations
#             sampled = clips[start_idx:start_idx + N]       
#         for sample in sampled:
#             count +=1
#         # sampled_clips_per_video[video_uid] = sampled
#             print("count:", count)
#             print("asdfasdfasdfasdf", sample)
#             # break
#     #     # print (ego4d_narrations[i]['narration_pass_1']['summaries'])
#     #     # print (ego4d_narrations[i]['narration_pass_1']['narrations'])
#     #     count +=1 
#     #     print(count)
#     #     print("length of data", len(ego4d_narrations))
#     #     print(video_uid)
#     #     print(ego4d_narrations[video_uid].keys())
#     #     string = input("please enter a field name")
#     #     print(ego4d_narrations[video_uid][string])
#     # break
# # # count = len(ego4d_narrations)  # Count number of documents (objects) in the list
# # # print(f"Number of documents: {count}")

# # for video in data["videos"]:
# #     for clip in video['clips']:
# #         count += 1
# #         print(count)
# #         if clip["clip_start_sec"] != 0:
# #             print("Hello")
# #             print(clip["clip_start_sec"])

