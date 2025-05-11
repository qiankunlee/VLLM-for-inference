# import requests
# from PIL import Image
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# import torch
# from transformers import AutoProcessor, LlavaForConditionalGeneration
# import csv
# import tqdm
# import json
# import pandas as pd
# from transformers import AutoProcessor
# from vllm import LLM, SamplingParams
# from qwen_vl_utils import process_vision_info

# test_file_path = "/home/qkli/VQA/EchoSight/evqa_data/test_10_rows.csv"
# test_file = pd.read_csv(test_file_path)

# model_id = "/home/qkli/huggingface/model/Qwen/Qwen2.5-VL-7B-Instruct"
# iNat_image_path = "/home/qkli/47data/vqa_data/evqa/iNaturalist/id_name"
# with open(iNat_image_path + "/train_id2name.json", "r") as f:
#     iNat_id2name = json.load(f)


# def get_image(image_id, dataset_name, iNat_id2name=None):

#     GLD_image_path = "/home/qkli/47data/vqa_data/evqa/landmark/train" 
#     iNat_image_path = "/home/qkli/47data/vqa_data/evqa/iNaturalist"
#     infoseek_test_path = "/home/qkli/47data/vqa_data/infoseek/images/infoseek_val"
#     infoseek_train_path = "/home/qkli/47data/vqa_data/infoseek/images/infoseek_train"
#     """_summary_
#         get the image file by image_id. image id are indexed by its first 3 letters in the corresponding folder. e.g. image_id = "abcde" will be stored in "a/b/c/abcde.jpg"
#     Args:
#         image_id : the image id
#     """
    
#     if dataset_name == "inaturalist":
#         file_name = iNat_id2name[image_id]
        
#         image_path = os.path.join(iNat_image_path, file_name)
#     elif dataset_name == "landmarks":
#         image_path = os.path.join(GLD_image_path, image_id[0], image_id[1], image_id[2], image_id + ".jpg")
#     elif dataset_name == "infoseek":
#         if os.path.exists(os.path.join(infoseek_test_path, image_id + ".jpg")):
#             image_path = os.path.join(infoseek_test_path, image_id + ".jpg")
#         elif os.path.exists(os.path.join(infoseek_test_path, image_id + ".JPEG")):
#             image_path = os.path.join(infoseek_test_path, image_id + ".JPEG")
#     else:
#         raise NotImplementedError("dataset name not supported")
#     return image_path

# llm = LLM(
#     model=model_id,
#     limit_mm_per_prompt={"image": 1, "video": 1},
#     max_num_seqs=32,
#     block_size=32,
#     max_model_len=2048
# )

# sampling_params = SamplingParams(
#     temperature=0.0,
#     top_p=0.001,
#     repetition_penalty=1.05,
#     max_tokens=256,
#     stop_token_ids=[],
# )
# processor = AutoProcessor.from_pretrained(model_id)

# test_file["llava_reasoning"] = ""
# test_file["image_path"] = ""

# # Prepare batch inputs
# batch_inputs = []
# batch_indices = []
# for i, test_example in tqdm.tqdm(test_file.iterrows(), total=len(test_file)):
#     question = test_example["question"]
#     image_path = get_image(
#         str(test_example["dataset_image_ids"]).split("|")[0],
#         str(test_example["dataset_name"]),
#         iNat_id2name,
#     )
    
#     test_file.at[i, "image_path"] = image_path
    
# #     image_messages = [
# #         {"role": "system", "content": "You are a helpful assistant."},
# #         {
# #             "role": "user",
# #             "content": [
# #                 {
# #                     "type": "image",
# #                     "image": image_path,
# #                 },
# #                 {"type": "text", "text": f"Question:{{{question}}}\n\
# # (1) Identify key visual elements related to the question: <list key relevant objects>.\
# # (2) Based on the visual context, the information needed for a good answer to the question should include: <describe required knowledge>.\
# # Follow this exact output format:\
# # (1) Key visual elements:\
# # (2) Required knowledge for a good answer:"},
# #             ],
# #         },
# #     ]

#     image_messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": image_path,
#                 },
#                 {"type": "text", "text": 
# "(1) Identify the key elemental features of the objects in the image: <list key visual elements>.\
# (2) Please think step by step to infer the characteristics of the object in other periods/states (such as different seasons, new and old versions, before and after use, etc: <Characteristics changes>\
# Follow this exact output format:\
# (1) Key visual elements:\
# (2) Characteristics changes:"
#                 },
#             ],
#         },
#     ]
#     prompt = processor.apply_chat_template(
#         image_messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
    
#     image_inputs, video_inputs = process_vision_info(image_messages, return_video_kwargs=False)
    
    
#     mm_data = {}
#     if image_inputs is not None:
#         mm_data["image"] = image_inputs
    
#     batch_inputs.append({
#         "prompt": prompt,
#         "multi_modal_data": mm_data,
#     })

# # 进行批量推理
# outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

# # 处理结果
# for idx, output in enumerate(outputs):
#     generated_text = output.outputs[0].text
#     test_file.at[idx, "llava_reasoning"] = generated_text

# # 保存结果
# test_file.to_csv("./test_qwen_reasoning2.csv", index=False)



import requests
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import csv
import tqdm
import json
import pandas as pd
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

test_file_path = "/home/qkli/VQA/EchoSight/evqa_data/test.csv"
test_file = pd.read_csv(test_file_path)

model_id = "/home/qkli/huggingface/model/Qwen/Qwen2.5-VL-7B-Instruct"
iNat_image_path = "/home/qkli/47data/vqa_data/evqa/iNaturalist/id_name"
with open(iNat_image_path + "/train_id2name.json", "r") as f:
    iNat_id2name = json.load(f)


def get_image(image_id, dataset_name, iNat_id2name=None):

    GLD_image_path = "/home/qkli/47data/vqa_data/evqa/landmark/train" 
    iNat_image_path = "/home/qkli/47data/vqa_data/evqa/iNaturalist"
    infoseek_test_path = "/home/qkli/47data/vqa_data/infoseek/images/infoseek_val"
    infoseek_train_path = "/home/qkli/47data/vqa_data/infoseek/images/infoseek_train"
    """_summary_
        get the image file by image_id. image id are indexed by its first 3 letters in the corresponding folder. e.g. image_id = "abcde" will be stored in "a/b/c/abcde.jpg"
    Args:
        image_id : the image id
    """
    
    if dataset_name == "inaturalist":
        file_name = iNat_id2name[image_id]
        
        image_path = os.path.join(iNat_image_path, file_name)
    elif dataset_name == "landmarks":
        image_path = os.path.join(GLD_image_path, image_id[0], image_id[1], image_id[2], image_id + ".jpg")
    elif dataset_name == "infoseek":
        if os.path.exists(os.path.join(infoseek_test_path, image_id + ".jpg")):
            image_path = os.path.join(infoseek_test_path, image_id + ".jpg")
        elif os.path.exists(os.path.join(infoseek_test_path, image_id + ".JPEG")):
            image_path = os.path.join(infoseek_test_path, image_id + ".JPEG")
    else:
        raise NotImplementedError("dataset name not supported")
    return image_path

llm = LLM(
    model=model_id,
    limit_mm_per_prompt={"image": 1, "video": 1},
    max_num_seqs=32,
    block_size=32,
    max_model_len=2048
)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=256,
    stop_token_ids=[],
)
processor = AutoProcessor.from_pretrained(model_id)

test_file["llava_reasoning"] = ""
test_file["image_path"] = ""

# Prepare batch inputs

batch_size = 128

for batch_start in tqdm.tqdm(range(0, len(test_file),batch_size)):
    batch_end = min(batch_start + batch_size, len(test_file))
    batch = test_file.iloc[batch_start:batch_end]

    
    batch_inputs = []
    batch_indices = []
    for i, test_example in batch.iterrows():
        
        batch_indices.append(i)
        question = test_example["question"]
        image_path = get_image(
            str(test_example["dataset_image_ids"]).split("|")[0],
            str(test_example["dataset_name"]),
            iNat_id2name,
        )
    
        test_file.at[i, "image_path"] = image_path

        image_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": "(1) Identify the key elemental features of the objects in the image: <list key visual elements>.\
(2) Please think step by step to infer the characteristics of the object in other periods/states (such as different seasons, new and old versions, before and after use, etc: <Characteristics changes>\
Follow this exact output format:\
(1) Key visual elements:\
(2) Characteristics changes:"
                    },
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            image_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        image_inputs, video_inputs = process_vision_info(image_messages, return_video_kwargs=False)
        
        
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        
        batch_inputs.append({
            "prompt": prompt,
            "multi_modal_data": mm_data,
        })
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        test_file.at[batch_indices[idx], "llava_reasoning"] = generated_text

test_file.to_csv("./test_qwen_reasoning.csv", index=False)

