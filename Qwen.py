import requests
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
    max_num_seqs=2,
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
batch_inputs = []
batch_indices = []
for i, test_example in tqdm.tqdm(test_file.iterrows(), total=len(test_file)):
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
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        image_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(image_messages,return_video_kwargs=True)
    
    
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    

    batch_indices.append(i)
    batch_inputs.append({
        "prompt": prompt,
        "multi_modal_data": mm_data,
    })

# 进行批量推理
outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

# 处理结果
for idx, output in enumerate(outputs):
    generated_text = output.outputs[0].text
    test_file.at[idx, "llava_reasoning"] = generated_text

# 保存结果
test_file.to_csv("./test_qwen_reasoning.csv", index=False)



# import requests
# from PIL import Image
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# import torch
# from transformers import AutoProcessor, LlavaForConditionalGeneration
# import csv
# import tqdm
# import json
# import pandas as pd
# from vllm import LLM, SamplingParams

# test_file_path = "/home/qkli/VQA/EchoSight/evqa_data/test.csv"
# test_file = pd.read_csv(test_file_path)

# model_id = "/home/qkli/huggingface/model/llava-hf/llava-1.5-7b-hf"
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


# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     low_cpu_mem_usage=True, 
# ).to(0)

# processor = AutoProcessor.from_pretrained(model_id)

# test_file["llava_reasoning"] = ""

# batch_size = 20  # You can increase this if you have more GPU memory

# # Process in batches
# for i in tqdm.tqdm(range(0, len(test_file), batch_size)):
#     batch_indices = range(i, min(i + batch_size, len(test_file)))
    
#     # Prepare batch data
#     batch_questions = []
#     batch_images = []
#     batch_conversations = []
    
#     for idx in batch_indices:
#         question = test_file.loc[idx, "question"]
#         image_path = get_image(
#             str(test_file.loc[idx, "dataset_image_ids"]).split("|")[0],
#             str(test_file.loc[idx, "dataset_name"]),
#             iNat_id2name,
#         )
        
#         conversation = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": f"Question:{{{question}}} \nIdentify the objects in the image related to the Question."},
#                     {"type": "image"},
#                 ],
#             },
#         ]
        
#         batch_questions.append(question)
#         batch_images.append(Image.open(image_path))
#         batch_conversations.append(conversation)
    
#     # Process batch
#     prompts = [processor.apply_chat_template(conv, add_generation_prompt=True) for conv in batch_conversations]
#     inputs = processor(
#         text=prompts, 
#         images=batch_images, 
#         return_tensors="pt", 
#         padding=True
#     ).to(0, torch.float16)
    
#     # Generate responses
#     outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    
#     # Decode and store results
#     for batch_idx, idx in enumerate(batch_indices):
#         output_text = processor.decode(outputs[batch_idx][2:], skip_special_tokens=True)
#         test_file.at[idx, "llava_reasoning"] = output_text

# # Save results
# test_file.to_csv("./test_llava_reasoning.csv", index=False)


