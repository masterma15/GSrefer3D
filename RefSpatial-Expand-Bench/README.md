---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: scene
    dtype: string
  - name: image
    dtype: image
  - name: mask
    dtype: image
  - name: object
    dtype: string
  - name: prompt
    dtype: string
  - name: suffix
    dtype: string
  - name: step
    dtype: int64
  splits:
  - name: location
    num_bytes: 42361250.0
    num_examples: 241
  - name: placement
    num_bytes: 38223951.0
    num_examples: 200
  download_size: 46828074
  dataset_size: 80585201.0
configs:
- config_name: default
  data_files:
  - split: location
    path: data/location-*
  - split: placement
    path: data/placement-*

---



<!-- New benchmark release announcement -->

<div style="background-color: #ecfdf5; border-left: 4px solid #10b981; padding: 0.75em 1em; margin-top: 1em; color: #065f46; font-weight: bold; border-radius: 0.375em;">
  🎉 This repository contains the new version of <strong>RefSpatial-Bench</strong> — <strong>RefSpatial-Expand-Bench</strong>!<br>
  Based on the original benchmark, the new version <strong>extends indoor scenes</strong> (e.g., factories, stores) and adds <strong>previously uncovered outdoor scenarios</strong> (e.g., streets, parking lots), providing a more comprehensive evaluation of spatial referring tasks.
</div>


<div style="background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 0.75em 1em; margin-top: 1em; color: #78350f; font-weight: bold; border-radius: 0.375em;">
  🏆 The paper associated with this benchmark, <strong>RoboRefer</strong>, has been accepted to <strong>NeurIPS 2025</strong>!<br>
  Thank you all for your attention and support! 🙌
</div>



<h1 style="display: flex; align-items: center; justify-content: center; font-size: 1.65em; font-weight: 600;">


  <img src="https://huggingface.co/datasets/BAAI/RefSpatial-Bench/resolve/main/assets/logo.png" style="height: 60px; flex-shrink: 0;">

  <span style="line-height: 1.2; margin-left: 0px; text-align: center;">
    RefSpatial-Expand-Bench: A Benchmark for Multi-step Spatial Referring
  </span>

</h1>

<!-- # RefSpatial-Expand-Bench: A Benchmark for Multi-step Spatial Referring with Reasoning -->

 <!-- [![Generic badge](https://img.shields.io/badge/🤗%20Datasets-BAAI/RefSpatial--Expand--Bench-blue.svg)](https://huggingface.co/datasets/JingkunAn/RefSpatial-Expand-Bench)  -->

<p align="center">
  <a href="https://zhoues.github.io/RoboRefer"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project-Homepage-blue" alt="HomePage"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2506.04308"><img src="https://img.shields.io/badge/arXiv-2506.04308-b31b1b.svg?logo=arxiv" alt="arXiv"></a>
  &nbsp;
  <a href="https://github.com/Zhoues/RoboRefer"><img src="https://img.shields.io/badge/Code-RoboRefer-black?logo=github" alt="Project Homepage"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/JingkunAn/RefSpatial"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-RefSpatial--Dataset-brightgreen" alt="Dataset"></a>
  &nbsp;
  <a href="https://huggingface.co/collections/Zhoues/roborefer-and-refspatial-6857c97848fab02271310b89"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-RoboRefer-yellow" alt="Weights"></a>
</p>



Welcome to **RefSpatial-Expand-Bench**, a challenging benchmark based on real-world cluttered scenes to evaluate more complex multi-step spatial referring with reasoning.

<img src="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fzhoues.github.io&labelColor=%232ccce4&countColor=%230158f9" alt="visitor badge" style="display: none;" />
<img src="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fanjingkun.github.io&labelColor=%232ccce4&countColor=%230158f9" alt="visitor badge" style="display: none;" />



<!-- ## 📝 Table of Contents

* [🎯 Tasks](#🎯-tasks)
* [🧠 Reasoning Steps](#🧠-reasoning-steps)
* [📁 Dataset Structure](#📁-dataset-structure)
  * [🤗 Hugging Face Datasets Format (data/ folder)](#🤗-hugging-face-datasets-format-data-folder)
  * [📂 Raw Data Format](#📂-raw-data-format)
* [🚀 How to Use Our Benchmark](#🚀-how-to-use-our-benchmark)
  * [🤗 Method 1: Using Hugging Face datasets Library](#🤗-method-1-using-hugging-face-datasets-library)
  * [📂 Method 2: Using Raw Data Files (JSON and Images)](#📂-method-2-using-raw-data-files-json-and-images)
  * [🧐 Evaluating Our RoboRefer/RoboPoint](#🧐-evaluating-our-roborefer-model)
  * [🧐 Evaluating Gemini 2.5 Series](#🧐-evaluating-gemini-25-pro)
  * [🧐 Evaluating the Molmo Model](#🧐-evaluating-the-molmo-model)
* [📊 Dataset Statistics](#📊-dataset-statistics)
* [🏆 Performance Highlights](#🏆-performance-highlights)
* [📜 Citation](#📜-citation)
  --- -->

## 🎯 Task Split

- Location Task: This task contains **241** samples, which requires model to predicts a 2D point indicating the **unique target object**.
- Placement Task: This task contains **200** samples, which requires model to predicts a 2D point within the **desired free space**.


## 🧠 Reasoning Steps

- We introduce *reasoning steps* (`step`) for each benchmark sample as the number of anchor objects and their spatial relations that help constrain the search space.
- A higher `step` value reflects greater reasoning complexity and a stronger need for spatial understanding and reasoning.


## 📁 Dataset Structure

We provide two formats:

<details>
<summary><strong>Hugging Face Datasets Format</strong></summary>


`data/` folder contains HF-compatible splits:

* `location`
* `placement`

Each sample includes:

| Field    | Description                                                  |
| :------- | :----------------------------------------------------------- |
| `id`     | Unique integer ID                                            |
| `scene`  | Indoor or outdoor                                            |
| `object` | Natural language description of target (object or free area), which is extracted from the `prompt` |
| `prompt` | Full Referring expressions                                   |
| `suffix` | Instruction for answer formatting (**different models may use different suffixes or none**; we provide the format used by RoboRefer) |
| `image`  | RGB image (`datasets.Image`)                                 |
| `mask`   | Binary mask image (`datasets.Image`)                         |
| `step`   | Reasoning complexity (number of anchor objects / spatial relations) |

</details>

<details>
<summary><strong>Raw Data Format</strong></summary>


For full reproducibility and visualization, we also include the original files under:

* `Location/`
* `Placement/`

Each folder contains:

```
Location/
├── image/        # RGB images (e.g., 0.png, 1.png, ...)
├── mask/         # Ground truth binary masks
└── question.json # List of referring prompts and metadata
```

Each entry in `question.json` has the following format:

```json
{
  "id": 40,
  "object": "the second object from the left to the right on the nearest platform",
  "prompt": "Please point out the second object from the left to the right on the nearest platform.",
  "suffix": "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], ...",
  "rgb_path": "image/40.png",
  "mask_path": "mask/40.png",
  "category": "location",
  "step": 2,
  "scene": "indoor"
}
```

</details>


## 🚀 How to Use RefSpaital-Bench


<!-- This section explains different ways to load and use the RefSpatial-Expand-Bench dataset. -->

The official evaluation code is available at https://github.com/Zhoues/RoboRefer.
The following provides a quick guide on how to load and use the RefSpatial-Expand-Bench.


<details>
<summary><strong>Method 1: Using Hugging Face Library</strong></summary>


You can load the dataset easily using the `datasets` library:

```python
from datasets import load_dataset

# Load the entire dataset (all splits: location, placement)
# This returns a DatasetDict
dataset_dict = load_dataset("JingkunAn/RefSpatial-Expand-Bench")

# Access a specific split, for example 'location'
location_split_hf = dataset_dict["location"]

# Or load only a specific split directly (returns a Dataset object)
# location_split_direct = load_dataset("JingkunAn/RefSpatial-Expand-Bench", name="location")

# Access a sample from the location split
sample = location_split_hf[0] 

# sample is a dictionary where 'rgb' and 'mask' are PIL Image objects
# To display (if in a suitable environment like a Jupyter notebook):
# sample["image"].show()
# sample["mask"].show()

print(f"Prompt (from HF Dataset): {sample['prompt']}")
print(f"Suffix (from HF Dataset): {sample['suffix']}")
print(f"Reasoning Steps (from HF Dataset): {sample['step']}")
```

</details>

<details>
<summary><strong>Method 2: Using Raw Data Files (JSON and Images)</strong></summary>



If you are working with the raw data format (e.g., after cloning the repository or downloading the raw files), you can load the questions from the `question.json` file for each split and then load the images and masks using a library like Pillow (PIL).

This example assumes you have the `location` and  `placement` folders (each containing `image/`, `mask/`, and `question.json`) in a known `base_data_path`.

```python
import json
import os
from PIL import Image

# Set the dataset split name and base directory path
split_name = "Location"
base_data_path = "."  # Or set to your actual dataset path

# Load question.json file
question_file = os.path.join(base_data_path, split_name, "question.json")
try:
    with open(question_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
except FileNotFoundError:
    print(f"File not found: {question_file}")
    samples = []

# Process the first sample if available
if samples:
    sample = samples[0]
    print(f"\n--- Sample Info ---")
    print(f"ID: {sample['id']}")
    print(f"Prompt: {sample['prompt']}")

    # Construct absolute paths to RGB image and mask
    rgb_path = os.path.join(base_data_path, split_name, sample["rgb_path"])
    mask_path = os.path.join(base_data_path, split_name, sample["mask_path"])

    # Load images using Pillow
    try:
        rgb_image = Image.open(rgb_path)
        mask_image = Image.open(mask_path)
        sample["image"] = rgb_image
        sample["mask"] = mask_image
        print(f"RGB image size: {rgb_image.size}")
        print(f"Mask image size: {mask_image.size}, mode: {mask_image.mode}")
    except FileNotFoundError:
        print(f"Image file not found:\n{rgb_path}\n{mask_path}")
    except Exception as e:
        print(f"Error loading images: {e}")
else:
    print("No samples loaded.")
```

</details>


<details>
<summary><strong>Evaluating RoboRefer / RoboPoint</strong></summary>


To evaluate RoboRefer on RefSpatial-Expand-Bench:

1. **Prepare Input Prompt:** 

   Concatenate `sample["prompt"]` and `sample["suffix"]` to form the complete instruction.

   ```python
   # Example for constructing the full input for a sample
   full_input_instruction = sample["prompt"] + " " + sample["suffix"]
   ```

2. **Model Prediction & JSON Parsing & Coordinate Scaling:** 

   - **Model Prediction**: After providingthe image (`sample["image"]`) and `full_input_instruction` to the RoboRefer, it outputs **normalized coordinate in a JSON format** like`[(x, y),...]`, where each `x and `y` value is normalized to a range of 0-1.

   - **JSON Parsing:** Parse this JSON string to extract the coordinate attributes (e.g., `x`, `y`).

   - **Coordinate Scaling:** 

     1. Use `sample["image"].size` to get `(width, height)` and scale to the original image dimensions (height for y, width for x). 

     ```python
     # Example: model_output_robo is [(0.234, 0.567)] from Roborefer/RoboPoint
     # sample["image"] is a PIL Image object loaded by the datasets library or loaded from the raw data
     
     def text2pts(text, width, height):
         pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
         matches = re.findall(pattern, text)
         points = []
         for match in matches:
             vector = [
                 float(num) if '.' in num else int(num) for num in match.split(',')
             ]
             if len(vector) == 2:    
                 x, y = vector
                 if isinstance(x, float) or isinstance(y, float):
                     x = int(x * width)
                     y = int(y * height)
                 points.append((x, y))
     
     width, height = sample["image"].size
     scaled_roborefer_points = text2pts(model_output_robo, width, height)
     
     # These scaled_roborefer_points are then used for evaluation against the mask.
     ```

3. **Evaluation:** Compare `scaled_roborefer_points` against `sample["mask"]`. The main metric is **average success rate** — the percentage of predictions falling within the mask.

</details>

<details>
<summary><strong>Evaluating Gemini Series</strong></summary>



To evaluate Gemini Series on RefSpatial-Expand-Bench:

1. **Prepare Input Prompt:** 

   Concatenate the string `"Locate the points of"` and `sample["object"] ` to form the complete instruction.

   ```python
   # Example for constructing the full input for a sample
   full_input_instruction = "Locate the points of " + sample["object"] + "."
   ```

2. **Model Prediction & JSON Parsing & Coordinate Scaling:** 

   * **Model Prediction:** After providing the image (`sample["image"]`) and `full_input_instruction` to the Gemini model series, it outputs **normalized coordinates in an JSON format** like `"```json\n[\n  {\"point\": [y, x], \"label\": \"free space\"}, ...\n]\n```"`, where each `y` and `x` value is normalized to a range of 0-1000.

   * **JSON Parsing:** Parse this JSON string to extract the coordinate attributes (e.g., `x1`, `y1`, `x2`, `y2`, etc.).

   * **Coordinate Conversion:** To use these coordinates for evaluation against the mask, they must be:

     1.  Divided by 1000.0 to normalize them to the 0.0-1.0 range.
     2.  Scaled to the original image dimensions (height for y, width for x).

     ```python
     # Example: model_output_gemini is "```json\n[\n  {\"point\": [438, 330], \"label\": \"free space\"}\n]\n```" from Gemini
     # and sample["image"] is a PIL Image object loaded by the datasets library or loaded from the raw data
     
     def json2pts(text, width, height):
        match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if not match:
            print("No valid code block found.")
            return np.empty((0, 2), dtype=int)
      
        json_cleaned = match.group(1).strip()
      
        try:
            data = json.loads(json_cleaned)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return np.empty((0, 2), dtype=int)
      
        points = []
        for item in data:
            if "point" in item and isinstance(item["point"], list) and len(item["point"]) == 2:
                y_norm, x_norm = item["point"]
                x = int(x_norm / 1000 * width)
                y = int(y_norm / 1000 * height)
                points.append((x, y))
      
        return np.array(points)
     
     width, height = sample["image"].size 
     scaled_gemini_points = json2pts(model_output_gemini, width, height)
     # These scaled_gemini_points are then used for evaluation against the mask.
     ```

3. **Evaluation:** Compare `scaled_gemini_points` against `sample["mask"]`. The main metric is **average success rate** — the percentage of predictions falling within the mask.

</details>

<details>
<summary><strong>Evaluating the Molmo</strong></summary>


To evaluate a Molmo model on this benchmark:

1. **Prepare Input Prompt:** 

   Concatenate `"Locate several points of"` and `sample["object"]` to form the complete instruction.

   ```python
   # Example for constructing the full input for a sample
   full_input_instruction = "Locate several points of " + sample["object"] + "."
   ```

2. **Model Prediction, XML Parsing, & Coordinate Scaling:** 

   - **Model Prediction**: After providing the image (`sample["image"]`) and `full_input_instruction` to the Molmo, it outputs **normalized coordinates in an XML format** like `<points x1="61.5" y1="40.4" x2="76.8" y2="21.8" ... />`, where each `x` and `y` value is normalized to a range of 0-100.

   - **XML Parsing:** Parse this XML string to extract the coordinate attributes (e.g., `x1`, `y1`, `x2`, `y2`, etc.).

   - **Coordinate Conversion:** 

     1.  Divide each coordinate by 100.0 to normalize it to the 0.0-1.0 range.
     2.  Scaled to the original image dimensions (height for y, width for x). 

     ```python
     # Example: model_output_molmo is '<points x1="61.5" y1="40.4" x2="76.8" y2="21.8"/>' from Molmo
     # and sample["image"] is a PIL Image object loaded by the datasets library or loaded from the raw data
     
     def xml2pts(xml_text, width, height):
     	import re
         pattern = re.compile(r'(x\d+)="(-?\d+\.?\d*)"\s+(y\d+)="(-?\d+\.?\d*)"')
         matches = pattern.findall(xml_text)
         points = [(int(float(x_val) / 100.0 * width), int(float(y_val) / 100.0 * height) ) for _, x_val, _, y_val in matches]
         return np.array(points)
     
     width, height = sample["image"].size 
     scaled_molmo_points = xml2pts(model_output_molmo, width, height)
     # These scaled_molmo_points are then used for evaluation.
     ```

3. **Evaluation:** Compare `scaled_molmo_points` against `sample["mask"]`. The main metric is **average success rate** — the percentage of predictions falling within the mask.
   </details>


## 📊 Dataset Statistics

Detailed statistics on `step` distributions and instruction lengths are provided in the table below.

| Task Type | Indoor  | Outdoor | Total   |
| --------- | ------- | ------- | ------- |
| Location  | 115     | 126     | 241     |
| Placement | 120     | 80      | 200     |
| **Total** | **235** | **206** | **441** |

| Task Type | Step           | Samples | Avg. Prompt Length |
| --------- | -------------- | ------- | ------------------ |
| Location  | Step 1         | 54      | 10.61              |
|           | Step 2         | 129     | 12.56              |
|           | Step 3         | 58      | 16.10              |
|           | **Avg. (All)** | **241** | **12.98**          |
| Placement | Step 1         | 3       | 15.00              |
|           | Step 2         | 86      | 15.14              |
|           | Step 3         | 75      | 16.95              |
|           | Step 4         | 29      | 22.24              |
|           | Step 5         | 7       | 22.71              |
|           | **Avg. (All)** | **200** | **17.11**          |



## 🏆 Performance Highlights

Detailed accuracy results of RoboRefer-2B-SFT and RoboRefer-8B-SFT Models on RefSpatial-Expand-Bench

#### **Location Task**

| Category | 2B SFT | 8B SFT |
| -------- | ------ | ------ |
| Overall  | 50.21  | 61.00  |
| Indoor   | 49.57  | 58.26  |
| Outdoor  | 50.79  | 63.49  |
| Step 1   | 61.11  | 72.22  |
| Step 2   | 52.71  | 62.02  |
| Step 3   | 34.48  | 48.28  |

#### **Placement Task**

| Category | 2B SFT | 8B SFT |
| -------- | ------ | ------ |
| Overall  | 48.50  | 60.00  |
| Indoor   | 50.83  | 60.00  |
| Outdoor  | 45.00  | 60.00  |
| Step 1   | 33.33  | 33.33  |
| Step 2   | 41.86  | 51.16  |
| Step 3   | 54.67  | 70.67  |
| Step 4   | 48.28  | 55.17  |
| Step 5   | 71.43  | 85.71  |



## 📫 Contact

If you have any questions about the benchmark, feel free to email Jingkun (anjingkun02@gmail.com) and Enshen (zhouenshen@buaa.edu.cn).
<img src="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fzhoues.github.io&labelColor=%232ccce4&countColor=%230158f9" alt="visitor badge" style="display: none;" />
<img src="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fanjingkun.github.io&labelColor=%232ccce4&countColor=%230158f9" alt="visitor badge" style="display: none;" />

## 📜 Citation

Please consider citing our work if this benchmark is useful for your research.

```
@article{zhou2025roborefer,
    title={RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics},
    author={Zhou, Enshen and An, Jingkun and Chi, Cheng and Han, Yi and Rong, Shanyu and Zhang, Chi and Wang, Pengwei and Wang, Zhongyuan and Huang, Tiejun and Sheng, Lu and others},
    journal={arXiv preprint arXiv:2506.04308},
    year={2025}
}
```