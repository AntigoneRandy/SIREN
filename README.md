# SIREN (S&P 2025)

This repository contains the official implementation for "Towards Reliable Verification of Unauthorized Data Usage in Personalized Text-to-Image Diffusion Models" (IEEE S&P 2025).

## Setup Instructions
This section provides a detailed guide to prepare the environment and execute the SIREN project. Please adhere to the steps outlined below.

### 1. Environment Setup
#### Create a Conda Environment:
Generate a new Conda environment named siren using Python 3.9:
```
conda create --name siren python=3.9
```
#### Activate the Environment:
Activate the newly created environment:
```
conda activate siren
```
### 2. Installation of Dependencies
Install required Python packages:
```
pip install -r requirements.txt
```

## Quickstart Guidelines
### 1. Dataset & Pre-trained Models Preparation
Our project supports datasets from both local sources and Hugging Face.
### Datasets
##### Local Datasets
You can use the ```--is_text``` flag to specify whether to read the text. The text-image correspondence in the dataset follows a naming convention, where, for example, ```1.png``` corresponds to ```1.txt```.

##### Huggingface Datasets
For datasets from Hugging Face, you need to control the ```--load_text_key``` and ```--load_image_key``` parameters.

### Models
Then, download the meta-learned encoder and detector models from [here](https://www.dropbox.com/scl/fo/7cc8da2xiinfj6yrz670k/AIVqv4eQr5Xytosv-H4Iuq4?rlkey=4rl4tl3khjfr8310st29usw0e&st=cq0t5gwd&dl=0) and place them into the ```./ckpt``` folder.


### 2. Fine-Tune
#### Pokemon encoder & decoder
For your convenience, we directly provide a fine-tuned encoder and decoder on the Pokémon dataset, allowing you to obtain them from [here](https://www.dropbox.com/scl/fo/hr9c2y0a2kcujyqdgamm4/AIxk2wmF074xbf756guoONs?rlkey=d5vzavlvcitax005vlykty5e9&st=y4mngear&dl=0).

#### Fine-Tuned Diffusion Model
You need to train a personalized diffusion model before fine-tuning encoder and decoder. You can train this diffusion model using the webui's code [here](https://github.com/kohya-ss/sd-scripts).
We will provide scripts for training on the Pokémon and CelebA datasets.

##### Pokémon
Training scripts: 
```
accelerate launch --gpu_ids='GPU_ID'  train_network.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --dataset_config="pokemon.toml" \
    --output_dir="OUTPUT_PATH" \
    --output_name="OUTPUT_NAME" \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_epochs=80 \
    --learning_rate=1e-4 \
    --optimizer_type="AdamW" \
    --mixed_precision="fp16" \
    --save_every_n_epochs=20 \
    --network_module=networks.lora \
    --network_dim=64 \
    --gradient_checkpointing \
    --gradient_accumulation_steps=1 \
    --cache_latents
```
You also need a dataset parameter file pokemon.toml:
```
[general]
enable_bucket = true                        # Whether to use Aspect Ratio Bucketing

[[datasets]]
resolution = 512                            # Training resolution
batch_size = 4                              # Batch size

  [[datasets.subsets]]
  image_dir = 'IMAGE_DATASET_PATH'                     # Specify the folder containing the training images
  caption_extension = '.txt'            # Caption file extension; change this if using .txt
  num_repeats = 1                          # Number of repetitions for training images
```

##### CelebA
Training scripts: 
```
accelerate launch --gpu_ids='GPU_ID'  train_network.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --dataset_config="celeba.toml" \
    --output_dir="OUTPUT_PATH" \
    --output_name="OUTPUT_NAME" \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_epochs=80 \
    --learning_rate=1e-4 \
    --optimizer_type="AdamW" \
    --mixed_precision="fp16" \
    --save_every_n_epochs=20 \
    --network_module=networks.lora \
    --network_dim=64 \
    --gradient_checkpointing \
    --gradient_accumulation_steps=1 \
    --cache_latents
```
You also need a dataset parameter file celeba.toml:
```
[general]
enable_bucket = true                        # Whether to use Aspect Ratio Bucketing

[[datasets]]
resolution = 512                            # Training resolution
batch_size = 4                              # Batch size

  [[datasets.subsets]]
  image_dir = 'IMAGE_DATASET_PATH'                     # Specify the folder containing the training images
  caption_extension = '.txt'            # Caption file extension; change this if using .txt
  num_repeats = 1                          # Number of repetitions for training images
```

#### Run Command
Take pokemon as an example, you can perform fine-tuning by running the following command:
```
accelerate launch --gpu_ids='GPU_ID' fine_tune.py \
    --dataset_path "DATASET_FROM_LOCAL_OR_HUB" \
    --epoch 60 \
    --save_n_epoch 20 \
    --output_path "OUTPUT_PATH" \
    --log_dir "LOG_PATH" \
    --diffusion_path "runwayml/stable-diffusion-v1-5" \
    --lora_path "LORA_PATH" \
    --trigger_word "pokemon" \
    --decoder_checkpoint "META_DECODER_PATH" \
    --encoder_checkpoint "MATE_ENCODER_PATH" \
    --is_diffuser
```
Please note that the ```lora_path``` here refers to the LoRA obtained from the previous Fine-Tuned Diffusion Model.

### 3. Coating
#### Local Datasets
For local datasets, you can apply coating to the image using the following command:
```
python coating.py \
    --dataset_path "DATASET_FROM_LOCAL" \
    --decoder_checkpoint "FINE_TUNE_DECODER_PATH" \
    --encoder_checkpoint "FINE_TUNE_ENCODER_PATH" \
    --output_path "OUTPUT_PATH"\
    --is_text \
    --gpu_id GPU_ID
```

#### Huggingface Datasets
For huggingface datasets, you can apply coating to the image using the following command:
```
python coating.py \
    --dataset_path "DATASET_FROM_HUGGINGFACE" \
    --decoder_checkpoint "FINE_TUNE_DECODER_PATH" \
    --encoder_checkpoint "FINE_TUNE_ENCODER_PATH" \
    --output_path "OUTPUT_PATH" \
    --load_text_key "TEXT_KEY" \
    --load_image_key "IMAGE_KEY" \
    --gpu_id GPU_ID
```

### 4. Detecting
#### Clean Dataset
You can train a personalized model using the original clean dataset, and then use this model to generate a dataset as a clean dataset.

For detecting the clean dataset, you can use the following command:
```
python detect.py \
    --dataset_path "CLEAN_DATASET" \
    --decoder_path "FINE_TUNE_DECODER_PATH" \
    --gpu_id GPU_ID \
    --output_path "OUTPUT_PATH" \
    --output_filename "CLEAN_DATASET_OUTPUT_FILENAME" 
```

#### Coating Dataset
You can train a personalized model using the coated dataset, and then use this model to generate a dataset as the suspicious model images.

For detecting this dataset, you can use the following command:
```
python detect.py \
    --dataset_path "COATING_DATASET" \
    --decoder_path "FINE_TUNE_DECODER_PATH" \
    --gpu_id GPU_ID \
    --output_path "OUTPUT_PATH" \
    --output_filename "COATING_DATASET_OUTPUT_FILENAME" 
```
### 5. Hypothesis Testing
You can run the following command to perform hypothesis testing:
```
python ks_test.py \
    --clean_path "CLEAN_DATASET_OUTPUT_FILENAME" \
    --coating_path "COATING_DATASET_OUTPUT_FILENAME" \
    --output "OUTPUT_PATH" \
    --repeat 10000 \
    --samples 30
```
### 6. Meta Learning (Optional)
#### Run Command
You can directly use the model after meta learning. If you want to perform meta-learning by yourself, running the following command:
```
accelerate launch --gpu_ids='GPU_ID' meta.py \
    --dataset_path "Multimodal-Fatima/COCO_captions_train" \
    --epoch 400 \
    --output_path "OUTPUT_PATH" \
    --log_dir "LOG_PATH" \
    --diffusion_path "runwayml/stable-diffusion-v1-5"
```
You may use the pre-trained HiDDeN models from [Stable Signature](https://github.com/facebookresearch/stable_signature/tree/main/hidden) as a warm initialization.

## Acknowledgements
Our SIREN's coating encoder and detector is inherited from [HiDDeN](https://arxiv.org/abs/1807.09937) with slightly modified the model architectures. We mainly use the implementation from [Stable Signature](https://github.com/facebookresearch/stable_signature/tree/main/hidden) and initialize the model using provided pre-trained weights before meta-learning. We also use some code from the [Adversarial Library](https://github.com/jeromerony/adversarial-library) as well as the [Diffusers Library](https://github.com/huggingface/diffusers). Thanks the authors and contributors for their great work!

## Citation
Feel free to contact Boheng Li (randy.bh.li@foxmail.com or via GitHub issue) if you have questions. Should this work assist your research, feel free to cite us via:
```
@inproceedings{li2024towards,
  title={Towards Reliable Verification of Unauthorized Data Usage in Personalized Text-to-Image Diffusion Models},
  author={Li, Boheng and Wei, Yanhao and Fu, Yankai and Wang, Zhenting and Li, Yiming and Zhang, Jie and Wang, Run and Zhang, Tianwei},
  booktitle={2025 IEEE Symposium on Security and Privacy (SP)},
  pages={73--73},
  year={2024},
  organization={IEEE Computer Society}
}
```
