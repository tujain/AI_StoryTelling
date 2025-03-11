# AI_StoryTelling

## Setup
Have the requirements in check using the 'requirements.txt'


## Create dataset
This step involves tokenizing the dataset in a specified folder that contains ``.json`` files, and then saving it into a folder `parsed_raw_pre` containing three ``.pt`` files of train, dev and test. 
```bash
python ./auto-encoder/pre_procesing.py --corpus data-bin/dummy_data --maxlen 256 --encoder_model google-t5/t5-small --decoder_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --hf_token your_hf_token
```
### Note: Here we are using the T5 and deepseek models, you can choose any model instead.

