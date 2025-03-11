# AI_StoryTelling

Have the requirements in check using the 'requirements.txt'

Put in you data in the data-bin folder, in my case i put some short stories in ndJson format.

'''
python ./auto-encoder/pre_procesing.py --corpus data-bin/dummy_data --maxlen 256 --encoder_model google-t5/t5-small --decoder_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --hf_token your_hf_token
'''
Here we are using the T5 and deepseek models, you can choose any model instead.

