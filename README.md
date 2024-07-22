# TTS (Text to Speech) demo for Shan language


## Models
- [Finetuned - Nova](https://huggingface.co/NorHsangPha/mms-tts-nova-train) on 600 samples [Shan-TTS-Nova](https://huggingface.co/datasets/NorHsangPha/Shan-TTS-Nova) datasets.
- [Finetuned - HomHom](https://huggingface.co/NorHsangPha/mms-tts-shn-train) on 184 samples [NorHsangPha/shn-tts-datasets](https://huggingface.co/datasets/NorHsangPha/shn-tts-datasets) datasets.
- [facebook/mms-tts-shn](https://huggingface.co/facebook/mms-tts-shn)


## Finetune model
My fine-tuned model aims to address some pronunciation and missing vocabulary issues with newer Shan consonants like 'ၾ'.
However, small, noisy, and low-quality datasets lead to noisy and robotic-sounding output.


## Text pre-processing

Using [ShanNLP](https://github.com/NoerNova/ShanNLP) to pre-process and handle Shan date and number.


## Usage
```bash
# install requirements
pip install -r requirements.txt
```

```bash
# run
python app.py

# or gradio debug mode
gradio app.py
```
