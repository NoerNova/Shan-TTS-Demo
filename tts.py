import os
from transformers import VitsModel, VitsTokenizer
import torch
from shannlp import util, word_tokenize


def preprocess_string(input_string: str):
    input_string = input_string.replace("(", "").replace(")", "")
    string_token = word_tokenize(input_string)
    num_to_shanword = util.num_to_shanword

    result = []
    for token in string_token:
        if token.strip().isdigit():
            result.append(num_to_shanword(int(token)))
        else:
            result.append(token)

    full_token = "".join(result)
    return full_token


def synthesize(model: str, input_string: str, speed: float = 1.0):
    auth_token = os.environ.get("TOKEN_READ_SECRET") or True

    model_id = {
        "original": "facebook/mms-tts-shn",
        "nova": "NorHsangPha/mms-tts-nova-train",
        "homhom": "NorHsangPha/mms-tts-shn-train",
    }[model]

    model = VitsModel.from_pretrained(model_id, token=auth_token)
    tokenizer = VitsTokenizer.from_pretrained(model_id, token=auth_token)

    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)

    processed_string = preprocess_string(input_string)
    inputs = tokenizer(processed_string, return_tensors="pt").to(device)

    torch.manual_seed(42)

    model.speaking_rate = speed
    model.noise_scale = 0.2
    model.noise_scale_w = 0.2
    model.length_scale = 1.0 / speed

    with torch.no_grad():
        output = model(**inputs).waveform

    output = output.squeeze().cpu().numpy()

    return ((16_000, output), processed_string)


TTS_EXAMPLES = [
    ["nova", "မႂ်ႇသုင်ၶႃႈ ယူႇလီၵိၼ်ဝၢၼ်ၵတ်းယဵၼ် လီယူႇၶႃႈၼေႃႈ။", 1.0],
    ["original", "ပဵၼ်ယၢမ်းဢၼ် ၸႂ်တိုၼ်ႇတဵၼ်ႈ ၽူင်ႉပိဝ် တႃႇၼုမ်ႇယိင်းၼုမ်ႇၸၢႆးၶဝ် ၸိူဝ်းဢၼ် တေလႆႈၶိုၼ်ႈႁဵၼ်းၼၼ်ႉယူႇ", 1.0],
    [
        "homhom",
        "မိူဝ်ႈပီ 1958 လိူၼ်မေႊ 21 ဝၼ်းၼၼ်ႉ ၸဝ်ႈၼွႆႉသေႃးယၼ်ႇတ ဢမ်ႇၼၼ် ၸဝ်ႈၼွႆႉ ဢွၼ်ႁူဝ် ၽူႈႁၵ်ႉၸိူဝ်ႉၸၢတ်ႈ 31 ၵေႃႉသေ တိူင်ႇၵၢဝ်ႇယၼ်ႇၸႂ် ၵိၼ်ၼမ်ႉသတ်ႉၸႃႇ တႃႇၵေႃႇတင်ႈပူၵ်းပွင် ၵၢၼ်လုၵ်ႉၽိုၼ်ႉ တီႈႁူၺ်ႈပူႉ ႁိမ်းသူပ်းၼမ်ႉၵျွတ်ႈ ၼႂ်းဢိူင်ႇမိူင်းႁၢင် ၸႄႈဝဵင်းမိူင်းတူၼ် ၸိုင်ႈတႆးပွတ်းဢွၵ်ႇၶူင်း လႅၼ်လိၼ်ၸိုင်ႈထႆး။",
        1.0,
    ],
    [
        "nova",
        "ပဵၼ်ၵၢၼ်ၾုၵ်ႇၾင်ၸႂ်ၵၼ်ၼႅၼ်ႈ ၼၵ်းပၵ်းၸႂ် ယွင်ႈၵုၼ်းယွင်ႈမုၼ်ဢူငဝ်း ၸိူဝ်းၽူႈလဵပ်ႈႁဵၼ်းႁူႉပိုၼ်း ၸဵမ်လဵၵ်ႉယႂ်ႇၼုမ်ႇထဝ်ႈ ၼႂ်းၸိူဝ်း ၽူႈႁၵ်ႉ ၸိူဝ်ႉၸၢတ်ႈလၢႆပၢၼ်လၢႆသႅၼ်းမႃး 66 ပီ ၼပ်ႉတင်ႈတႄႇ 1958 ဝၼ်းတီႈ 21 လိူၼ်မေႊ။",
        1.0,
    ],
]
