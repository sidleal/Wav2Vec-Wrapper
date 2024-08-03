import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import re
import jiwer
import jiwer.transforms as tr
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

cer_transform = tr.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)

# It's the jiwer default transform
wer_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

def compute_cer(reference, hypothesis):
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    cer = jiwer.wer(reference, hypothesis, truth_transform=cer_transform, hypothesis_transform=cer_transform)
    return cer

def compute_wer(reference, hypothesis):
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    wer = jiwer.wer(reference, hypothesis, truth_transform=wer_transform, hypothesis_transform=wer_transform) 
    return wer


alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõũúû1234567890%\-\n/\\ "

def replace_special_tokens_and_normalize(text):
    text = text.lower()
    
    map_words = {
        "éh": "eh",
        "ehm": "eh",
        "ehn": "eh",
        "hum": "uh",
        "hm": "uh",
        "uhm": "uh",
        "hã": "ah",
        "ãh": "ah",
        "ã":  "ah",
        "hmm": "uh",
        "mm": "uh",
        "mhm": "uh"
    }

    text = re.sub("h+", "h", text)
    text = re.sub("[^{}]".format(alphabet+" "), " ", text)
    text = re.sub("[ ]+", " ", text)
    
    words = text.split(' ')
    new_words = []
    for word in words:
        if word == '' or word == ' ':
            continue
        if word in map_words:
            new_words.append(map_words[word])
        else:
            new_words.append(word)

    return " ".join(new_words)

def calculate_wer_cer(reference, hypothesis):
    if reference.strip() == '' or hypothesis.strip() == '':
        return 1, 1    
    wer = compute_wer(reference, hypothesis)
    cer = compute_cer(reference, hypothesis)
    return wer, cer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "RodrigoLimaRFL/distil-whisper-nurc-sp-fine-tuned"
#model_id = "RodrigoLimaRFL/distil-large-nurc-sp"
#model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

#dataset_stream = load_dataset("RodrigoLimaRFL/nurc-sp-hugging-face", split="validation", streaming=True, trust_remote_code=True)
#dataset = list(dataset_stream.take(7000))

#dataset = load_dataset('audiofolder', data_dir="d:/datasets/NURC-SP", split="test")
dataset = load_dataset('audiofolder', data_dir="d:/datasets/CORAA-v1", split="test")
#dataset = load_dataset('audiofolder', data_dir="d:/datasets/common_voice", split="test")

results = []
i = 0
with open("nurcsp-distil-test-log.txt", mode="w", encoding="utf-8") as f:
    for out in tqdm(pipe(KeyDataset(dataset, "audio"))):
        f.write(f"{i} - {dataset[i]['audio']['path']}\n")
        results.append(out['text'])
        i+=1

total_wer = 0.0
total_cer = 0.0
qtd = 0
with open("nurcsp-distil-tun-test-output-coraa.tsv", mode="w", encoding="utf-8") as f:
    f.write("file_path\toriginal\tprediction\toriginal normalized\tprediction normalized\twer\tcer\n")
    for i, text in enumerate(results):
        qtd += 1
        audio_path = dataset[i]['audio']['path']
        text_original = dataset[i]['text']
        text = text.replace('\n','')
        text_original = text_original.replace('\n','')
        text_original_norm = replace_special_tokens_and_normalize(text_original)
        text_norm = replace_special_tokens_and_normalize(text)
        wer, cer = calculate_wer_cer(text_original_norm, text_norm)
        total_wer += wer
        total_cer += cer
        f.write(f"{audio_path.replace('\\', '/')[20:]}\t{text_original}\t{text}\t{text_original_norm}\t{text_norm}\t{wer}\t{cer}\n")

print(qtd, "WER:", total_wer, total_wer/qtd, "CER:", total_cer, total_cer/qtd)