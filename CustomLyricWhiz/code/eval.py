import jiwer
import os
import json
# from tools.utils import remove_emoji, compute_wer, convert_digits_to_words, transformation
import re
from vietnam_number import n2w
import unicodedata
import glob


BANG_XOA_DAU = str.maketrans(
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴáàảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ",
    "A"*17 + "D" + "E"*11 + "I"*5 + "O"*17 + "U"*11 + "Y"*5 + "a"*17 + "d" + "e"*11 + "i"*5 + "o"*17 + "u"*11 + "y"*5
)

def remove_diacritics(txt: str) -> str:
    if not unicodedata.is_normalized("NFC", txt):
        txt = unicodedata.normalize("NFC", txt)
    return txt.translate(BANG_XOA_DAU)


def normalize_VNmese(text):
    return unicodedata.normalize('NFKC', text)

def load_predicted_text(pred_file):
    """Load predicted text from a JSON file."""
    with open(pred_file, 'r') as _file:
        segments = json.load(_file)
        text = [seg['text'][:-1] for seg in segments] #Remove the last char, which is usually a dot
    return ' '.join(text)

def load_ground_truth(gt_file):
    """Load ground truth text from a TXT file and strip timestamps."""
    with open(gt_file, 'r') as _file:
        raw_text = _file.read()
        # Remove timestamps like [mm:ss.xx]
        return re.sub(r'\[\d{2}:\d{2}\.\d{2}\]', '', raw_text).strip()
    
def remove_punc(text):
    sanitized = re.sub(r"[!@#$%^&*()+=,?]", "", text)
    return sanitized

def convert_numbers_in_string(input_string):
    """
    Extract numeric substrings from input, convert them to Vietnamese words,
    and replace them in the original string.
    """
    # Use regex to find all numeric sequences
    numeric_parts = re.findall(r'\d+', input_string)

    # Replace numeric sequences with their word equivalents
    for num in numeric_parts:
        word = n2w(num)  # Convert to Vietnamese words
        input_string = input_string.replace(num, word, 1)  # Replace only the first occurrence

    return input_string

def preprocess_text(text):
    """Normalize and preprocess text for comparison."""
    text = text.lower().strip()
    text = text.replace('\n', ' ')
    # text = remove_emoji(text)
    # text = text.replace(' am', 'm').replace('\'d', ' would')
    # text = transformation(text)

    # text = convert_digits_to_words(text)
    text = remove_punc(text)
    # text = convert_numbers_in_string(text)
    return text

def compare_files(pred_file, gt_file):
    """Compare predicted and ground truth texts, returning error metrics."""
    pred_text = preprocess_text(load_predicted_text(pred_file))
    gt_text = preprocess_text(load_ground_truth(gt_file))

    # Compute error metrics using jiwer
    errors = jiwer.compute_measures(truth=gt_text, hypothesis=pred_text)
    cer_score = jiwer.cer(truth=gt_text, hypothesis=pred_text)
    errors['cer'] = cer_score
    
    return errors

def evaluate_predictions(pred_dir, gt_dir):
    """Evaluate all predicted files against ground truth files."""
    all_metrics = {'mer': 0, 'wer': 0, 'wil': 0, 'wip': 0, 'cer': 0}
    count = 0
    all_gt = glob.glob(os.path.join(gt_dir, '*.lrc'))
    all_gt_songname = [remove_diacritics(os.path.basename(x).split('.origin.lrc')[0]) for x in all_gt]

    for pred_file in os.listdir(pred_dir):
        if not pred_file.endswith('.json'):
            continue  # Skip non-JSON files

        # Match predicted file with corresponding ground truth
        song_name = os.path.basename(pred_file).split('.mp3.json')[0]
        song_name_normalized = remove_diacritics(song_name)
        
        gt_file = all_gt[all_gt_songname.index(song_name_normalized)]

        if not os.path.exists(gt_file):
            print('-------------------')
            print(os.path.basename(gt_file).split == os.path.basename(pred_file))
            print(pred_file)
            print(gt_file)
            print(f"Warning: No ground truth found for {song_name}")
            continue

        # Compare and collect metrics
        errors = compare_files(os.path.join(pred_dir, pred_file), gt_file)
        for key in all_metrics:
            all_metrics[key] += errors[key]

        count += 1

    # Print summary
    if count > 0:
        print(f"Total songs evaluated: {count}")
        print(f"Average MER: {all_metrics['mer'] / count:.3f}")
        print(f"Average WER: {all_metrics['wer'] / count:.3f}")
        print(f"Average WIL: {all_metrics['wil'] / count:.3f}")
        print(f"Average WIP: {all_metrics['wip'] / count:.3f}")
        print(f"Average CER: {all_metrics['cer'] / count:.3f}")
    else:
        print("No valid predictions found.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python script.py <pred_dir> <gt_dir>")
        sys.exit(1)

    pred_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    evaluate_predictions(pred_dir, gt_dir)

    # text = load_ground_truth('/home/anh/Documents/vietnamese-song-scraping/out/quoc_master_1st_100_preprocessed/4 Phút 20 Giây (Về Nhà Với Anh Đi).origin.lrc')
    # text = load_predicted_text('/home/anh/Documents/vietnamese-song-scraping/out/PhoWhisper-small/quoc_master_1st_100_preprocessed_nospeech-unremove/4 Phút 20 Giây (Về Nhà Với Anh Đi).mp3.json')
    # text = preprocess_text(text)
    # print(text)