import tg2sv_change
import os

def auto(dic_path,json_path,wav_path):
    tg2sv_change.run(dic_path,json_path,wav_path)

if __name__ == "__main__":
    dic_path = r'字典/japanese-romaji-dict.txt'
    db_path = r'F:\Download\SVDBCreator_Release_1.0.0\baini_JP\wav'
    folders = [f for f in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, f))]
    print(folders)
    for folder in folders:
        json_path = os.path.join(db_path, folder, 'TextGrid', 'json', 'word_phone.json')
        wav_path = os.path.join(db_path, folder)
        print(json_path, wav_path)
        auto(dic_path,json_path,wav_path)


