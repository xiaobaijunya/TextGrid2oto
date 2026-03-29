from lab_generate import wavname2lab
import os


def run(wav_path,index_path,cuts):
    dic = {}
    with open(index_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(',')
        dic[line[0].split('.')[0]] = wavname2lab.process_wav_name(line[1],cuts)
    print(dic)
    
    for root, dirs, files in os.walk(wav_path):
        for wav_file in files:
            if wav_file.endswith('.wav'):
                wav_name = os.path.splitext(wav_file)[0]
                if wav_name in dic:
                    lab_content = dic[wav_name]
                else:
                    lab_content = wavname2lab.process_wav_name(wav_file, cuts)
                
                lab_file_name = wav_name + '.lab'
                lab_file_path = os.path.join(root, lab_file_name)
                with open(lab_file_path, 'w', encoding='utf-8') as lab_file:
                    lab_file.write(lab_content)
                print(f"已生成 {wav_file}: {lab_content}")







if __name__ == "__main__":
    index_path = r'E:\UTAU\voice\KYE Arpasing english\index.csv'
    cuts = '_,-'
    wav_path = r'E:\UTAU\voice\KYE Arpasing english'
    run(wav_path,index_path,cuts)