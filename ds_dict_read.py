


def run(ds_dictpath):
    ds_dict = {}
    vowels = []
    consonant =[]
    CV_pinyin = []
    V_pinyin = []
    with open(ds_dictpath, 'r') as f:
        for line in f:
            line = line.split()
            if len(line) == 3:
                CV_pinyin.append(line[0])
                consonant.append(line[1])
                vowels.append(line[2])
            elif len(line) == 2:
                V_pinyin.append(line[0])
                vowels.append(line[1])
    CV_pinyin = set(CV_pinyin)
    V_pinyin = set(V_pinyin)
    vowels = set(vowels)
    consonant = set(consonant)
    print(len(CV_pinyin),CV_pinyin)
    print(len(V_pinyin),V_pinyin)
    print(len(consonant),consonant)
    print(len(vowels),vowels)

if __name__ == "__main__":
    ds_dictpath = 'opencpop-extension.txt'
    run(ds_dictpath)