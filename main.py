import wavname2lab

# nuitka --standalone --onefile main.py

if __name__ == '__main__':
    print('sofa-UTAU自动标注')
    print('1.生成lab')
    path=input('请输入wav的路径：')
    cut=input('请输入分隔符默认为_')
    if not cut:
        cut = '_'
    wavname2lab.run(path,cut)

