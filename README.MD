# **TextGrid2oto**

这是一个把TextGrid转换为oto的程序，本程序目前**仅支持两段式标记**的转换。请您在使用前先了解utau的**presamp.ini**字典以及**SOFA模型字典**，目前暂时**不支持多语种标记**的转换，但您可以通过修改多语种字典为您需要的单语种，进行使用。

## 现已支持**CVVC,VCV,CVVR,扩张整音四种utau标记类型

## 使用此程序，你需要准备：

1. [SOFA](https://github.com/qiuqiao/SOFA)：TextGrid标记生成
2. [qiuqiao/SOFA · Discussions · GitHub](https://github.com/qiuqiao/SOFA/discussions)：在SOFA社区选择一个您的语言的SOFA模型
3. SOFA模型自带的字典(一般在模型文件里面)
4. [vlabeler](https://github.com/sdercolin/vlabeler)：TextGrid标记修改软件（需要安装TextGrid标注插件（好像是这样））
5. utau音制工具：setParam 或者 [vlabeler](https://github.com/sdercolin/vlabeler)的utau标注模式（推荐）
6. utau录音完成的wav文件(请先了解utau声库如何制作)
7. utau录音表配套的：presamp.ini（**现已支持CVVC和VCV**，请参考presamp文件夹内的文件）
8. 一台**Windows10及以上版本**的电脑（好像python自3.8.10后就不再支持win7了，很抱歉）

## 使用方法：

1. 双击运行**TextGrid2oto.exe**，按照指引一步一步的生成文件。
2. 或者根据提示修改**run-config.txt**，拖入单个或者多个run-config.txt，进行自动生成。

## ***注意！！***

1. 生成lab的时候，请确保您的wav名称的拼音或者罗马音 和实际音频内容可以对应
2. 你的模型字典中的 单词 要和 presamp 一一对应，如果 单词 在模型字典或者presamp 中不存在，则会被sofa和该程序删除，会导致您的部分的oto和实际音素有差别，您可以直接修改sofa字典中的单词部分，但请不要修改对应的音素
3. 您不必让 模型字典 的 音素 和您的 presamp 的音素部分完全一致，您只需要保持 单词列 的一致性即可
4. 程序完全不读取TextGrid的words层，所以如果您需要修改TextGrid标记只需修改phones层
5. 您可以修改cv_sum和vc_sum，来改变CV和VC每一条线的占比，不建议修改左线和预发声这种位置比较固定的线的数值。。。
6. 您可以修改cv_offset和vc_offset来调整oto的偏移，可以实现一些例如辅音长度缩短和CVR音源的连贯性优化

## 模型推荐：

中文两段式模型：[SOFA_Models](https://github.com/BaiShuoQwQ/SOFA_Models)：【支持opencpop-extension字典的两段式模型】

## 联系我们：

QQ群聊：1036935644

### 未来支持。。。

1.更好的日语CVVC和日语VCV支持

2.UTAU专用标注模型

3.。。。

