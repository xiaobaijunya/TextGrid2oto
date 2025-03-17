# **TextGrid2oto**

这是一个把TextGrid转换为oto的程序，本程序目前**仅支持两段式标记**的转换。请您在使用前先了解utau的**presamp.ini**字典以及**SOFA模型字典**，目前暂时**不支持多语种标记**的转换，但您可以通过修改多语种字典为您需要的单语种，进行使用。

## 使用此程序，你需要准备：

1. [SOFA](https://github.com/qiuqiao/SOFA)：TextGrid标记生成
2. [qiuqiao/SOFA · Discussions · GitHub](https://github.com/qiuqiao/SOFA/discussions)：在SOFA社区选择一个您的语言的SOFA模型
3. SOFA模型自带的字典(一般在模型文件里面)
4. utau录音完成的wav文件(请先了解utau声库如何制作)
5. utau录音表配套的：presamp.ini（如果是 VCV 或者 单独音，暂时需要自己编辑一个presamp获取支持）
6. 一台**Windows10及以上版本**的电脑（好像python自3.8.10后就不再支持win7了，很抱歉）

## 使用方法：

你可以按照指引一步一步的生成文件，也可以拖入单个或者多个run-config.txt，进行自动生成。

你需要先在程序中生成.lab文件才可以使用SOFA进行TextGrid标记生成

注意你的模型字典中的 拼音 要和 presamp 一一对应，如果 拼音 不存在，则会被删除

## 模型推荐：

​		中文两段式模型：[SOFA_Models](https://github.com/BaiShuoQwQ/SOFA_Models)：【支持opencpop-extension字典的两段式模型】

##联系我们

QQ群聊：1036935644


