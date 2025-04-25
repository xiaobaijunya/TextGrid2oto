# **TextGrid2oto**

这是一个把TextGrid转换为oto的程序，本程序*现**已支持多段标注模型***。请您在使用前先了解utau的**presamp.ini**字典以及**SOFA模型字典**，目前暂时**不支持多语种标记**的转换，但您可以通过修改多语种字典为您需要的单语种，进行使用。

## 现已支持	***CVVC,VCV,CVVR,扩张整音***	四种utau标记类型

## *现已支持多段标注模型*

## 使用此程序，你需要准备：

1. [SOFA](https://github.com/qiuqiao/SOFA)：TextGrid标记生成
2. [qiuqiao/SOFA · Discussions · GitHub](https://github.com/qiuqiao/SOFA/discussions)：在SOFA社区选择一个您的语言的SOFA模型
3. SOFA模型自带的字典(一般在模型文件里面)
4. [vlabeler](https://github.com/sdercolin/vlabeler)：TextGrid标记修改软件（[下载vlabeler的textgrid插件](https://github.com/sdercolin/vlabeler-textgrid)）
5. utau音制工具：setParam 或者 [vlabeler](https://github.com/sdercolin/vlabeler)的utau标注模式（推荐）
6. utau录音完成的wav文件(请先了解utau声库如何制作)
7. utau录音表配套的：presamp.ini（**现已支持CVVC和VCV**，请参考presamp文件夹内的文件）
8. 一台**Windows10及以上版本**的电脑（好像python自3.8.10后就不再支持win7了，很抱歉）

## 使用方法：

1. 在config文件夹里面选择一个音源类型的配置
2. 修改config文件里面必改的前三项
3. 将config.txt拖到程序时上
4. 程序运行生成lab文件
5. 前往sofa，选择对应语言的模型，生成**textgrid标记**
6.  **（可选）**中文两段式模型使用sofa-dic文件夹中的SOFA-UTAUCHN-dic.txt，专门为utau声库制作的字典（**注意：部分模型不一定支持此字典中的音素**）
7. 返回程序，按Enter键继续生成
8.  **（可选）**检查程序提示的缺少的音素
9.  **（可选）**查找原因，反馈给开发者
10. oto生成完毕

## ***注意！！***

1. 生成lab的时候，请确保您的wav名称的拼音或者假名 和实际音频内容可以一一对应
2. 你的模型字典中的 单词 要和 presamp 可以对应，如果 单词 在模型字典或者presamp 中不存在，则会被sofa和该程序删除，会导致您的部分的oto和实际音素有差别
3. 您可以直接修改sofa字典中的单词部分，但不要修改音素名
4. 您不必让 模型字典 的 音素名 和您的 presamp 的音素名完全一致，您只需要保持 单词 的一致性即可
5. 程序完全不读取TextGrid的words层，所以如果您需要修改TextGrid标记只需修改phones层
6. 您可以修改cv_sum和vc_sum，来改变CV和VC每一条线的占比，不建议修改左线和预发声这种位置比较固定的线的数值。。。
7. 您可以修改cv_offset和vc_offset来调整oto的偏移，可以实现一些例如辅音长度缩短和CVR音源的连贯性优化
8. 如果缺少CV部分，可以检查sofa模型字典的 单词 是否与presamp里面的不同，可以修改sofa字典的拼音列，让它符合presamp

## uatu采样征集

提交采样，助力我们训练全新的专用于拼接声库的标注模型

huggingface（需要梯子）：https://baibaibai-utau-vocalbank-collect.hf.space/

阿里魔塔社区（直连）：https://modelscope.cn/studios/xiaobaijunya/utau_Vocalbank_collect

## 模型推荐：

中文两段式模型：[SOFA_Models](https://github.com/BaiShuoQwQ/SOFA_Models)：【支持opencpop-extension字典的两段式模型】

日语模型：[ Japanese Quadrilx SOFA model ](https://github.com/Muzium/muzium.github.io/releases/tag/v1.0.2)【支持假名和罗马音音源】

## 联系我们：

QQ群聊：1036935644

### 未来更新计划。。。

1.全新的专用标注模型

2.给重构过的CVVC和CVR生成，重新调优数值

3.重构webUI结构
