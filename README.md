# sentiment
CCF大数据比赛，基于主题的文本情感分析
https://github.com/EliasCai/sentiment 代码持续更新，欢迎大家关注，希望有所帮助，共同提升 
个人介绍：工作从事大数据开发，熟悉机器学习和深度学习的使用
比赛经验：曾参加场景分类（AiChallenger）、口碑商家客流量预测（天池）、用户贷款风险预测（DataCastle）、
          摩拜算法天战赛（biendata）等，寻找队友冲击前排，希望不吝收留！
版本：v1.1
环境：python3; tensorflow-1.0.0; keras-2.0.6
邮箱：elias8888#qq.com
使用：将data文件夹中的三个csv文件放到py文件同个文件夹下面即可运行
Finish：a
使用jieba进行分词，并用LSTM对第一个情感关键词进行预测，10轮epochs后验证样本的准确率为0.70
Todo：
1、将情感关键词添加到jieba的字典里
2、将第2、3个关键词添加到样本，将预测的概率大于阈值的位置作为情感关键词输出
3、完成主题和情感正负面的分析
4、完善LSTM的网络
5、试试CNN的效果
