[TOC]

### pytorch中 model.named_parameters()、model.parameters()、model.state_dict().items()的区别:  
***

1. model.named_parameters()，迭代打印 model.named_parameters()，将会打印每一次迭代元素的 名字和param;  [可以修改parameters.requires_grad=False]  
2. mdoel.parameters(),迭代打印 model.parameters()将会打印每一次迭代元素的 parame而不会打印名字,这是和model.named_parameters()的区别,两者都可以修改 requires_grad的属性;   
3. model.state_dict().items() 每次迭代打印该选项的话,将会打印所有的  name-param，但是这里所有的 param都是  requires_grad=False, 没有办法改变 requires_grad的属性,所以
requires_grad的属性只能通过1-2种方式;   
4.改变了requires_grad之后要修改 optimizer的属性;     
optimizer = optim.SGD(  
            filter(lambda p: p.requires_grad, model.parameters()),   #只更新  requires_grad=True的参数  
            lr=cfg.TRAIN.LR,  
            momentum=cfg.TRAIN.MOMENTUM,   
            weight_decay=cfg.TRAIN.WD,  
            nesterov=cfg.TRAIN.NESTEROV)  
5. 随机参数初始化  
def init_weights(m):  
    if isinstance(m,nn.Conv2d):  
        torch.nn.init.xavier_uniform(m.weight.data)  
model.apply(init_weights)  

***

### pytorch中模型的存储和导入    

***  
### pytorch中训练模型的流程图:     
1.  [模型定义]
2. 数据处理-----> 2. 数据封装 ----> 4. 将数据放入模型-------> 5. 计算损失函数--->  6. 方向传播  
7.模型预测   8.模型保存、导入 
1-6形成一个闭环，也是最简单的 模型训练过程. 下面详细分开,然后各自添加一些.  
1. 数据处理: 1)NLP中的去除停用词 2)if 中文，中文分词  3)词典        4)不同任务用的特征层面，一般前3个足够，有的任务会加特征，比如:句法信息，词性信息，NER等 
2. 喂给模型的其实本质是一个多维的矩阵-tesnor,为了模型能高校-快速的处理数据，一般对数据进行封装，也就是batch_size的封装，比如: 一次喂给模型10个句子，然后模型一下子判断10个句子的分类标签。这里的batch_size就是10.  对应到pytorch: torch.utils.data那个模块下面的函数。  然后根据需求可以进行对应。    
3. 模型的定义: 本质上就是  把模型/网络的架构画出来，然后使用forwawrd函数把各个模块串联起来，一方面进行前向传播，另一方面进行反向传播时候计算梯度。  对应pytorch:   
4. 数据放入模型: 其实就是 数据当作参数喂给模型，也就是 forward函数里面接收的第一个参数，就是x,然后模型中forward函数按照定义好的流程图一步步对于x进行处理，得到最终的结果。    对应 pytorch: 
5. 计算损失函数: 也就是计算 模型预测的标签 和  数据自己的标签之间的距离，  这个距离可以有很多种衡量方法，也就是 很多很多的损失函数,对应pytorch:  
6. 反向传播: A: 对于模型中的参数，所有可导的参数进行求梯度，也就是  按照 前向传播的公式，使用  计算出来的损失函数  求导数，然后保存起来
            B: 优化器优化:  也就是当变量  t有了倒数，以及原来的值的情况下，求他更新以后的值，有很多种计算方法，也就是对应不同的优化器，  对应pytorch:  
7. 1-6基本算是模型完整的部分，有这些也算是模型完成了，其他的就都是一些辅助的东西。   比如； 想要检测模型的好坏，那就写一个测试函数，看看模型预测的结果怎么样，说白了就是不同于loss的评价，按照人的直观进行评价，比如准确率，同样，有很多评价标准，  对应于 pytorch:  
8. 同样模型可以训练好了保存下来，保存的东西，其实是两个东西，也就是组成模型的两个部分: 1) 模型的架构，但是架构本质上一段代码，基本上代码有了，所以保存会浪费空间，所以不建议保存。  2）模型的参数: 这部分肯定是要保存的，其实本质上模型学到的东西也就是这些东西。  打个比方: 神经网络表达式为: y=2x+3,其中2,3是通过两个坐标点学到的,那么 ax+b这个框架就是模型的框架，在你的代码里面定义好了，a=2,b=3这个是参数，也就是模型学到的，需要保存下来，下来给定x的时候就直接可以知道他的标签。  对应于: pytorch： 
9. 数据并行: [先不看]对应于 pytorch: 
10. GPU训练: 优点:  对应于pytorch代码: 
1-10学会了我觉得就已经完全够了，这个完全是按照需求出发的，你有了需求，然后你去找 pytorch是怎样封装的函数满足你的需求的。   其他的都是一些细枝末节的东西:  
        比如: 上面的 model.state_dict（）这个函数是看模型的参数字典的，看清楚了他的功能是查看和修改，所以我们如果不需要那就完全可以不用这个函数，等等都是，都是需要的时候再调用。  
注: 你先按照60分钟那几个教程，把这个框架搭起来，然后就是往里面填东西，每填一个东西，其实都是在解决你的一个需求,比如: 你想把 tensor a 和 b加起来，那就去找  tensor的操作那一节里面，加法操作。 比如: 你想要 封装batch的时候batch_size设置的大一些，那你应该去看看 batch_size这个参数在哪个函数，比如: troch.utils.data.Dataloader（）里面有这个参数，那你就去找这个函数，改这个参数。    就按照这种思维先  搭框架吧，这也是以后你写代码的框架，看代码的框架。  然后学到的知识点，就按照上面 1-10的总结写，题目你可以自己起，之外的知识点，列在11-12---等吧。   其实说白了，说来说去，还是我之前一直和你絮叨的  60分钟教程，包括 maml等等，你可以对照，那一行代码不在60分钟里面，除非一些小的点  


#### pytorch中的detach() 与 detach_()
https://www.cnblogs.com/jiangkejie/p/9981707.html  

### pytorch模型可视化  
1. TensorboardX 包   
2. graphviz 包 https://blog.csdn.net/GYGuo95/article/details/78821617   



### VS Code 插件整理     

### parser.add_argument()  action=‘store_true’参数    
action参数主要是为了区分预先设置的值，其中可以分为4种情况
1. action='store_true',default=False   
   1.1 使用时候  
2. action='store_true',default = True  
3. action='store_true',default=None  
https://blog.csdn.net/tsinghuahui/article/details/89279152
  

### pytorch_bert中一些参数解释  
1. args.local_rank：训练是否使用分布式训练,以及分布式训练的参数， if != -1 : 分布式训练 ; if == -1 : 不采用分布式训练; 
2. args.no_cuda : 是否不用cuda训练，if true,not use cuda
  

### pytorch中分布式训练总结    


### python包 tqdm总结       


### python中的 下划线:  ————**————  ——*——  

















### pytorch runtime error(59):device-side assert triggered at XXX  
原因: 数据 和 label 对不齐的原因  
* 词表 和  文章中单词不匹配 
  *  https://blog.csdn.net/u011394059/article/details/78664642  
  *  词表问题

* 数据label超出自己定义的范围，比如: label: 5,定义[01234]   
    * https://blog.csdn.net/weixin_37142859/article/details/93844144


### python 中的 深拷贝 & 浅拷贝 & 赋值
{
1. python argparse中action参数: https://blog.csdn.net/tsinghuahui/article/details/89279152 
  default=True/False,如果调用则全部为 True，如果不调用，则为 Default,默认为False; 
2. python 中的赋值、深拷贝、浅拷贝: https://www.cnblogs.com/huangbiquan/p/7795152.html  
赋值: 内存相同,不会开辟新的内存空间，一个变另外一个变;   
浅拷贝: 创建新对象,内容是原对象的引用，仅仅只是拷贝了一层； 三种形式: 切片操作,工厂函数,copy中的copy(); 
深拷贝: 只有一种形式,copy模块中的deepcopy()函数; 拷贝出来的是一个全新的对象,与之前的对象没有任何关联; 拷贝了对象的所有元素,包括多层嵌套的元素;
}



### Conda 查看服务i列表  
1. conda env list  
2. conda info --e    

### python spacy [E050] Can't find model 'en_core_web_sm'  
python -m spacy download en_core_web_sm 问题解决  https://www.cnblogs.com/mindReader007/p/10801783.html     



### python OS 模块  
1. python中的os.path.dirname与os.path.dirname(__file__)的用法  |  https://www.cnblogs.com/wxj1129549016/p/9513530.html    
1.1 os.path.dirname(path)  
功能: 去掉当前脚本的文件名，返回目录  
1.2 os.path.dirname(__file__):  
* 1） 所在脚本是以 完整路径运行，则输出该脚本所在的完整路径   
* 2）所在脚本是以相对路径被运行的，那么将输出空目录。  

2. os.path.abspath(path) : 返回绝对路径     



### python中  numpy的形状改变: 
numpy库ndarray多维数组的维度变换方法：reshape、resize、swapaxes、flatten等详解与实例  |   https://blog.csdn.net/brucewong0516/article/details/79185282   



### pytorch中的   scatter_()  和 gather() 
https://blog.csdn.net/qq_43581151/article/details/97372502 
https://blog.csdn.net/qq_39004117/article/details/95665418    
   







### python爬虫入门到放弃  
1.



