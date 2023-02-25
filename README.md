单频点预测

该respository记录了本人学习过程中使用深度学习方法的一些代码

此项目是使用新型神经网络对传输线的S参数曲线进行预测，本项目数据集来源于ADS软件仿真，通过传输线的线宽，线间距和线长对其S参数曲线进行预测。

该新型神经网络来源于论文《A Novel Deep Neural Network Topology for Parametric Modeling of Passive Microwave Components》，主要是将传输线的几何参数和频率参数分开处理，通过输入几何参数和频率对S参数进行预测，实质是单频点预测。个人根据论文的思想进行了代码复现

Network.py文件夹是代码的入口
