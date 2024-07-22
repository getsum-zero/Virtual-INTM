## Virtual-INTM
* The UI design of this project is based on Tkinter / [ttkbootstrap][1]
* The core module of this project is based on [brainpy][2]



### Packaging
* Make sure you have the correct python environment
* Execute instructions
```shell
Pyinstaller -F pysim.py   				# or
Pyinstaller -F -w pysim.py 				# Packaging without console
Pyinstaller -F -w -i XXX.ico pysim.py 	# Pack and specify exe icon packaging
```

### Quick Use

run `pysim.exe`:

* On the initial screen, enter the MEA size (default 8 x 8)

<div align="center">
	<img src="./img/input.png" alt="image-input" width="300">
</div>

* We can get the following main interface:
<div align="center">
	<img src="./img/main.png" alt="image-main" width="300">
</div>

* Digital Twin Model:
    * Clik `Digital Twin Model` and choose `data/2.1.1/2.1.1.mat`
    * Clik `Preset` and choose `data/2.1.1/args.yaml`
    * Click the `Run` button at the top of the interface
    * We can get the results in logs/outputs:
<div align="center">
	<img src="./img/real_heatmap.png" alt="image-real_heatmap" width="400"> <img src="./img/frame_heatmap.png" alt="image-frame_heatmap" width="400">
</div> 


* 结果：

  <img src="./img/fitting.png" alt="image-20240411223200113" style="zoom:50%;" /><img src="./img/res.gif" alt="res" style="zoom:40%;" />

  



### 介绍

模拟器页面主要包括6个组件：

* 模式选择：包括**拟合**真实数据 (fitting)  和  模拟 (simulation)
* 构建刺激范式：控制哪些MEA发放刺激
* 核心参数（必须选择）：选择神经元类型、突触类型、长期可塑性模型
* 运行参数
* 拓扑结构
* 其他参数

### Note 

* 受`brainpy`和`jax`限制，目前只能用`cpu`；
  由于使用`cpu`，因此速度较慢，需要1~2分钟左右；后续增加等待时间进度条
* 由于jax原因，导致打包出现失败，这里简单将出现问题的代码注释
  https://github.com/google/jax/issues/17705
   `File "jax\_src\interpreters\mlir.py", line 711, in lower_jaxpr_to_module
  jaxlib.mlir._mlir_libs._site_initialize.<locals>.MLIRError: Verification failed`

### Note
* 由于神经元数目和突触数目变大，造成更多随机性；优秀的结果需要相同参数下的更多尝试
* 每一个data目录下包括三个文件：
  * `.mat` 原始脉冲数据文件
  * `.txt` 刺激范式文件，8x8；可以借助UI界面的load工具载入
  * `.yaml` 参数配置文件，包含一组精细调节的参数；可以借助UI界面的Preset工具载入
  * `.pkl` 拓扑组织文件，主要用于simulation
  * `outputs` 存储相关的模拟的结果


### Todo

[1]: https://ttkbootstrap.readthedocs.io/en/latest/zh/ " ttkbootstrap"
[2]: https://github.com/brainpy/BrainPy "BrainPy(Github)"
