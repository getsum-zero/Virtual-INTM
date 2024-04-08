## Pysim

本项目UI设计基于Tkinter/  [ttkbootstrap][1]

本项目核心模块基于 [brainpy][2] ,更多信息详见原仓库

### 打包

执行指令

```shell
Pyinstaller -F pysim.py   				# 打包exe
Pyinstaller -F -w pysim.py 				# 不带控制台的打包
Pyinstaller -F -w -i XXX.ico pysim.py 	# 打包指定exe图标打包
```

### 使用

执行 `pysim.exe` ：

* 进入初始界面，输入MEA尺寸（默认$8 \times 8$​​ ）

  <details>
  <summary>初始界面</summary>
  <img src="./img/input.png" alt="image-input" style="zoom:70%;" />
  </details>

* 提交后进入主界面；点击 `Select a Mode` / `Simulation`，选择 `egs/`文件夹，点击运行即可

  <details>
  <summary>运行</summary>
  <img src="./img/main.png" alt="image-20240408172836239" style="zoom:50%;" />
  </details>

* 结果：

  <img src="./img/res.gif" alt="res" style="zoom:80%;" />

  





### Note 

* 受`brainpy`和`jax`限制，目前只能用`cpu`；
  由于使用`cpu`，因此速度较慢，需要1~2分钟左右；后续增加等待时间进度条
* 由于jax原因，导致打包出现失败，这里简单将出现问题的代码注释
  https://github.com/google/jax/issues/17705
   File "jax\_src\interpreters\mlir.py", line 711, in lower_jaxpr_to_module
  jaxlib.mlir._mlir_libs._site_initialize.<locals>.MLIRError: Verification failed

### Todo

[1]: https://ttkbootstrap.readthedocs.io/en/latest/zh/ " ttkbootstrap"
[2]: https://github.com/brainpy/BrainPy "BrainPy(Github)"
