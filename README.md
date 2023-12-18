# LFM_Simulation_and_Analysis

(注意：代码是有一些错误的，仅供参考）

## 1.LFM信号分析

### （1）仿真LFM信号；
![图片](https://github.com/SmallC1oud/LFM_Simulation_and_Analysis/assets/77475570/13f41adf-c920-4708-9917-45f183024ebb)

![图片](https://github.com/SmallC1oud/LFM_Simulation_and_Analysis/assets/77475570/90baae45-344d-4afc-ac32-88975256c513)

![图片](https://github.com/SmallC1oud/LFM_Simulation_and_Analysis/assets/77475570/9f2059ca-6170-4b20-9760-80bb7a064fb8)

![图片](https://github.com/SmallC1oud/LFM_Simulation_and_Analysis/assets/77475570/c28fe570-97a2-4fab-a146-27c8a056bca9)


### （2）观察不同过采样率下的DFT结果；

![图片](https://github.com/SmallC1oud/LFM_Simulation_and_Analysis/assets/77475570/e7479bd1-ddd4-41d6-93ab-3dc7a50320c8)

![图片](https://github.com/SmallC1oud/LFM_Simulation_and_Analysis/assets/77475570/d616b62f-48c4-4a70-82a2-04a60fbfcc3f)


### （3）观察不同TBP的LFM信号的频谱。

![图片](https://github.com/SmallC1oud/LFM_Simulation_and_Analysis/assets/77475570/e13d3088-2820-4c3d-adba-180f11a9f69e)


## 2.针对"基带LFM信号"实现脉冲压缩仿真

IRW冲激响应宽度，指冲激响应的3dB宽度，其数值等于脉冲分辨率，时间量纲下的3dB分辨率可以表示为
$\rho = \frac{0.886}{| K | T} \approx \frac{1}{| K | T}$

PSLR最大旁瓣与主瓣峰值的高度比，称为峰值旁瓣比。
$PSLR=10log_{10} \left( \frac{P_{sidelobe}}{P_{mainlobe}} \right)$

ISLR积分旁瓣比，旁瓣能量与主瓣能量的比值(计算中主峰和旁瓣以靠近峰值的两个零点为分界线)
$ISLR=10log_{10} \left( \frac{P_{total}-P_{main}}{P_{main}} \right)$


### （1）实现无误差的脉冲压缩，计算指标（IRW、PSLR、ISLR）

![图片](https://github.com/SmallC1oud/LFM_Simulation_and_Analysis/assets/77475570/5017dafc-078d-46cb-87ae-c38154f6b031)


### （2）观察频域加窗的影响，计算指标（IRW、PSLR、ISLR）

![图片](https://github.com/SmallC1oud/LFM_Simulation_and_Analysis/assets/77475570/64728ab5-9570-4ab5-a295-ac1a46e027b4)

![图片](https://github.com/SmallC1oud/LFM_Simulation_and_Analysis/assets/77475570/a2ff1607-a834-4abf-a28a-29e4c670c897)

![图片](https://github.com/SmallC1oud/LFM_Simulation_and_Analysis/assets/77475570/ecb84ca3-b01b-469b-aa23-6acda4ef910d)


