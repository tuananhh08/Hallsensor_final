IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. 74, 2025 

2536511 

## Theoretical Data-Driven MobilePosenet: Lightweight Neural Network for Accurate Calibration-Free 5-DOF Magnet Localization 

Wenxuan Xie , Yuelin Zhang , Jiwei Shan , _Graduate Student Member, IEEE_ , Hongzhe Sun , _Graduate Student Member, IEEE_ , Jiewen Tan , _Graduate Student Member, IEEE_ , and Shing Shin Cheng , _Member, IEEE_ 

**_Abstract_ —Permanent magnet tracking using the external sensor array is crucial for the accurate localization of wireless capsule endoscope robots. Traditional tracking algorithms, based on the magnetic dipole model and Levenberg-Marquardt (LM) algorithm, face challenges related to computational delays and the need for initial position estimation. More recently proposed neural network-based approaches often require extensive hardware calibration and real-world data collection, which are time-consuming and labor-intensive. To address these challenges, we propose MobilePosenet, a lightweight neural network architecture that leverages depthwise separable convolutions to minimize computational cost and a channel attention mechanism to enhance localization accuracy. Besides, the inputs to the network integrate the sensors’ coordinate information and random noise, compensating for the discrepancies between the theoretical model and the actual magnetic fields and thus allowing MobilePosenet to be trained entirely on theoretical data. Experimental eva** **<mark>luations conducted in a 90</mark>** **_<mark>×</mark>_** **<mark>90</mark>** **_<mark>×</mark>_** **<mark>80 mm workspace demon</mark> -** **<mark>strate that MobilePosenet exhibits excellent 5-DOF localization accuracy (1</mark>** <mark>.</mark> **<mark>54</mark>** **_<mark>±</mark>_** **<mark>1</mark>** <mark>.</mark> **<mark>03 mm and 2</mark>** <mark>.</mark> **<mark>24</mark>**<sup>◦</sup> **_<mark>±</mark>_** **<mark>1</mark>** <mark>.</mark> **<mark>84</mark>**<sup>◦</sup> **<mark>) and less inference time (0.9 ms) against state-of-the-art (SOTA</mark> ) methods trained on** **<mark>real-world data</mark> . Since network training relies solely on theoretical data, MobilePosenet can eliminate the hardware calibration and real-world data collection process, improving the generalizability of this permanent magnet localization method and the potential for rapid adoption in di** ff **erent clinical settings.** 

**_Index Terms_ —Deep learning, five-degree-of-freedom, permanent magnet tracking, pose estimation, sensor array.** 

### I. INTRODUCTION 

# **W** 

**W** IRELESSwidely adoptedcapsulenoninvasiveendoscopytool(WCE)for gastrointestinalhas become a examination [1], [2]. However, the autonomous movement of the capsule inside the body presents significant challenges for real-time tracking of its position and orientation. To address 

Received 16 October 2024; revised 16 April 2025; accepted 15 May 2025. Date of publication 18 June 2025; date of current version 24 June 2025. This work was supported in part by the Research Grant Council (RGC) of Hong Kong under Grant CUHK 14217822, Grant CUHK 14207823, and Grant AoE/E-407/24-N; in part by the Innovation and Technology Commission of Hong Kong under Grant ITS/225/23, Grant ITS/235/22, Grant ITS/224/23, and Grant MHP/096/22 and Multi-Scale Medical Robotics Center, InnoHK; and in part by The Chinese University of Hong Kong Direct Grant. The Associate Editor coordinating the review process was Dr. Octavian Postolache. _(Corresponding author: Shing Shin Cheng.)_ 

The authors are with the Department of Mechanical and Automation Engineering and T Stone Robotics Institute, The Chinese University of Hong Kong, Hong Kong (e-mail: sscheng@cuhk.edu.hk). 

Digital Object Identifier 10.1109/TIM.2025.3575965 

this, permanent magnet tracking has been proposed, which involves embedding a permanent magnet into the capsule and employing an external magnetic sensor array to locate it. This approach enables localization of the capsule within the body, enhancing the safety and efficiency of the treatment process. 

Modeling the magnetic field using numerical integration [3], [4] or the magnetic dipole model [5], [6] are two common approaches in permanent magnet tracking based on the sensor array. Numerical integration methods used to solve for sixdegree-of-freedom (DOF) poses are computationally intensive and nonconvex [3], and are thus not suitable for real-time tracking. In contrast, the magnetic dipole model simplifies the magnetic field calculation, enabling real-time tracking and retrieval of 5-DOF pose information [5], [6], [7], [8]. 

When combined with sensor measurements, the modeled magnetic field data can be used to estimate the pose of the permanent magnet. Among the various optimizationbased algorithms proposed to solve this inverse problem for pose estimation, the Levenberg–Marquardt (LM) algorithm is widely used due to its robust performance [5], [6], [7], [9]. However, as a local optimization method, LM [10] requires an initial position guess for iterative optimization. Deviations between the initial guess and the true position can degrade both the efficiency and accuracy of LM. To mitigate this issue, several methods, such as linear algorithms [6], particle swarm optimization (PSO) [4], and feedforward neural networks (FFNNs) [11], have been proposed to improve the initial guess. While PSO and FFNNs offer efficient initial guesses for the LM algorithm, they are computationally intensive. For instance, PSO can take up to 830 s to estimate the pose of a 6-DOF ring magnet [4], and FFNNs require extensive computational resources and training time. 

The computational complexity of traditional methods has prompted recent research to investigate neural network-based methods for addressing the challenges of permanent magnet tracking [3], [12], [13], [14]. Sebkhi et al. [12] developed a f <mark>ully automated 5-D positional stage, generating a dataset containing 1.7 million samples. However, the data collection process is labor-intensive and time-consuming, requiring almost 6 months for a dataset with 1 mm positional resolution.</mark> To alleviate the need for large datasets, Guo et al. [14] utilized a residual neural network (ResNet) to process triaxial 

> © 2025 The Authors. This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/ 

2536511 

IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. 74, 2025 

calib methods 

magnetometer data, interpreting the three axes as analogous to RGB channels, i.e., treating each axis as a separate color channel. They collected 5312 data points spanning six magnet poses for training and evaluation. Su et al. [3] proposed generating a synthetic dataset based on prior knowledge for model pretraining, followed by fine-tuning with a limited set of real-world samples. Nonetheless, fine-tuning with sparse real data introduces the risk of overfitting. 

To mitigate the problem that arises from gen ra từ real-data training or fine-tuning with limited real data, recent studies have shown that t <mark>heoretical data-driven</mark> neural dipole model networks offer a promising alternative by <mark>replacing real-world da</mark> ta collection with <mark>synthetic datasets generated from physical models.</mark> Qin et al. [11] proposed a hybrid approach combining a FFNN and an LM algorithm for magnet localization. In their method, the FFNN was trained on a simulation dataset of 20 691 samples generated via finite-element analysis to obtain an initial 6-D pose estimate of a cylindrical magnet. The LM algorithm then refined this estimate by solving the nonlinear inverse problem based on the magnetic dipole model. Similarly, Wu et al. [15] introduced a neural network-based approach that utilizes <mark>f</mark> i <mark>nite element method (FEM) simulations</mark> to generate synthetic training data, avoiding the inaccuracies associated with the magnetic dipole model at short distances. Their results demonstrate that FEM-based data enhance localization accuracy, particularly for nonspherical magnets. However, in real-world tracking scenarios, inevitable system calibration errors degrade input data quality and negatively impact the accuracy of MLP and CNN estimates. 

It is noteworthy that the localization accuracy of both the aforementioned <mark>LM-based and neural network-based methods is highly dependent on the quality of sensor dat</mark> a. Sensor noise can significantly degrade tracking accuracy, typically caused by manufacturing errors in the sensors and magnets. <mark>Several studies [9], [16], [17] have proposed methods to calibrate the noise resulting from these errors. However, these approaches are often complex and unsuitable for large-scal</mark> e clinical applications. Therefore, the development of magnetic tracking methods that can tolerate manufacturing errors without requiring calibration is necessary. 

To address challenges such as dependence on initial guesses, overfitting risks, and the requirement for real-world datasets and hardware calibration, we propose MobilePosenet, a lightweight neural ne <mark>twork designed for real-time perma</mark> - <mark>nent magnet tracking. MobilePosenet leverages dept</mark> hwise separable convolutions and a channel attention mechanism, significantly reducing computational costs while maintaining the ability to capture essential features. By incorporating the <mark>additional sensor location a</mark> nd artificially <mark>added random noise as part</mark> of the newly proposed input representation, MobilePosenet effectively mitigates discrepancies between the idealized magnetic dipole model and real-world magnetic fields, en <mark>abling the network to be trained exclusively on theoretical data generated from the magnetic dipole model with highly accurate localization result</mark> . Comprehensive experiments demonstrate that MobilePosenet achieves superior 5-DOF localization performance compared with state-of-the-art (SOTA) models trained on real-world data. 



Fig. 1. Coordinate system for magnet’s localization: _O_ 1 is the central position of the magnet, _H_ 0 is the direction of the magnet, _Ri_ is the distance between the _i_ th sensor and _O_ 1, _l_ is magnet length, and _d_ is magnet diameter. 

The main contributions of this work are summarized as follows. 

- 1) MobilePosenet is pro <mark>posed as a lightweight architecture for 5-DOF</mark> permanent magnet localization, integrating a channel attention mechanism and depthwise separable convolutions. The channel attention mechanism enhances accuracy by assigning higher weights to features most relevant for positioning through a reweighting process, while depthwise separable convolutions reduce computational cost by decomposing standard convolutions into depthwise and pointwise convolutions. These combined techniques result in improvement in positioning accuracy and execution speed compared to existing methods. 

- 2) A novel <mark>training me</mark> thod that utilizes only t <mark>he theoretical dataset generated by the ma</mark> gnetic dipole model is proposed for the first time. The network <mark>input combines theoretical magnetic f</mark> l <mark>ux densities with the coordinate information of triaxial magnetometers, with random noise a</mark> dded to compensate for discrepancies between the magnetic dipole model and the real magnetic field. This input methodology, integrated with a channel attention mechanism, eliminates the need for sensor array calibration and real-world data collection, enabling MobilePosenet to be easily adopted in different clinical settings. 

II. METHODS 

### _A. Magnetic Dipole Model_ 

In this work, an external sensor array is employed to locate a cylindrical permanent magnet. Previous research by Hu [18] demonstrates that when the distance between the permanent magnet and the sensor exceeds eight times the magnet’s radius, the magnetic <mark>dipole model can ap</mark> proximately simulate its magnetic field, thus enabling its 5-DOF pose estimation. 

Fig. 1 depicts a cylindrical permanent magnet in space with length _l_ , diameter _d_ , and surface magnetization _M_ . Let _Xi_ = ( _xi_ − _a_ , _yi_ − _b_ , _zi_ − _c_ )<sup>T</sup> denote the vector representing the relative position of a spatial point ( _xi_ , _yi_ , _zi_ ) with respect to the magnet’s center at ( _a_ , _b_ , _c_ ). The magnetic flux density at this point can be calculated as 



2536511 

XIE et al.: THEORETICAL DATA-DRIVEN MobilePosenet: LIGHTWEIGHT NEURAL NETWORK 

where _BT_ = (µ _r_ µ0π _r_<sup>2</sup> _lM_ )/4π, µ _r_ is the relative permeability of the medium, µ0 is the magnetic permeability of air, _r_ is the radius of the magnet, and _H_ 0 is the normalized vector representing the orientation of the magnet’s magnetism. 

Suppose there are _N_ sensors in the space and the _i_ th sensor is located at ( _xi_ , _yi_ , _zi_ ), the magnetic flux density at the _i_ th sensor location can be represented by 



where _Bix_ , _Biy_ , and _Biz_ represent the three components of the magnetic induction intensity of the point. Specifically, it is expressed as follows: 



where _Ri_ = (( _xi_ − _a_ )<sup>2</sup> + ( _yi_ − _b_ )<sup>2</sup> + ( _zi_ − _c_ )<sup>2</sup> )<sup>1/2</sup> . 

It is worth noting that the direction vector _H_ 0 does not correspond directly to the rotation angle. Suppose that the yaw angle is ϕ, the pitch angle is θ, and the roll angle is φ, then _H_ 0 can be calculated as follows: 



where _RB_ is the rotation matrix defined as follows: 



_B. Theoretical Data Training and Neural Network Input Design_ 

In this work, each sensor in the sensor array is a triaxial magnetometer. The measurements of the triaxial magnetometer can be considered as a <mark>three-channel (RGB)</mark> image pixel. Previous studies such as [3] and [14] utilize convolutional neural networks to capture correlations among adjacent sensors in the sensor array to predict the magnet’s pose. However, these studies rely solely on sensor array readings as input for the neural networks, typically requiring real-world datasets to train or fine-tune the models. As noted in [12], <mark>collecting realworld data for neural network training is a tim</mark> e-consuming task, making it difficult to sample these data at small enough intervals for a dense dataset. 

To address these challenges, our proposed MobilePosenet is trained exclusively using theoretical datasets generated from the magnetic dipole model. This approach avoids the challenges of training or fine-tuning the network using real data. However, in practical localization scenarios, the magnetic dipole model demonstrates inherent limitations in accurately simulating the magnetic fields generated by permanent magnets, primarily due to factors such as imperfections in sensor 

manufacturing, environmental noise, and nonuniform magnetization. Training neural networks solely on theoretical datasets risk overfitting, which may degrade localization performance. <mark>Therefore, to provide the neural network with more localization-relevant information and thereby improve accuracy, we draw inspiration from the use of sensor array positions in the LM algorithm. O</mark> ur neural network input comprises not only the theoretical magnetic flux densities of the magnetometers but also their corresponding positions, organized into two 3 _×_ 4 _×_ 4 tensors. 

- 1) Each magnetometer is considered as a three-channel pixel point, with the channels representing ( _Bx_ , _By_ , _Bz_ ). 

- 2) The position of each magnetometer is considered as a three-channel pixel point, with the channels representing the ( _X_ , _Y_ , _Z_ ) coordinates of the sensor. 

Given that <mark>16 magnetometers</mark> are used to form the sensor array, each component of the input is resized to 3 _×_ 4 _×_ 4. During model training, the two input components are concatenated to form a final input of size 6 _×_ 4 _×_ 4, where the six channels are denoted by ( _Bx_ , _By_ , _Bz_ , _X_ , _Y_ , _Z_ ). Besides, random noise is added to the three magnetic flux density channels, following a normal distribution _N_ (0, σ<sup>2</sup> ), where σ is th <mark>e standard deviation and σ = 1 µT.</mark> 

The proposed input methodology enables the neural network to capture critical information, including sensor coordinates and random noise, thereby enabling the network to recognize the dis <mark>crepancies between the theoretical model and real mag</mark> - netic fields. This approach mitigates the loss of information due to the absence of real-world data in the training or fine-tuning proce <mark>ss, thereby contributing to enhanced positioning accuracy. The potential limitation of the proposed</mark> thiếu xót <mark>input method is that it only accounts for unavoidable noise</mark> của <mark>introduced by the manufacturing and calibration processes,</mark> phương <mark>assuming a standard deviation of 1 µT.</mark> However, it does not pháp consider external magnetic field interference that may arise from ferromagnetic metals or actuation magnetic fields. Such interference can be mitigated by modeling the disturbance sources or employing additional convolutional neural networks designed to suppress external noise [3], [19], [20]. A more detailed discussion on this topic is provided in Section IV. 

### _C. Inverted Residual Attention Block_ 

To enhance the localization accuracy of the neural network while minimizing computational cost, depthwise separable convolution is used as the base layer of MobilePosenet while the channel attention mechanism is introduced to recalibrate the extracted features. As shown in Fig. 2, we leverage the inverted residual block introduced in [21] and integrate it with the squ <mark>eeze-and-excitation block (SEBlock)</mark> [22] to construct the inverted residual attention block (IRAB). This lightweight module forms the foundational block of MobilePosenet. The information flow is shown in Table I. 

In the IRAB, depthwise separable convolutions are employed to achieve a balance between computational efficiency and model performance. By significantly reducing the number of parameters (NPs) and operations, the network can achieve real-time tracking with only a minor loss in 

2536511 

IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. 74, 2025 



<!-- Start of picture text -->
H sensor array position Conv Conv | [raaprive H<br>>) Gr kh 8 tk] |S (fae H<br>H a fo} fre | |e B Relu (ut) H<br>H Say, H MobilePosenet<br>H i Sling 1 zero H{€<br>i<br>: concait | padding H a) lal le<br>H sor i EH EH E HO @ @ owe<br>i} i erco x 688) |HH soneu/| §]=| | ¥)alel |lz qal lal >|E <:® :mutiplySEBlockyy input<br>H P= OS<br>H iie usnn py H ZT 5 R em it |||3H] 3b |2 Re lu(x)=lu6(x}-min( m ax(03x)ax(0,x),6)<br>{ 2 Hoe) H otherwise\ 13] fa) |B = : (: toverted Residual<br>H a“ _ random noise H 5 3 x Attention Block<br>r= TK<br>i% (0,02),¢ = IT i s| 13] lz shortcut: skip connection<br>| magnetic dipole model y iH a} lat fe<br>Model Input Inverted Residual Attention Block<br><!-- End of picture text -->

Fig. 2. Structure and workflow of MobilePosenet: the network performs end-to-end predictions of the magnet’s pose. The input consists of sensor readings and coordinates, while the output consists of the magnet’s position and orientation. In the network architecture, ConvBnRelu represents a composite operation that includes a Conv2d layer, batch normalization, and the ReLU activation function. IRAB denotes an inverted residual attention block. AdaptiveAvgPool refers to an adaptive average pooling layer. Dwise indicates depthwise separable convolution. SEBlock is a channel attention mechanism that enhances feature extraction by reweighting the input channels. 

#### TABLE I 

INFORMATION FLOW OF IRAB 



<!-- Start of picture text -->
Input Operator Output<br>SS<br>kxhxw 1x1 conv2d, ReLU6 tkxhxw<br>thx hxw 3x3 dwise s =s , ReLU6 tk x (h/s) x (w/s)<br>tk x (h/s) x (w/s) — linear 1x1 conv2d Kk! x (h/s) x (w/s)<br>kx (h/s) x (w/s)  SEBlock k’ x (h/s) x (w/s)<br>Note: Inverted Residual Attention Block transforming from k to k” chann -<br>els, with stride s, and expansion factor #, dwise means depthwise separable<br>convolution.<br><!-- End of picture text -->

accuracy. To mitigate this loss, the SEBlock is introduced, as it adaptively adjusts the weights of feature channels, thereby enhancing localization accuracy. 

To analyze the computational efficiency of the IRAB, consider an input tensor _Li_ with dimensions _hi × wi × di_ , where _hi_ and _wi_ denote the spatial resolution, and _di_ represents the <mark>number of input channels. A convolutional kernel of size</mark> _<mark>k × k</mark>_ is applied to produce an output tensor _L j_ with dimensions _hi × wi × d j_ , where _d j_ is the number of output channels. The computational cost of a standard convolution operation is _hi × wi × di × d j × k_<sup>2</sup> . However, by decomposing the standard convolution into depthwise and pointwise convolutions, depthwise separable convolution reduces this cost to _hi × wi × di ×_ ( _k_<sup>2</sup> + _d j_ ). As a result, the computational cost of the IRAB is approximately (1/ _k_<sup>2</sup> ) of that of standard convolutions, significantly improving efficiency. 

While depthwise separable convolutions offer computational advantages, they introduce a slight reduction in accuracy. This is effectively compensated by the SEBlock, which dynamically adjusts ch <mark>annel-wise weights to</mark> emphasize features critical to permanent magnet localization while suppressing irrelevant or redundant information. By selectively enhancing significant features, the network can more effectively capture variations in the magnetic field and the position of the permanent magnet, thereby improving both robustness and generalization. Beyond 

TABLE II 

INFORMATION FLOW OF MOBILEPOSENET 



<!-- Start of picture text -->
[Anput | Operator Tt Te [Is |<br>6x4xd4 ZeroPad2d(2) - 6 - -<br>6x8xs8 ConvBnRelu - 32 1 1<br>32x 8x8 IRAB 1 16 1 1<br>16x 8x8 | IRAB 6] 32 }2/ 1<br>32x8x8 | IRAB 6| 64 | 2) 2<br>64x4x4 IRAB 6 | 128) 2) 1<br>128 x 4.x 4 | IRAB 1 | 256} 1 | 1<br>256 x 4x 4 | ConvBNReLU - 512} 1 1<br>512 x 4x4 | AdaptiveAvgPool2d(1,1) | - S12 | - -<br>512 1x1 | flatten - | 512] - | -<br>512x1 Linear - 3 - -<br>512 x1 Linear : 3 : :<br>Note: ConvBnRelu refers to a composite operation consisting of a Conv2d<br>layer, batch normalization, and the ReLU activation function. IRAB denotes an<br>inverted residual attention block. ”—” indicates that the layer has no associated<br>parameters. Each row in the table describes a sequence of one or more<br>identical layers, repeated n times. All layers within the same sequence have<br>the same number c of output channels. The first layer of each sequence has a<br>stride s, while all subsequent layers use a stride of 1. All spatial convolutions<br>employ 3 x 3 kernels. The expansion factor ¢ is consistently applied to the<br>input size, as shown in Table I.<br><!-- End of picture text -->

feature enhancement, the SEBlock also plays a vital role in mitigating the impact of noise in the input data. Its adaptive weighting mechanism enables the IRAB to prioritize highquality data, assigning greater importance to these regions during the localization process. By reducing the influence of noise and unreliable signals, this targeted weighting strategy further enhances localization accuracy, ensuring more reliable performance in practical applications. 

### _D. MobilePosenet_ 

MobilePosenet is an end-to-end model for permanent magnet pose estimation that maps the sensor array-related inputs directly to the position and orientation of the permanent magnet. This approach provides an efficient solution for 

2536511 

XIE et al.: THEORETICAL DATA-DRIVEN MobilePosenet: LIGHTWEIGHT NEURAL NETWORK 



Fig. 3. Experimental platform comprises a triaxial magnetometer array, a calibration board, a plastic magnet shell with a height of 10 mm and an inner diameter of 10 mm, and a cylindrical N35 permanent magnet with a diameter and height of 10 mm ( _BT_ = 8.18 _×_ 10<sup>−2</sup> T). The calibration board measures 100 _×_ 100 mm, with 15 mm between adjacent holes. The magnetometer array consists of 16 triaxial magnetometers arranged in a 4 _×_ 4 configuration. 

real-time magnet localization. Fig. 2 shows the workflow and network structure of MobilePosenet, with the information flow detailed in Table II. MobilePosenet employs a 6 _×_ 4 _×_ 4 input tensor ( _Bx_ , _By_ , _Bz_ , _X_ , _Y_ , _Z_ ), constructed using theoretical sensor readings from an array of 16 triaxial magnetometers, along with their corresponding sensor positions, to predict the pose of a permanent magnet. The model output comprises two 1 _×_ 3 tensors: one for the magnet’s position ( _a_ , _b_ , _c_ ) and the other for its orientation ( _m_ , _n_ , _p_ ). <mark>Zero-padding is applied to expand the input tensor to a size of 6</mark> _<mark>×</mark>_ <mark>8</mark> _<mark>×</mark>_ <mark>8, ensuring that the original sensor data remains centered. This operation preserves the integrity of the magnetic f</mark> i <mark>eld informati</mark> on and prevents edge sensor data loss during convolution, thereby enhancing the localization accuracy of the neural network. The padded input is subsequently processed by multiple IRABs. Depthwise separable convolution-based IRAB, coupled with the small input size, renders MobilePosenet an exceptionally lightweight framework. MobilePosenet contains 568 070 NPs and performs 10 565 104 floating-point operations (FLOPs). These computational costs represent only 20.2% and 50% of those reported in [3]. The lower computational overhead indicates that this network could be deployed for high-frequency localization across a broad range of devices with minimal computational hardware demand. 

### III. EXPERIMENTS 

### _A. Hardware Setup_ 

sensor cách nhau 3.33 cm 

As shown in Fig. 3, a 4 _×_ 4 magnetometer array was used to measure magnetic induction intensities. The array consists of 16 triaxial magnetometers (STMicroelectronics LIS3MDL), evenly mounted on a printed circuit board (PCB) with a spacing of <mark>(100/3) mm.</mark> This magnetometer array costs approximately $100, making it both affordable and widely deployable in most facilities or institutions that offer gastrointestinal endoscopy. A 100 _×_ 100 mm calibration board, featuring a grid of 49 positions with adjacent holes spaced 15 mm apart, was employed for precise placement of permanent magnets. Each magnet has a diameter and height of 10 mm, and its magn <mark>etic f</mark> l <mark>ux density</mark> _<mark>B</mark> T_ <mark>is 8.18</mark> _<mark>×</mark>_ <mark>10</mark><sup>−2</sup> <mark>T, a</mark> s 





















































































Fig. 4. Six magnet orientations correspond to the boundary values [−1, 1] of the orientation vector ( _m_ , _n_ , _p_ ). 

calculated using the method described in [9]. The distance between the magnetometer array and the calibration board is adjustable using plastic columns, allowing for flexible positioning to accommodate different experimental conditions. <mark>The magnetometers operate within a measurement range of</mark> _<mark>±</mark>_ <mark>1600 µT, and data acquisition is managed by a microcontroller unit (STMicroelectronics STM32F103RCT6), which transmits the collected data to a computer running Ubuntu 22.04. The computational setup includes an Intel i7-13700K CPU and an NVIDIA GeForce RTX 4090 GPU, providing substantial processing power for data analysis and model evaluation.</mark> 

### _B. Dataset and Data Preprocessing_ 

To illustrate the dataset collection process, we first define the workspace in which the dataset resides. The center of the sensor array is defined as the origin of the workspace, with coordinates (0, 0, 0). The sensor plane is defined as the origin plane of the _z_ -axis in this workspace. The variables _x_ , _y_ , and _z_ represent the distances of the center point of the permanent magnet from the origin along the respective axes. The rotation angles of the permanent magnet are defined in Section II-A. Since the magnetic dipole model inherently loses information about rotation around its own axis, the system operates in a 5-DOF workspace. <mark>The working volume of the system is def</mark> i <mark>ned as</mark> _<mark>x</mark>_ <mark>∈ [−45, 45] mm,</mark> _<mark>y</mark>_ <mark>∈ [−45, 45] mm,</mark> _<mark>z</mark>_ <mark>∈ [40, 120] mm, and ϕ, θ ∈ [−180, 180]</mark><sup>◦</sup> <mark>. T</mark> he horizontal ( _x_ - and _y_ -axes) range is determined by the distribution of holes in the calibration board. A recent work [14] demonstrates that the optimal height range for localization of a cylindrical magnet with dimensions of 10 _×_ 10 mm lies between <mark>48 and 118 mm.</mark> Therefore, a vertical range of 40–120 mm was selected. In this workspace, two datasets were collected: a theoretical dataset and a real datase <mark>t. The theoretical dataset was generated by the magnetic dipole model and used for model training, w</mark> hile the real dataset was collected on our self-made platform and used exclusively as the ground truth to evaluate the model’s performance and was not involved in the training process. 

The theoretical dataset was calculated based on the magnetic dipole model. Equations (3)–(5) are used to calculate the magnetic field intensity component ( _Bx_ , _By_ , _Bz_ ) at each sampling 

đo magnetic flux 

IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. 74, 2025 



2536511 

point. Learning-based magnetic tracking heavily depends on the size and resolution of the training dataset [3]. To ensure that the training space covers the actual working space, the sampling range of the dataset is defined as _x_ ∈ [−50, 50] mm, _y_ ∈ [−50, 50] mm, and _z_ ∈ [30, 130] mm, which slightly exceeds the actual workspace. Sampling points were uniformly distributed across this range, with a spacing of 5 mm. At each location, the rotation angles ϕ and θ were sampled at 30<sup>◦</sup> intervals. Additionally, random offsets up to _±_ 5 mm for position and _±_ 30<sup>◦</sup> for angle were applied to each sampling point. This sampling process was repeated three times for each point. Furthermore, to ensure that the boundary values of the orientation vector ( _m_ , _n_ , _p_ ) within the range [−1, 1] are included in the training dataset, we separately sampled six specific orientations, as shown in Fig. 4, without adding any offsets. Consequently, the theoretical dataset comprises 4 862 025 data points. <mark>From this dataset, 2% is randomly selected for the test set, another 2% for the validation set, and the remaining 96% for the training set.</mark> 

The <mark>real dataset was collected using the calibration board,</mark> with the sampling range corresponding to the workspace. The trừ đi <mark>recorded magnetic induction values in the real dataset were</mark> từ <mark>the sensor readings minus the geomagnetic f</mark> i <mark>eld measured</mark> trường <mark>when no magnet was nearby.</mark> Sampling planes were established đất along the _z_ -axis from 40 to 120 mm at 10 mm intervals. Each plane contains 49 uniformly distributed sampling points, with 15 mm spacing between adjacent points. The sensor readings were recorded for six magnet poses shown in Fig. 4 at each point. The real dataset consists <mark>of 2646 pieces of data.</mark> 

Notably, there are differences between real data and the đoạn corresponding theoretical data due to manufacturing errors, này inaccuracies of the magnetic dipole model, and sensor saturachưa tion. The standard devia <mark>tion of this dif</mark> f <mark>erences is</mark> _<mark>±</mark>_ <mark>15.27 µT,</mark> làm rõ with the maximum deviation of 647 µT at the height of 4 <mark>0 mm</mark> (primarily due to sensor saturation). 

### _C. Network Training Detail and Evaluation Metrics_ 

MobilePosenet was implemented using PyTorch 2.1 and CUDA 11.8. The model was trained using the Adam optimizer over 256 epochs. In the training process, differe <mark>nt random noises were added to the input sensor readings at eac</mark> h epoch. The initial learning rate was set to 1 _×_ 10<sup>−4</sup> , and t <mark>he cosine annealing schedule was employed to gradually reduce the learning rate to 0 over the course of training.</mark> The training was conducted on the training set of the theoretical dataset. The model weights corresponding to the best-performing epoch on the validation set <mark>were saved as the f</mark> i <mark>nal model weights.</mark> The batch size was set to <mark>3200 and the total training time was approximately 6.4 h.</mark> 

To better adjust the loss function and evaluate the localization performance of the model, we separately calculate the position error and orientation error. The position error _E p_ and the orientation error _EO_ <mark>can be expressed as follows:</mark> 



























































Fig. 5. Positioning errors at various heights. The height refers to the distance between the permanent magnet center and the sensor plane. 

where ( _ar_ , _br_ , _cr_ , _mr_ , _nr_ , _pr_ ) is the ground truth and ( _ap_ , _bp_ , _cp_ , _mp_ , _np_ , _pp_ ) is the predicted value. The orientation error can also be expressed as an angle error, calculated using 2 sin<sup>−1</sup> (( _EO_ /2)). In the subsequent evaluations, the angle error in this form is reported instead since it is a more commonly used representation. 

Considering that the working space size is defined as 90 _×_ 90 _×_ 80 mm, the data distribution of position and orientation is different. To better balance the network’s ability to predict the position and orientation of permanent magnets, we add weight β to _E p_ to balance the loss proportion. The final training loss function is shown as follows: 



<mark>After model training, we evaluated the model on the test set of the theoretical dataset. The trained model achieved a mean position error of 0.88 mm and an angle error of 0.79</mark><sup>◦</sup> <mark>.</mark> 

test data gen trước rồi mới test data thực 

_D. Evaluation of Localization Accuracy_ 

This section evaluates the localization accuracy of MobilePosenet usin <mark>g the real dataset as the ground truth.</mark> To demonstrate the effectiveness of the proposed approach, we <mark>compared its performance against the LM algorithm by an</mark> a- lyzing locali <mark>zation errors on the same datasets.</mark> Considering that the accuracy and <mark>convergence of the LM algorithm is sensitive to the initial guess, two sets of biases were appl</mark> ied to the ground truth as the initial guesses for the LM algorithm, as detailed below. 

Bench với thuật toán LM 

1) _bias1_ : (3 mm, 3 mm, –3 mm, 0.2, –0.2, 0.2). 

2) _bias2_ : (20 mm, –20 mm, 20 mm, 0.3, 0.3, –0.3). 

Fig. 5 presents the localization errors of MobilePosenet at various heights in the real dataset. The average position error across all samples is 1.54 _±_ 1.03 mm, with an angle error of 2.24<sup>◦</sup> _±_ 1.84<sup>◦</sup> . In contrast, the LM algorithm, with an initial position bias of bias1, exhibits an average position error of 2.76 _±_ 1.58 mm and an angle error of 5.18<sup>◦</sup> _±_ 2.70<sup>◦</sup> . When initialized with bias2, the LM algorithm results in an average position error of 2.88 _±_ 1.73<sup>◦</sup> mm and an angle error of 5.28<sup>◦</sup> _±_ 2.91<sup>◦</sup> . In addition to larger errors, the LM algorithm also encountered localization failures, which were not observed with MobilePosenet-based localization. These findings clearly demonstrate the superior performance 

2536511 

XIE et al.: THEORETICAL DATA-DRIVEN MobilePosenet: LIGHTWEIGHT NEURAL 

NETWORK 

LM fail 

TABLE III 

COMPUTATION LATENCY OF DIFFERENT METHODS 



of MobilePosenet, emphasizing its robustness in avoiding localization failures and achieving higher accuracy even when trained exclusively on theoretical datasets. 

<mark>Notably, the LM-based algorithm exhibited localization failures under conditions of signif</mark> i <mark>cant initial position bias or substantial input noise. With an initial position bias of bias2, 17.3% of the 2646 samples in the real dataset failed to localize, with the failure rate rising to 46.9% at the height of 40 mm in the near f</mark> i <mark>eld. These results highlight the limitations of the LM algorithm, li</mark> kely due to significant inaccuracies in the magnetic dipole model when simulating the near field of permanent magnets [23]. I <mark>n the near f</mark> i <mark>eld, positioning errors are relatively high due to the limited f</mark> i <mark>tting capabilities of the magnetic dipole model and the occurrence of sensor saturation.</mark> As the permanent magnet moves away from the sensor array, the saturation effect diminishes, thereby reducing the simulation error of the magnetic dipole model and consequently decreasing the positioning error. However, as the permanent magnet continues to move farther from the sensor array, the positioning error increases again due to the limitations of the sensors’ effective working range and measurement capabilities. 



### _E. Evaluation of Localization Latency_ 

The execution speed of the positioning algorithm is a critical metric in real-world applications. To evaluate the positioning latency of MobilePosenet, we selected a test sample at each height and conducted positioning tests, comparing the latency with that of the LM algorithm at the corresponding point. Since the positioning latency of the LM algorithm is highly dependent on the initial guess, we introduced two sets of biases as initial guesses for the LM algorithm and recorded the corresponding latency for each. The biases used are the same as described in Section III-D. 

Table III shows that the computational latency of MobilePosenet is significantly lower than that of the LM algorithm. This is because MobilePosenet is a lightweight network based on depth separable convolution, allowing it to run efficiently on various devices without requiring an initial guess or iterative computation. In contrast, the LM algorithm is an 

iteration-based optimization method, and its computational delay depends heavily on both the initial guess and the noise level in the input data. When there is a substantial deviation in the initial guess or a high noise level, the execution time of the LM algorithm can increase significantly, and it may even fail to converge. In this experiment, the computational latency of the LM algorithm is higher than that reported in other studies, such as in [3] and [14]. This discrepancy may arise because the convergence rate of this algorithm is sensitive to noise levels. In this study, the input data were not calibrated, resulting in a higher noise level, which consequently increased the computation delay. 

### _F. Comparison With Existing Approaches_ 

Tracking and localizing permanent magnets is a highly specialized task. In addition to the algorithms, tracking performance is influenced by hardware differences across tracking systems, such as the size of the permanent magnets, calibration procedures, and manufacturing tolerances. Therefore, comparing errors and execution times across methods holds limited significance without replicating the hardware systems and datasets. To provide a more comprehensive comparison, we have compiled Table IV, which contrasts the implementation requirements and reported performance of our tracking system with several existing SOTA tracking systems and algorithms. <mark>Additionally, we reproduced multiple baseline algorithms to compare their performance with MobilePosenet within our tracking system. To ensure a fair comparison, all model inputs were integrated with coordinate information, and dif</mark> f <mark>erent models were trained under the same conditions.</mark> 

Table IV summarizes the implementation requirements and reported performance of various tracking systems. Traditional methods, such as the LM algorithm, rely on accurate initial point estimation and low-noise data, often requiring extensive hardware calibration to achieve high-precision positioning. Similarly, existing neural network-based approaches typically depend on <mark>real-world datasets for training or f</mark> i <mark>ne-t</mark> uning, which can be time-consuming and highly dependent on system-specific calibration, potentially limiting their scalability in large-scale applications. In contrast, permanent magnet tracking using MobilePosenet overcomes these limitations by enabling high-precision and high-frequency positioning in an end-to-end manner, without the need for hardware calibration or real data training. The simple implementation conditions of MobilePosenet allow for the generation of diverse training datasets within minutes, thereby making it adaptable to various positioning spaces and systems. 

Even without considering its flexible and simple implementation, MobilePosenet demonstrates significant advantages in both accuracy and speed. It achieves a position error of (1.54 mm) and an angle error of 2.24<sup>◦</sup> , with an execution time of only 0.91 ms, making it <mark>the fastest amo</mark> ng all current SOTA methods. In terms of positioning accuracy, while ResN <mark>et-LM slightly outperforms MobilePosenet in both position error (0.9 mm) and angle error (1.51</mark><sup>◦</sup> <mark>),</mark> its execution time is significantly longer (79.3 ms), and it is also limited to predicting only six specific poses of the magnet due to the dataset constraints. Similarly, DNN achieves a position 

2536511 

IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. 74, 2025 

TABLE IV 

COMPARISON OF DIFFERENT LOCALIZATION SYSTEMS 



error of 1.4 mm, but it is restricted to only 3-DoF position estimation. Compared to AMagposenet, MobilePosenet offers 5-DoF tracking with a lower position error and execution time. <mark>Although Amagposenet shows a slight improvement in angle error, this advantage is primarily due to f</mark> i <mark>ne-tuning on real-world data. H</mark> owever, the inherent noise randomness in such data increases the risk of overfitting, which can degrade localization accuracy in other regions of the workspace [19]. In contrast to SAE-LIN, MobilePosenet achieves comparable localization accuracy while significantly reducing execution time, without requiring real-world data or complex regional ensemble architectures. Unlike traditional methods, such as linear algorithms and the LM algorithm, MobilePosenet does not require an initial guess or calibration, yet still delivers superior accuracy, making it a more reliable and efficient solution for permanent magnet tracking applications. Although the workspace employed in this study is relatively limited, MobilePosenet is trained exclusively on theoretical data, enabling the flexible adaptation of the sampling range to accommodate larger workspaces. 

For a fair comparison independent of hardware systems, MobilePosenet was evaluated against baseline algorithms selected from commonly used SOTA convolutional neural network models. The selection process was guided by the fact that our input size, 6 _×_ 4 _×_ 4, differs significantly from the standard input size of 3 _×_ 244 _×_ 244, making more complex networks inefficient for processing such small inputs in the magnetic tracking task. Among the selected baseline methods, MobilePosenet achieves the smallest position and angle errors. Although EfficientNet employs a similar channel attention mechanism and utilizes neural architecture search (NAS), its position error (2.12 mm) and angle error (2.59<sup>◦</sup> ) are higher. While MobileNetV2 offers slightly faster execution times (0.61 ms), MobilePosenet provides superior accuracy, making it the most effective solution for real-time localization tasks. 

Additionally, by reproducing and evaluating various baseline algorithms within our system, valuable insights were obtained about the performance of neural networks in tracking permanent magnets. Our experiments reveal that increasing 

the NPs does not necessarily result in improved localization accuracy, as evidenced by the lower performance of VGG11 compared to AlexNet. Notably, the inclusion of residual connections, as implemented in ResNet, significantly enhances localization accuracy, outperforming conventional convolutional neural networks that lack such connections. While replacing standard convolutions with depthwise separable convolutions reduces both computational costs and execution time, it also leads to a noticeable decline in tracking accuracy. For instance, ResNet achieves superior localization performance compared to MobileNetV2, despite having an execution time of approximately five times longer. Furthermore, the incorporation of attention mechanisms substantially improves localization accuracy, with only a marginal increase in delay. Although EfficientNet shares architectural similarities with MobilePosenet, such as inverted residual block and SEBlock, its slight decrease in accuracy suggests that the inclusion of dropout layers may not contribute to better generalization performance. 

### _G. Ablation Study_ 

To assess the contribution of each component in our model, we conducted a series of ablation experiments. Additionally, we explored the application of the attention mechanism by introducing the convolutional block attention module (CBAM) [26] to replace the original channel attention mechanism, SEBLock. This modification is intended to determine whether the inclusion of a spatial attention mechanism could more effectively capture the correlations between different sensors, thereby further enhancing the model’s localization capabilities. All modifications were trained under identical conditions to ensure a fair comparison. 

Table V shows the results of the ablation study. The baseline model has a position error of 1.54 _±_ 1.03 mm and an angle error of 2.24<sup>◦</sup> _±_ 1.84<sup>◦</sup> on the real dataset. To assess the impact of random noise in the input on model training, we removed the noise added to the sensor readings during model training. As a result, the position error increased to 1.88 _±_ 1.26 mm, while the angle error rose to 2.45<sup>◦</sup> _±_ 2.18<sup>◦</sup> . Furthermore, 

2536511 

XIE et al.: THEORETICAL DATA-DRIVEN MobilePosenet: LIGHTWEIGHT NEURAL NETWORK 

TABLE V 

ABLATION STUDY ON MODEL PERFORMANCE 

among the 294 samples at the height of 120 mm, position errors exceeding 10 mm were observed for the first time, with a total of three such instances. These results confirm that incorporating random noise during training enhances the model’s robustness, especially when the magnet is positioned farther from the sensor array. 

Beyond the impact of noise, we further investigated the importance of key <mark>input features and model components.</mark> When the coordi <mark>nate information of ea</mark> ch sensor in the array is bỏ tọa độ excluded from the input, the position error increases to 1.94 _±_ sensors 1.52 mm, and the angle error rises to <mark>2.69</mark><sup>◦</sup> _<mark>±</mark>_ <mark>2.37</mark><sup>◦</sup> <mark>. T</mark> hese ra khỏi results underscore the significance of coordinate information input in enhancing the model’s localization capabilities. The channel attention mechanism plays a critical role in enhancing the neural network’s ability to accurately locate permanent magnets. After removing the SEBlock, the positional error increases to 2.97 _±_ 2.62 mm, and the angle error increases to 3.41<sup>◦</sup> _±_ 3.36<sup>◦</sup> . When the SEBlock was replaced with the CBAM module to incorporate spatial attention, the model with the CBAM module achieved a positional error of 1.52 _±_ 1.05 mm and an angle error of 2.32<sup>◦</sup> _±_ 1.87<sup>◦</sup> , with a total training time of approximately 12.2 h. These results indicate that incorporating spatial attention (CBAM) does not enhance localization accuracy but significantly increases computational cost, doubling the training time. This may be attributed to the limited effectiveness of spatial attention when applied to small input dimensions (6 _×_ 4 _×_ 4). Given these results, SEBlock was selected over CBAM as the attention mechanism in the final model. 

_H. Evaluation of System Robustness and Long-Term Stability_ 

To evaluate the robustness of our proposed MobilePosenetbased localization method in possible clinical applications, we used multiple s <mark>ilicone tubes with diferent</mark> f <mark>outer</mark> diameters to cover the magnet to simulate the obscuration of the magnet by tissues with different thicknesses. All silicone tubes have an inner diameter of 10 mm and an outer diameter ranging from 12 to 16 mm. The top and bottom of each tube are sealed with silicone patches. In addition, considering that the magnetic sensor array may have sensitivity drift and the permanent magnet may be oxidized, resulting in changes in the magnetic constant, we collected and evaluated a new occlusion dataset after exposing the magnetic sensor array and permanent magnet <mark>to air for about 4 mont</mark> hs. The new localization experiment can also be used to evaluate the overall long-term stability of the localization system, as well as its repeatability and robustness against measurement uncertainty. 

Fig. 6. Positioning error at various heights when the permanent magnet is obstructed by silicone tubes of different thicknesses. The height refers to the distance between the center of the permanent magnet and the sensor plane. 

The results provide quantitative insights into the robustness and long-term consistency of the proposed localization system in clinical applications. 

The sealed magnet was placed in a newly made plastic shell for testing, allowing us to obtain the ground truth of the position of the magnet after it was wrapped by the silicone tube. Using the hardware platform described in Section III-A, we collected five datasets, each corresponding to a different silicone tube outer diameter. Each dataset contained 2646 sample points, and the positioning errors at various heights were compared with those obtained without silicone tube coverage. The positioning errors for <mark>all six datasets</mark> are shown in Fig. 6. The experimental results of the group without silicone tube obstruction are from the dataset collected 4 months ago, and the remaining results are evaluated using the newly collected dataset. No significant changes in the error distribution were observed across any experimental group when compared to the results obtained without silicone tube coverage. The average positioning errors for the five siliconecovered groups were approximately 1.65 _±_ 1.03 mm and 2.24<sup>◦</sup> _±_ 1.83<sup>◦</sup> . The differences in positioning errors among groups with different silicone tube thicknesses were less than 0.05 mm at various heights, demonstrating that the thickness of the silicone tube surrounding the magnet does not impact localization accuracy. This result aligns with our expectations, as the magnetic permeability of silicone is similar to that of human tissue and close to that of air. 

Compared to the results obtained 4 months earlier, the average positioning error in the new dataset increased by approximately 0.12 mm, while the angular error remained unchanged. The slight increase in positioning error may be due to the long time interval between data collections, during which the magnetic sensor array and permanent magnet could have undergone sensitivity drift or slight oxidation from prolonged exposure to humid air [27]. Given that the average positioning error increased by only 0.12 mm, our results indicate that the proposed localization system and method exhibit long-term stability and robustness to measurement uncertainty, further supporting the potential clinical applicability of our proposed positioning method. 

### IV. DISCUSSION 

Beyond clinical applications, permanent magnet tracking is widely used in finger tracking [28], AGV navigation [29], 

2536511 

IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. 74, 2025 

and stylus positioning [30], with varying requirements for workspace, magnet size, and accuracy. The proposed lowlatency tracking method, based on MobilePosenet and a sensor array, is trained using magnetic dipole model data and applies to any scenario where the magnet behaves as a dipole and its field is detectable. However, sensor-array-based localization relies on detecting magnetic fields, making it ineffective for very small magnets (e.g., 1 mm in diameter and length) due to weak signals. In addition, since magnetic field strength decreases with the cube of distance, the method’s effective range depends on magnet size, while accuracy is influenced by training data density and sensor precision. Within a suitable range, it enables high-precision real-time pose estimation, demonstrating potential for advanced magnetic localization and control applications. 

Since MobilePosenet is based on neural networks, its training process leverages a large amount of theoretically generated noisy data. This enables the model to automatically extract stable features from the magnetic field data and offers a certain tolerance to magnetic noise in real data caused by manufacturing errors. In future work, inspired by [19] and [20], we will explore integrating an anomaly detection and measurement recovery mechanism along with a regional ensemble learning strategy into MobilePosenet to further improve the system’s robustness in complex interference environments. In particular, the autoencoder-based anomaly detection and neighborhood data compensation method proposed in [20] can correct abnormal data after detecting interference regions. This effectively reduces the adverse effects of external magnetic interference (such as hard-iron and soft-iron effects) on localization accuracy. Moreover, the regional modeling and multiview fusion strategy used in [19] refines the magnetic field characteristics through local modeling and employs a gating mechanism to dynamically suppress noise, thereby enhancing the system’s resilience to noise and interference. By combining these advanced techniques, we expect to further improve the positioning accuracy and robustness of MobilePosenet in multiinterference environments, while maintaining its real-time performance. This will provide a more reliable solution for high-precision magnetic localization and control applications 

Furthermore, it is important to note that MobilePosenet relies on magnetic field signals generated by the permanent magnets embedded in the WCE for positioning. Although clinical studies have confirmed the safety of WCEs with builtin permanent magnets for gastric examinations [31], potential electromagnetic compatibility (EMC) issues must still be carefully evaluated in clinical environments. These issues include possible interference with nearby medical or electronic equipment, as well as risks associated with static magnetic fields for patients with implanted devices such as pacemakers. Future system designs should comply with established EMC standards and incorporate patient safety assessment protocols to ensure safe operation in medical settings. 

### V. CONCLUSION 

In this study, we propose MobilePosenet, a lightweight neural network architecture optimized for real-time permanent magnet localization, with accurate, robust, and 

efficient localization capabilities. MobilePosenet leverages depthwise separable convolutions and a channel attention mechanism to achieve high localization accuracy with minimal computational cost. To the best of our knowledge, this is the first application of these techniques in the context of permanent magnet tracking, enabling a significant reduction in computational complexity without compromising performance. Additionally, we introduce a novel training strategy based entirely on theoretical datasets generated from the magnetic dipole model, eliminating the need for labor-intensive real-world data collection and sensor calibration, thereby enhancing the system’s practicality and flexibility. The innovative input method, which combines triaxial magnetometer readings with coordinate information and random noise, effectively compensates for discrepancies between idealized models and real magnetic fields, thereby enhancing localization accuracy. However, despite its excellent performance, the primary limitation of MobilePosenet lies in its exclusive reliance on theoretically generated training data. This may lead to a decrease in positioning accuracy in real environments with complex magnetic disturbances that cannot be captured by the dipole model with random noise. Addressing this challenge will be a focus of future work, where we aim to adopt hybrid training strategies that combine real and synthetic data to improve robustness under practical conditions. 

### ACKNOWLEDGMENT 

The content is solely the responsibility of the authors and does not reflect the views of the sponsors. 

### REFERENCES 

- [1] W. Weitschies, R. K¨otitz, D. Cordini, and L. Trahms, “High-resolution monitoring of the gastrointestinal transit of a magnetically marked capsule,” _J. Pharmaceutical Sci._ , vol. 86, no. 11, pp. 1218–1222, Nov. 1997. 

- [2] T. D. Than, G. Alici, H. Zhou, and W. Li, “A review of localization systems for robotic endoscopic capsules,” _IEEE Trans. Biomed. Eng._ , vol. 59, no. 9, pp. 2387–2399, Sep. 2012. 

- [3] S. Su, S. Yuan, M. Xu, H. Gao, X. Yang, and H. Ren, “AMagPoseNet: Real-time 6-DoF magnet pose estimation by dual-domain few-shot learning from prior model,” _IEEE Trans. Ind. Informat._ , vol. 19, no. 9, pp. 9722–9732, Sep. 2023. 

- [4] S. Song et al., “6-D magnetic localization and orientation method for an annular magnet based on a closed-form analytical model,” _IEEE Trans. Magn._ , vol. 50, no. 9, pp. 1–11, Sep. 2014. 

- [5] C. R. Taylor, H. G. Abramson, and H. M. Herr, “Low-latency tracking of multiple permanent magnets,” _IEEE Sensors J._ , vol. 19, no. 23, pp. 11458–11468, Dec. 2019. 

- [6] C. Hu, M. Li, S. Song, W. Yang, R. Zhang, and M. Q.-H. Meng, “A cubic 3-axis magnetic sensor array for wirelessly tracking magnet position and orientation,” _IEEE Sensors J._ , vol. 10, no. 5, pp. 903–913, May 2010. 

- [7] Y. Xu, K. Li, Z. Zhao, and M. Q.-H. Meng, “A novel system for closedloop simultaneous magnetic actuation and localization of WCE based on external sensors and rotating actuation,” _IEEE Trans. Autom. Sci. Eng._ , vol. 18, no. 4, pp. 1640–1652, Oct. 2021. 

- [8] S. Song et al., “Integrated design and decoupled control of anchoring and drug release for wireless capsule robots,” _IEEE_ / _ASME Trans. Mechatronics_ , vol. 27, no. 5, pp. 2897–2907, Oct. 2022. 

- [9] S. Su et al., “Investigation of the relationship between tracking accuracy and tracking distance of a novel magnetic tracking system,” _IEEE Sensors J._ , vol. 17, no. 15, pp. 4928–4937, Aug. 2017. 

- [10] X. Wang, M. Q.-H. Meng, and C. Hu, “A localization method using 3-axis magnetoresistive sensors for tracking of capsule endoscope,” in _Proc. Int. Conf. IEEE Eng. Med. Biol. Soc._ , Aug. 2006, pp. 2522–2525. 

2536511 

XIE et al.: THEORETICAL DATA-DRIVEN MobilePosenet: LIGHTWEIGHT NEURAL NETWORK 

- [11] Y. Qin, B. Lv, H. Dai, and J. Han, “An hFFNN-LM based real-time and high precision magnet localization method,” _IEEE Trans. Instrum. Meas._ , vol. 71, pp. 1–9, 2022. 

- [12] N. Sebkhi et al., “A deep neural network-based permanent magnet localization for tongue tracking,” _IEEE Sensors J._ , vol. 19, no. 20, pp. 9324–9331, Oct. 2019. 

- [13] B. Lv, Y. Chen, H. Dai, S. Su, and M. Lin, “PKBPNN-based tracking range extending approach for TMR magnetic tracking system,” _IEEE Access_ , vol. 7, pp. 63123–63132, 2019. 

- [14] P. Guo, H. Dai, Q. Yang, Q. Huang, and H. Yao, “An improved magnetic tracking approach based on ResNet-LM fusion algorithm,” _IEEE Trans. Instrum. Meas._ , vol. 71, pp. 1–10, 2022. 

- [15] M. Wu, T. Langerak, O. Hilliges, and J. Zarate, “Using synthetic data in supervised learning for robust 5-DoF magnetic marker localization,” _IEEE Trans. Magn._ , vol. 60, no. 8, pp. 1–11, Aug. 2024. 

- [16] M. Li, S. Song, C. Hu, W. Yang, L. Wang, and M. Q.-H. Meng, “A new calibration method for magnetic sensor array for tracking capsule endoscope,” in _Proc. IEEE Int. Conf. Robot. Biomimetics (ROBIO)_ , Dec. 2009, pp. 1561–1566. 

- [17] C. Hu, M. Meng, and M. Mandal, “The calibration of 3-axis magnetic sensor array system for tracking wireless capsule endoscope,” in _Proc. IEEE_ / _RSJ Int. Conf. Intell. Robots Syst._ , Oct. 2006, pp. 162–167. 

- [18] C. Hu, “Localization and orientation system for robotic wireless capsule endoscope,” Ph.D. dissertation, Dept. Elect. Comput. Eng., Univ. Alberta, Alberta, AB, Canada, 2006. 

- [19] J. Lu, F. Zheng, S. Zhang, Q. Yang, and C. Hu, “Regional perceptrons ensemble for permanent magnetic localization,” _IEEE Sensors J._ , vol. 24, no. 17, pp. 28056–28067, Sep. 2024. 

- [20] S. Su, H. Lan, H. Gao, H. Dai, and H. Ren, “Enhancing anti-interference of magnetic tracking: A MagRobustNet-based framework with selfsupervised anomaly detection and measurements recovery,” _IEEE Trans. Ind. Informat._ , vol. 21, no. 1, pp. 781–790, Jan. 2025. 

- [21] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in _Proc. IEEE_ / _CVF Conf. Comput. Vis. Pattern Recognit._ , Jun. 2018, pp. 4510–4520. 

- [22] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in _Proc. IEEE_ / _CVF Conf. Comput. Vis. Pattern Recognit._ , Jun. 2018, pp. 7132–7141. 

- [23] F. Y. Wu, S. Foong, and Z. Sun, “A hybrid field model for enhanced magnetic localization and position control,” _IEEE_ / _ASME Trans. Mechatronics_ , vol. 20, no. 3, pp. 1278–1287, Jun. 2015. 

- [24] C. Hu, M. Q.-H. Meng, and M. Mandal, “A linear algorithm for tracing magnet position and orientation by using three-axis magnetic sensors,” _IEEE Trans. Magn._ , vol. 43, no. 12, pp. 4096–4101, Dec. 2007. 

- [25] M. Wang, Q. Shi, S. Song, and M. Q.-H. Meng, “A novel magnetic tracking approach for intrabody objects,” _IEEE Sensors J._ , vol. 20, no. 9, pp. 4976–4984, May 2020. 

- [26] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: Convolutional block attention module,” in _Proc. Eur. Conf. Comput. Vis._ , Sep. 2018, pp. 3–19. 

- [27] M. Ribeiro, M. Ortner, and D. Spitzer, “Long term magnetic sensor system reliability assessment,” in _Proc. 11th Int. Conf. Sens. Technol. (ICST)_ , Dec. 2017, pp. 1–6. 

- [28] Y. Ma, Z.-H. Mao, W. Jia, C. Li, J. Yang, and M. Sun, “Magnetic hand tracking for human-computer interface,” _IEEE Trans. Magn._ , vol. 47, no. 5, pp. 970–973, May 2011. 

- [29] S. Su, H. Dai, S. Cheng, and Z. Chen, “Improved magnetic guidance approach for automated guided vehicles by error analysis and prior knowledge,” _IEEE Trans. Intell. Transp. Syst._ , vol. 22, no. 11, pp. 6843–6852, Nov. 2021. 

- [30] R.-H. Liang, K.-Y. Cheng, C.-H. Su, C.-T. Weng, B.-Y. Chen, and D.-N. Yang, “GaussSense: Attachable stylus sensing using magnetic sensor grid,” in _Proc. 25th Annu. ACM Symp. User Interface Softw. Technol._ , Oct. 2012, pp. 319–326. 

- [31] Z. Liao et al., “Feasibility and safety of magnetic-controlled capsule endoscopy system in examination of human stomach: A pilot study in healthy volunteers,” _J. Interventional Gastroenterol._ , vol. 2, no. 4, pp. 155–160, Oct. 2012. 

