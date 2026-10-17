## AMagPoseNet: Real-Time Six-DoF Magnet Pose Estimation by Dual-Domain Few-Shot Learning From Prior Model

Shijian Su , Sishen Yuan, Mengya Xu , Huxin Gao , Xiaoxiao Yang, and Hongliang Ren [URL 🔗](https://orcid.org/0000-0002-4338-7079)

Abstract—Traditional magnetic tracking approaches based on mathematical models and optimization algo- rithms are computationally intensive, depend on initial guesses, and do not guarantee convergence to a global op- timum. Although fully supervised data-driven deep learning can solve the above issues, the demand for a comprehen- sive dataset hampers its applicability in magnetic tracking. Thus, we propose an annular magnet pose estimation net- work (called AMagPoseNet) based on dual-domain few-shot learning from a prior mathematical model, which consists of two subnetworks: PoseNet and CaliNet. PoseNet learns to estimate the magnet pose from the prior mathemati- cal model, and CaliNet is designed to narrow the gap be- tween the mathematical model domain and the real-world domain. Experimental results reveal that the AMagPoseNet outperforms the optimization-based method regarding

localization accuracy (1.87±1.14 mm, 1.89±0.81◦), robust- ness (nondependence on initial guesses), and computa-

tional latency (2.08±0.02 ms). In addition, the six-degree-of- freedom pose of the magnet could be estimated when dis- criminative magnetic field features are provided. With the assistance of the mathematical model, the AMagPoseNet

Manuscript received 17 August 2022; revised 1 December 2022; ac- cepted 25 December 2022. Date of publication 4 January 2023; date of current version 24 July 2023. This work was supported in part by the Hong Kong Research Grants Council through the Collaborative Re- search Fund under Grant CRF C4063-18G and the General Research Fund under Grant GRF 14216022, in part by the Chinese University of Hong Kong Mainland Postdoctoral Sponsorship Program, and in part by the Shenzhen–Hong Kong–Macau Technology Research Program (Type C) under Grant 202108233000303. Paper no. TII-22-3526. (Cor- responding author: Hongliang Ren.)

This work involved human subjects or animals in its research. Ap- proval of all ethical and experimental procedures and protocols was granted by the Animal Ethics Committee of Qilu Hospital of Shangdong University under Application No. Dwll-2021-021, and performed in line with the Guiding Principles in the Care and Use of Animals.

Shijian Su and Sishen Yuan are with the Department of Electronic Engineering, Chinese University of Hong Kong, Hong Kong (e-mail: shijiansu@foxmail.com; ssyuan@link.cuhk.edu.hk). [URL 🔗](mailto:ssyuan@link.cuhk.edu.hk)

Mengya Xu and Huxin Gao are with the Department of Biomedical Engineering, College of Design and Engineering, National University of Singapore, Singapore 117575, and also with the Department of Electronic Engineering, Chinese University of Hong Kong, Hong Kong (e-mail: e0320813@u.nus.edu). [URL 🔗](mailto:e0320813@u.nus.edu)

Xiaoxiao Yang is with the Department of Gastroenterology, Qilu Hos- pital of Shandong University, Jinan 250012, China (e-mail: yangxiaox- iao@mail.sdu.edu.cn). [URL 🔗](mailto:yangxiaoxiao@mail.sdu.edu.cn)

Hongliang Ren is with the Department of Electronic Engi- neering, Chinese University of Hong Kong, Hong Kong (e-mail: hlren@ee.cuhk.edu.hk). [URL 🔗](mailto:hlren@ee.cuhk.edu.hk)

Color versions of one or more figures in this article are available at https://doi.org/10.1109/TII.2022.3233675. [URL 🔗](https://doi.org/10.1109/TII.2022.3233675)

Digital Object Identifier 10.1109/TII.2022.3233675 [URL 🔗](https://doi.org/10.1109/TII.2022.3233675)

This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see http://creativecommons.org/licenses/by/4.0/

requires only a few real-world samples and has excellent performance, showing great potential for practical biomed- ical and industrial applications.

Index Terms—Deep learning, magnetic tracking, mathe- matical model, neural network, six-degree-of-freedom (DoF) pose estimation.

## I. INTRODUCTION

P ERMANENT magnetic tracking is an established tech- [URL 🔗](#page-0)

nique for locating medical instruments inside the human

body [1]. Owing to its no line-of-sight limitations, low cost, and no power consumption for tracking targets, the magnetic track- ing technique was also employed for vehicle pose estimation in industrial applications [2], [3]. [URL 🔗](#page-0)

However, only three-degree-of-freedom (DoF) position and two-DoF orientation can be obtained when using the radical magnetized cylindrical magnet [4] or spherical magnet [5] due to their symmetric magnetic field distribution along the mag- netic moment direction. Nonrotational symmetric magnets have been suggested to avoid such a problem. Yang et al. [6] and Cichon et al. [7] adopted a cuboid magnet, and Song et al. [8] utilized a radially magnetized annular magnet to obtain the six-DoF pose. Although various indirect methods have been presented, such as employing two magnets with opposite mag- netic directions [9] and fusing with inertial sensing [10],the space occupation and hardware complexity have increased. In contrast to the cuboid magnet, the annular magnet with a hollow design offers better compatibility with commercialwireless cap- sule endoscopy (WCE) structural integration, such as enabling the deployment of required cylindrical electronic modules. In addition, annular magnets are also preferred for the functional design ofmagnetically driven capsule robots [11]. Hence, in this study, we address the challenge of the six-DoF magnetic local- ization of a single circular magnet with a radial magnetization field. [URL 🔗](#page-0)

Several equivalent models for calculating the spatialmagnetic field distribution of annular magnets include magnetic dipole, magnetizing current, and magnetic charge [12]. The magnetic dipole model with simple expression is generally adopted in magnetic tracking and magnetic actuation [5], [11], [13], [14], [15], [16]. However, only the five-DoF pose could be estimated because the magnetic dipole model loses the description of the [URL 🔗](#page-0)


rotation information around the magnetic moment [17]. In ad- dition, the magnetic dipole model will introduce approximation errors for nonspherical magnets, especially in the near field [18]. Thus, the equivalent magnetizing current model was adopted in [6], [7], and [8] to accurately calculate the magnetic field distribution for annular and cuboid magnets. Nevertheless, the yielded model based on numerical integration is highly com- plex and nonconvex, hence unsuitable for immediate real-time magnetic tracking. [URL 🔗](#page-0)

To solve the underlying inverse magnetostatic problem (pose estimation), theLevenberg–Marquardt(LM)algorithm iswidely used for the magnetic dipole model owing to its outstanding execution speed and calculation accuracy [5], [13], [14], [15], [16]. The LM algorithm is a local optimization algorithm that relies on an initial guess and model gradients. For the numerical- integration-based magnetic model, swarm intelligence algo- rithms, such as particle swarm optimization (PSO), were typ- ically used [6], [8]. However, depending on populations and iterations, the PSO algorithm is computationally intensive and might fall into a local optimum. For instance, it took up to 830 s to estimate the annular magnet’s six-DoF pose even though the dipole model and the LM algorithm were utilized to restrict the search boundaries of PSO [8]. [URL 🔗](#page-0)

Recently, deep learning has emerged in a few studies [19], [20], [21] in magnetic tracking. Sebkhi et al. [19] developed a data collection setup and spent a week gathering a dataset [URL 🔗](#page-0)

composed of ∼1.7 million samples in a 10 × 10 × 10 cm3 working space with a positional resolution of 5 mm and an angular resolution of 10◦. The trained five-layer fully connected (FC) network model only outputs a three-DoF position with a median error of 1.4 mm. As stated by the authors, if the positional resolution is increased to 1 mm, the duration of data collection will increase to six months, which is unacceptable for practical applications. To reduce the dependence on the dataset, a regression model trained by 20 691 samples [20] and a classification model trained by 66 000 samples [21] were proposed. However, owing to their lowlocalization accuracy, the trained models were just used to provide a five-DoF initial pose for the LM algorithm at the initial start-up. Thus, how to train a high-precision localizationmodel with a fewreal-world samples is the primary issue for learning-based magnetic tracking. [URL 🔗](#page-0)

Fortunately, prior knowledge-based few-shot learning makes it possible to learn a new task that contains only a few sam- ples [22]. There are various ways to integrate prior knowledge into deep learning. For example, we can use prior knowledge to augment the training dataset [23], narrow the search space of parameters [2], and optimize the search strategy [24]. Thus, we propose an annular magnet pose estimation network (named AMagPoseNet) based on dual-domain few-shot learning. First, leveraging the mathematical model of magnet sensing as prior knowledge, we train an end-to-end convolutional neural net- work (CNN) model that maps from sensor measurement to the magnet’s pose. After that, instead of collecting a sufficient and comprehensive dataset, we narrow the gap between the mathematical model domain (MMD) and the real-world domain (RWD), requiring only a few real-world samples for few-shot learning. [URL 🔗](#page-0)

*Fig. 1. Modeling diagram of an annular magnet based on the equiv- alent magnetizing current and the Biot–Savart law. According to the superposition principle, the magnetic induction intensity produced by the annular magnet is identical to the subtraction of the magnetic induction intensity produced by two cylindrical magnets with different radii.*

The main contributions of this study are summarized as

follows.

- 1) This article introduces an approach that significantly re- duces the dependence on real-world data with the assis- tance ofthe prior mathematical model.The fact that only a few real-world samples (780 = 1300 × 0.6) were used in this study proves the practicality of the proposed method.

- 2) It improves the orientation representation to match well with the symmetrical property of the magnetic field dis- tribution, where the magnetic field symmetry is a natural property for regular-shaped magnets.

- 3) It proposes a network structure consisting of CaliNet and PoseNet as well as a corresponding training strategy to bridge the gap between the MMD and the RWD.

Benefiting from deep learning techniques, the AMagPoseNet can estimate the six-DoF pose of the annular magnet in real time and outperforms the LM-based approach in terms of localization accuracy, robustness, and computational latency.

## II. PRIOR MATHEMATICAL MODEL

## A. Magnetic Induction Intensity ofan AnnularMagnet

As illustrated in Fig. 1, a magnet coordinate frame {m} is established at the magnet’s center, and the magnetic moment [URL 🔗](#page-0)

direction is from Zm− to Zm+. From the equivalent magnetiz- ing current model, magnetic currents are only present around the permanent magnet’s outer surface. With the Biot–Savart law, the magnetic induction intensity dBl at an arbitrary point pi(px,py,pz) generated by a surface current element can be expressed as

|P0|3

where μ0 is the permeability of free space, I denotes the current magnitude, dl indicates the current direction, and P0 is the vector from the current element to point pi.


Furthermore, the magnetic induction intensity dBz at point pi can be calculated by a surface current Iabcda with the current

path a→b→c→d→a, i.e.,

where ab, bc, cd, and da are the segment paths of the current.

Assuming that the radius of the cylinder magnet is r,the magnetic induction intensity at point pi can be estimated by

−r

da

*Fig. 2. Magnetic field distribution around the annular magnet. When*

The Simpson rule, a numerical integral method, can be em- ployed to calculate the above double integral expression. The detailed numerical expression can be referred to [8]. Thus, the magnetic induction intensity at point pi generated by the cylinder magnet with radius r can be presented as [URL 🔗](#page-0)

*the annular magnet rotates 90◦ and −90◦ about Zm-axis, it generates the same distribution in space, i.e., the magnetometer array gives the same measurements in both orientations.*

calculated as follows:

Bc ≈ fc(pi,r).

The subtraction of the magnetic induction intensity generated by two cylindrical magnets with different radii that yield an equivalent annular magnet’s magnetic induction intensity at point pi is estimated by the superposition principle

Equation (8) is the magnetic field model of the annular mag- net, from which we can predict the magnetic induction intensity at any location in space. [URL 🔗](#page-0)

Ba ≈ fa(pi,r1,r2)= fc(pi,r1) − fc(pi,r2)

where r1 and r2 are the outer and inner radii of the annular magnet, respectively.

## PRIOR-MATHEMATICAL-MODEL-ASSISTED LEARNING III. POSE ESTIMATION BASED ON

## B. Coordinate Transformation

A magnetometer array is usually adopted to perceive the magnetic fields generated by the annular magnet in space. As

shown in Fig. 1, a system coordinate frame {s} is built at the center of the magnetometer array. The annular magnet’s [URL 🔗](#page-0)

magnetic induction intensity in (5) is modeled in the frame {m}. The six-DoF pose of the annular magnet describes the translation [URL 🔗](#page-0)

and rotation of the frame {m} with respect to the frame {s}.

Assume that v =[x, y, z, α, β, γ] is the six-DoF pose of the

annular magnet in the frame {s}, where tsm =[x, y, z]T is the translation vector, and (α, β, γ) are the Euler angles at which the magnet rotates sequentially about the Xs-, Ys-, and Zs-axes. Such a rotation sequence can be represented as

where Rsm is the rotation matrix.

The sensor position at point pi in frame {s} and frame {m} can be denoted as ps i =[ps xi,ps yi,ps zi]T and pm i =

[pm xi,pm yi,pm zi]T, respectively, and ps i and pm i have the following

relationship:

pm i = R−1 sm(ps i − tsm).

Given the annular magnet’s pose v, the magnetic induction

intensity at the ith magnetometer in the frame {s} can be

## A. Orientation Representation

The annular magnet’s orientation can be described in various representations, including Euler angles, quaternion, and rotation matrix.

Generally, the original Euler angles range is defined as α ∈

[−180◦, 180◦), β ∈ [−180◦, 180◦), and γ ∈ [−180◦, 180◦). However, the magnetic field distribution generated by the an- nular magnet is symmetrical along the Xm–Zm plane. Any γi

in the range between (−180◦, 0) obviously will have a corre- sponding γj in the range between (0, 180◦), resulting in the annular magnet producing the same magnetic field distribution.

As illustrated in Fig. 2, γi = −90◦ and γj = 90◦ indicate two completely different orientations, while they cause the same sensor measurements. It should be noted that the ambiguity of orientation representation will lead to nonconvergence in pose regression. [URL 🔗](#page-0)

Alternatively, the quaternion is a continuous and smooth representation of orientation, which is suitable for interpolation operation. However, there is ambiguity in quaternion, such as a

quaternion q represents the same orientation as −q. The rotation matrix is a redundant parameter representation of orientation. It is difficult to impose constraints in the learning process to ensure


*Fig. 3. Architecture of the AMagPoseNet composed of PoseNet and CaliNet. PoseNet learns to estimate the magnet pose from the prior mathematical model, and CaliNet is designed to minimize the gap between the prior MMD and the RWD. The nonlinear activation function after*

*each 3 × 3 CNN is ReLU. Conv refers to a CNN, BN means batch normalization, and FC denotes a fully connected layer. The training process of AMagPoseNet is divided into three stages: training PoseNet, pretraining CaliNet, and fine-tuning CaliNet.*

the orthogonality of the output rotation matrix. In addition, as components in quaternion and rotation matrices are highly coupled, matching them with the symmetric properties of the magnetic field distribution is challenging.

The original Euler angles adopt three separate rotation angles to intuitively describe the orientation of a rigid body. We define

γ ∈ [−90◦, 90◦) to match the symmetrical characteristics of the magnetic field distribution. However, the original Euler angle faces a similar problem with quaternion: periodicity leads to very different angular values for similar sensor data [25].For [URL 🔗](#page-0)

example, α might jump from −180◦ to 180◦ though the annular magnet is just a smooth rotation. To avoid this problem, we map each Euler angle to a 2-D unit circle through a pair of sine and cosine functions, and we name it the Euler variant. The orientation of the annular magnet can be represented as

Scaling γ to 2γ guarantees the numerical continuity of eγ when the annular magnet rotates periodically. Consequently, the pose of the annular magnet can be expressed as vnet =

[tT sm, eα, eβ, eγ] ∈ R1×9.

## B. Neural Network forPose Estimation

A triaxis magnetometer measurement can be viewed as an image pixel with red, green, and blue values. Inspired by this, we employ the CNN as the fundamental network layer of the AMagPoseNet to capture the local spatial correlation between adjacent sensors.

Fig. 3 shows the network architecture of the AMagPoseNet, which consists of two subnetworks: PoseNet and CaliNet. PoseNet learns to estimate the magnet pose from the prior mathematical model. PoseNet is similar to ResNet-18 [26] in that it adds residual connections to facilitate the backpropagation of the gradient. The primary changes from ResNet-18 are the reduction of the kernel size in the first convolutional layer and the network depth and the splitting into four FC output branches to accommodate the data dimensionality and the pose regression task.Asmooth and injective regression loss is defined in Euclidean space to learn to predict themagnet’s pose (position and orientation) [URL 🔗](#page-0)

L(M)= Lp(M)+ βLo(M)

whereMmeans the input magnetometer data, Lp(M) indicates

the position regression loss, and

Lo(M) denotes the orientation

regression loss. Since the position and orientation are expressed

in different units, a scale factor

β is adopted to balance the two

regression tasks, and two losses are defined as follows:

where ·δ denotes δ-norm and the symbols with ˆ· indicate PoseNet outputs.

There is a gap between theMMDand the RWDsince themea- surements from the magnetometer array are imperfect and mixed with sensor noise. A calibration procedure has been proposed for the optimization-based magnetic tracking method [13], [16]. [URL 🔗](#page-0)


*TABLE I*

*NPSAND FLOPSFOR DIFFERENT MODELS*

| Model Tnput Size NPs FLOPs |
| --- |
| ResNet-1§ 1.17 x 107 1.82 x 107 CaliNet 3.86 x 101 9.65 x 10° PoseNet 3x5x%5 2.77 x 10° 1.69 x 107 AMagPoseNet 3 x 5 x 5 2.81 1.78 x 107 |

However, the general calibration procedure is based on the in- verse solution of the magnetic dipole model, which is unsuitable for this study using the numerical-integration-based magnetic field model.

Thus, we design a preposition network named CaliNet that is placed between the sensor array output and PoseNet input to narrow the gap, and the δ-norm is utilized to evaluate the calibration performance of CaliNet

After training, CaliNet C(·) could correct the input sensor data Mˆcontaining noises to make it closer to the desired outputsM.

To evaluate the efficiency of the AMagPoseNet, the number of parameters (NPs) and floating-point operations (FLOPs) are employed as metrics for the computational complexity and the memory cost. Both the CNNand FC layers in the AMagPoseNet do not contain bias terms. Thus, NPs and FLOPs in a given layer can be calculated according to the following formula [27]. [URL 🔗](#page-0)

- 1) CNNlayer: NPs =K2 ×Cin ×Cout,FLOPs W× (Cin ×K2) ×Cout, where K is the kernel width (assumedtobesymmetric), H and W are width and height of the input feature map, respectively, and Cin and Cout represent the number of input and output channels, respectively. = 2 ×H ×

- 2) BN layer: NPs = 2 ×Cin,FLOPs = 2 × 2 ×Cin, where Cin is the number of input channels.

- 3) FC layer: NPs = Din ×Dout,FLOPs = 2 ×Din ×Dout, where Din is the input dimensionality and Dout is the output dimensionality.

Table I shows the calculation results of different models, where ResNet-18, a common backbone network in computer vi- sion, is used as a benchmark for comparison. Qasaimeh et al. [28] measured the performance ofResNet-18 on embedded platforms and showed that ResNet-18 could achieve 5.17 frames/s on ARM Cortex A57 CPU and 145 frames/s on Jetson TX2 GPU. Compared to ResNet-18, the AMagPoseNet has only 24% of its NPs and 0.98% of its computation (FLOPs). It should be noted that the calculation of FLOPs depends not only on the NPs but also on the input data size. In Table I,3 × 224 × 224 [URL 🔗](#page-0)

is the standard input size for ResNet-18, while 3 × 5 × 5isthe data size measured by our magnetometer array. The comparison indicates that AMagPoseNet is a lightweight network with the potential to be deployed on embedded platforms.

## C. NetworkTraining Process

To take advantage of prior knowledge and reduce the de- pendence on real-world samples, we train the AMagPoseNet in

*Fig. 4. Experimental hardware setup comprising a magnetometer ar- ray, a calibration board, and an annular NdFeB magnet (grade: N35, outer radius r1 = 5 mm, inner radius r2 = 3.5 mm, and length L = 20 mm). The annular magnet embedded in a 3-D printed shell can be precisely placed on the slots of the calibration board.*

combination with the mathematical model detailed in Section II. Typically, both 1-norm and 2-norm can be employed in (11) and (12), but we employ 1-norm because it performs more robust on datasets that contain saturated sensor data (outliers). The specific training process is summarized as follows. [URL 🔗](#page-0)

- 1) Training PoseNet based on the prior mathematical model. The main task of PoseNet is to learn the inverse solution of the mathematical model in (8). A generated dataset Dmodel(Mmodel, vmodel), sampled from the mathemat- ical model over the working space, is provided to train PoseNet. Section IV-A presents an example illustrating how to obtain a generated dataset from the prior mathe- matical model. [URL 🔗](#page-0)

- 2) Pretraining CaliNet with the prior mathematical model. The neural network trained by a few samples might lead to overfitting and poor generalization [22]. Hence, we first pretrain CaliNet with the generated dataset Dmodel. The generated sensor data Mmodel is used as the desired output of CaliNet, and as the input data after adding Gaussian noise obeying N(0,σ2), where σ = 1 μTin this study. [URL 🔗](#page-0)

- 3) Fine-tuning CaliNet by real-world sensor data and prior mathematical model. A real-world dataset Dreal(Mreal, vreal) is created by sampling from the actual magnetometer array. Here, vreal is a subset of vmodel. CaliNet is further fine-tuned based on Dreal, where Mreal is used as the input data. A subset of Mmodel corresponding to vreal is utilized as the desired outputs, i.e., CaliNet is trained under label-shared dual domain.

## IV. EXPERIMENTS

## A. Experimental Setup

1) Hardware Platform: As shown in Fig. 4,a5×5 mag- netometer array (STMicroelectronics LIS3MDL) was adopted to perceive the magnetic induction intensities. A total of 25 magnetometers were welded evenly on a printed circuit board at [URL 🔗](#page-0)


*Fig. 5. Dataset visualization. (a) Position distribution of the generated dataset. (b) Position distribution of the real-world dataset. (c) Imaging of magnetic induction intensity (Left: generated from the prior mathemati- cal model. Right: sampled from the actual magnetometer array).*

40-mm intervals. The measurement range of the magnetometers

is set to ±1200 μT. Amicrocontroller unit (STMicroelectronics STM32H743) collected the magnetometer data and then trans- mitted it to a personal computer running Ubuntu 20.04 with an AMDRyzen 9 5950X CPU and an NVIDIAGeForce RTX 3090

GPU.

2) Datasets: A working volume of the system is defined as x ∈ [−80, 80] mm, y ∈ [−80, 80] mm, z ∈ [20, 160] mm, α ∈

[−180◦, 180◦), β ∈ [−90◦, 90◦), and γ ∈ [−90◦, 90◦). Sam- plingwithin theworking volume yields two datasets: a generated

datasetDmodel and a real-world datasetDreal, as shown in Fig. 5. The two datasets have been uploaded to IEEE DataPort for sharing with other researchers (http://ieee-dataport.org/10101). [URL 🔗](http://ieee-dataport.org/10101)

The generated dataset was obtained based on the prior math- ematical model. The expected sampling points were evenly distributed in the working space, with a position interval of 10 mm and an angular interval of 20◦. Each expected sam- pling point had a random offset added to it, where the posi- orientation offset was in the range between (−20◦,20◦). Fi- tion offset was in the range between (−10, 10) mm, and the

nally, 293.76 (15×15 × 12 × 17 × 8×8) million samples were gathered into the generated dataset. At each height (Zs-axis), we randomly selected 2% samples as the validation dataset, 2% as the test dataset, and the remaining 96% as the training dataset.

The real-world dataset was collected with the assistance of the calibration board, as shown in Fig. 4. The height above the magnetometer array varied from 33 to 153 mm with a 10-mm interval. At each height, we sampled 100 points, which were composed of25 kinds ofpositions and four kinds oforientations, thus yielding a total of 1300 samples. To verify the generaliz- ability of the AMagPoseNet in the case of a few samples, we extracted five from 25 kinds ofpositions and one from four kinds of orientations as the test dataset (40% of total samples), and the rest as the training dataset (60% of total samples). In this way, [URL 🔗](#page-0)

none of the positions in the test dataset are equal to the ones in the training dataset.

Here, an example is shown to illustrate how to yield the generated dataset from the prior mathematical model. Assume the annular magnet’s pose v = [−30 mm, 45 mm, 103 mm,

0◦,0◦,90◦] and the sensor position at point ps i = [−40 mm, 40 mm, 0 mm]T, as shown in Fig. 5(c). The magnetic induc- tion intensity at the position ps i can be calculated using (8), i.e., [URL 🔗](#page-0)

Correspondingly, the actual magnetometer measurement at this position is

The sensor position is based on the hardware configuration of the magnetometer array shown in Fig. 4. Since our magnetometer [URL 🔗](#page-0)

array is composed of 5×5 triaxial magnetometers, 75 magnetic induction intensity measurements will be acquired. To facilitate the training of the CNN, the 75 data are recomposed to three

channels, each with a size of 5×5 pixel cells.

3) Training Details: The AMagPoseNet was implemented in Pytorch. The training process was divided into three steps, as detailed in Section III-C. We adopted the Adam optimizer to train our model on the two training datasets with a base learning [URL 🔗](#page-0)

rate of 1 × 10−4, reduced with the cosine annealing schedule in a total of 50 training epochs. The batch size was specified as 512 for the generated dataset and 8 for the real-world dataset. It took about 6 h in total to train the AMagPoseNet on the computer configured with anAMD5950XCPUand an NVIDIA 3090 GPU. After pretraining on the generated training dataset, PoseNet can achieve a localization accuracy that Ep =1.15 mm, Eα = 0.88◦, Eβ = 0.51◦, and Eγ = 28.71◦ on the generated test dataset.

4) Evaluation Metrics: The position error Ep and the orien-

tation error Eo are defined as

= |αr − αe| + |βr − βe| + |γr − γe|

where (xr,yr,zr,αr,βr,γr) is the ground truth and (xe, ye,ze,αe,βe,γe) is the estimated value.

The annular magnet’s three-DoF position and two-DoF ori- entation can also be predicted based on the magnetic dipole model and optimization algorithms. The orientation is repre- sented by a unit direction vector (m, n, p) of the magnetic moment. The relationship between (m, n, p) and (α, β, γ) is as follows:

[m, n, p]T = Rsm · [0, 0, 1]T .

Then, we have


*Fig. 6. Predictions of PoseNet based on quaternion and Euler variant.*

*Note that the sensor values are similar at α = −180◦ and α = 179◦,but are represented by two different quaternions.*

## B. Comparison ofOrientation Representations

In this experiment, PoseNet was trained based on quaternion and Euler variant, respectively (for quaternion, the three FC layers with 256 inputs and two outputs in the last layer of PoseNet were modified to one FC layer with 256 inputs and four outputs). We generated 360 test samples when the magnet was placed at (0, 0, 50) mm and rotated around the Xs-axis from

−180◦ to 180◦ with 1◦ interval (excluding 180◦). The ground truth and estimated values inferred from the PoseNet model are shown in Fig. 6. [URL 🔗](#page-0)

Fig. 6 shows that PoseNet based on the Euler variant could estimate values that fit well with the ground truth over the entire period, while the quaternion-based PoseNet performs poorly on periodic critical points. The reason is that similar sensor data on periodic critical points were assigned quite different quaternions in pose regression. [URL 🔗](#page-0)

## C. Dual-Domain Few-Shot Learning Behavior

Apart from the training schemeoffine-tuning CaliNet adopted in this study, another training scheme is skipping CaliNet and directly fine-tuning PoseNet on the real-world dataset. To more obviously exhibit the advantage of dual-domain few-shot learn- ing, in this comparison experiment, none of the positions in the test dataset is equal to the ones in the training dataset.

The loss curves of these two training schemes are shown in Fig. 7(a), which reveals that fine-tuning PoseNet results in sig- nificant overfitting over the training dataset since there is a large gap between the training loss and the test loss. In contrast, the gap is obviously smaller if we fine-tune CaliNet. The main reason is that PoseNet maps from high-dimensional measurements to a low-dimensional magnet pose, which is a complex task. In contrast, CaliNet maps from one measurement space to another equal-dimensional measurement space. The primary purpose of CaliNet is to eliminate measurement noises and measurement offsets caused by sensor position and orientation deviations. The [URL 🔗](#page-0)

*Fig. 7. Dual-domain few-shot learning behavior. (a) Loss curves during fine-tuning CaliNet and PoseNet on the real-world dataset. (b) t-SNE of samples under different domains after fine-tuning CaliNet. CaliNet output domain is closer to the MMD, indicating that CaliNet could narrow the gap between the MMD and the RWD.*

experimental results indicate that CaliNet could be well trained on the real-world dataset containing only a few samples, while PoseNet is susceptible to overfitting.

To visualize the efficacy of the trained CaliNet, we performed dimensionality reduction analysis on data from three domains related to CaliNet: input data (real-world data), CaliNet output data, and desired output data (generated data from the math- ematical model). The t-distributed stochastic neighbor embed- ding (t-SNE) of samples under different domains is shown in Fig. 7(b). There is a domain shift between the MMD and the RWD, while CaliNet could alleviate the problem, i.e., CaliNet migrates the real-world data distribution toward the generated data distribution. [URL 🔗](#page-0)

## D. Localization Accuracy at Different Heights

The AMagPoseNet is trained based on dual-domain few-shot learning. To show that the proposed magnetic tracking method is more effective, we compared our method with the LM-based method on the same experimental platform for the following reasons.

- 1) Five-DoF pose estimation based on the magnetic dipole model and the LMalgorithm is a well-established method and has been adopted in most studies [5], [13], [14], [15], [16] due to its outstanding performance in accuracy and speed. [URL 🔗](#page-0)

- 2) The performance of a magnetic tracking system is highly dependent on both hardware and algorithms. The per- formance of the same tracking algorithm is likely to be different on different hardware platforms. Influencing factors include the calibration procedure [21], permanent magnet shape [6], [8], number of magnetometers [15], sensor arrangement [16], sensor types [13], [15], and so on. [URL 🔗](#page-0)

Depending on whether the real-world training dataset is used, our proposed method and the LM-based method are subdivided into the following four types:

- 1) PoseNet, which is only trained on the generated dataset;

- 2) AMagPoseNet, which includes CaliNet and is trained by dual-domain few-shot learning;


*Fig. 8. Localization errors at different heights. Height indicates the dis- tance between the magnet and the magnetometer array in the Zs-axis. (a) Position error versus height. (b) Orientation error versus height.*

- 3) LM_ONLY, which regards the hardware design specifica- tions as the position and orientation ofthe magnetometers, rather than performing a calibration procedure;

- 4) LM_CALI, as it carries out a calibration procedure on the real-world training dataset for correcting the position and orientation of the magnetometers [13]. [URL 🔗](#page-0)

These four methods were tested on the real-world dataset. Since only α and β could be estimated by the LM-based method, we let Eγ = 0 when calculating the orientation error using (13). [URL 🔗](#page-0)

Fig. 8 shows that the localization accuracy of the proposed method (1.87±1.14 mm, 1.89±0.81◦) outperforms the LM- [URL 🔗](#page-0)

based method (2.07±1.19 mm, 2.70±1.75◦) within the working space, with or without the participation of real-world datasets. The difference is more significant at lower heights because:

- 1) some magnetometers closer to the magnet are saturated at the heights of33 and 43mm,which is also the main reason for the deterioration of our proposed method’s accuracy;

- 2) the LM-based method utilizes the magnetic dipole model to predict sensor measurements. However, it has a signif- icant approximation error in the near field [18]. [URL 🔗](#page-0)

Comparing PoseNet and LM_ONLY shows that the PoseNet trained only with the assistance of the prior mathematical model could achieve comparable localization accuracy to theLM-based method. This result illustrates that PoseNet has the potential to be generalized to the real-world dataset after being sufficiently trained on the generated dataset.

## E. Computational Latency and Robustness Testing

The computational latency of the algorithm is one of the key metrics in real-time applications. We compared AMag- PoseNet against LM_CALI on ten test samples randomly cho- sen from the real-world test dataset, as shown in Table II. The ground truth of each sample added an offset voffset = (Δx,Δy,Δz,Δm,Δn,Δp) was utilized as the initial guess of LM_CALI. Since LM_CALI is sensitive to the initial guess, three groups of offsets were set: [URL 🔗](#page-0)

- 1) voffset1 = (2 mm, −2 mm, 2 mm, 0.2, −0.2, 0.2);

- 2) voffset2 = (10 mm, −10 mm, 10 mm, 0.2, −0.2, 0.2);

*TABLE II*

*COMPARISON OF THE COMPUTATIONAL LATENCY*

- 3) voffset3 = (30 mm, −30 mm, 30 mm, 0.3, −0.3, 0.3).

AMagPoseNet and LM_CALI were implemented in Python and ran on the AMD 5950X CPU without any acceleration. The two algorithms were executed 100 times on each sample.

Table II shows that the computational latency of AMag- PoseNet is less than 32% ofLM_CALI and nearly does not vary with different magnet poses. The LM algorithm is based on an initial guess and gradient descent for iterative optimization. It requires the computation of the second-order Taylor expansion term (Hessian matrix), and the convergence time will be varied when different initial guesses are assigned. When the initial guess is far from the ground truth (e.g., high-speed movement), it might diverge or fall into a local optimum. On the contrary, AMagPoseNet is a single feedforward neural network whose outcomes depend only on the input sensor data after the network has been trained, avoiding the risk of falling into local optima. Thus, the AMagPoseNet is superior to LM_CALI in terms of computational complexity and robustness (nondependence on initial guesses). [URL 🔗](#page-0)

## F. Evaluation of6-D Pose Estimation

In the experiment, we found that the annular magnet rotating around Zs-axis produces much less variation in magnetic field distribution than rotating around Xs-axis or Ys-axis. This char- acteristic makes γ more susceptible to noise disturbances in pose regression, and the estimation error on γ is significantly larger than that on α and β, comparing Figs. 8 and 9. Therefore, we reduced the working volume of the Zs-axis to the range between (30, 100)mmand investigated the accuracyoftheAMagPoseNet for estimating the γ angle on the generated dataset and the real-world dataset. [URL 🔗](#page-0)

As shown in Fig. 9, it is difficult for the AMagPoseNet to estimate γ accurately on the generated test dataset even without adding any noise. Moreover, the real-world test dataset contains various disturbances, such as external ambient magnetic inter- ferences and internal measurement noises. We added different levels of Gaussian noise to the generated test dataset to mimic these disturbances. From the results, the estimation error is closer to that of the real-world test dataset after adding the Gaussian noise with the variance of 1 μT on the generated test dataset. Note again that the real-world dataset includes [URL 🔗](#page-0)


*Fig. 9. Comparison of γ error between the generated test dataset and the real-world test dataset, where the length of the annular magnet is 20 mm. Different levels of Gaussian noise were added to the generated test dataset to evaluate its impact on the estimation performance.*

*Fig. 10. Estimation errors on for different lengths of the annular mag- net. These are the evaluation results on generated test datasets in which the Gaussian noise with the variance of 1 μT was added.*

some outliers at 33- and 43-mm height due to sensor saturation, resulting in the deterioration of AMagPoseNet performance.

Increasing the length of the annular magnet can provide more discriminative magnetic features for γ estimation. We yielded different generated datasets for various lengths of the annular magnet, and the test results are shown in Fig. 10. [URL 🔗](#page-0)

Fig. 10 shows that increasing the length of the annular magnet could significantly improve the estimation accuracy on γ. When the length is increased to 40 mm or above, the AMagPoseNet [URL 🔗](#page-0)

achieves an estimation accuracy better than 4.66±3.06◦, verify- ing that the AMagPoseNet can estimate the six-DoF pose when discriminative magnetic field features are provided.

## G. In Vivo Testing

We designed aWCEwith an outer layer ofthe annular magnet. A polytetrafluoroethylene tube was passed through the working cavity of the flexible endoscope (Smart GS-60DQ, HUACO, Beijing, China), where one end of the tube was tightly fixed to the WCE inside the pig’s stomach, and a surgeon manipulated the other end. In the experiment, the surgeon manipulated the WCE to move 30 mm of displacement by the tube. The outputs of AMagPoseNet were recorded in real time during the entire movement. All the procedures were conducted in accordance with the Guiding Principles in the Care and Use ofAnimals and

*Fig. 11. In vivo experiment. (a) Animal experiment site. (b) Some snapshots of endoscope imaging. (c) Endoscope and WCE. (d) WCE movement trajectory. There are some vibrations when the WCE stops moving. We average the points containing vibrations and treat the mean as the end point. The trajectory length and the Euclidean distance between the start and end points are 32.52 and 28.60 mm, respectively.*

were approved by the Animal Ethics Committee ofQilu Hospital of Shangdong University (Dwll-2021-021).

It is challenging to obtain the ground truth of the WCE’s pose in vivo. The trajectory length Lt and the Euclidean distance Ld were used to evaluate the localization performance of the AM- agPoseNet indirectly. Fig. 11 shows that Lt and Ld are around 30 mm, which verifies the feasibility of the AMagPoseNet for in vivo application to some extent. [URL 🔗](#page-0)

## H. Discussion

With the help of the prior mathematical model, the AMag- PoseNet outperforms traditional optimization-based methods, while only a few real-world samples need to be collected manually. Compared to the use of automated data acquisition equipment [19], the proposed approach could rapidly generate a comprehensive training dataset in the working space with- out potential electromagnetic interference, equipment freedom constraints, and hardware and development costs. In addition, the prior mathematical model of the annular magnet is derived from the basic Biot–Savart law. The modeling method is also suitable for other shapes of magnets, such as cuboid [7] and cylinder magnet [13]. Thus, the AMagPoseNet is generalizable and could be applied to other shapes of magnets. [URL 🔗](#page-0)

Magnetic field symmetry is a natural property for regular- shaped magnets, and our improved orientation representation (Euler variant) has the following benefits:

- 1) improving the prediction accuracy of AMagPoseNet be- cause the symmetric magnetic field distribution and the periodicity of the sensor measurements are well de- scribed, as illustrated in Section IV-B; [URL 🔗](#page-0)


*Fig. 12. Magnetic field distribution for various shapes of magnets (First row: distribution of magnetic lines of force around magnets. Second row: the magnetic field on a plane when γ = 0◦. Third row: the magnetic field on a plane when γ = 45◦). Cuboid and annular magnets can provide discriminative magnetic field features when they rotate along the magnetic moment direction (N-S axis).*

- 2) decoupling γ angle from the orientation. Comparing Figs. 8 and 9, α and β could be accurately estimated even though there is a lack of discriminative magnetic field features for γ. [URL 🔗](#page-0)

Providing discriminativemagnetic field features is a prerequi-

site for AMagPoseNet to estimate γ accurately. Fig.

[12](#page-0)

presents

the magnetic field distributions for various shapes of magnets simulated by the finite-element analysis software COMSOL Multiphysics. When the magnet is rotated along the magnetic moment direction (N-S axis), the magnetic field distribution generated around the cylindrical magnet and spherical magnet remains unchanged. Thus, the γ angle cannot be estimated for these two magnets. In contrast, the cuboid and annular magnets could produce discriminative magnetic field features. The more discriminative the features of the magnetic field are, the more accurate the estimated γ angle will be. To improve the discrim- ination of magnetic induction intensity measured by the sensor array, in addition to increasing the length of the annular magnet, we can also:

- 1) optimize the magnetometer layout, e.g., adopting a 3-D spatial layout [15], [16] instead of a 2-D planar layout; [URL 🔗](#page-0)

- 2) combine two magnets with different magnetic moment directions [9], [14]. [URL 🔗](#page-0)

Onedeficiency in the six-DoF pose estimation is that γ is in the half-cycle range between [−90◦,90◦) rather than the full-cycle

range between [−180◦, 180◦). The deficiency arises from the symmetry distribution of the magnet’s magnetic field. Since the motion of the rigid body is continuous,we can extend the γ angle to the full-period range based on the historical pose information.

## V. CONCLUSION

Learning-based magnetic tracking strongly depends on the size and resolution of the training dataset, while collecting a comprehensive and sufficient dataset is labor intensive and time consuming. This article proposed an end-to-end neural

network model (AMagPoseNet) to solve the underlying inverse magnetostatic problem by dual-domain few-shot learning from the prior mathematical model. Compared with the traditional optimization-based method, AMagPoseNet has the following advantages:

- 1) higher localization accuracy (1.87±1.14 mm, 1.89± 0.81◦), especially in the near field;

- 2) enhanced robustness, as the AMagPoseNet is just a single feedforward neural network that does not rely on initial guesses and avoids the risk of falling into local optima;

- 3) lower computational latency (2.08±0.02 ms) since the magnet pose is directly regressed from a single feedfor- ward network rather than iterative optimization;

- 4) real-time estimation of six-DoF pose if discriminative magnetic field features are provided.

In terms of practical applications, the suggested method can track WCEs, magnetic surgical instruments, and the tips of conventional flexible endoscopes. Thanks to it, physicians will be able to know the exact pose ofmedical instruments inside the human body.

Saturated sensor measurements deteriorate the tracking sys- tem’s performance at lower heights. Future research will re- construct normal data from saturated sensor measurements and thus enhance the localization accuracy of magnetic tracking in the near field.

## REFERENCES

- [1] P. E. Dupont et al., “A decade retrospective of medical robotics research from 2010 to 2020,” Sci. Robot., vol. 6, no. 60, 2021, Art. no. eabi8017.

- [2] S. Su, H. Dai, S. Cheng, and Z. Chen, “Improved magnetic guidance approach for automated guided vehicles by error analysis and prior knowl- edge,” IEEE Trans. Intell. Transp. Syst., vol. 22, no. 11, pp. 6843–6852, Nov. 2021.

- [3] J. Wang et al., “From model to algorithms: Distributed magnetic sensor system for vehicle tracking,” IEEE Trans. Ind. Informat., to be published, doi: 10.1109/TII.2022.3156663. [URL 🔗](https://dx.doi.org/10.1109/TII.2022.3156663)

- [4] C. Di Natali, M. Beccani, N. Simaan, and P. Valdastri, “Jacobian-based iterative method for magnetic localization in robotic capsule endoscopy,” IEEE Trans. Robot., vol. 32, no. 2, pp. 327–338, Apr. 2016.

- [5] C. R. Taylor, H. G. Abramson, and H. M. Herr, “Low-latency track- ing of multiple permanent magnets,” IEEE Sens. J., vol. 19, no. 23, pp. 11458–11468, Dec. 2019.

- [6] W. Yang, C. Hu,M. Q.-H. Meng, S. Song, and H. Dai, “A six-dimensional magnetic localization algorithm for a rectangular magnet objective based on a particle swarm optimizer,” IEEE Trans. Magn., vol. 45, no. 8, pp. 3092–3099, Aug. 2009.

- [7] D. Cichon, R. Psiuk, H. Brauer, and H. Töpfer, “A Hall-sensor-based localization method with six degrees of freedom using unscented Kalman filter,” IEEE Sens. J., vol. 19, no. 7, pp. 2509–2516, Apr. 2019.

- [8] S. Song et al., “6-D magnetic localization and orientation method for an annular magnet based on a closed-form analytical model,” IEEE Trans. Magn., vol. 50, no. 9, Sep. 2014, Art. no. 5000411.

- [9] S. Song, X. Qiu, W. Liu, and M. Q.-H. Meng, “An improved 6-D pose detection method based on opposing-magnet pair system and constraint multiple magnets tracking algorithm,” IEEE Sens. J., vol. 17, no. 20, pp. 6752–6759, Oct. 2017.

- [10] H. Zhang, Y. Li, and Z. Li, “6-D spatial localization of wireless magneti- cally actuated capsule endoscopes based on the fusion of hall sensor array and IMU,” IEEE Sens. J., vol. 22, no. 13, pp. 13424–13433, Jul. 2022.

- [11] S. Song et al., “Integrated design and decoupled control of an- choring and drug release for wireless capsule robots,” IEEE/ASME Trans. Mechatronics, vol. 27, no. 5, pp. 2897–2907, Oct. 2022, doi: 10.1109/TMECH.2021.3125673.

- [12] Y. Zhang et al., “Comparative study on equivalent models calculating magnetic force between permanent magnets,” J. Intell. Manuf. Special Equip., vol. 1, no. 1, pp. 43–65, 2020.


Mengya Xu received the B.Eng. degree in information engineering from Soochow University, Suzhou, China, in 2018, and the M.Sc. degree in electrical and computer engineering in 2019 from the National University of Singapore, Singapore, where she is currently working toward the Ph.D degree in biomedical engineering under the supervision of Prof. Hongliang Ren.

- [13] S. Su et al., “Investigation of the relationship between tracking accuracy and tracking distance of a novel magnetic tracking system,” IEEE Sens. J., vol. 17, no. 15, pp. 4928–4937, Aug. 2017.

- [14] Y. Xu, K. Li, Z. Zhao, and M. Q.-H. Meng, “A novel system for closed- loop simultaneous magnetic actuation and localization of WCE based on external sensors and rotating actuation,” IEEE Trans. Autom. Sci. Eng., vol. 18, no. 4, pp. 1640–1652, Oct. 2021.

- [15] C. Hu,M. Li, S. Song, R. Zhang, andM. Q.-H. Meng,“A cubic 3-axismag- netic sensor array for wirelessly trackingmagnet position and orientation,” IEEE Sens. J., vol. 10, no. 5, pp. 903–913, May 2010.

- [16] C. Hu et al., “Locating intra-body capsule object by three-magnet sensing system,” IEEE Sens. J., vol. 16, no. 13, pp. 5167–5176, Jul. 2016.

- [17] D. Son, S.Yim, andM. Sitti, “A5-D localizationmethod for a magnetically manipulated untethered robot using a 2-D array of hall-effect sensors,” IEEE/ASME Trans. Mechatronics, vol. 21, no. 2, pp. 708–716, Apr. 2016.

- [18] F. Y. Wu, S. Foong, and Z. Sun, “A hybrid field model for enhanced mag- netic localization and position control,” IEEE/ASMETrans.Mechatronics, vol. 20, no. 3, pp. 1278–1287, Jun. 2015.

- [19] N. Sebkhi et al., “A deep neural network-based permanent magnet local- ization for tongue tracking,” IEEE Sens. J., vol. 19, no. 20, pp. 9324–9331, Oct. 2019.

- [20] Y. Qin, B. Lv, H. Dai, and J. Han, “An HFFNN-LM based real-time and high precision magnet localization method,” IEEE Trans. Instrum. Meas., vol. 71, 2022, Art. no. 2509009.

- [21] Y. Fu and Y.-X. Guo, “Guideline on initial point finding and search bounds setting for biomedical magnetic localization,”IEEESens. J., vol. 22, no. 16, pp. 16492–16504, Aug., 2022, doi: 10.1109/JSEN.2022.3190389.

- [22] Y. Wang, Q. Yao, J. T. Kwok, and L. M. Ni, “Generalizing from a few examples: A survey on few-shot learning,” ACM Comput. Surv., vol. 53, no. 3, pp. 1–34, 2020.

- [23] B. Liu, X. Wang, M. Dixit, R. Kwitt, and N. Vasconcelos, “Feature space transfer for data augmentation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2018, pp. 9090–9098.

- [24] R. Keshari, M. Vatsa, R. Singh, and A. Noore, “Learning structure and strength of CNN filters for small sample size training,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2018, pp. 9349–9358.

- [25] J. Wu, L. Ma, and X. Hu, “Delving deeper into convolutional neural networks for camera relocalization,” in Proc. IEEE Int. Conf. Robot. Autom., 2017, pp. 5644–5651.

- [26] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 770–778.

- [27] P. Molchanov, S. Tyree, T. Karras, T. Aila, and J. Kautz, “Pruning convolutional neural networks for resource efficient inference,” 2017, arXiv:1611.06440.

- [28] M. Qasaimeh et al., “Benchmarking vision kernels and neural network inference accelerators on embedded platforms,” 2021, Art. no. 101896. J. Syst. Archit., vol. 113,

Shijian Su received the B.S., M.S., and Ph.D. degrees in electronic science and technology from Fuzhou University, Fuzhou, China, in 2012, 2015, and 2021, respectively.

From 2015 to 2018, he was a Researcher with the Quanzhou Institute of Equipment Man- ufacturing, Haixi Institutes, Chinese Academy of Sciences, Quanzhou, China. He is currently a Postdoctoral Fellow with the Department of Electronic Engineering, The Chinese Univer- sity of Hong Kong, Hong Kong. His research

interests include magnetic tracking, intelligent perception, and robot localization.

Sishen Yuan received the B.S. degree in me- chanical design manufacture and automation from the Harbin Institute of Technology at Wei- hai, Weihai, China, in 2018, and the M.S. degree in mechatronic engineering from the Harbin In- stitute of Technology, Shenzhen, China, in 2021. He is currently working toward the Ph.D. degree in electronic engineering with the Chinese Uni- versity of Hong Kong, Hong Kong.

His current research interests include mag- netic medical robots covering interventional flex-

ible robotics, capsule robotics, and the design and actuation of magnetic origami robots.

Her research interests include vision- language multimodality-based surgical scene understanding.

Huxin Gao received the B.Eng. degree in mechanical engineering from Wuhan University, Wuhan, China, in 2018. He is currently working toward the Ph.D. degree in biomedical engineering with the National University of Singapore, Singapore, under the supervision of Prof. Hongliang Ren.

His research interests include surgical robotics, robotic simulation, reinforcement learning, and autonomous control.

Xiaoxiao Yang received the B.S. degree in clinical medicine in 2019 from the School of Medicine, Shandong University, Jinan, China, where she is currently working toward the M.D. degree in medicine with the First Clinical Col- lege, under the supervision of Prof. Xiuli Zuo.

Her research interests include the application of artificial intelligence and endoscopic robot in gastroenterology.

Hongliang Ren received the Ph.D. degree in electronic engineering (specialized in biomedi- cal engineering) from the Chinese University of Hong Kong, Hong Kong, in 2008.

He has been navigating his academic journey through the Chinese University of Hong Kong; University of California, Berkeley, Berkeley, CA, USA; Johns Hopkins University, Baltimore, MD, USA; Children’s Hospital Boston, Boston, MA, USA; Harvard Medical School, Boston; Chil- dren’s National Medical Center, Washington,

DC, USA; and National University of Singapore, Singapore. He is cur- rently an Associate Professor with the Department of Electronic En- gineering, The Chinese University of Hong Kong. His research inter- ests include robotics, mechatronics, artificial intelligence, actuators, and sensors.

Dr. Ren is the recipient of the IFMBE/IAMBE Early Career Award 2018, the Interstellar Early Career Investigator Award 2018, the ICBHI Young Investigator Award 2019, and the Health Longevity Catalyst Award 2022 by the National Academy of Medicine and the Research Grants Council. He is an Associate Editor for IEEE TRANSACTIONS ON

AUTOMATION SCIENCE AND ENGINEERING and Medical and Biological En-

gineering and Computing. He was an Active Organizer and Contributor on the committees of numerous robotics conferences, including a variety of roles in the flagship IEEE International Conference on Robotics and Automation, IEEE/RSJ International Conference on Intelligent Robots and Systems, as well as other domain conferences such as IEEE Inter- national Conference on Robotics and Biomimetics, IEEE International Conference on Biomedical Robotics and Biomechatronics, and IEEE International Conference on Information and Automation.
