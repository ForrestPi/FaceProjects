传统使用3DMM重建三维人脸的缺陷：

* 必须定位人脸特征点（大多数是人脸68个特征点）
* 必须要求检测得到的特征点准确
* 计算量过大
* 估计出来的3DMM系数往往存在较大偏差


随着深度学习的发展，越来越多的CNN结构被提出，使得图像特征的提取变得越来越简单有效。因此，基于CNN的三维人脸重建的方法也被陆续提出。

近几年基于CNN方法的三维人脸重建主要分为以下几类：

估计3DMM系数。Tran et al. CVPR2017 [1](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tran_Regressing_Robust_and_CVPR_2017_paper.pdf)    
估计非线性3DMM系数.Tran et al. CVPR2018 [2](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_Nonlinear_3D_Face_CVPR_2018_paper.pdf)    
end-to-end的三维人脸重建. Feng et al. ECCV2018[3](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Feng_Joint_3D_Face_ECCV_2018_paper.pdf);     
Aaron et al. ICCV2017[4](http://openaccess.thecvf.com/content_ICCV_2017/papers/Jackson_Large_Pose_3D_ICCV_2017_paper.pdf)    
人脸细节重建. Trigeorgis et al. CVPR2017[5](https://ibug.doc.ic.ac.uk/media/uploads/documents/normal_estimation__cvpr_2017_-4.pdf)


## projects
[Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction)    
[3DDFA](https://github.com/cleardusk/3DDFA)    
[face3d](https://github.com/YadiraF/face3d)    