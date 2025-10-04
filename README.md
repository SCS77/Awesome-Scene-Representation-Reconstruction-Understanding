<h1 align="center">
  <b>Awesome-Scene-Reconstruction-Understanding (NeRF、3DGS)</b>
</h1>

## <b><span style="color:white; background-color:red;">README.md for Reconstruction and Proceedings (For detailed papers on specific application domains, please refer to: Representation-Reconstruction-Understanding.md)</span></b>

📦 Curated collection of papers, datasets, codebases, and resources in the field of scene reconstruction and understanding using NeRF and 3DGS.

## 😄😄 <span style="color:red;">Under Construction</span>  😄😄


## [🔥 Scene Reconstruction](#scene-reconstruction)  

### • [NeRF(Neural Radiance Fields)](#nerf) 
### • [3DGS(3D Gaussian Splatting)](#3dgs)  


## [🚀 Scene Understanding](#scene-understanding)  

### • [Scene Segmentation](#scene-segmentation)  

---

## Scene Reconstruction

### NeRF

#### Acceleration
[[code]](https://github.com/bmild/nerf) [[paper]](https://arxiv.org/abs/2003.08934) [[project]](https://www.matthewtancik.com/nerf) **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** [ECCV 2020]  🔥🔥  
[[code]](https://github.com/facebookresearch/NSVF) [[paper]](https://arxiv.org/pdf/2007.11571) [[project]](https://lingjie0206.github.io/papers/NSVF/) **NSVF: Neural Sparse Voxel Fields** [NeurIPS 2020]  
[[code]](https://github.com/Xharlie/pointnerf) [[paper]](https://arxiv.org/pdf/2201.08845) [[project]](https://xharlie.github.io/projects/project_sites/pointnerf/index.html) **Point-NeRF: Point-based Neural Radiance Fields** [CVPR 2020]  🔥  
[[code]](https://github.com/computational-imaging/automatic-integration) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Lindell_AutoInt_Automatic_Integration_for_Fast_Neural_Volume_Rendering_CVPR_2021_paper.pdf) [[project]](http://www.computationalimaging.org/publications/automatic-integration/) **AutoInt: Automatic Integration for Fast Neural Volume Rendering** [CVPR 2021]  
[[code]](https://github.com/ubc-vision/derf) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Rebain_DeRF_Decomposed_Radiance_Fields_CVPR_2021_paper.pdf) [[project]](https://ubc-vision.github.io/derf/) **DeRF: Decomposed Radiance Fields** [CVPR 2021]  
[[code]](https://github.com/creiser/kilonerf) [[paper]](https://thomasneff.github.io/adanerf/adanerf_supplementary.pdf) [[project]](https://creiser.github.io/kilonerf/) **KiloNeRF: Speeding Up Neural Radiance Fields with Thousands of Tiny MLPs** [ICCV 2021]  🔥  
[[code]](https://github.com/NVlabs/instant-ngp) [[paper]](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf) [[project]](https://nvlabs.github.io/instant-ngp/) **Instant Neural Graphics Primitives with a Multiresolution Hash Encoding** [TOG 2022]  🔥🔥  
[[code]](https://github.com/sunset1995/DirectVoxGO) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Direct_Voxel_Grid_Optimization_Super-Fast_Convergence_for_Radiance_Fields_Reconstruction_CVPR_2022_paper.pdf) [[project]](https://sunset1995.github.io/dvgo/) **Direct Voxel Grid Optimization: Super-Fast Convergence for Radiance Fields Reconstruction** [CVPR 2022]  🔥  
[[code]](https://github.com/dvlab-research/EfficientNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_EfficientNeRF__Efficient_Neural_Radiance_Fields_CVPR_2022_paper.pdf) **EfficientNeRF – Efficient Neural Radiance Fields** [CVPR 2022]  
[[code]](https://github.com/lwwu2/diver) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_DIVeR_Real-Time_and_Accurate_Neural_Radiance_Fields_With_Deterministic_Integration_CVPR_2022_paper.pdf) **DIVeR: Real-time and Accurate Neural Radiance Fields with Deterministic Integration for Volume Rendering** [CVPR 2022]  
[[code]](https://github.com/zju3dv/ENeRF) [[paper]](https://arxiv.org/abs/2112.01517) [[project]](https://zju3dv.github.io/enerf/) **ENeRF: Efficient Neural Radiance Fields for Interactive Free-viewpoint Video** [SIGGRAPH Asia 2022]  
[[code]](https://github.com/thomasneff/AdaNeRF) [[paper]](https://thomasneff.github.io/adanerf/adanerf_supplementary.pdf) [[project]](https://thomasneff.github.io/adanerf/) **AdaNeRF: Adaptive Sampling for Real-time Rendering of Neural Radiance Fields** [ECCV 2022]  
[[code]](https://github.com/apchenstu/TensoRF) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920332.pdf) [[project]](https://apchenstu.github.io/TensoRF/) **TensoRF: Tensorial Radiance Fields** [ECCV 2022]  🔥  
[[code]](https://github.com/google-research/multinerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Barron_Mip-NeRF_360_Unbounded_Anti-Aliased_Neural_Radiance_Fields_CVPR_2022_paper.html) [[project]](https://jonbarron.info/mipnerf360/) **Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields** [CVPR 2022]  🔥  
[[code]](https://github.com/dunbar12138/DSNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_Depth-Supervised_NeRF_Fewer_Views_and_Faster_Training_for_Free_CVPR_2022_paper.pdf) [[project]](https://www.cs.cmu.edu/~dsnerf/) **Depth-supervised NeRF: Fewer Views and Faster Training for Free** [CVPR 2022]  🔥  
[[code]](https://github.com/sxyu/svox2) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Fridovich-Keil_Plenoxels_Radiance_Fields_Without_Neural_Networks_CVPR_2022_paper.pdf) [[project]](https://github.com/sxyu/svox2.git) **Plenoxels: Radiance Fields without Neural Networks** [CVPR 2022]  🔥  
[[code]](https://github.com/NVIDIAGameWorks/kaolin-wisp) [[paper]](https://arxiv.org/abs/2206.07707) [[project]](https://nv-tlabs.github.io/vqad/) **Variable Bitrate Neural Fields** [SIGGRAPH 2022]  🔥  
[[code]](https://github.com/wolfball/PlenVDB) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_PlenVDB_Memory_Efficient_VDB-Based_Radiance_Fields_for_Fast_Training_and_CVPR_2023_paper.pdf) [[project]](https://plenvdb.github.io/) **PlenVDB: Memory Efficient VDB-Based Radiance Fields for Fast Training and Rendering** [CVPR 2023]  
[[code]](https://github.com/Heng14/DyLiN) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_DyLiN_Making_Light_Field_Networks_Dynamic_CVPR_2023_paper.pdf) [[project]](https://dylin2023.github.io/) **DyLiN: Making Light Field Networks Dynamic** [CVPR 2023]  
[[code]](https://github.com/snap-research/MobileR2L) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Real-Time_Neural_Light_Field_on_Mobile_Devices_CVPR_2023_paper.pdf) [[project]](https://snap-research.github.io/MobileR2L/) **Real-time neural light field on mobile devices** [CVPR 2023]  
[[paper]](https://arxiv.org/abs/2302.14859) [[project]](https://bakedsdf.github.io/) **BakedSDF: Meshing Neural SDFs for Real-Time View Synthesis** [SIGGRAPH 2023]  
[[code]](https://github.com/google/hypernerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Turki_HybridNeRF_Efficient_Neural_Rendering_via_Adaptive_Volumetric_Surfaces_CVPR_2024_paper.pdf) [[project]](https://haithemturki.com/hybrid-nerf/) **HybridNeRF: Efficient Neural Rendering via Adaptive Volumetric Surfaces** [CVPR 2024]  🔥  
[[code]](https://github.com/VISION-SJTU/Lightning-NeRF) [[paper]](https://arxiv.org/abs/2403.05907) **Lightning NeRF: Efficient Hybrid Scene Representation for Autonomous Driving** [ICRA 2024]  






#### Quality & Realism Improvement

[[code]](https://github.com/tancik/fourier-feature-networks) [[paper]](https://proceedings.neurips.cc/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf) [[project]](https://bmild.github.io/fourfeat/) **Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains** [NeurIPS 2020]  🔥  
[[code]](https://github.com/cgtuebingen/NeRD-Neural-Reflectance-Decomposition) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Boss_NeRD_Neural_Reflectance_Decomposition_From_Image_Collections_ICCV_2021_paper.pdf) [[project]](https://markboss.me/publication/2021-nerd/) **NeRD: Neural Reflectance Decomposition from Image Collections** [ICCV 2021]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Srinivasan_NeRV_Neural_Reflectance_and_Visibility_Fields_for_Relighting_and_View_CVPR_2021_paper.pdf) [[project]](https://pratulsrinivasan.github.io/nerv/) **NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis** [CVPR 2022]  
[[code]](https://github.com/google-research/multinerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Verbin_Ref-NeRF_Structured_View-Dependent_Appearance_for_Neural_Radiance_Fields_CVPR_2022_paper.pdf?_hsenc=p2ANqtz-8tadeadAJeGMwdMK0dCQgL4tcspDr7QP-jHu5vlS_dI1xLF0CUSZRiUo_5SCHuLAIP0XSO) [[project]](https://gcl-seminar.github.io/Awesome-Graphics-Papers/papers/CVPR/2022/RefNeRF/) **Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields** [CVPR 2022]  🔥  
[[code]](https://github.com/cwchenwang/NeRF-SR) [[paper]](https://cg.cs.tsinghua.edu.cn/papers/MM-2022-NeRF-SR.pdf) [[project]](https://cwchenwang.github.io/NeRF-SR/) **NeRF-SR: High-Quality Neural Radiance Fields using Supersampling** [ACM 2022]  
[[code]](https://github.com/oppo-us-research/NeuRBF) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_NeuRBF_A_Neural_Fields_Representation_with_Adaptive_Radial_Basis_Functions_ICCV_2023_paper.pdf) [[project]](https://oppo-us-research.github.io/NeuRBF-website/) **NeurBF: A Neural Fields Representation with Adaptive Radial Basis Functions** [ICCV 2023]  
[[code]](https://github.com/3D-FRONT-FUTURE/NeuDA) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Cai_NeuDA_Neural_Deformable_Anchor_for_High-Fidelity_Implicit_Surface_Reconstruction_CVPR_2023_paper.pdf) [[project]](https://3d-front-future.github.io/neuda/) **NeuDA: Neural deformable anchor for high-fidelity implicit surface reconstruction** [CVPR 2023]  
[[code]](https://github.com/ActiveVisionLab/nope-nerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Bian_NoPe-NeRF_Optimising_Neural_Radiance_Field_With_No_Pose_Prior_CVPR_2023_paper.pdf) [[project]](https://nope-nerf.active.vision/) **NoPe-NeRF: Optimising neural radiance field with no pose prior** [CVPR 2023]  
[[code]](https://github.com/rover-xingyu/L2G-NeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Local-to-Global_Registration_for_Bundle-Adjusting_Neural_Radiance_Fields_CVPR_2023_paper.pdf) [[project]](https://rover-xingyu.github.io/L2G-NeRF/) **Local-to-global Registration for Bundle-adjusting Neural Radiance Fields** [CVPR 2023]  
[[code]](https://github.com/suLvXiangXin/zipnerf-pytorch) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Barron_Zip-NeRF_Anti-Aliased_Grid-Based_Neural_Radiance_Fields_ICCV_2023_paper.pdf) **Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields** [ICCV 2023]  🔥  
[[code]](https://github.com/cwchenwang/NeRF-SR) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_RefSR-NeRF_Towards_High_Fidelity_and_Super_Resolution_View_Synthesis_CVPR_2023_paper.pdf) [[project]](https://cwchenwang.github.io/NeRF-SR/) **RefSR-NeRF: Towards High-Fidelity and Super-Resolution View Synthesis** [CVPR 2023]  
[[paper]](https://github.com/thucz/PanoGRF) [[paper]](https://3dvar.com/Chen2023PanoGRF.pdf) [[project]](https://thucz.github.io/PanoGRF/) **PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline Panoramas** [NeurIPS 2023]  
[[code]](https://zyqz97.github.io/GP_NeRF/) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_GP-NeRF_Generalized_Perception_NeRF_for_Context-Aware_3D_Scene_Understanding_CVPR_2024_paper.pdf) [[project]](https://zyqz97.github.io/GP_NeRF/) **GP-NeRF: Generalized Perception NeRF for Context-Aware 3D Scene Understanding** [CVPR 2024]  
[[code]](https://github.com/sony/NeISF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_NeISF_Neural_Incident_Stokes_Field_for_Geometry_and_Material_Estimation_CVPR_2024_paper.pdf) [[project]](https://sony.github.io/NeISF/) **NeISF: Neural Incident Stokes Field for Geometry and Material Estimation** [CVPR 2024]  
[[code]](https://github.com/lyclyc52/SANeRF-HQ) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_SANeRF-HQ_Segment_Anything_for_NeRF_in_High_Quality_CVPR_2024_paper.pdf) [[project]](https://lyclyc52.github.io/SANeRF-HQ/) **SANeRF-HQ: Segment Anything for NeRF in High Quality** [CVPR 2024]  🔥  
[[code]](https://github.com/autonomousvision/murf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_MuRF_Multi-Baseline_Radiance_Fields_CVPR_2024_paper.pdf) [[project]](https://haofeixu.github.io/murf/) **MuRF: Multi-Baseline Radiance Fields** [CVPR 2024]  
[[code]](https://github.com/s3anwu/pbrnerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_PBR-NeRF_Inverse_Rendering_with_Physics-Based_Neural_Fields_CVPR_2025_paper.pdf) **PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields** [CVPR 2025]  
[[paper]](https://github.com/baskargroup/SC-NeRF) [[paper]](https://arxiv.org/abs/2503.21958) [[project]](https://baskargroup.github.io/SC-NeRF/) **SC-NeRF: NeRF-based Point Cloud Reconstruction using a Stationary Camera for Agricultural Applications** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025W/CV4Metaverse/papers/Zhang_IL-NeRF_Incremental_Learning_for_Neural_Radiance_Fields_with_Camera_Pose_CVPRW_2025_paper.pdf) [[project]](https://3d-front-future.github.io/neuda/) **IL-NeRF: Incremental Learning for Neural Radiance Fields with Camera Pose Learning** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025W/CV4Metaverse/papers/Seo_ARC-NeRF_Area_Ray_Casting_for_Broader_Unseen_View_Coverage_in_CVPRW_2025_paper.pdf) [[project]](https://shawn615.github.io/arc-nerf/) **ARC-NeRF: Area Ray Casting for Broader Unseen View Coverage in Few-shot Object Insertion** [CVPR 2025]  
[[code]](https://github.com/linjohnss/FrugalNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_FrugalNeRF_Fast_Convergence_for_Extreme_Few-shot_Novel_View_Synthesis_without_CVPR_2025_paper.pdf) [[project]](https://linjohnss.github.io/frugalnerf/) **FrugalNeRF: Fast Convergence for Extreme Few-shot Novel View Synthesis** [CVPR 2025]  
[[code]](https://github.com/wen-yuan-zhang/NeRFPrior) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_NeRFPrior_Learning_Neural_Radiance_Field_as_a_Prior_for_Indoor_CVPR_2025_paper.pdf) **NeRFPrior: Learning Neural Radiance Field as a Prior for Signed Distance Fields** [CVPR 2025]  





#### Dynamic & Deformable Scenes


[[code]](https://github.com/albertpumarola/D-NeRF) [[paper]](https://arxiv.org/abs/2011.13961) [[project]](https://www.albertpumarola.com/research/D-NeRF/index.html) **D-NeRF: Neural Radiance Fields for Dynamic Scenes** [CVPR 2021]  🔥  
[[code]](https://github.com/zhengqili/Neural-Scene-Flow-Fields) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Neural_Scene_Flow_Fields_for_Space-Time_View_Synthesis_of_Dynamic_CVPR_2021_paper.pdf) [[project]](https://www.cs.cornell.edu/~zl548/NSFF/) **Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes** [CVPR 2021]  🔥  
[[code]](https://github.com/google/nerfies) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Park_Nerfies_Deformable_Neural_Radiance_Fields_ICCV_2021_paper.pdf) [[project]](https://nerfies.github.io/) **Nerfies: Deformable Neural Radiance Fields** [ICCV 2021]  🔥  
[[code]](https://github.com/facebookresearch/nonrigid_nerf) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Tretschk_Non-Rigid_Neural_Radiance_Fields_Reconstruction_and_Novel_View_Synthesis_of_ICCV_2021_paper.pdf) [[project]](https://vcai.mpi-inf.mpg.de/projects/nonrigid_nerf/) **Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Deforming Scene from Monocular Video** [ICCV 2021]  
[[code]](https://github.com/gafniguy/4D-Facial-Avatars) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Gafni_Dynamic_Neural_Radiance_Fields_for_Monocular_4D_Facial_Avatar_Reconstruction_CVPR_2021_paper.pdf) [[project]](https://gafniguy.github.io/4D-Facial-Avatars/) **Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction** [CVPR 2021]  🔥  
[[code]](https://github.com/nogu-atsu/NARF) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Noguchi_Neural_Articulated_Radiance_Field_ICCV_2021_paper.pdf) **Neural Articulated Radiance Field** [ICCV 2021]  
[[code]](https://github.com/JanaldoChen/Anim-NeRF?tab=readme-ov-file) [[paper]](https://arxiv.org/abs/2106.13629) **Animatable Neural Radiance Fields from Monocular RGB Videos** [arXiv 2021]  
[[code]](https://github.com/googleinterns/IBRNet) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_IBRNet_Learning_Multi-View_Image-Based_Rendering_CVPR_2021_paper.pdf) [[project]](https://ibrnet.github.io/) **IBRNet: Learning Multi-View Image-Based Rendering** [CVPR 2021]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Xian_Space-Time_Neural_Irradiance_Fields_for_Free-Viewpoint_Video_CVPR_2021_paper.pdf) [[project]](https://video-nerf.github.io/) **Space-time Neural Irradiance Fields for Free-Viewpoint Video** [CVPR 2021]  
[[code]](https://github.com/yilundu/nerflow) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Du_Neural_Radiance_Flow_for_4D_View_Synthesis_and_Video_Processing_ICCV_2021_paper.pdf) **Neural Radiance Flow for 4D View Synthesis and Video Processing** [ICCV 2021]  
[[code]](https://github.com/gaochen315/DynamicNeRF) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Dynamic_View_Synthesis_From_Dynamic_Monocular_Video_ICCV_2021_paper.pdf) [[project]](https://free-view-video.github.io/) **Dynamic View Synthesis from Dynamic Monocular Video** [ICCV 2021]  🔥  
[[code]](https://github.com/zju3dv/neuralbody) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Peng_Neural_Body_Implicit_Neural_Representations_With_Structured_Latent_Codes_for_CVPR_2021_paper.pdf) [[project]](https://zju3dv.github.io/neuralbody/) **Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans** [CVPR 2021]  
[[code]](https://github.com/google/hypernerf) [[paper]](https://arxiv.org/pdf/2106.13228) [[project]](https://hypernerf.github.io/) **HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields** [SIGGRAPH 2021]  🔥  
[[code]](https://github.com/lingjie0206/Neural_Actor_Main_Code) [[paper]](https://dl.acm.org/doi/epdf/10.1145/3478513.3480528) [[project]](https://vcai.mpi-inf.mpg.de/projects/NeuralActor/) **Neural Actor: Neural Free-view Synthesis of Human Actors with Pose Control** [SIGGRAPH Asia 2021]  
[[code]](https://github.com/hustvl/TiNeuVox) [[paper]](https://arxiv.org/abs/2205.15285) [[project]](https://jaminfong.cn/tineuvox/) **TiNeuVox: Fast Dynamic Radiance Fields with Time-Aware Neural Voxels** [SIGGRAPH Asia 2021]  
[[code]](https://github.com/ChikaYan/d2nerf) [[paper]](https://papers.nips.cc/paper_files/paper/2022/file/d2cc447db9e56c13b993c11b45956281-Paper-Conference.pdf) [[project]](https://papers.nips.cc/paper_files/paper/2022/hash/d2cc447db9e56c13b993c11b45956281-Abstract-Conference.html) **D2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video** [NeurIPS 2022]  🔥  
[[code]](https://github.com/AlgoHunt/StreamRF) [[paper]](https://papers.nips.cc/paper_files/paper/2022/file/57c2cc952f388f6185db98f441351c96-Paper-Conference.pdf) [[project]](https://papers.nips.cc/paper_files/paper/2022/hash/57c2cc952f388f6185db98f441351c96-Abstract-Conference.html) **Streaming Radiance Fields for 3D Video Synthesis** [NeurIPS 2022]  
[[code]](https://github.com/fengres/mixvoxels) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Neural_3D_Video_Synthesis_From_Multi-View_Video_CVPR_2022_paper.pdf) [[project]](https://fengres.github.io/mixvoxels/) **Mixed Neural Voxels for Fast Multi-view Video Synthesis** [CVPR 2022]  🔥  
[[code]](https://github.com/google/dynibar) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_DynIBaR_Neural_Dynamic_Image-Based_Rendering_CVPR_2023_paper.pdf) [[project]](https://dynibar.github.io/) **DynIBaR: Neural Dynamic Image-Based Rendering** [CVPR 2023]  🔥  
[[code]](https://github.com/DSaurus/Tensor4D) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shao_Tensor4D_Efficient_Neural_4D_Decomposition_for_High-Fidelity_Dynamic_Reconstruction_and_CVPR_2023_paper.pdf) [[project]](https://liuyebin.com/tensor4d/tensor4d.html) **Tensor4D: Efficient Neural 4D Decomposition for High-fidelity Dynamic Reconstruction and Rendering** [CVPR 2023]  
[[code]](https://github.com/facebookresearch/hyperreel) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Attal_HyperReel_High-Fidelity_6-DoF_Video_With_Ray-Conditioned_Sampling_CVPR_2023_paper.pdf) [[project]](https://hyperreel.github.io/) **HyperReel: High-Fidelity 6-DoF Video with Ray-Conditioned Samplings** [CVPR 2023]  
[[code]](https://github.com/Caoang327/HexPlane) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_HexPlane_A_Fast_Representation_for_Dynamic_Scenes_CVPR_2023_paper.pdf) **HexPlane: A Fast Representation for Dynamic Scenes** [CVPR 2023]  🔥   
[[code]](https://github.com/facebookresearch/robust-dynrf) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Robust_Dynamic_Radiance_Fields_CVPR_2023_paper.pdf) [[project]](https://robust-dynrf.github.io/) **Robust Dynamic Radiance Fields** [CVPR 2023]  🔥  
[[code]](https://github.com/MightyChaos/fsdnerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Flow_Supervision_for_Deformable_NeRF_CVPR_2023_paper.pdf) [[project]](https://mightychaos.github.io/projects/fsdnerf/) **Flow Supervision for Deformable NeRF** [CVPR 2023]  
[[code]](https://github.com/jefftan969/dasr) [[paper]](https://jefftan969.github.io/dasr/paper.pdf) [[project]](https://jefftan969.github.io/dasr/) **Distilling Neural Fields for Real-Time Articulated Shape Reconstruction** [CVPR 2023]  
[[code]](https://yifanjiang19.github.io/alignerf/demo/compare.html) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_AligNeRF_High-Fidelity_Neural_Radiance_Fields_via_Alignment-Aware_Training_CVPR_2023_paper.pdf) [[project]](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_AligNeRF_High-Fidelity_Neural_Radiance_Fields_via_Alignment-Aware_Training_CVPR_2023_paper.pdf) **AligNeRF: High-Fidelity Neural Radiance Fields via Alignment-Aware Training** [CVPR 2023]  
[[code]](https://github.com/NVlabs/BundleSDF) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.pdf) [[project]](https://bundlesdf.github.io/) **BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown** [CVPR 2023]  🔥  
[[code]](https://github.com/YilingQiao/DMRF) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Qiao_Dynamic_Mesh-Aware_Radiance_Fields_ICCV_2023_paper.pdf) [[project]](https://mesh-aware-rf.github.io/) **Dynamic Mesh-Aware Radiance Fields** [ICCV 2023]  
[[code]](https://github.com/fengres/mixvoxels) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Mixed_Neural_Voxels_for_Fast_Multi-view_Video_Synthesis_ICCV_2023_paper.pdf) [[project]](https://fengres.github.io/mixvoxels/) **MixVoxels: Mixed Neural Voxels for Fast Multi-view Video Synthesis** [ICCV 2023]  
[[code]](https://github.com/kaichen-z/DynPoint) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/dbdc7a9779ce0278c6e43b62c7e97759-Paper-Conference.pdf) [[project]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dbdc7a9779ce0278c6e43b62c7e97759-Abstract-Conference.html) **DynPoint: Dynamic Neural Point For View Synthesis** [NeurIPS 2023]  
[[code]](https://github.com/seoha-kim/Sync-NeRF) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28057) [[project]](https://seoha-kim.github.io/sync-nerf/) **Sync-NeRF: Generalizing Dynamic NeRFs to Unsynchronized Videos** [AAAI 2024]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Otonari_Entity-NeRF_Detecting_and_Removing_Moving_Entities_in_Urban_Scenes_CVPR_2024_paper.pdf) [[project]](https://otonari726.github.io/entitynerf/) **Entity-NeRF: Detecting and Removing Moving Entities in Urban Scenes** [CVPR 2024]  
[[code]](https://github.com/xingyi-li/s-dyrf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_S-DyRF_Reference-Based_Stylized_Radiance_Fields_for_Dynamic_Scenes_CVPR_2024_paper.pdf) [[project]](https://xingyi-li.github.io/s-dyrf/) **S-DyRF: Reference-Based Stylized Radiance Fields for Dynamic Scenes** [CVPR 2024]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_DS-NeRV_Implicit_Neural_Video_Representation_with_Decomposed_Static_and_Dynamic_CVPR_2024_paper.pdf) [[project]](https://haoyan14.github.io/DS-NeRV/) **DS-NeRV: Implicit Neural Video Representation with Decomposed Static and Dynamic Codes** [CVPR 2024]  
[[code]](https://github.com/FYTalon/pienerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Feng_PIE-NeRF_Physics-based_Interactive_Elastodynamics_with_NeRF_CVPR_2024_paper.pdf) [[project]](https://fytalon.github.io/pienerf/?ref=aiartweekly) **PIE-NeRF: Physics-based Interactive Elastodynamics with NeRF** [CVPR 2024]  
[[code]](https://github.com/huiqiang-sun/DyBluRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_DyBluRF_Dynamic_Neural_Radiance_Fields_from_Blurry_Monocular_Video_CVPR_2024_paper.pdf) [[project]](https://kaist-viclab.github.io/dyblurf-site/) **Dynamic Deblurring Neural Radiance Fields for Blurry Monocular Video** [CVPR 2024]  
[[code]](https://github.com/merlresearch/Gear-NeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Gear-NeRF_Free-Viewpoint_Rendering_and_Tracking_with_Motion-aware_Spatio-Temporal_Sampling_CVPR_2024_paper.pdf) [[project]](https://merl.com/research/highlights/gear-nerf) **Gear-NeRF: Free-Viewpoint Rendering and Tracking with Motion-aware Spatio-Temporal Sampling** [CVPR 2024]  
[[code]](https://github.com/huiqiang-sun/DyBluRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kumar_DynaMoDe-NeRF_Motion-aware_Deblurring_Neural_Radiance_Field_for_Dynamic_Scenes_CVPR_2025_paper.pdf) [[project]](https://huiqiang-sun.github.io/dyblurf/) **Motion-aware Deblurring Neural Radiance Field for Dynamic Scenes** [CVPR 2025]  



#### Large-Scale & Unbounded Scenes

[[code]](https://github.com/facebookresearch/NSVF) [[paper]](https://arxiv.org/pdf/2007.11571) [[project]](https://lingjie0206.github.io/papers/NSVF/) **NSVF: Neural Sparse Voxel Fields** [NeurIPS 2020]  
[[code]](https://github.com/creiser/kilonerf) [[paper]](https://thomasneff.github.io/adanerf/adanerf_supplementary.pdf) [[project]](https://creiser.github.io/kilonerf/) **KiloNeRF: Speeding Up Neural Radiance Fields with Thousands of Tiny MLPs** [ICCV 2021]  
[[code]](https://github.com/simicvm/snerg) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Hedman_Baking_Neural_Radiance_Fields_for_Real-Time_View_Synthesis_ICCV_2021_paper.pdf) **SNeRG: Baking Neural Radiance Fields for Real-Time View Synthesis** [SIGGRAPH 2021]  
[[code]](https://github.com/Kai-46/nerfplusplus) [[paper]](https://arxiv.org/abs/2010.07492) **NeRF++: Analyzing and Improving Neural Radiance Fields for Unbounded 3D Scenes** [CVPR 2021]  🔥  
[[code]](https://github.com/city-super/BungeeNeRF) [[paper]](https://city-super.github.io/citynerf/img/1947.pdf) [[project]](https://city-super.github.io/citynerf/) **CityNeRF: Building NeRF at City Scale** [ECCV 2021]  
[[code]](https://github.com/google-research/multinerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Barron_Mip-NeRF_360_Unbounded_Anti-Aliased_Neural_Radiance_Fields_CVPR_2022_paper.html) [[project]](https://jonbarron.info/mipnerf360/) **Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields** [CVPR 2022]  🔥  
[[code]](https://github.com/sunset1995/DirectVoxGO) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Direct_Voxel_Grid_Optimization_Super-Fast_Convergence_for_Radiance_Fields_Reconstruction_CVPR_2022_paper.pdf) [[project]](https://sunset1995.github.io/dvgo/) **Direct Voxel Grid Optimization: Super-Fast Convergence for Radiance Fields Reconstruction** [CVPR 2022]  🔥  
[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Tancik_Block-NeRF_Scalable_Large_Scene_Neural_View_Synthesis_CVPR_2022_paper.pdf) [[project]](https://waymo.com/research/block-nerf/) **Block-NeRF: Scalable Large Scene Neural View Synthesis** [CVPR 2022]  
[[paper]](https://github.com/thucz/PanoGRF) [[paper]](https://3dvar.com/Chen2023PanoGRF.pdf) [[project]](https://thucz.github.io/PanoGRF/) **PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline Panoramas** [NeurIPS 2023]  
[[code]](https://github.com/wolfball/PlenVDB) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_PlenVDB_Memory_Efficient_VDB-Based_Radiance_Fields_for_Fast_Training_and_CVPR_2023_paper.pdf) [[project]](https://plenvdb.github.io/) **PlenVDB: Memory Efficient VDB-Based Radiance Fields for Fast Training and Rendering** [CVPR 2023]  
[[code]](https://github.com/wuminye/TeTriRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_TeTriRF_Temporal_Tri-Plane_Radiance_Fields_for_Efficient_Free-Viewpoint_Video_CVPR_2024_paper.pdf) [[project]](https://wuminye.github.io/projects/TeTriRF/) **TeTriRF: Temporal Tri-Plane Radiance Fields for Efficient Free-Viewpoint Video** [CVPR 2023]  
[[code]](https://github.com/MiZhenxing/Switch-NeRF) [[paper]](https://openreview.net/forum?id=PQ2zoIZqvm) [[project]](https://mizhenxing.github.io/switchnerf/) **Switch-NeRF: Learning Scene Decomposition with Mixture of Experts for Large-scale Neural Radiance Fields** [ICLR 2023]  
[[code]](https://github.com/cmusatyalab/mega-nerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Turki_Mega-NERF_Scalable_Construction_of_Large-Scale_NeRFs_for_Virtual_Fly-Throughs_CVPR_2022_paper.pdf) [[project]](https://meganerf.cmusatyalab.org/) **Mega-NeRF: Scalable Radiance Fields for Large Outdoor Scenes** [CVPR 2023]  
[[SLIDES]](https://cvpr.thecvf.com/media/cvpr-2024/Slides/30207_zHbzPLg.pdf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Grounding_and_Enhancing_Grid-based_Models_for_Neural_Fields_CVPR_2024_paper.pdf) [[project]](https://sites.google.com/view/cvpr24-2034-submission/home) **Grounding and Enhancing Grid-Based Models for Neural Fields** [CVPR 2024]  
[[code]](https://github.com/autonomousvision/murf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_MuRF_Multi-Baseline_Radiance_Fields_CVPR_2024_paper.pdf) [[project]](https://haofeixu.github.io/murf/) **MuRF: Multi-Baseline Radiance Fields** [CVPR 2024]  
[[code]](https://github.com/showlab/DynVideo-E) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_DynVideo-E_Harnessing_Dynamic_NeRF_for_Large-Scale_Motion-_and_View-Change_Human-Centric_CVPR_2024_paper.pdf) **DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing** [CVPR 2024]  
[[code]](https://github.com/google/hypernerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Turki_HybridNeRF_Efficient_Neural_Rendering_via_Adaptive_Volumetric_Surfaces_CVPR_2024_paper.pdf) [[project]](https://haithemturki.com/hybrid-nerf/) **HybridNeRF: Efficient Neural Rendering via Adaptive Volumetric Surfaces** [CVPR 2024]  🔥  
[[code]](https://github.com/zyqz97/Aerial_lifting) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Aerial_Lifting_Neural_Urban_Semantic_and_Building_Instance_Lifting_from_CVPR_2024_paper.pdf) [[project]](https://zyqz97.github.io/Aerial_Lifting/) **Aerial Lifting: Neural Urban Semantic and Building Instance Lifting from Aerial Imagery** [CVPR 2024]  
[[paper]](https://github.com/JiahuiLei/MoSca) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lei_MoSca_Dynamic_Gaussian_Fusion_from_Casual_Videos_via_4D_Motion_CVPR_2025_paper.pdf) [[project]](https://jiahuilei.com/projects/mosca/) **MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds** [CVPR 2024]  
[[code]](https://github.com/sail-sg/InfNeRF) [[paper]](https://arxiv.org/abs/2403.14376) [[project]](https://jiabinliang.github.io/InfNeRF.io/) **Inf-NeRF: Towards Infinite Scale NeRF Rendering with O(log n) Space Complexity** [SIGGRAPH Asia 2024]  
[[code]](https://github.com/chobao/) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Bao_Free360_Layered_Gaussian_Splatting_for_Unbounded_360-Degree_View_Synthesis_from_CVPR_2025_paper.pdf) [[project]](https://zju3dv.github.io/free360/) **Free360: Layered Gaussian Splatting for Unbounded 360-Degree View Synthesis from Extremely Sparse and Unposed Views** [CVPR 2025]  



#### Sparse Inputs & Generalization

[[code]](https://github.com/autonomousvision/graf) [[paper]](https://proceedings.neurips.cc/paper/2020/file/e92e1b476bb5262d793fd40931e0ed53-Paper.pdf) **GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis** [NeurIPS 2020]  
[[code]](https://github.com/ajayjain/DietNeRF) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Jain_Putting_NeRF_on_a_Diet_Semantically_Consistent_Few-Shot_View_Synthesis_ICCV_2021_paper.pdf) [[project]](https://www.ajayj.com/dietnerf) **Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis** [ICCV 2021]  
[[code]](https://github.com/vincentfung13/MINE) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_MINE_Towards_Continuous_Depth_MPI_With_NeRF_for_Novel_View_ICCV_2021_paper.pdf) [[project]](https://vincentfung13.github.io/projects/mine/) **MINE: Towards Continuous Depth MPI with NeRF for Novel View Synthesis** [ICCV 2021]  
[[code]](https://github.com/autonomousvision/giraffe) [[paper]](https://www.cvlibs.net/publications/Niemeyer2021CVPR.pdf) [[project]](https://m-niemeyer.github.io/project-pages/giraffe/index.html) **GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields** [CVPR 2021]  
[[code]](https://github.com/marcoamonteiro/pi-GAN) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chan_Pi-GAN_Periodic_Implicit_Generative_Adversarial_Networks_for_3D-Aware_Image_Synthesis_CVPR_2021_paper.pdf) [[project]](https://marcoamonteiro.github.io/pi-GAN-website/) **pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis** [CVPR 2021]  
[[code]](https://github.com/munshisanowarraihan/nerf-meta) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Tancik_Learned_Initializations_for_Optimizing_Coordinate-Based_Neural_Representations_CVPR_2021_paper.pdf) [[project]](https://www.matthewtancik.com/learnit) **Learned Initializations for Optimizing Coordinate-Based Neural Representations (MetaNeRF)** [CVPR 2021]  
[[code]](https://github.com/apchenstu/mvsnerf) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_MVSNeRF_Fast_Generalizable_Radiance_Field_Reconstruction_From_Multi-View_Stereo_ICCV_2021_paper.pdf) [[project]](https://apchenstu.github.io/mvsnerf/) **MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo** [ICCV 2021]  
[[code]](https://github.com/jchibane/srf) [[paper]](https://ieeexplore.ieee.org/document/9578451) [[project]](https://virtualhumans.mpi-inf.mpg.de/srf/) **Stereo Radiance Fields (SRF): Learning View Synthesis from Sparse Views of Novel Scenes** [CVPR 2021]  🔥  
[[code]](https://github.com/ajayjain/dietnerf) [[paper]](https://arxiv.org/abs/2104.00677) [[project]](https://ajayj.com/dietnerf) **DietNeRF: Reducing Spatial Bias in NeRF for Unsupervised Material Transfer** [arXiv 2021]  
[[code]](https://github.com/SheldonTsui/GOF_NeurIPS2021) [[paper]](https://arxiv.org/abs/2111.00969) [[project]](https://sheldontsui.github.io/projects/GOF) **Generative Occupancy Fields for 3D Surface-Aware Image Synthesis** [NeurIPS 2021]  
[[code]](https://github.com/barbararoessle/dense_depth_priors_nerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Roessle_Dense_Depth_Priors_for_Neural_Radiance_Fields_From_Sparse_Input_CVPR_2022_paper.pdf) [[project]](https://barbararoessle.github.io/dense_depth_priors_nerf/) **Dense Depth Priors for Neural Radiance Fields from Sparse Input Views** [CVPR 2022]  
[[code]](https://github.com/xxlong0/SparseNeuS) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920210.pdf) [[project]](https://www.xxlong.site/SparseNeuS/) **SparseNeuS: Fast Generalizable Neural Surface Reconstruction from Sparse Views** [ECCV 2022]  
[[code]](https://github.com/mjmjeong/InfoNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_InfoNeRF_Ray_Entropy_Minimization_for_Few-Shot_Neural_Volume_Rendering_CVPR_2022_paper.pdf) [[project]](https://cv.snu.ac.kr/research/InfoNeRF/) **InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering** [CVPR 2022]  
[[code]](https://github.com/stelzner/srt) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sajjadi_Scene_Representation_Transformer_Geometry-Free_Novel_View_Synthesis_Through_Set-Latent_Scene_CVPR_2022_paper.pdf) [[project]](https://srt-paper.github.io/) **Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representation** [CVPR 2022]  
[[code]](https://github.com/facebookresearch/StyleNeRF) [[paper]](https://jiataogu.me/style_nerf/files/StyleNeRF.pdf) [[project]](https://jiataogu.me/style_nerf/) **StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis** [ICLR 2022]  
[[code]](https://github.com/VITA-Group/SinNeRF) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820712.pdf) [[project]](https://vita-group.github.io/SinNeRF/) **SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single Imag** [ECCV 2022]  
[[code]](https://github.com/simicvm/snerg) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Single-Stage_Diffusion_NeRF_A_Unified_Approach_to_3D_Generation_and_ICCV_2023_paper.pdf) [[project]](https://hanshengchen.com/ssdnerf/) **Single-stage Diffusion NeRF: A Unified Approach to 3D Generation and Reconstruction** [ICCV 2023]  
[[code]](https://github.com/Kitsunetic/SDF-Diffusion) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shim_Diffusion-Based_Signed_Distance_Fields_for_3D_Shape_Generation_CVPR_2023_paper.pdf) [[project]](https://kitsunetic.github.io/sdf-diffusion/) **Diffusion-Based Signed Distance Fields for 3D Shape Generation** [CVPR 2023]  
[[code]](https://github.com/ActiveVisionLab/nope-nerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Bian_NoPe-NeRF_Optimising_Neural_Radiance_Field_With_No_Pose_Prior_CVPR_2023_paper.pdf) [[project]](https://nope-nerf.active.vision/) **NoPe-NeRF: Optimising neural radiance field with no pose prior** [CVPR 2023]  🔥  
[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_RODIN_A_Generative_Model_for_Sculpting_3D_Digital_Avatars_Using_CVPR_2023_paper.pdf)) [[project]](https://3d-avatar-diffusion.microsoft.com/) **Rodin: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion** [CVPR 2023]  🔥  
[[code]](https://github.com/zubair-irshad/NeO-360) [[paper]](https://arxiv.org/pdf/2308.12967) [[project]](https://zubair-irshad.github.io/projects/neo360.html) **NeO360: Neural Fields for Sparse View Synthesis of Outdoor Scenes** [ICCV 2023]  
[[code]](https://github.com/NIRVANALAN/LN3Diff) [[paper]](https://arxiv.org/pdf/2403.12019) [[project]](https://nirvanalan.github.io/projects/ln3diff/) **Ln3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation** [ECCV 2024]  
[[paper]](https://github.com/JiahuiLei/MoSca) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Chou_GSNeRF_Generalizable_Semantic_Neural_Radiance_Fields_with_Enhanced_3D_Scene_CVPR_2024_paper.pdf) [[project]](https://timchou-ntu.github.io/gsnerf/) **GSNeRF: Generalizable Semantic Neural Radiance Fields with Enhanced 3D Scene Understanding** [CVPR 2024]  🔥  
[[code]](https://github.com/leejielong/DiSR-NeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_DiSR-NeRF_Diffusion-Guided_View-Consistent_Super-Resolution_NeRF_CVPR_2024_paper.pdf) **DiSR-NeRF: Diffusion-Guided View-Consistent Super-Resolution NeRF** [CVPR 2024]  
[[code]](https://github.com/Youngju-Na/UFORecon) [[paper]](https://arxiv.org/abs/2403.05086) [[project]](https://youngju-na.github.io/uforecon.github.io/) **UFORecon: Generalizable Sparse-View Surface Reconstruction from Arbitrary and Unfavorable Set** [CVPR 2024]   
[[code]](https://github.com/ruili3/Know-Your-Neighbors) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Know_Your_Neighbors_Improving_Single-View_Reconstruction_via_Spatial_Vision-Language_Reasoning_CVPR_2024_paper.pdf) [[project]](https://ruili3.github.io/kyn/) **Know Your Neighbors: Improving Single-View Reconstruction via Spatial Vision-Language Reasoning** [CVPR 2024]  
[[code]](https://github.com/linjohnss/FrugalNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_FrugalNeRF_Fast_Convergence_for_Extreme_Few-shot_Novel_View_Synthesis_without_CVPR_2025_paper.pdf) [[project]](https://linjohnss.github.io/frugalnerf/) **FrugalNeRF: Fast Convergence for Extreme Few-shot Novel View Synthesis** [CVPR 2025]  
[[code]](https://github.com/baskargroup/SC-NeRF) [[paper]](https://arxiv.org/abs/2503.21958) [[project]](https://baskargroup.github.io/SC-NeRF/) **SC-NeRF: NeRF-based Point Cloud Reconstruction using a Stationary Camera for Agricultural Applications** [CVPR 2025]  




#### Generative Models & Editing

[[code]](https://github.com/autonomousvision/graf) [[paper]](https://proceedings.neurips.cc/paper/2020/file/e92e1b476bb5262d793fd40931e0ed53-Paper.pdf) **GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis** [NeurIPS 2020]  
[[code]](https://github.com/autonomousvision/giraffe) [[paper]](https://www.cvlibs.net/publications/Niemeyer2021CVPR.pdf) [[project]](https://m-niemeyer.github.io/project-pages/giraffe/index.html) **GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields** [CVPR 2021]  🔥  
[[code]](https://github.com/marcoamonteiro/pi-GAN) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chan_Pi-GAN_Periodic_Implicit_Generative_Adversarial_Networks_for_3D-Aware_Image_Synthesis_CVPR_2021_paper.pdf) [[project]](https://marcoamonteiro.github.io/pi-GAN-website/) **pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis** [CVPR 2021]  
[[code]](https://github.com/stevliu/editnerf) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Editing_Conditional_Radiance_Fields_ICCV_2021_paper.pdf) [[project]](http://editnerf.csail.mit.edu/) **Editing Conditional Radiance Fields** [ICCV 2021]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kaneko_AR-NeRF_Unsupervised_Learning_of_Depth_and_Defocus_Effects_From_Natural_CVPR_2022_paper.pdf) [[project]](https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/ar-nerf/) **AR-NeRF: Unsupervised Learning of Depth and Defocus Effects from Natural Images** [CVPR 2022]  
[[code]](https://github.com/Kitsunetic/SDF-Diffusion.git) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Shim_Diffusion-Based_Signed_Distance_Fields_for_3D_Shape_Generation_CVPR_2023_paper.html) **Diffusion-Based Signed Distance Fields for 3D Shape Generation** [CVPR 2023]  
[[code]](https://github.com/SamsungLabs/SPIn-NeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Mirzaei_SPIn-NeRF_Multiview_Segmentation_and_Perceptual_Inpainting_With_Neural_Radiance_Fields_CVPR_2023_paper.pdf) [[project]](https://spinnerf3d.github.io/) **SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting with Neural Radiance Fields** [CVPR 2023]   
[[code]](https://github.com/zju3dv/SINE) [[paper]](http://www.cad.zju.edu.cn/home/gfzhang/papers/sine/sine.pdf) [[project]](https://zju3dv.github.io/sine/) **SINE: Semantic-driven image-based NeRF editing with prior-guided editing field** [CVPR 2023]  
[[code]](https://github.com/kerrj/lerf) [[paper]](https://arxiv.org/abs/2303.09553) [[project]](https://www.lerf.io/) **LERF: Language Embedded Radiance Fields** [ICCV 2023]  
[[code]](https://github.com/graphdeco-inria/nerfshop) [[paper]](http://www-sop.inria.fr/reves/Basilic/2023/JKKDLD23/nerfshop.pdf) [[project]](https://repo-sam.inria.fr/fungraph/nerfshop/) **NeRFshop: Interactive Editing of Neural Radiance Fields** [I3D 2023]  🔥  
[[code]](https://github.com/simicvm/snerg) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Single-Stage_Diffusion_NeRF_A_Unified_Approach_to_3D_Generation_and_ICCV_2023_paper.pdf) [[project]](https://hanshengchen.com/ssdnerf/) **Single-stage Diffusion NeRF: A Unified Approach to 3D Generation and Reconstruction** [ICCV 2023]  
[[code]](https://github.com/ayaanzhaque/instruct-nerf2nerf) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Haque_Instruct-NeRF2NeRF_Editing_3D_Scenes_with_Instructions_ICCV_2023_paper.pdf) [[project]](https://instruct-nerf2nerf.github.io/) **Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions** [ICCV 2023]  🔥  
[[code]](https://github.com/hbai98/Componerf) [[paper]](https://vlislab22.github.io/componerf/static/videos/componerf.pdf) [[project]](https://vlislab22.github.io/componerf/) **CompoNeRF: Text-Guided Multi-object Compositional NeRF with Editable 3D Scene Layout** [ICCV 2023]  
[[code]](https://github.com/BillyXYB/FaceDNeRF) [[paper]](https://arxiv.org/abs/2306.00783) **FaceDNeRF: Semantics-Driven Face Reconstruction with Diffusion Priors** [NeurIPS 2023]  
[[code]](https://github.com/Dongjiahua/VICA-NeRF) [[paper]](https://openreview.net/pdf?id=Pk49a9snPe) [[project]](https://dongjiahua.github.io/VICA-NeRF/) **ViCA-NeRF: View-Consistency-Aware 3D Editing of Neural Radiance Fields** [Neurips 2023]  
[[code]](https://github.com/r4dl/LAENeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Radl_LAENeRF_Local_Appearance_Editing_for_Neural_Radiance_Fields_CVPR_2024_paper.pdf) [[project]](https://r4dl.github.io/LAENeRF/) **LAENeRF: Local Appearance Editing for Neural Radiance Fields** [CVPR 2024]  
[[code]](https://github.com/nerfdeformer/nerfdeformer) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_NeRFDeformer_NeRF_Transformation_from_a_Single_View_via_3D_Scene_CVPR_2024_paper.pdf) **NeRFDeformer: NeRF Transformation from a Single View via 3D Scene Flows** [CVPR 2024]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Mazzucchelli_IReNe_Instant_Recoloring_of_Neural_Radiance_Fields_CVPR_2024_paper.pdf) [[project]](https://iviazz97.github.io/irene/) **IReNe: Instant Recoloring of Neural Radiance Fields** [CVPR 2024]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Kaneko_Improving_Physics-Augmented_Continuum_Neural_Radiance_Field-Based_Geometry-Agnostic_System_Identification_with_CVPR_2024_paper.pdf) [[project]](https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/lpo/) **Improving Physics-Augmented Continuum Neural Radiance Field-Based Geometry-Agnostic System Identification with Lagrangian Particle Optimization** [CVPR 2024]  
[[code]](https://github.com/cgtuebingen/SIGNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Dihlmann_SIGNeRF_Scene_Integrated_Generation_for_Neural_Radiance_Fields_CVPR_2024_paper.pdf) [[project]](https://signerf.jdihlmann.com/) **SIGNeRF: Scene Integrated Generation for Neural Radiance Fields** [CVPR 2024]  
[[code]](https://github.com/JasonLSC/NeRFCodec_public) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_NeRFCodec_Neural_Feature_Compression_Meets_Neural_Radiance_Fields_for_Memory-Efficient_CVPR_2024_paper.pdf) **NeRFCodec: Neural Feature Compression Meets Neural Radiance Fields for Memory-Efficient Scene Representation** [CVPR 2024]  
[[code]](https://github.com/cnhaox/NeRF-HuGS) [[paper]](https://arxiv.org/pdf/2403.17537) [[project]](https://cnhaox.com/NeRF-HuGS/) **NeRF-HuGS: Improved Neural Radiance Fields in Non-static Scenes Using Heuristics-Guided Segmentation** [CVPR 2024]  🔥  
[[code]](https://github.com/jhq1234/ED-NeRF) [[paper]](https://arxiv.org/pdf/2310.02712) [[project]](https://jhq1234.github.io/ed-nerf.github.io/) **ED-NeRF: Efficient Text-Guided Editing of 3D Scene With Latent Space NeRF** [ICLR 2024]   
[[paper]](https://arxiv.org/abs/2402.08622) [[project]](https://mfischer-ucl.github.io/nerf_analogies/) **NeRF Analogies: Example-Based Visual Attribute Transfer for NeRFs,** [CVPR 2024]  
[[code]](https://github.com/ethanweber/nerfiller) [[paper]](https://arxiv.org/abs/2312.04560) [[project]](https://ethanweber.me/nerfiller/) **NeRFiller: Completing Scenes via Generative 3D Inpainting** [CVPR 2024]  
[[code]](https://github.com/kcshum/pose-conditioned-NeRF-object-fusion) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Shum_Language-driven_Object_Fusion_into_Neural_Radiance_Fields_with_Pose-Conditioned_Dataset_CVPR_2024_paper.pdf) **Language-driven Object Fusion into Neural Radiance Fields with Pose-Conditioned Dataset Updates** [CVPR 2024]   
[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_DynVideo-E_Harnessing_Dynamic_NeRF_for_Large-Scale_Motion-_and_View-Change_Human-Centric_CVPR_2024_paper.pdf) [[project]](https://zju3dv.github.io/sine/) **DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing** [CVPR 2024]  
[[code]](https://github.com/sinoyou/nelf-pro) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/You_NeLF-Pro_Neural_Light_Field_Probes_for_Multi-Scale_Novel_View_Synthesis_CVPR_2024_paper.pdf) [[project]](https://sinoyou.github.io/nelf-pro/) **NeLF-Pro: Neural Light Field Probes for Multi-Scale Novel View Synthesis** [CVPR 2024]  
[[paper]](https://openreview.net/forum?id=8HwI6UavYc) [[project]](https://replaceanything3d.github.io/) **ReplaceAnything3D: Text-Guided Object Replacement in 3D Scenes** [NeurIPS 2024]   
[[code]](https://github.com/DP-Recon/DP-Recon) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Ni_Decompositional_Neural_Scene_Reconstruction_with_Generative_Diffusion_Prior_CVPR_2025_paper.pdf) [[project]](https://dp-recon.github.io/) **Decompositional Neural Scene Reconstruction with Generative Diffusion Prior** [CVPR 2025]  
[[code]](https://github.com/kwanyun/FFaceNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yun_FFaceNeRF_Few-shot_Face_Editing_in_Neural_Radiance_Fields_CVPR_2025_paper.pdf) [[project]](https://kwanyun.github.io/FFaceNeRF_page/) **FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields** [CVPR 2025]  



### 3DGS

[[code]](https://github.com/graphdeco-inria/gaussian-splatting) [[paper]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) [[project]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) **3D Gaussian Splatting for Real-Time Radiance Field Rendering** [TOG 2023]  🔥🔥  

#### Acceleration - Optimization Algorithms, Hardware Acceleration, Adaptive Rendering

><span style="color:lightblue;">💡💡 Optimization Algorithms: Progressive frequency and sampling dominate early 2025.</span>  
><span style="color:lightblue;">💡💡 Hardware Acceleration: Mid-year shifts to GPU-specific rasterization.</span>  
><span style="color:lightblue;">💡💡 Adaptive Rendering: Late 2025 emphasizes mobile and real-time adjustments.</span>  
><span style="color:lightblue;">💡💡 Challenges and Projects: Memory bottlenecks addressed through quantization</span>  


[[code]](https://github.com/SJTU-MVCLab/SeeLe) [[paper]](https://arxiv.org/abs/2503.05168) [[project]](https://seele-project.netlify.app/) **SEELE: A Unified Acceleration Framework for Real-Time Gaussian Splatting** [arXiv 2023]  
[[code]](https://github.com/Sharath-girish/efficientgaussian) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07976.pdf)  **EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS** [ECCV 2024]  
[[code]](https://github.com/r4dl/StopThePop) [[paper]](https://dl.acm.org/doi/10.1145/3658187) [[project]](https://r4dl.github.io/StopThePop/) **StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering** [TOG 2024]  
[[code]](https://github.com/fatPeter/mini-splatting) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09866.pdf) **Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians** [ECCV 2024]  
[[code]](https://github.com/fatPeter/mini-splatting2) [[paper]](https://arxiv.org/pdf/2411.12788) **Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification** [arXiv 2024]  🔥  
[[code]](https://github.com/SaiJoshika/FreGS-3D-Gaussian-Splatting) [[paper]](https://arxiv.org/abs/2403.06908) [[project]](https://rogeraigc.github.io/FreGS-Page/) **FreGS: 3D Gaussian Splatting with Progressive Frequency Regularizations** [CVPR 2024]  
[[code]](https://github.com/donydchen/mvsplat) [[paper]](https://arxiv.org/abs/2403.14627) [[project]](https://donydchen.github.io/mvsplat/) **MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images** [ECCV 2024]  🔥  
[[code]](https://github.com/WeiPhil/nbvh) [[paper]](https://weiphil.s3.eu-central-1.amazonaws.com/neural_bvh.pdf) [[project]](https://weiphil.github.io/portfolio/neural_bvh) **N-BVH: Neural ray queries with bounding volume hierarchies** [ SIGGRAPH 2024]  
[[code]](https://github.com/hiroxzwang/adrgaussian) [[paper]](https://dl.acm.org/doi/10.1145/3680528.3687675) [[project]](https://hiroxzwang.github.io/publications/adrgaussian/) **AdR-Gaussian: Adaptive Radius** [SIGGRAPH Asia 2024]  
[[code]](https://github.com/InternLandMark/FlashGS) [[paper]](https://arxiv.org/pdf/2408.07967) [[project]](https://maxwellf1.github.io/flashgs_page/) **FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-resolution Rendering** [arXiv 2024]  
[[code]](https://github.com/humansensinglab/taming-3dgs) [[paper]](https://humansensinglab.github.io/taming-3dgs/docs/paper_lite.pdf) [[project]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) **Taming 3DGS: High-Quality Radiance Fields with Limited Resources** [SIGGRAPH Asia 2024]  
[[code]](https://github.com/zju3dv/EasyVolcap) [[paper]](https://drive.google.com/file/d/1J9Hl7KPMOkBmrt6UTFf_u9sQcjbblILA/view) [[project]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) **Representing Long Volumetric Video with Temporal Gaussian Hierarchy** [TOG 2024]  🔥🔥  
[[paper]](https://arxiv.org/pdf/2412.07293) [[project]](https://openreview.net/forum?id=EJZfcKXdiT) **Event-3DGS: Event-based 3D Reconstruction Using 3D Gaussian Splatting** [NeurIPS 2024]  
[[paper]](https://arxiv.org/pdf/2412.07608) **Faster and Better 3D Splatting via Group Training** [arxiv 2024]  
[[code]](https://github.com/YouyuChen0207/DashGaussian) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_DashGaussian_Optimizing_3D_Gaussian_Splatting_in_200_Seconds_CVPR_2025_paper.pdf) [[project]](https://dashgaussian.github.io/) **VDashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds** [CVPR 2025]  🔥🔥  
[[code]](https://github.com/Accelsnow/gaussian-splatting-distwar) [[paper]](https://arxiv.org/abs/2401.05345) **DISTWAR atomic reduction optimization on 3D Gaussian Splatting** [arXiv 2025]  
[[code]](https://github.com/city-super/Octree-GS) [[paper]](https://ieeexplore.ieee.org/document/10993308) [[project]](https://city-super.github.io/octree-gs/) **Octree-GS: Towards Consistent Real-time Rendering** [TPAMI 2025]  🔥  
[[code]](https://github.com/cvlab-kaist/PF3plat) [[paper]](https://openreview.net/pdf?id=VjI1NnsW4t) [[project]](https://cvlab-kaist.github.io/PF3plat/) **PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting** [ICML 2025]  🔥  
[[code]](https://github.com/tuallen/speede3dgs) [[paper]](https://arxiv.org/pdf/2506.07917) [[project]](https://speede3dgs.github.io/) **Speedy Deformable 3D Gaussian Splatting: Fast Rendering and Compression of Dynamic Scenes** [arXiv 2025]  🔥  
[[code]](https://github.com/hjhyunjinkim/MH-3DGS) [[paper]](https://arxiv.org/abs/2506.12945) [[project]](https://hjhyunjinkim.github.io/MH-3DGS/) **Metropolis-Hastings Sampling for 3D Gaussian Reconstruction** [NeurIPS 2025]  
[[code]](https://github.com/NVlabs/LongSplat) [[paper]](https://arxiv.org/abs/2508.14041) [[project]](https://linjohnss.github.io/longsplat/) **LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos** [ICCV 2025]  🔥  
[[code]](https://github.com/j-alex-hanson/speedy-splatg) [[paper]](https://arxiv.org/pdf/2412.00578) [[project]](https://speedysplat.github.io/) **Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives** [CVPR 2025]  
[[code]](https://github.com/ant-research/PlanarSplatting) [[paper]](https://arxiv.org/abs/2412.03451) [[project]](https://icetttb.github.io/PlanarSplatting/) **PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes** [CVPR 2025]  🔥  
[[code]](https://github.com/YaourtB/GPS_plus) [[paper]](https://arxiv.org/pdf/2411.11363) [[project]](https://yaourtb.github.io/GPS-Gaussian+) **GPS-Gaussian+: Generalizable Pixel-Wise 3D Gaussian Splatting for Real-Time Human-Scene Rendering from Sparse Views** [T-PAMI 2025]  
[[code]](https://github.com/Yuhuoo/EasySplat) [[paper]](https://arxiv.org/abs/2501.01003) **EasySplat: View-Adaptive Learning** [ICME 2025]  
[[code]](https://github.com/VITA-Group/VideoLifter) [[paper]](https://arxiv.org/abs/2501.01949) [[project]](https://videolifter.github.io/) **VideoLifter: Lifting Videos to 3D with Fast Hierarchical Stereo Alignment** [CVPR 2025]  🔥  
[[code]](https://github.com/fraunhoferhhi/Improving-ADC-3DGS) [[paper]](https://arxiv.org/pdf/2503.14274) **3D Gaussian Splatting for Real-Time Radiance Field Rendering** [ICPR 2025]  
[[code]](https://github.com/lukasHoel/3DGS-LM) [[paper]](https://arxiv.org/abs/2409.12892) [[project]](https://lukashoel.github.io/3DGS-LM/) **3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt** [ICCV 2025]  🔥  
[[paper]](https://m-niemeyer.github.io/radsplat/static/pdf/niemeyer2024radsplat.pdf) [[project]](https://lukashoel.github.io/3DGS-LM/) **RadSplat: Radiance Field-Informed Gaussian Splatting for Robust Real-Time Rendering with 900+ FPS** [3DV 2025]  
[[paper]](https://arxiv.org/abs/2409.12892) **TC-GS: A Faster and Flexible 3DGS Module Utilizing Tensor Cores** [arXiv 2025]  
[[paper]](https://arxiv.org/pdf/2505.08510) **3DGS$^2$: Near Second-order Converging 3D Gaussian Splatting** [arxiv 2025]  
[[paper]](https://arxiv.org/pdf/2505.08510) **FOCI: Trajectory Optimization on Gaussian Splats** [arxiv 2025]  
[[paper]](https://arxiv.org/pdf/2501.14534) **Trick-GS: A Balanced Bag of Tricks for Efficient Gaussian Splatting** [arxiv 2025]  
[[paper]](https://arxiv.org/pdf/2501.00342) **SG-Splatting: Accelerating 3D Gaussian Splatting with Spherical Gaussians** [arxiv 2025]  





#### Quality & Realism Improvement - Geometric Precision, Materials & Lighting, Special Conditions

><span style="color:lightblue;">💡💡 Geometric Precision: Multi-scale alignments.</span>  
><span style="color:lightblue;">💡💡 Materials & Lighting: Residual tone mappers.</span>  
><span style="color:lightblue;">💡💡 Special Conditions: Blur-agnostic and watermarking.</span>  

[[code]](https://github.com/lzhnb/GS-IR) [[paper]](https://arxiv.org/abs/2311.16473) [[project]](https://lzhnb.github.io/project-pages/gs-ir.html) **GS-IR: 3D Gaussian Splatting for Inverse Rendering** [arXiv 2025]  🔥  
[[paper]](https://arxiv.org/abs/2403.10814) [[project]](https://lukashoel.github.io/3DGS-LM/) **DarkGS: Learning Neural Illumination and 3D Gaussians Relighting for Robotic Exploration in the Dark** [ICRA 2024]  
[[code]](https://github.com/Anttwo/SuGaR) [[paper]](https://arxiv.org/abs/2311.12775) [[project]](https://anttwo.github.io/sugar/) **SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering** [CVPR 2024]  🔥  
[[code]](https://github.com/Asparagus15/GaussianShader) [[paper]](https://arxiv.org/abs/2311.17977) [[project]](https://asparagus15.github.io/GaussianShader.github.io/) **GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces** [CVPR 2024]   🔥  
[[code]](https://github.com/kcheng1021/GaussianPro) [[paper]](https://arxiv.org/abs/2402.14650) [[project]](https://kcheng1021.github.io/gaussianpro.github.io/) **GaussianPro: 3D Gaussian Splatting with Progressive Propagation** [ICML 2024]  🔥  
[[code]](https://github.com/hbb1/2d-gaussian-splatting) [[paper]](https://arxiv.org/abs/2403.17888) [[project]](https://surfsplatting.github.io/) **2D Gaussian Splatting for Geometrically Accurate Radiance Fields** [SIGGRAPH 2024]  🔥  
[[code]](https://github.com/turandai/gaussian_surfels) [[paper]](https://arxiv.org/pdf/2404.17774) [[project]](https://turandai.github.io/projects/gaussian_surfels/) **High-quality Surface Reconstruction using Gaussian Surfels** [ACM SIGGRAPH 2024]  🔥  
[[code]](https://github.com/LetianHuang/op43dgs) [[paper]](https://arxiv.org/abs/2402.00752) [[project]](https://letianhuang.github.io/op43dgs/) **On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy** [ECCV 2024]  
[[code]](https://github.com/benhenryL/Deblurring-3D-Gaussian-Splatting) [[paper]](https://arxiv.org/abs/2401.00834) [[project]](https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/) **Deblurring-3D-Gaussian-Splatting** [ECCV 2024]  
[[code]](https://github.com/NJU-3DV/Relightable3DGaussian) [[paper]](https://arxiv.org/abs/2311.16043) [[project]](https://nju-3dv.github.io/projects/Relightable3DGaussian/) **Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing** [ECCV 2024]  🔥  
[[code]](https://github.com/lzhnb/Analytic-Splatting) [[paper]](https://arxiv.org/abs/2403.11056) [[project]](https://lzhnb.github.io/project-pages/analytic-splatting/) **Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration** [ECCV 2024]  
[[code]](https://github.com/snldmt/BAGS) [[paper]](https://arxiv.org/pdf/2403.04926) [[project]](https://nwang43jhu.github.io/BAGS/) **BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Modeling** [ECCV 2024]  🔥  
[[code]](https://github.com/yanyan-li/GeoGaussian) [[paper]](https://arxiv.org/abs/2403.11324) [[project]](https://yanyan-li.github.io/project/gs/geogaussian.html) **GeoGaussian: Geometry-aware Gaussian Splatting for Scene Rendering** [ECCV 2024]  
[[code]](https://github.com/Jumponthemoon/WeatherGS) [[paper]](https://arxiv.org/pdf/2412.18862) [[project]](https://jumponthemoon.github.io/weather-gs/) **WeatherGS: 3D Scene Reconstruction in Adverse Weather Conditions via Gaussian Splatting** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2501.14231) **Micro-macro Wavelet-based Gaussian Splatting for 3D Reconstruction from Unconstrained Images** [AAAI 2025]  
[[code]](https://github.com/LiuJF1226/GaussHDR) [[paper]](https://arxiv.org/abs/2503.10143) [[project]](https://liujf1226.github.io/GaussHDR/) **GaussHDR: High Dynamic Range Gaussian Splatting** [CVPR 2025]  🔥  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wan_S2Gaussian_Sparse-View_Super-Resolution_3D_Gaussian_Splatting_CVPR_2025_paper.pdf) [[project]](https://jeasco.github.io/S2Gaussian/) **S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting** [CVPR 2025]  
[[code]](https://github.com/kuai-lab/cvpr25_3D-GSW) [[paper]](https://arxiv.org/abs/2409.13222) **3D-GSW: 3D Gaussian Splatting for Robust Watermarking** [CVPR 2025]  
[[code]](https://github.com/cuiziteng/Luminance-GS) [[paper]](https://arxiv.org/abs/2504.01503v2) [[project]](https://cuiziteng.github.io/Luminance_GS_web/) **Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment** [CVPR 2025]  
[[code]](https://github.com/garrisonz/LabelGS) [[paper]](https://arxiv.org/pdf/2508.19699) **LabelGS: Label-Aware 3D Gaussian Splatting for 3D Scene Segmentation** [PRCV 2025]  
[[code]](https://github.com/HanzhiChang/MeshSplat) [[paper]](https://arxiv.org/pdf/2508.17811) [[project]](https://hanzhichang.github.io/meshsplat_web/) **MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splattingt** [arXiv 2025]  
[[code]](https://github.com/XiaoBin2001/Improved-GS) [[paper]](https://arxiv.org/pdf/2508.12313) [[project]](https://xiaobin2001.github.io/improved-gs-web/) **Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2509.02141) [[project]](https://mohitm1994.github.io/GRMM/) **GRMM: Real-Time High-Fidelity Gaussian Morphable Head Model with Learned Residuals** [arXiv 2025]  




#### Dynamic & Deformable Scenes - 4D Representations, Motion Decomposition, Long Videos

><span style="color:lightblue;">💡💡 4D Representations: Flow-guided extensions.</span>  
><span style="color:lightblue;">💡💡 Motion Decomposition: Progressive segmentation.</span>  
><span style="color:lightblue;">💡💡 Long Videos: Robust unposed methods.</span>  

[[code]](https://github.com/jiawei-ren/dreamgaussian4d) [[paper]](https://arxiv.org/abs/2312.17142) [[project]](https://jiawei-ren.github.io/projects/dreamgaussian4d/) **DreamGaussian4D:Generative 4D Gaussian Splatting** [Arxiv 2023]  🔥  
[[code]](https://github.com/JonathonLuiten/Dynamic3DGaussians) [[paper]](https://arxiv.org/pdf/2308.09713) [[project]](https://dynamic3dgaussians.github.io/) **Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis** [3DV 2024]  🔥🔥    
[[code]](https://github.com/mikeqzy/3dgs-avatar-release) [[paper]](https://arxiv.org/abs/2312.09228) [[project]](https://neuralbodies.github.io/3DGS-Avatar/index.html) **3DGS-Avatar: Animatable Avatars via Deformable 3D Gaussian Splatting** [CVPR 2024]  
[[code]](https://github.com/ingra14m/Deformable-3D-Gaussians) [[paper]](https://arxiv.org/abs/2309.13101) [[project]](https://ingra14m.github.io/Deformable-Gaussians/) **Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction** [CVPR 2024]  🔥🔥  
[[code]](https://github.com/zju3dv/4K4D)  [[paper]](https://drive.google.com/file/d/1Y-C6ASIB8ofvcZkyZ_Vp-a2TtbiPw1Yx/view) **4K4D: Real-Time 4D View Synthesis at 4K Resolution** [CVPR 2024]  🔥🔥  
[[code]](https://github.com/VITA-Group/4DGen) [[paper]](https://vita-group.github.io/4DGen/) [[project]](https://vita-group.github.io/4DGen/) **4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency** [Arxiv 2024]  
[[code]](https://github.com/oppo-us-research/SpacetimeGaussians) [[paper]](https://arxiv.org/pdf/2312.16812) [[project]](https://oppo-us-research.github.io/SpacetimeGaussians-website/) **Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis** [CVPR 2024]  🔥  
[[code]](https://github.com/hustvl/4DGaussians) [[paper]](https://arxiv.org/pdf/2310.08528v2) [[project]](https://guanjunwu.github.io/4dgs/index.html) **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering** [CVPR 2024]  🔥🔥  
[[code]](https://github.com/zhichengLuxx/GaGS) [[paper]](https://arxiv.org/pdf/2404.06270) [[project]](https://npucvr.github.io/GaGS/) **3D Geometry-aware Deformable Gaussian Splatting for Dynamic View Synthesis** [CVPR 2024]  
[[code]](https://github.com/lizhe00/AnimatableGaussians) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Animatable_Gaussians_Learning_Pose-dependent_Gaussian_Maps_for_High-fidelity_Human_Avatar_CVPR_2024_paper.pdf) [[project]](https://animatable-gaussians.github.io/) **Animatable Gaussians: Learning Pose-Dependent Gaussian Maps for High-Fidelity Human Avatar Modeling** [CVPR 2024]  🔥🔥  
[[code]](https://github.com/moqiyinlun/HiFi4G_Dataset) [[paper]](https://arxiv.org/abs/2312.03461) [[project]](https://nowheretrix.github.io/HiFi4G/) **HiFi4G: High-Fidelity Human Performance Rendering via Compact Gaussian Splatting** [CVPR 2024]  
[[code]](https://github.com/skhu101/GauHuman) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_GauHuman_Articulated_Gaussian_Splatting_from_Monocular_Human_Videos_CVPR_2024_paper.pdf) [[project]](https://skhu101.github.io/GauHuman/) **GauHuman: Articulated Gaussian Splatting from Monocular Human Videos** [CVPR 2024]  
[[code]](https://github.com/YuelangX/Gaussian-Head-Avatar) [[paper]](https://arxiv.org/abs/2312.03029) [[project]](https://yuelangx.github.io/gaussianheadavatar/) **Gaussian Head Avatar:Ultra High-fidelity Head Avatar via Dynamic Gaussians** [CVPR 2024]  🔥  
[[code]](https://github.com/xwx0924/SurgicalGaussian) [[paper]](https://arxiv.org/abs/2407.05023) [[project]](https://surgicalgaussian.github.io/) **SurgicalGaussian: Deformable 3D Gaussians for High-Fidelity Surgical Scene Reconstruction** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2312.00112) [[project]](https://agelosk.github.io/dynmf/) **DynMF: Neural Motion Factorization for Real-time Dynamic View Synthesis with 3D Gaussian Splatting** [ECCV 2024]  
[[code]](https://github.com/markomih/SplatFields) [[paper]](https://github.com/markomih/SplatFields) [[project]](https://markomih.github.io/SplatFields/) **SplatFields:Neural Gaussian Splats for Sparse 3D and 4D Reconstruction** [ECCV 2024]  
[[code]](https://github.com/RuijieZhu94/MotionGS) [[paper]](https://openreview.net/pdf?id=6FTlHaxCpR) [[project]](https://github.com/RuijieZhu94/MotionGS) **MotionGS: Exploring Explicit Motion Guidance for Deformable 3D Gaussian Splatting** [NeurIPS 2024]  
[[code]](https://github.com/juno181/Ex4DGS) [[paper]](https://arxiv.org/abs/2410.15629) [[project]](https://leejunoh.com/Ex4DGS/) **Fully Explicit Dynamic Gaussian Splatting** [NeurIPS 2024]  
[[paper]](https://papers.neurips.cc/paper_files/paper/2024/hash/e95da8078ec8389533c802e368da5298-Abstract-Conference.html) [[project]](https://deep-diver.github.io/neurips2024/posters/0syctgl4in/) **4D Gaussian Splatting in the Wild with Uncertainty-Aware Regularization** [NeurIPS 2024]  
[[code]](https://github.com/fudan-zvg/4d-gaussian-splatting) [[paper]](https://arxiv.org/abs/2310.10642) [[project]](https://fudan-zvg.github.io/4d-gaussian-splatting/) **Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting** [ICLR 2024]  🔥  
[[code]](https://github.com/waczjoan/D-MiSo) [[paper]](https://arxiv.org/abs/2405.14276) **D-MISO: Editing Dynamic 3D Scenes Using Multi-Gaussians Soup** [NeurIPS 2024]  
[[code]](https://github.com/xg-chu/GPAvatar) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Feng_GPAvatar_High-fidelity_Head_Avatars_by_Learning_Efficient_Gaussian_Projections_CVPR_2025_paper.pdf) [[project]](https://xg-chu.site/project_gpavatar/) **GPAvatar: High-Fidelity Head Avatars by Learning Efficient Gaussian Projections** [ICLR 2024]  
[[code]](https://github.com/gqk/HiCoM) [[paper]](https://arxiv.org/pdf/2411.07541) **HiCoM: Hierarchical Coherent Motion for Dynamic Streamable Scenes with 3D Gaussian Splatting** [NeurIPS 2024]  
[[code]](https://github.com/weify627/4D-Rotor-Gaussians) [[paper]](https://arxiv.org/abs/2402.03307) [[project]](https://weify627.github.io/4drotorgs/) **4D-Rotor Gaussian Splatting: Towards Efficieent Novel View Synthesis for Dynamic Scenes** [SIGGRAPH 2024]  
[[code]](https://github.com/GuanxingLu/ManiGaussian) [[paper]](https://arxiv.org/abs/2403.08321) [[project]](https://guanxinglu.github.io/ManiGaussian/) **ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation** [ECCV 2024]  
[[code]](https://github.com/facebookresearch/D3GA) [[paper]](https://arxiv.org/pdf/2311.08581) [[project]](https://zielon.github.io/d3ga/) **D3GA - Drivable 3D Gaussian Avatars** [3DV 2025]  
[[code]](https://github.com/mlzxy/motion-blender-gs) [[paper]](https://arxiv.org/abs/2503.09040) [[project]](https://mlzxy.github.io/motion-blender-gs/) **Motion Blender Gaussian Splatting** [CoRL 2025]  
[[code]](https://github.com/wgsxm/OmniPhysGS) [[paper]](https://arxiv.org/abs/2501.18982) [[project]](https://wgsxm.github.io/projects/omniphysgs/) **OmniPhysGS: 3D Constitutive Gaussians for General Physics-based Dynamics Generation** [ICLR 2025]  
[[code]](https://github.com/brownvc/GauFRe) [[paper]](https://lynl7130.github.io/gaufre/static/pdfs/WACV_2025___GauFRe%20(1).pdf) [[project]](https://lynl7130.github.io/gaufre/index.html) **GauFRe🧇: Gaussian Deformation Fields for Real-time Dynamic Novel View Synthesis** [WACV 2025]  




#### Large-Scale & Unbounded Scenes

><span style="color:lightblue;">💡💡 City-Level Reconstructions: Progressive propagation.</span>  
><span style="color:lightblue;">💡💡 Driving Scenes: Moving object resistance.</span>  
><span style="color:lightblue;">💡💡 Parallel Optimization: Scaling parameters.</span>  

[[code]](https://github.com/kangpeilun/VastGaussian) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Lin_VastGaussian_Vast_3D_Gaussians_for_Large_Scene_Reconstruction_CVPR_2024_paper.pdf) [[project]](https://vastgaussian.github.io/) **VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction** [CVPR 2024]  🔥  
[[code]](https://github.com/Linketic/CityGaussian) [[paper]](https://arxiv.org/pdf/2404.01133) [[project]](https://dekuliutesla.github.io/citygs/) **CityGaussian Series for High-quality Large-Scale Scene Reconstruction with Gaussians** [ECCV 2024]  🔥  
[[code]](https://github.com/zju3dv/CoSurfGS) [[paper]](https://arxiv.org/abs/2412.17612) [[project]](https://gyy456.github.io/CoSurfGS/#:~:text=To%20address%20these%20issues%2C%20we%20propose%20CoSurfGS%2C%20a,based%20on%20distributed%20learning%20for%20large-scale%20surface%20reconstruction.) **CoSurfGS: Collaborative 3D Surface Gaussian Splatting with Distributed Learning for Large Scene Reconstruction** [ECCV 2024]  
[[code]](https://github.com/chengweialan/DeSiRe-GS) [[paper]](https://arxiv.org/abs/2411.11921) **DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction for Urban Driving Scenes** [Arxiv 2024]  
[[code]](https://github.com/EastbeanZhang/Gaussian-Wild) [[paper]](https://arxiv.org/abs/2403.08321) [[project]](https://eastbeanzhang.github.io/GS-W/) **Gaussian in the Wild: 3D Gaussian Splatting for Unconstrained Image Collections** [ECCV 2024]  🔥  
[[code]](https://github.com/autonomousvision/gaussian-opacity-fields) [[paper]](https://drive.google.com/file/d/1_IEpaSqDP4DzQ3TbhKyjhXo6SKscpaeq/view) [[project]](https://niujinshuchong.github.io/gaussian-opacity-fields/) **Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes** [SIGGRAPH ASIA 2024]  🔥  
[[code]](https://github.com/zhaofuq/LOD-3DGS) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3687762) [[project]](https://zhaofuq.github.io/LetsGo/) **LetsGo: Large-Scale Garage Modeling and Rendering via LiDAR-Assisted Gaussian Primitives** [SIGGRAPH Asia 2024]  🔥  
[[code]](https://github.com/AIBluefisher/DOGS) [[paper]](https://aibluefisher.github.io/DOGS/) [[project]](https://aibluefisher.github.io/DOGS/) **DOGS: Distributed-Oriented Gaussian Splatting for Large-Scale 3D Reconstruction Via Gaussian Consensus** [NeurIPS 2024]  
[[code]](https://github.com/VITA-Group/LightGaussian) [[paper]](https://arxiv.org/pdf/2311.17245) [[project]](https://lightgaussian.github.io/) **LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS** [NeurIPS 2024]  🔥  
[[code]](https://github.com/xiliu8006/3DGS-Enhancer) [[paper]](https://arxiv.org/abs/2410.16266) [[project]](https://xiliu8006.github.io/3DGS-Enhancer-project/) **3DGS-Enhancer: Enhancing Unbounded 3D Gaussian Splatting with View-consistent 2D Diffusion Priors** [NeurIPS 2024]  
[[code]](https://github.com/graphdeco-inria/hierarchical-3d-gaussians) [[paper]](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/hierarchical-3d-gaussians_low.pdf) [[project]](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/) **A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets** [ECCV 2024]  🔥🔥  
[[code]](https://github.com/nyu-systems/Grendel-GS) [[paper]](https://arxiv.org/abs/2406.18533) [[project]](https://daohanlu.github.io/scaling-up-3dgs/) **Gaussian Splatting at Scale with Distributed Training System** [ICLR 2025]  🔥  
[[code]](https://github.com/Linketic/CityGaussian/tree/CityGaussian_V2.0) [[paper]](https://dekuliutesla.github.io/CityGaussianV2/static/paper/CityGaussianV2.pdf) [[project]](https://dekuliutesla.github.io/CityGaussianV2/) **CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes** [ICLR 2025]  🔥  
[[code]](https://github.com/3DV-Coder/FGS-SLAM) [[paper]](https://arxiv.org/pdf/2503.01109) **FGS-SLAM: Fourier-based Gaussian Splatting for Real-time SLAM with Sparse and Dense Map Fusion** [IROS 2025]  
[[code]](https://github.com/Jixuan-Fan/Momentum-GS) [[paper]](https://arxiv.org/abs/2412.04887) [[project]](https://jixuan-fan.github.io/Momentum-GS_Page/) **Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction** [ICCV 2025]  



#### Sparse Inputs & Generalization

><span style="color:lightblue;">💡💡 Single/Few Views: Sparse synthesis.</span>  
><span style="color:lightblue;">💡💡 DFeed-Forward: Instant models.</span>  
><span style="color:lightblue;">💡💡 Generalization: Neural splats.</span>  

- [[code]](https://github.com/ForMyCat/SparseGS) [[paper]](https://arxiv.org/abs/2312.00206) [[project]](https://formycat.github.io/SparseGS-Real-Time-360-Sparse-View-Synthesis-using-Gaussian-Splatting/) **SparseGS: Real-Time 360° Sparse View Synthesis using Gaussian Splatting** [arXiv 2023]  
- [[code]](https://github.com/VAST-AI-Research/TriplaneGaussian) [[paper]](https://arxiv.org/abs/2312.09147) [[project]](https://zouzx.github.io/TriplaneGaussian/) **Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers** [arXiv 2023]  🔥  

- [[code]](https://github.com/VITA-Group/FSGS) [[paper]](https://arxiv.org/abs/2312.00451) [[project]](https://zehaozhu.github.io/FSGS/) **FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting** [ECCV 2024]  🔥  
- [[code]](https://github.com/dcharatan/pixelsplat) [[paper]](https://davidcharatan.com/pixelsplat/) [[project]](https://davidcharatan.com/pixelsplat/) **pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction** [ECCV 2024]  🔥🔥  
- [[code]](https://github.com/szymanowiczs/splatter-image) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Szymanowicz_Splatter_Image_Ultra-Fast_Single-View_3D_Reconstruction_CVPR_2024_paper.pdf) **Splatter Image: Ultra-Fast Single-View 3D Reconstruction** [CVPR 2024]  🔥🔥  
- [[code]](https://github.com/Fictionarry/DNGaussian) [[paper]](https://arxiv.org/abs/2403.06912) [[project]](https://guanxinglu.github.io/ManiGaussian/) **DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization** [CVPR 2024]  🔥  
- [[code]](https://github.com/Chrixtar/latentsplat) [[paper]](https://geometric-rl.mpi-inf.mpg.de/latentsplat/static/assets/latentSplat.pdf) [[project]](https://geometric-rl.mpi-inf.mpg.de/latentsplat/) **latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction** [ECCV 2024]  
- [[code]](https://github.com/SkyworkAI/Gamba) [[paper]](https://arxiv.org/abs/2403.18795) [[project]](https://florinshen.github.io/gamba-project/) **Gamba: Marry Gaussian Splatting with Mamba for Single View 3D Reconstruction** [TPAMI 2024]  🔥  
- [[code]](https://github.com/jiaw-z/CoR-GS) [[paper]](https://arxiv.org/pdf/2405.12110) [[project]](https://jiaw-z.github.io/CoR-GS/) **CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization** [ECCV 2024]  
- [[code]](https://github.com/markomih/SplatFields) [[paper]](https://github.com/markomih/SplatFields) [[project]](https://markomih.github.io/SplatFields/) **SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction** [ECCV 2024]  
- [[code]](https://github.com/donydchen/mvsplat) [[paper]](https://arxiv.org/abs/2403.14627) [[project]](https://donydchen.github.io/mvsplat/) **MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images** [ECCV 2024]  🔥🔥  

- [[paper]](https://arxiv.org/abs/2503.04314) [[project]](https://jeasco.github.io/S2Gaussian/) **S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting** [CVPR 2025]  
- [[code]](https://github.com/Open3DVLab/HiSplat) [[paper]](https://arxiv.org/pdf/2410.06245) [[project]](https://open3dvlab.github.io/HiSplat/) **HiSplat: Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction** [ICLR 2025]  
- [[code]](https://github.com/gurutvapatle/AD-GS) [[paper]](https://arxiv.org/abs/2509.11003) [[project]](https://gurutvapatle.github.io/publications/2025/ADGS.html) **AD-GS: Alternating Densification for Sparse-Input 3D Gaussian Splatting** [ACM SIGGRAPH Asia 2025]  
- [[code]](https://github.com/ranrhuang/SPFSplat) [[paper]](https://arxiv.org/abs/2403.08321) [[project]](https://ranrhuang.github.io/spfsplat/) **No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views** [ICCV 2025]  
- [[code]](https://github.com/ueoo/DropGaussian) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Park_DropGaussian_Structural_Regularization_for_Sparse-view_Gaussian_Splatting_CVPR_2025_paper.pdf) **DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting** [CVPR 2025]  
- [[code]](https://github.com/xingyoujun/transplat) [[paper]](https://xingyoujun.github.io/transplat/) [[project]](https://xingyoujun.github.io/transplat/) **TranSplat: Generalizable 3D Gaussian Splatting from Sparse Multi-View Images with Transformers** [AAAI 2025]  
- [[code]](https://github.com/chobao/) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Bao_Free360_Layered_Gaussian_Splatting_for_Unbounded_360-Degree_View_Synthesis_from_CVPR_2025_paper.pdf) [[project]](https://zju3dv.github.io/free360/) **Free360: Layered Gaussian Splatting for Unbounded 360-Degree View Synthesis from Extremely Sparse and Unposed Views** [CVPR 2025]  






#### Generative Models

><span style="color:lightblue;">💡💡 Single/Few Views: Sparse synthesis.</span>  
><span style="color:lightblue;">💡💡 DFeed-Forward: Instant models.</span>  
><span style="color:lightblue;">💡💡 Generalization: Neural splats.</span>  

- [[code]](https://github.com/dreamgaussian/dreamgaussian) [[paper]](https://arxiv.org/abs/2309.16653) [[project]](https://dreamgaussian.github.io/) **DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation** [ICLR 2024]  🔥🔥    

- [[code]](https://github.com/hustvl/GaussianDreamer) [[paper]](https://arxiv.org/abs/2310.08529) [[project]](https://taoranyi.com/gaussiandreamer/) **GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models** [CVPR 2024]  🔥  
- [[code]](https://github.com/weiqi-zhang/DiffGS) [[paper]](https://arxiv.org/abs/2410.19657) [[project]](https://junshengzhou.github.io/DiffGS/) **DiffGS: Functional Gaussian Splatting Diffusion** [NeurIPS 2024]  

- [[code]](https://github.com/hzxie/GaussianCity) [[paper]](https://arxiv.org/abs/2406.06526) **Generative Gaussian Splatting for Unbounded 3D City Generation** [CVPR 2025]  
- [[code]](https://github.com/gohyojun15/SplatFlow) [[paper]](https://arxiv.org/abs/2411.16443) [[project]](https://gohyojun15.github.io/SplatFlow/) **SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis** [CVPR 2025]  
- [[code]](https://github.com/chenguolin/DiffSplat) [[paper]](https://arxiv.org/abs/2501.16764) [[project]](https://chenguolin.github.io/projects/DiffSplat/) **DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation** [ICLR 2025]  🔥  




#### Style Transfer & Scene Editing

><span style="color:lightblue;">💡💡 Style Transfer: Neural transfers.</span>  
><span style="color:lightblue;">💡💡 Scene Editing: Language-aware.</span>  

- [[code]](https://github.com/minghanqin/LangSplat) [[paper]](https://arxiv.org/pdf/2312.16084) [[project]](https://langsplat.github.io/) **LangSplat: 3D Language Gaussian Splatting** [CVPR 2024]  🔥  

- [[code]](https://github.com/Kunhao-Liu/StyleGaussian) [[paper]](https://arxiv.org/abs/2403.07807) [[project]](https://kunhao-liu.github.io/StyleGaussian/) **StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting** [SIGGRAPH Asia 2024]  
- [[code]](https://github.com/ActiveVisionLab/gaussctrl) [[paper]](https://arxiv.org/abs/2403.08733) [[project]](https://gaussctrl.active.vision/) **GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing** [ECCV 2024]  🔥  
- [[code]](https://github.com/HarukiYqM/Reference-based-Scene-Stylization) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/076c1fa639a7190e216e734f0a1b3e7b-Paper-Conference.pdf) **Reference-based Controllable Scene Stylization** [NeurIPS 2024]  
- [[code]](https://github.com/umangi-jain/gaussiancut) [[paper]](https://openreview.net/pdf?id=Ns0LQokxa5) [[project]](https://umangi-jain.github.io/gaussiancut/) **GaussianCut: Interactive Segmentation via Graph Cut for 3D Gaussian Splatting** [NeurIPS 2024]  🔥  
- [[code]](https://github.com/bernard0047/style-splat) [[paper]](https://arxiv.org/abs/2407.09473) [[project]](https://bernard0047.github.io/stylesplat/) **StyleSplat: 3D Object Style Transfer with Gaussian Splatting** [arXiv 2024]  
- [[code]](https://github.com/HaroldChen19/gaussianvton) [[paper]](https://arxiv.org/abs/2405.07472) [[project]](https://haroldchen19.github.io/gsvton/) **GaussianVTON: 3D Human Virtual Try-ON via Multi-Stage Gaussian Splatting Editing with Image Prompting** [arXiv 2024]  

- [[code]](https://github.com/Kristen-Z/StylizedGS) [[paper]](https://arxiv.org/abs/2404.05220) [[project]](https://kristen-z.github.io/stylizedgs/) **StylizedGS: Controllable Stylization for 3D Gaussian Splatting** [TPAMI 2025]  
- [[code]](https://github.com/nianticlabs/morpheus) [[paper]](https://nianticlabs.github.io/morpheus/resources/Morpheus.pdf) [[project]](https://nianticlabs.github.io/morpheus/) **Text-Driven 3D Gaussian Splat Shape and Color Stylization** [CVPR 2025]  
- [[code]](https://github.com/juhyeon-kwon/Instruct-4DGS) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Kwon_Efficient_Dynamic_Scene_Editing_via_4D_Gaussian-based_Static-Dynamic_Separation_CVPR_2025_paper.html) [[project]](https://hanbyelcho.info/instruct-4dgs/) **Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation** [CVPR 2025]  
- [[code]](https://github.com/vpx-ecnu/ABC-GS) [[paper]](https://arxiv.org/pdf/2503.22218) [[project]](https://vpx-ecnu.github.io/ABC-GS-website/) **ABC-GS: Alignment-Based Controllable Style Transfer for 3D Gaussian Splatting** [ICME 2025]  
- [[paper]](https://arxiv.org/pdf/2506.09565) [[project]](https://semanticsplat.github.io/) **SemanticSplat: Feed-Forward 3D Scene Understanding with Language-Aware Gaussian Fields** [arXiv 2025]  





#### Robotics & SLAM

><span style="color:lightblue;">💡💡 Visual SLAM: Dense and semantic.</span>  
><span style="color:lightblue;">💡💡 Localization Navigation: Hierarchical planning.</span>  
><span style="color:lightblue;">💡💡 Active Reconstruction: Gaussian herding.</span>  

- [[code]](https://github.com/ShuhongLL/SGS-SLAM) [[paper]](https://arxiv.org/pdf/2402.03246) [[project]](https://www.youtube.com/watch?v=y83yw1E-oUo) **SGS-SLAM: Semantic Gaussian Splatting For Neural Dense SLAM** [ECCV 2024]  🔥  
- [[code]](https://github.com/hjr37/CG-SLAM) [[paper]](https://arxiv.org/abs/2403.16095) [[project]](https://zju3dv.github.io/cg-slam/) **Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field** [ECCV 2024]  
- [[code]](https://github.com/muskie82/MonoGS) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Matsuki_Gaussian_Splatting_SLAM_CVPR_2024_paper.pdf) [[project]](https://rmurai.co.uk/projects/GaussianSplattingSLAM/) **Gaussian Splatting SLAM** [CVPR 2024]  🔥🔥  
- [[code]](https://github.com/yanchi-3dv/diff-gaussian-rasterization-for-gsslam) [[paper]](https://arxiv.org/pdf/2311.11700) [[project]](https://gs-slam.github.io/) **GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting** [CVPR 2024]  
- [[code]](https://github.com/Touch-GS) [[paper]](https://arxiv.org/pdf/2403.09875) [[project]](https://arm.stanford.edu/touch-gs) **Touch-GS: Visual-Tactile Supervised 3D Gaussian Splatting** [IROS 2024]  
- [[paper]](https://openreview.net/forum?id=EyEE7547vy) [[project]](https://tyxiong23.github.io/event3dgs) **Event3DGS: Event-Based 3D Gaussian Splatting for High-Speed Robot Egomotion** [CoRL 2024]  
- [[code]](https://github.com/MrSecant/GaussianGrasper) [[paper]](https://arxiv.org/abs/2403.09637) [[project]](https://mrsecant.github.io/GaussianGrasper/) **GaussianGrasper: 3D Language Gaussian Splatting for Open-Vocabulary Robotic Grasping** [LRA 2024]  
- [[paper]](https://rffr.leggedrobotics.com/works/teleoperation/RadianceFieldsForTeleoperation.pdf) [[project]](https://rffr.leggedrobotics.com/works/teleoperation/) **Radiance Fields for Robotic Teleoperation** [IROS 2024]  
- [[code]](https://github.com/jimazeyu/GraspSplats) [[paper]](https://arxiv.org/pdf/2409.02084) [[project]](https://graspsplats.github.io/) **GraspSplats: Efficient Manipulation with 3D Feature Splatting** [CoRL 2024]  
- [[paper]](https://arxiv.org/pdf/2409.17624) [[project]](https://zijunfdu.github.io/HGS-Planner/) **HGS-Planner: Hierarchical Planning Framework for Active Scene Reconstruction Using 3D Gaussian Splatting** [ECCV 2024]  
- [[code]](https://github.com/PRBonn/PIN_SLAM) [[paper]](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2024tro.pdf) **PIN-SLAM: LiDAR SLAM using a Point-Based Implicit Neural Representation** [T-RO 2024]  🔥  

- [[code]](https://github.com/rmurai0610/MASt3R-SLAM) [[paper]](https://edexheim.github.io/mast3r-slam/) [[project]](https://edexheim.github.io/mast3r-slam/) **MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors** [CVPR 2025]  🔥🔥  
- [[code]](https://github.com/XiaohanLei/GaussNav) [[paper]](https://arxiv.org/abs/2403.11625) [[project]](https://xiaohanlei.github.io/projects/GaussNav/) **GaussNav: Gaussian Splatting for Visual Navigation** [TPAMI 2025]  
- [[code]](https://github.com/JohannaXie/GauSS-MI) [[paper]](https://www.roboticsproceedings.org/rss21/p030.pdf) **GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction** [RSS 2025]  




#### Semantic Understanding & Segmentation

><span style="color:lightblue;">💡💡 Open-Vocabulary: Language-embedded.</span>  
><span style="color:lightblue;">💡💡 Instance Segmentation: Boundary-enhanced.</span>  

- [[code]](https://github.com/XuHu0529/SAGS) [[paper]](https://arxiv.org/pdf/2401.17857) **SAGD: Boundary-Enhanced Segment Anything in 3D Gaussian via Gaussian Decomposition** [arXiv 2024]  🔥  
- [[code]](https://github.com/THU-luvision/OmniSeg3D) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Ying_OmniSeg3D_Omniversal_3D_Segmentation_via_Hierarchical_Contrastive_Learning_CVPR_2024_paper.pdf) [[project]](https://oceanying.github.io/OmniSeg3D/) **OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning** [CVPR 2024]  🔥  
- [[code]](https://github.com/ShijieZhou-UCLA/feature-3dgs) [[paper]](https://arxiv.org/abs/2312.03203) [[project]](https://feature-3dgs.github.io/) **Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields** [CVPR 2024]  🔥🔥  
- [[code]](https://github.com/buaavrcg/LEGaussians) [[paper]](https://arxiv.org/abs/2403.08321) [[project]](https://guanxinglu.github.io/ManiGaussian/) **LEGaussians: Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding** [CVPR 2024]  🔥  
- [[code]](https://github.com/sharinka0715/semantic-gaussians) [[paper]](https://arxiv.org/pdf/2403.15624) [[project]](https://sharinka0715.github.io/semantic-gaussians/) **Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting** [ECCV 2024]  🔥  
- [[paper]](https://arxiv.org/abs/2404.14249) [[project]](https://gbliao.github.io/CLIP-GS.github.io/) **CLIP-GS: CLIP-Informed Gaussian Splatting for Real-time and View-consistent 3D Semantic Understanding** [ECCV 2024]  🔥  
- [[code]](https://github.com/GuanxingLu/ManiGaussian) [[paper]](https://arxiv.org/abs/2311.18482) [[project]](https://buaavrcg.github.io/LEGaussians/) **ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation** [ECCV 2024]  🔥  
- [[paper]](https://arxiv.org/pdf/2401.05925) [[project]](https://david-dou.github.io/CoSSegGaussians/) **CoSSegGaussians: Compact and Swift Scene Segmenting 3D Gaussians with Dual Feature Fusion** [arXiv 2024]  🔥  
- [[code]](https://mbjurca.github.io/rt-gs2/) [[paper]](https://mbjurca.github.io/rt-gs2/) [[project]](https://mbjurca.github.io/rt-gs2/) **RT-GS2: Real-Time Generalizable Semantic Segmentation for 3D Gaussian Representations of Radiance Fields** [BMVC 2024]  
- [[paper]](https://arxiv.org/abs/2503.22204) [[project]](https://vulab-ai.github.io/Segment-then-Splat/) **Segment then Splat: A Unified Approach for 3D Open-Vocabulary Segmentation based on Gaussian Splatting** [arXiv 2024]  🔥  

- [[code]](https://github.com/kaist-ami/Dr-Splat) [[paper]](https://arxiv.org/abs/2502.16652) [[project]](https://drsplat.github.io/) **Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration** [CVPR 2025]  
- [[paper]](https://openaccess.thecvf.com/content/CVPR2025W/OpenSUN3D/papers/Wiedmann_DCSEG_Decoupled_3D_Open-Set_Segmentation_using_Gaussian_Splatting_CVPRW_2025_paper.pdf) **DCSEG: Decoupled 3D Open-Set Segmentation using Gaussian Splatting** [CVPR 2025]  🔥  
- [[code]](https://github.com/lhj-git/InstanceGasuusian_code) [[paper]](https://arxiv.org/pdf/2411.19235) [[project]](https://lhj-git.github.io/InstanceGaussian/) **InstanceGaussian: Appearance-Semantic Joint Gaussian Representation for 3D Instance-Level Perception** [CVPR 2025]  
- [[code]](https://github.com/lifuguan/LangSurf) [[paper]](https://arxiv.org/pdf/2412.17635) [[project]](https://langsurf.github.io/) **LangSurf: Language-Embedded Surface Gaussians for 3D Scene Understanding** [arXiv 2025]  
- [[code]](https://github.com/HorizonRobotics/GLS) [[paper]](https://arxiv.org/pdf/2411.18066) [[project]](https://jiaxiongq.github.io/GLS_ProjectPage/) **GLS: Geometry-aware 3D Language Gaussian Splatting** [arXiv 2025]  
- [[code]](https://github.com/weijielyu/Gaga) [[paper]](https://arxiv.org/abs/2404.07977) [[project]](https://www.gaga.gallery/) **Gaga: Group Any Gaussians via 3D-aware Memory Bank** [arXiv 2025]  🔥  
- [[code]](https://github.com/Quyans/GOI-Hyperplane) [[paper]](https://arxiv.org/pdf/2405.17596) [[project]](https://quyans.github.io/GOI-Hyperplane) **GOI: Find 3D Gaussians of Interest with an Optimizable Open-vocabulary Semantic-space Hyperplane** [ACM MM 2025]  🔥 
- [[code]](https://github.com/google-research/foundation-model-embedded-3dgs) [[paper]](https://arxiv.org/abs/2401.01970) [[project]](https://xingxingzuo.github.io/fmgs/) **FMGS: Foundation Model Embedded 3D Gaussian Splatting for Holistic 3D Scene Understanding** [IJCV 2025]  




#### Physics Simulation & Interaction

><span style="color:lightblue;">💡💡 Open-Vocabulary: Language-embedded.</span>  
><span style="color:lightblue;">💡💡 Instance Segmentation: Boundary-enhanced.</span>  

- [[code]](https://github.com/jiawei-ren/dreamgaussian4d) [[paper]](https://arxiv.org/abs/2312.17142) [[project]](https://jiawei-ren.github.io/projects/dreamgaussian4d/) **DreamGaussian4D: Generative 4D Gaussian Splatting** [arXiv 2023]  🔥  

- [[code]](https://github.com/XPandora/PhysGaussian) [[paper]](https://arxiv.org/abs/2311.12198) [[project]](https://xpandora.github.io/PhysGaussian/) **PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics** [CVPR 2024]  🔥🔥  
- [[code]](https://github.com/waczjoan/GASP) [[paper]](https://arxiv.org/abs/2409.05819) [[project]](https://waczjoan.github.io/GASP/) **GASP: Gaussian Splatting for Physic-Based Simulations** [arXiv 2024]  
- [[code]](https://ucla.app.box.com/s/yt4i3wm2i5m2ubace4l1cj3untr13ud1) [[paper]](https://arxiv.org/abs/2401.16663) [[project]](https://yingjiang96.github.io/VR-GS) **VR-GS: A Physical Dynamics-Aware Interactive Gaussian Splatting System in Virtual Reality** [SIGGRAPH 2024]  🔥🔥  
- [[code]](https://github.com/Colmar-zlicheng/Spring-Gaus) [[paper]](https://arxiv.org/abs/2403.09434) [[project]](https://zlicheng.com/spring_gaus/) **Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians** [ECCV 2024]  🔥  
- [[code]](https://github.com/liuff19/Physics3D) [[paper]](https://arxiv.org/abs/2406.04338) [[project]](https://guanxinglu.github.io/ManiGaussian/) **Physics3D: Learning Physical Properties of 3D Gaussians via Video Diffusion** [arXiv 2024]  🔥  
- [[code]](https://github.com/a1600012888/PhysDreamer) [[paper]](https://arxiv.org/abs/2404.13026) [[project]](https://physdreamer.github.io/) **PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation** [ECCV 2024]  🔥  
- [[code]](https://github.com/bdaiinstitute/embodied_gaussians) [[paper]](https://openreview.net/forum?id=AEq0onGrN2) [[project]](https://embodied-gaussians.github.io/) **Physically Embodied Gaussian Splatting: A Visually Learnt and Physically Grounded 3D Representation for Robotics** [CRL 2024]  🔥  

- [[code]](https://github.com/wgsxm/OmniPhysGS) [[paper]](https://arxiv.org/abs/2501.18982) [[project]](https://wgsxm.github.io/projects/omniphysgs/) **OmniPhysGS: 3D Constitutive Gaussians for General Physics-based Dynamics Generation** [ICLR 2025]  
- [[code]](https://github.com/wangmiaowei/DecoupledGaussian) [[paper]](https://arxiv.org/abs/2503.05484v1) [[project]](https://wangmiaowei.github.io/DecoupledGaussian.github.io/) **DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction** [CVPR 2025]  🔥  
- [[code]](https://github.com/YuLiu-LY/ArtGS) [[paper]](https://arxiv.org/pdf/2502.19459) [[project]](https://articulate-gs.github.io/) **ArtGS: Building Interactable Replicas of Complex Articulated Objects via Gaussian Splatting** [ICLR 2025]  
- [[code]](https://github.com/EnVision-Research/Gaussian-Property) [[paper]](https://gaussian-property.github.io/) [[project]](https://gaussian-property.github.io/) **GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs** [ICCV 2025]  



#### Compression & Storage Optimization

- [[code]](https://github.com/YihangChen-ee/HAC) [[paper]](https://arxiv.org/abs/2403.14530) [[project]](https://yihangchen-ee.github.io/project_hac/) **HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression** [ECCV 2024]  🔥  
- [[code]](https://github.com/KeKsBoTer/c3dgs) [[paper]](https://arxiv.org/abs/2401.02436) [[project]](https://niedermayr.dev/c3dgs/) **Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis** [arXiv 2024]  🔥  
- [[code]](https://github.com/wyf0912/ContextGS) [[paper]](https://arxiv.org/pdf/2405.20721) **ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Model** [ECCV 2024]  
- [[code1]](https://github.com/maincold2/Compact-3DGS) [[code2]](https://github.com/maincold2/Dynamic_C3DGS/) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_Compact_3D_Gaussian_Representation_for_Radiance_Field_CVPR_2024_paper.pdf) [[project]](https://maincold2.github.io/c3dgs/) **Compact 3D Gaussian Representation for Radiance Field** [CVPR 2024]  🔥  

- [[code]](https://github.com/YihangChen-ee/HAC-plus) [[paper]](https://arxiv.org/abs/2501.12255) [[project]](https://yihangchen-ee.github.io/project_hac++/) **HAC++: Towards 100X Compression of 3D Gaussian Splatting** [TPAMI 2025]  🔥  
- [[code]](https://github.com/maincold2/OMG) [[paper]](https://arxiv.org/abs/2503.16924) [[project]](https://maincold2.github.io/omg/) **Optimized Minimal 3D Gaussian Splatting** [NeurIPS 2025]  
- [[code]](https://github.com/dai647/DF_3DGS) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Dai_Efficient_Decoupled_Feature_3D_Gaussian_Splatting_via_Hierarchical_Compression_CVPR_2025_paper.pdf) **Efficient Decoupled Feature 3D Gaussian Splatting via Hierarchical Compression** [CVPR 2025]  
- [[code]](https://github.com/w-m/3dgs-compression-survey) [[paper]](https://github.com/w-m/3dgs-compression-survey) **3DGS.zip: A survey on 3D Gaussian Splatting Compression Methods** [CGF 2025]  🔥  




#### Special Scenes & Applications  
><span style="color:lightblue;">💡💡 Panoramic: Gaussian panoramas.</span>  
><span style="color:lightblue;">💡💡 Underwater: Dynamic adaptations.</span>  
><span style="color:lightblue;">💡💡 Medical: Color enhancements.</span>  
><span style="color:lightblue;">💡💡 Transparent Objects: Geometric guidance.</span>  

- [[code]](https://github.com/ShijieZhou-UCLA/dreamscene360) [[paper]](https://arxiv.org/abs/2404.06903) [[project]](https://dreamscene360.github.io/) **DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting** [ECCV 2024]  🔥  
- [[code]](https://github.com/zju3dv/street_gaussians/) [[paper]](https://arxiv.org/abs/2401.01339) [[project]](https://zju3dv.github.io/street_gaussians/) **Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting** [ECCV 2024]  🔥🔥  
- [[code]](https://github.com/Linketic/CityGaussian) [[paper]](https://arxiv.org/pdf/2404.01133) [[project]](https://dekuliutesla.github.io/citygs/) **CityGaussian Series: High-Quality Large-Scale Scene Reconstruction with Gaussians** [ECCV 2024]  🔥  
- [[code]](https://github.com/Linketic/CityGaussian/tree/CityGaussian_V2.0) [[paper]](https://dekuliutesla.github.io/CityGaussianV2/static/paper/CityGaussianV2.pdf) [[project]](https://dekuliutesla.github.io/CityGaussianV2/) **CityGaussianV2: Efficient & Geometrically Accurate Reconstruction for Large-Scale Scenes** [ICLR 2025]  🔥  

- [[code]](https://github.com/water-splatting/water-splatting) [[paper]](https://water-splatting.github.io/paper.pdf) [[project]](https://water-splatting.github.io/) **WaterSplatting: Fast Underwater 3D Scene Reconstruction Using Gaussian Splatting** [3DV 2025]  
- [[code]](https://github.com/dxyang/seasplat/) [[paper]](https://arxiv.org/abs/2409.17345) [[project]](https://seasplat.github.io/) **SeaSplat: Representing Underwater Scenes with 3D Gaussian Splatting & Physically Grounded Image Formation** [ICRA 2025]  
- [[code]](https://github.com/PKU-VCL-Geometry/GeoSplatting) [[paper]](https://arxiv.org/abs/2410.24204) [[project]](https://pku-vcl-geometry.github.io/GeoSplatting/) **GeoSplatting: Geometry-Guided Gaussian Splatting for Physically-Based Inverse Rendering** [ICCV 2025]  

- [[code]](https://github.com/xhd0612/GaussianRoom) [[paper]](https://arxiv.org/abs/2405.19671) [[project]](https://xhd0612.github.io/GaussianRoom.github.io/) **GaussianRoom: Improving 3D Gaussian Splatting with SDF Guidance & Monocular Cues for Indoor Scene Reconstruction** [ICRA 2025]  

- [[code]](https://github.com/horizon-research/Fov-3DGS) [[paper]](https://linwk20.github.io/assets/pdf/asplos25_vr.pdf) [[project]](https://horizon-lab.org/metasapiens/) **MetaSapiens: Real-Time Neural Rendering with Efficiency-Aware Pruning & Accelerated Foveated Rendering** [ASPLOS 2025]  
- [[code]](https://github.com/alibaba/MNN) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_TaoAvatar_Real-Time_Lifelike_Full-Body_Talking_Avatars_for_Augmented_Reality_via_CVPR_2025_paper.pdf) [[project]](https://pixelai-team.github.io/TaoAvatar/) **TaoAvatar: Real-Time Lifelike Full-Body Talking Avatars for AR via 3D Gaussian Splatting** [CVPR 2025]  🔥🔥🔥  

- [[code]](https://github.com/YuQiao0303/Fancy123) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_Fancy123_One_Image_to_High-Quality_3D_Mesh_Generation_via_Plug-and-Play_CVPR_2025_paper.pdf) [[project]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_Fancy123_One_Image_to_High-Quality_3D_Mesh_Generation_via_Plug-and-Play_CVPR_2025_paper.pdf) **Fancy123: One Image to High-Quality 3D Mesh Generation via Plug-and-Play Deformation** [CVPR 2025]  





### Scene Segmentation

#### Semantic segmentation, instance segmentation, panoramic segmentation

[[code]](https://github.com/charlesq34/pointnet.git) [[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation** [CVPR 2017]  🔥  
[[code]](https://github.com/charlesq34/pointnet2.git) [[paper]](https://dl.acm.org/doi/abs/10.5555/3295222.3295263) **PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space** [NIPS 2017]  🔥  
[[code]](https://github.com/collector-m/VoxelNet_CVPR_2018_PointCloud) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.pdf) **VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection** [CVPR 2018]  
[[code]](https://github.com/yangyanli/PointCNN) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2018/file/f5f8590cd58a54e94377e6ae2eded4d9-Paper.pdf) **PointCNN: Convolution On X-Transformed Points** [NIPS 2018]  🔥  
[[code]](https://github.com/laughtervv/SGPN) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_SGPN_Similarity_Group_CVPR_2018_paper.pdf) **SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation** [CVPR 2018]  
[[code]](https://github.com/WangYueFt/dgcnn.git) [[paper]](https://dl.acm.org/doi/abs/10.1145/3326362) **Dynamic Graph CNN for Learning on Point Clouds** [TOG 2019]  🔥  
[[code]](https://github.com/HuguesTHOMAS/KPConv.git) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Thomas_KPConv_Flexible_and_Deformable_Convolution_for_Point_Clouds_ICCV_2019_paper.pdf) **KPConv: Flexible and Deformable Convolution for Point Clouds** [ICCV 2019]  🔥  
[[code]](https://github.com/Sekunde/3D-SIS) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_3D-SIS_3D_Semantic_Instance_Segmentation_of_RGB-D_Scans_CVPR_2019_paper.pdf) **3D-SIS: 3D Semantic Instance Segmentation of RGB-D Scans** [CVPR 2019]  🔥  
[[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf) **Panoptic Segmentation** [CVPR 2019]  
[[code]](https://github.com/Yang7879/3D-BoNet) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2019/file/d0aa518d4d3bfc721aa0b8ab4ef32269-Paper.pdf) **3D-BoNet: Learning Object Boundaries for 3D Instance Segmentation** [NeurIPS 2019]  
[[code]](https://github.com/QingyongHu/RandLA-Net.git) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_RandLA-Net_Efficient_Semantic_Segmentation_of_Large-Scale_Point_Clouds_CVPR_2020_paper.pdf) **RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds** [CVPR 2020]  🔥  
[[code]](https://github.com/hszhao/PointWeb) [[paper]](https://llijiang.github.io/papers/cvpr19_pointweb.pdf) **PointWeb: Enhancing Local Neighborhood Features for Point Cloud Processing** [CVPR 2020]  
[[code]](https://github.com/bowenc0221/panoptic-deeplab) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Panoptic-DeepLab_A_Simple_Strong_and_Fast_Baseline_for_Bottom-Up_Panoptic_CVPR_2020_paper.pdf) **Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Panoptic Segmentation** [CVPR 2020]  🔥  
[[code]](https://github.com/POSTECH-CVLab/point-transformer.git) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf) **Point Transformer** [ICCV 2021]  🔥  
[[code]](https://github.com/DeepSceneSeg/EfficientPS) [[paper]](https://arxiv.org/abs/2004.02307) **EfficientPS: Efficient Panoptic Segmentation** [IJCV 2021]  
[[code]](https://github.com/Pointcept/PointTransformerV2.git) [[paper]](https://papers.nips.cc/paper_files/paper/2022/hash/d78ece6613953f46501b958b7bb4582f-Abstract-Conference.html) **Point Transformer V2: Grouped Vector Attention and Partition-based Pooling** [NIPS 2022]  
[[code]](https://github.com/valeoai/rangevit) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Ando_RangeViT_Towards_Vision_Transformers_for_3D_Semantic_Segmentation_in_Autonomous_CVPR_2023_paper.pdf) **RangeViT: Towards Vision Transformers for 3D Semantic Segmentation** [CVPR 2023]  
[[code]](https://github.com/drprojects/superpoint_transformer) [[paper]](https://arxiv.org/abs/2306.08045) **Efficient 3D Semantic Segmentation with Superpoint Transformer** [ICCV 2023]  🔥  
[[code]](https://github.com/OpenMask3D/openmask3d) [[paper]](https://papers.nips.cc/paper_files/paper/2023/file/d77b5482e38339a8068791d939126be2-Paper-Conference.pdf) **Unified-Lift: Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting** [NeurIPS 2023]  
[[code]](https://github.com/yashbhalgat/Contrastive-Lift) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1cb5b3d64bdf3c6642c8d9a8fbecd019-Abstract-Conference.html) **Contrastive Lift: 3D Object Instance Segmentation by Slow-Fast Contrastive Fusion** [NeurIPS 2023]  
[[code]](https://github.com/aminebdj/3D-OWIS) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/801750bc49fdc3d498e9ee63479f315e-Abstract-Conference.html) **3D Indoor Instance Segmentation in an Open-World** [NeurIPS 2023]  
[[code]](https://github.com/Pointcept/PointTransformerV3.git) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf) **Point Transformer V3: Simpler, Faster, Stronger** [CVPR 2024]  🔥🔥    
[[code]](https://github.com/IRMVLab/SNI-SLAM) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_SNI-SLAM_Semantic_Neural_Implicit_SLAM_CVPR_2024_paper.pdf) **SNI-SLAM: Semantic Neural Implicit SLAM** [CVPR 2024]  
[[code]](https://github.com/Pointcept/Pointcept) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Peng_OA-CNNs_Omni-Adaptive_Sparse_CNNs_for_3D_Semantic_Segmentation_CVPR_2024_paper.pdf) **OA-CNNs: Omni-Adaptive Sparse CNNs for 3D Semantic Segmentation** [CVPR 2024]  🔥  
[[code]](https://github.com/rozdavid/unscene3d) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Rozenberszki_UnScene3D_Unsupervised_3D_Instance_Segmentation_for_Indoor_Scenes_CVPR_2024_paper.pdf) **UnScene3D: Unsupervised 3D Instance Segmentation for Indoor Scenes** [CVPR 2024]  
[[code]](https://github.com/SooLab/Part2Object) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02657.pdf) **Part2Object: Hierarchical Unsupervised 3D Instance Segmentation** [ECCV 2024]  
[[code]](https://github.com/RyanG41/SA3DIP) [[paper]](https://arxiv.org/abs/2411.03819) **SA3DIP: Segment Any 3D Instance with Potential 3D Priors** [NeurIPS 2024]  
[[code]](https://github.com/apple/ml-kpconvx) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Thomas_KPConvX_Modernizing_Kernel_Point_Convolution_with_Kernel_Attention_CVPR_2024_paper.pdf) **KPConvX: Modernizing Kernel Point Convolution with Kernel Attention** [CVPR 2024]  
[[code]](https://github.com/astra-vision/PaSCo) [[paper]](https://arxiv.org/abs/2312.02158) **PaSCo: Urban 3D Panoptic Scene Completion with Uncertainty Awareness** [CVPR 2024]  
[[code]](https://github.com/weiguangzhao/BFANet) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_BFANet_Revisiting_3D_Semantic_Segmentation_with_Boundary_Feature_Analysis_CVPR_2025_paper.pdf) **BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis** [CVPR 2025]  
[[code]](https://github.com/DP-Recon/DP-Recon) [[paper]](https://arxiv.org/abs/2503.14830) **Decompositional Neural Scene Reconstruction with Generative Diffusion Prior** [CVPR 2025]  🔥  
[[code]](https://kuai-lab.github.io/cvpr2025protoocc/) [[paper]](https://arxiv.org/abs/2503.15185) **3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation** [CVPR 2025]  
[[code]](https://github.com/TyroneLi/CUA_O3D) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Cross-Modal_and_Uncertainty-Aware_Agglomeration_for_Open-Vocabulary_3D_Scene_Understanding_CVPR_2025_paper.pdf) **Cross-Modal and Uncertainty-Aware Agglomeration for Open-Vocabulary 3D Scene Understanding** [CVPR 2025]  
[[code]](https://github.com/visinf/cups) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Hahn_Scene-Centric_Unsupervised_Panoptic_Segmentation_CVPR_2025_paper.pdf) **Scene-Centric Unsupervised Panoptic Segmentation** [CVPR 2025]  
[[code]](https://europe.naverlabs.com/) [[paper]](https://arxiv.org/pdf/2506.21348) **PanSt3R: Multi-view Consistent Panoptic Segmentation** [ICCV 2025]  🔥  


#### NeRF

| Code | Paper | Title | Venue | Year |
|------|-------|-------|-------|------|
| [GitHub](https://github.com/Harry-Zhi/semantic_nerf) | [Paper](https://arxiv.org/abs/2103.15875) | **Semantic-NeRF: Semantic Neural Radiance Fields** 🔥 | ICCV | 2021 |
| [GitHub](https://github.com/zyqz97/GP-NeRF) | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_GP-NeRF_Generalized_Perception_NeRF_for_Context-Aware_3D_Scene_Understanding_CVPR_2024_paper.pdf) | **GP-NeRF: Generalized Perception NeRF for Context-Aware 3D Scene Understanding** | CVPR | 2023 |
| [GitHub](https://github.com/kerrj/lerf) | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kerr_LERF_Language_Embedded_Radiance_Fields_ICCV_2023_paper.pdf) | **LERF: Language Embedded Radiance Fields 🔥** | ICCV | 2023 |
| [GitHub](https://github.com/oppo-us-research/NeuRBF) | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_NeuRBF_A_Neural_Fields_Representation_with_Adaptive_Radial_Basis_Functions_ICCV_2023_paper.pdf) | **NeurBF: A Neural Fields Representation with Adaptive Radial Basis Functions 🔥** | ICCV | 2023 |
| [GitHub](https://github.com/Jumpat/SegmentAnythingin3D) | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/525d24400247f884c3419b0b7b1c4829-Paper-Conference.pdf) | **Segment Anything in 3D with NeRFs 🔥🔥** | NeurIPS | 2023 |
| [GitHub](https://github.com/pcl3dv/OV-NeRF) | [Paper](https://ieeexplore.ieee.org/document/10630553) | **OV-NeRF: Open-Vocabulary Neural Radiance Fields with Vision and Language Foundation Models for 3D Semantic Understanding** | IEEE TCSVT | 2023 |
| [GitHub](https://github.com/IRMVLab/SNI-SLAM) | [Paper](https://arxiv.org/pdf/2311.11016) | **SNI-SLAM: Semantic Neural Implicit SLAM 🔥** | CVPR | 2024 |
| [GitHub](https://github.com/ZechuanLi/GO-N3RDet) | [Paper](https://openaccess.thecvf.com/content/CVPR_2025/papers/Li_GO-N3RDet_Geometry_Optimized_NeRF-enhanced_3D_Object_Detector_CVPR_2025_paper.pdf) | **GO-N3RDet: Geometry Optimized NeRF-enhanced 3D Object Detector** | CVPR | 2025 |
| [GitHub](https://github.com/DP-Recon/DP-Recon) | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ni_Decompositional_Neural_Scene_Reconstruction_with_Generative_Diffusion_Prior_CVPR_2025_paper.pdf) | **Decompositional Neural Scene Reconstruction with Generative Diffusion Prior 🔥** | CVPR | 2025 |
| [GitHub](https://github.com/mRobotit/Cues3D) | [Paper](https://www.sciencedirect.com/science/article/pii/S1566253525002374) | **Cues3D: Unleashing the power of sole NeRF for consistent and unique 3D instance segmentation** | Information Fusion | 2025 |

#### 3DGS

| Code | Paper | Title | Venue | Year |
|------|-------|-------|-------|------|
| [GitHub](https://github.com/minghanqin/LangSplat) | [Paper](https://arxiv.org/pdf/2312.16084) | [Project](https://langsplat.github.io/) **LangSplat: 3D Language Gaussian Splatting 🔥🔥** | CVPR | 2024 |
| [GitHub](https://github.com/THU-luvision/OmniSeg3D) | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Ying_OmniSeg3D_Omniversal_3D_Segmentation_via_Hierarchical_Contrastive_Learning_CVPR_2024_paper.pdf) | **OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning 🔥** | CVPR | 2024 |
| [GitHub](https://github.com/ShijieZhou-UCLA/feature-3dgs) | [Paper](https://arxiv.org/abs/2312.03203) | [Project](https://feature-3dgs.github.io/) **Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields 🔥🔥** | CVPR | 2024 |
| [GitHub](https://github.com/XuHu0529/SAGS) | [Paper](https://arxiv.org/pdf/2401.17857) | **SAGD: Boundary-Enhanced Segment Anything in 3D Gaussian via Gaussian Decomposition 🔥** | arXiv | 2024 |
|  |[Paper](https://arxiv.org/abs/2404.14249) | **CLIP-GS: CLIP-Informed Gaussian Splatting for Real-time and View-consistent 3D Semantic Understanding 🔥** | ECCV | 2024 |
| [GitHub](https://github.com/wxrui182/GSemSplat) | [Paper](https://arxiv.org/abs/2412.16932) | **GSemSplat: Generalizable Semantic 3D Gaussian Splatting from Uncalibrated Image Pairs** | arXiv | 2024 |
| [GitHub](https://github.com/sharinka0715/semantic-gaussians) | [Paper](https://arxiv.org/pdf/2403.15624) | **Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting 🔥** | ECCV | 2024 |
| [GitHub](https://github.com/HorizonRobotics/GLS) | [Paper](https://arxiv.org/pdf/2411.18066) | **GLS: Geometry-aware 3D Language Gaussian Splatting** | arXiv | 2024 |
| [GitHub](https://github.com/weijielyu/Gaga) | [Paper](https://arxiv.org/abs/2404.07977) | **Gaga: Group Any Gaussians via 3D-aware Memory Bank 🔥** | arXiv | 2024 |
| [GitHub](https://github.com/JojiJoseph/3dgs-gradient-segmentation) | [Paper](https://arxiv.org/abs/2409.11681) | **Gradient-Driven 3D Segmentation and Affordance Transfer in Gaussian Splatting Using 2D Masks 🔥** | arXiv | 2024 |
| [GitHub](https://github.com/insait-institute/OccamLGS) | [Paper](https://arxiv.org/abs/2412.01807) | **Occam's LGS: A Simple Approach for Language Gaussian Splatting 🔥** | arXiv | 2024 |
| [GitHub](https://github.com/WHU-USI3DV/GAGS) | [Paper](https://arxiv.org/abs/2412.13654) | **GAGS: Granularity-Aware Feature Distillation for Language Gaussian Splatting 🔥** | arXiv | 2024 |
| [GitHub](https://github.com/buaavrcg/LEGaussians) | [Paper](https://openaccess.thecvf.com/content/WACV2025W/LLVMAD/papers/Chahe_Query3D_LLM-Powered_Open-Vocabulary_Scene_Segmentation_with_Language_Embedded_3D_Gaussians_WACVW_2025_paper.pdf) | **Query3D: LLM-Powered Open-Vocabulary Scene Segmentation with Language Embedded 3D Gaussian 🔥** | CVPR | 2024 |
| [GitHub](https://github.com/MyrnaCCS/contrastive-gaussian-clustering) | [Paper](https://arxiv.org/abs/2404.12784) | **Contrastive Gaussian Clustering: Weakly Supervised 3D Scene Segmentation** | ICPR | 2024 |
| [GitHub](https://github.com/umangi-jain/gaussiancut) | [Paper](https://openreview.net/pdf?id=Ns0LQokxa5) | [Project](https://umangi-jain.github.io/gaussiancut/) **GaussianCut: Interactive segmentation via graph cut for 3D Gaussian Splatting** | NeurIPS | 2024 |
|  |[Paper](https://arxiv.org/pdf/2412.00392) | **GradiSeg: Gradient-Guided Gaussian Segmentation with Enhanced 3D Boundary Precision 🔥🔥** | arXiv | 2024 |
|  |[Paper](https://arxiv.org/pdf/2412.10231) |  **SuperGSeg: Open-Vocabulary 3D Segmentation with Structured Super-Gaussians 🔥🔥** | arXiv | 2024 |
| [GitHub](https://github.com/wangyuyy/PLGS) | [Paper](https://arxiv.org/pdf/2410.17505v2) | **PLGS: Robust Panoptic Lifting with 3D Gaussian Splatting** | TIP | 2025 |
|  |[Paper](https://openaccess.thecvf.com/content/CVPR2025W/OpenSUN3D/papers/Wiedmann_DCSEG_Decoupled_3D_Open-Set_Segmentation_using_Gaussian_Splatting_CVPRW_2025_paper.pdf) | **DCSEG: Decoupled 3D Open-Set Segmentation using Gaussian Splatting 🔥** | CVPR | 2025 |
| [GitHub](https://github.com/lifuguan/LangSurf) | [Paper](https://arxiv.org/pdf/2412.17635) | [Project](https://langsurf.github.io/) **LangSurf: Language-Embedded Surface Gaussians for 3D Scene Understanding 🔥** | arXiv | 2025 |
| [GitHub](https://github.com/mlzxy/motion-blender-gs) | [Paper](https://arxiv.org/abs/2503.09040) | **Motion Blender Gaussian Splatting for Dynamic Scene Reconstruction** | CoRL | 2025 |
| [GitHub](https://github.com/uynitsuj/pogs) | [Paper](https://arxiv.org/abs/2503.05189) | **Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot Manipulation of Irregularly Shaped Objects** | arXiv | 2025 |
|  |[Paper](https://arxiv.org/abs/2506.09565) |  **SemanticSplat: Feed-Forward 3D Scene Understanding with Semantic Gaussians 🔥** | arXiv | 2025 |
| [GitHub](https://github.com/RuijieZhu94/ObjectGS) | [Paper](https://arxiv.org/pdf/2507.15454) | **ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting 🔥** | ICCV | 2025 |
| [GitHub](https://github.com/Runsong123/Unified-Lift) | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Rethinking_End-to-End_2D_to_3D_Scene_Segmentation_in_Gaussian_Splatting_CVPR_2025_paper.pdf) | **Unified-Lift: Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting 🔥** | CVPR | 2025 |
| [GitHub](https://any3dis.github.io/) | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_Any3DIS_Class-Agnostic_3D_Instance_Segmentation_by_2D_Mask_Tracking_CVPR_2025_paper.pdf) | **Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking 🔥** | CVPR | 2025 |
| [GitHub](https://github.com/zhaihongjia/PanoGS) | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhai_PanoGS_Gaussian-based_Panoptic_Segmentation_for_3D_Open_Vocabulary_Scene_Understanding_CVPR_2025_paper.pdf) | **PanoGS: Gaussian-based Panoptic Segmentation for 3D Open Vocabulary Scene Understanding 🔥** | CVPR | 2025 |
| [GitHub](https://github.com/BITyia/DroneSplat) | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_DroneSplat_3D_Gaussian_Splatting_for_Robust_3D_Reconstruction_from_In-the-Wild_CVPR_2025_paper.pdf) | **DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery 🔥** | CVPR | 2025 |
| [GitHub](https://github.com/Zhao-Yian/iSegMan) | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_iSegMan_Interactive_Segment-and-Manipulate_3D_Gaussians_CVPR_2025_paper.html) | **iSegMan: Interactive Segment-and-Manipulate 3D Gaussians** | CVPR | 2025 |
| [GitHub](https://github.com/GAP-LAB-CUHK-SZ/SAMPro3D) | [Paper](https://ieeexplore.ieee.org/document/11125552) | **SAMPro3D: Locating SAM Prompts in 3D for Zero-Shot Instance Segmentation 🔥** | 3DV | 2025 |
|  |[Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Rethinking_End-to-End_2D_to_3D_Scene_Segmentation_in_Gaussian_Splatting_CVPR_2025_paper.pdf) |  **Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting 🔥** | CVPR | 2025 |

