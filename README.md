<h1 align="center">
  <b>Awesome-Scene-Representation-Reconstruction-Understanding</b>
</h1>

📦 A curated collection of papers, datasets, codebases, and resources on Scene Representation, Reconstruction, and Understanding. This list covers both classical and modern approaches, including NeRF, 3D Gaussian Splatting (3DGS), multi-view geometry, SLAM, semantic parsing, neural scene rendering, and embodied perception.

## 😄😄 <span style="color:red;">Under Construction</span>  😄😄

## [👍 Scene Representation](#scene-representation)  
## [🔥 Scene Reconstruction](#scene-reconstruction)  
## [🚀 Scene Understanding](#scene-understanding)  

---

## Scene Representation
> #### <span style="color:lightblue;">💡 Scene Representation: This refers to the data structures or mathematical models used to store, manipulate, and render reconstructed scene information. It serves as the “language” computers employ to describe the three-dimensional world. The choice of representation—whether discrete point clouds, meshes, or continuous neural fields—directly determines the efficiency and possibilities of subsequent processing.</span>  

> #### <span style="color:Lightpink;">💡💡 Representation methods for three-dimensional scenes can be broadly categorized into two main types: explicit representation and implicit representation. Explicit representations directly define geometric shapes, such as point clouds, voxels, and polygonal meshes, which describe an object's surface or volume through a set of discrete elements (points, cubes, or polygons). In contrast, implicit representations define geometry through a function, where the object's surface is typically a level set of that function (e.g., the zero level set). Symbolic distance functions (SDFs) represent a classic implicit representation, while the recently emerging neural radiance fields (NeRFs) learn a continuous function via neural networks to represent the volumetric properties of an entire scene. The latest 3D Gaussian Splatter (3DGS) can be viewed as a modern hybrid approach, employing explicit primitives (Gaussian functions) whose parameters are learned through an optimization process similar to neural fields.</span>  


>#### Table:Comparison of Representation Methods Across Different Scenarios
| Representation Method | Core Primitive               | Data Structure             | Nature | Key Advantages                                                 | Key Limitations                                                              |
|------------------------|------------------------------|-----------------------------|--------|----------------------------------------------------------------|-------------------------------------------------------------------------------|
| Point Cloud            | 3D points (x, y, z, …)       | Unordered point set         | Explicit | Flexible; directly acquired from sensors                      | Unstructured; lacks topology; irregular and incomplete                        |
| Voxel Grid             | Cubic volume elements (voxels) | Regular or sparse 3D grid  | Explicit | Structured layout; suitable for 3D CNNs                       | High memory consumption; limited by resolution                                |
| Polygonal Mesh         | Vertices, edges, faces       | Graph / half-edge structure | Explicit | Well-defined topology; efficient rendering; editable          | Fixed topology; difficult to represent complex or non-manifold geometry       |
| NeRF                  | -                            | Neural network (MLP)        | Implicit | Photorealistic rendering; models complex view-dependent effects | Slow training and inference; difficult to modify or edit                      |
| 3DGS (3D Gaussian Splatting) | 3D Gaussian primitives  | Gaussian parameter set       | Hybrid | Real-time rendering; high visual fidelity                     | Large storage footprint; editing challenges; often depends on SfM             |



### Point cloud
> #### <span style="color:lightblue;">💡 Point Clouds: This is the most direct output format for many sensors (such as LiDAR) and reconstruction algorithms (such as SfM and MVS). It represents a simple collection of three-dimensional coordinate points (X, Y, Z), often accompanied by attributes like color (RGB) and normals. Point clouds offer the advantages of simple structure and ease of acquisition. However, their drawback lies in the absence of explicit topological connection information, making it challenging to perform high-quality rendering or surface analysis directly.</span>

[[code]](https://github.com/colmap/colmap.git) [[paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf) **Structure-from-Motion Revisited**  [CVPR 2016]  
[[code]](https://github.com/charlesq34/pointnet.git) [[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)  **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation** [CVPR 2017]  
[[code]](https://github.com/charlesq34/pointnet2.git) [[paper]](https://dl.acm.org/doi/abs/10.5555/3295222.3295263) **PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space**  [NIPS 2017]  
[[code]](https://github.com/WangYueFt/dgcnn.git) [[paper]](https://dl.acm.org/doi/abs/10.1145/3326362) **Dynamic Graph CNN for Learning on Point Clouds** [TOG 2019]  
[[code]](https://github.com/HuguesTHOMAS/KPConv.git) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Thomas_KPConv_Flexible_and_Deformable_Convolution_for_Point_Clouds_ICCV_2019_paper.pdf) **KPConv: Flexible and Deformable Convolution for Point Clouds** [ICCV 2019]  
[[code]](https://github.com/DylanWusee/pointconv.git) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_PointConv_Deep_Convolutional_Networks_on_3D_Point_Clouds_CVPR_2019_paper.pdf) **PointConv: Deep Convolutional Networks on 3D Point Clouds** [CVPR 2019]  
[[code]](https://github.com/POSTECH-CVLab/point-transformer.git) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf) **Point Transformer** [ICCV 2021]  
[[code]](https://github.com/Pointcept/PointTransformerV2.git) [[paper]](https://papers.nips.cc/paper_files/paper/2022/hash/d78ece6613953f46501b958b7bb4582f-Abstract-Conference.html) **Point Transformer V2: Grouped Vector Attention and Partition-based Pooling** [NIPS 2022]  
[[code]](https://github.com/Pointcept/PointTransformerV3.git) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf) **Point Transformer V3: Simpler, Faster, Stronger** [CVPR 2024]  
[[code]](https://github.com/guochengqian/PointNeXt.git) [[paper]](https://papers.nips.cc/paper_files/paper/2022/hash/9318763d049edf9a1f2779b2a59911d3-Abstract-Conference.html)**PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies** [NIPS 2022]  
[[code]](https://github.com/Gardlin/PCR-CG.git) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700439.pdf)**PCR-CG: PCR-CG: Point Cloud Registration via Color and Geometry** [ECCV 2022]  
[[code]](https://github.com/Pointcept/Pointcept.git) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Sonata_Self-Supervised_Learning_of_Reliable_Point_Representations_CVPR_2025_paper.pdf) **Sonata: Self-Supervised Learning of Reliable Point Representations** [CVPR 2025]  
[[code]](https://github.com/QingyongHu/RandLA-Net.git) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_RandLA-Net_Efficient_Semantic_Segmentation_of_Large-Scale_Point_Clouds_CVPR_2020_paper.pdf) **RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds** [CVPR 2020]  
[[code]](https://github.com/jrryzh/pointr.git) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_PoinTr_Diverse_Point_Cloud_Completion_With_Geometry-Aware_Transformers_ICCV_2021_paper.pdf) **PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers** [ICCV 2021]  
[[code]](https://github.com/facebookresearch/SparseConvNet.git) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Graham_3D_Semantic_Segmentation_CVPR_2018_paper.pdf) **3D Semantic Segmentation with Submanifold Sparse Convolutional Networks** [CVPR 2018]  

#### Point Cloud Processing Tool
[[code]](https://github.com/PointCloudLibrary/pcl.git) PCL  
>A comprehensive C++ point cloud processing library widely used for modeling, segmentation, and registration.  

[[code]](https://github.com/kzampog/cilantro.git) cilantro  
>A lightweight, fast C++ library focused on efficient point cloud data processing, providing features such as kd-trees, normal estimation, registration, and clustering.

[[code]](https://github.com/isl-org/Open3D.git) Open3D  
>A modern 3D data processing library supporting Python interfaces for point cloud/mesh operations  

[[code]](https://github.com/fwilliams/point-cloud-utils.git) point-cloud-utils  
>An easy-to-use Python library that provides many common features, such as reading and writing multiple mesh formats, point cloud sampling, distance calculation, and normal estimation.  

[[code]](https://github.com/torch-points3d/torch-points3d.git) torch-points3d  
>A PyTorch framework based on PyTorch Geometric and Hydra for running and evaluating point cloud deep learning models on standard benchmarks.  

[[code]](https://github.com/pyg-team/pytorch_geometric.git) PyTorch Geometric (PyG)  
>A broader geometric deep learning library that provides a rich set of tools and models for deep learning on irregular structured data such as graphs, point clouds, and 3D meshes.  

[[code]](https://github.com/open-mmlab/OpenPCDet.git) OpenPCDet  
>Object Detection Library Based on Point Clouds (for LiDAR Object Detection).  

[[code]](https://github.com/google/draco.git) draco  
>Google's open-source 3D mesh/point cloud compression library accelerates storage and transmission.  

[[code]](https://github.com/daavoo/pyntcloud.git) pyntcloud  
>Python Point Cloud Processing Library.  



### Voxel
> #### <span style="color:lightblue;">💡 Voxel Grids: As an extension of pixels in three-dimensional space, voxel grids divide space into regular cubic units (voxels). Each voxel can store occupancy information (i.e., whether the space is occupied), color, or other attributes. This structured representation facilitates Boolean operations and volumetric analysis, but its memory consumption increases cubically with resolution and can produce jagged edges due to discretization.</span>

[[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Rematas_Neural_Voxel_Renderer_Learning_an_Accurate_and_Controllable_Rendering_Tool_CVPR_2020_paper.pdf) **Neural Voxel Renderer: Learning an Accurate and Controllable Rendering Tool** [CVPR 2020]  
[[code]](https://github.com/facebookresearch/NSVF.git) [[paper]](https://proceedings.neurips.cc/paper/2020/hash/b4b758962f17808746e9bb832a6fa4b8-Abstract.html)  **Neural Sparse Voxel Fields** [NIPS 2020]  
[[code]](https://github.com/IGLICT/OctField.git) [[paper]](https://proceedings.neurips.cc/paper/2021/hash/698d51a19d8a121ce581499d7b701668-Abstract.html) **OctField: Hierarchical Implicit Functions for 3D Modeling** [NIPS 2021]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_OcTr_Octree-Based_Transformer_for_3D_Object_Detection_CVPR_2023_paper.pdf) **OcTr: Octree-based Transformer for 3D Object Detection** [CVPR 2023]  
[[code]](https://github.com/NVlabs/svraster.git) [[paper]](https://svraster.github.io/SVRaster.pdf) **SparseVoxelsRasterization:Real-timeHigh-fidelityRadianceFieldRendering** [CVPR 2025]   
[[code]](https://github.com/autonomousvision/voxgraf.git) [[paper]](https://www.cvlibs.net/publications/Schwarz2022NEURIPS.pdf) **VoxGRAF:Fast 3D-Aware Image Synthesis with Sparse Voxel Grids** [NeurIPS 2022]  
[[code]](https://github.com/autonomousvision/occupancy_networks.git) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953655) **Occupancy Networks: Learning 3D Reconstruction in Function Space** [CVPR 2019]  
[[code]](https://github.com/cvlab-yonsei/HVPR.git) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Noh_HVPR_Hybrid_Voxel-Point_Representation_for_Single-Stage_3D_Object_Detection_CVPR_2021_paper.pdf) **HVPR: Hybrid Voxel-Point Representation for Single-stage 3D Object Detection** [CVPR 2021]  
[[code]](https://github.com/GWxuan/TSP3D.git) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Text-guided_Sparse_Voxel_Pruning_for_Efficient_3D_Visual_Grounding_CVPR_2025_paper.pdf) **Text-guided Sparse Voxel Pruning for Efficient 3D Visual Grounding** [CVPR 2025]  


#### Voxel Processing Tool
[[code]](https://github.com/NVIDIA/MinkowskiEngine.git) MinkowskiEngine  
>High-dimensional sparse convolution library (includes point cloud convolution).

[[code]](https://github.com/traveller59/spconv.git) spconv  
>A library providing highly optimized implementations for spatial sparse convolutions, with particular emphasis on NVIDIA Tensor Core support for ultimate performance.

[[code]](https://github.com/AcademySoftwareFoundation/openvdb.git) openvdb  
>C++ library providing storage and manipulation of sparse voxel meshes (for use in film effects, etc.)

[[code]](https://github.com/ethz-asl/voxblox.git) Voxblox  
>A TSDF voxel terrain library released by ETH ASL  

[[code]](https://github.com/victorprad/InfiniTAM.git) InfiniTAM  
>An open-source RGB-D voxel fusion framework developed by Oxford and Cambridge Universities.

[[code]](https://github.com/OctoMap/octomap.git) OctoMap  
>A classic octree-based occupancy grid library.

[[code]](https://github.com/PRBonn/vdbfusion.git) VDBFusion   
>Efficient TSDF fusion using OpenVDB (Python/C++).

[[code]](https://github.com/NVIDIAGameWorks/kaolin.git) Kaolin(NVIDIA)  
>Deep learning 3D library with voxel data structure support and visualization.




### Polygon Mesh
> #### <span style="color:lightblue;">💡 Polygon Meshes: This has long been the dominant representation method in computer graphics. It consists of vertices, edges, and faces (typically triangles or quadrilaterals), explicitly defining an object's surface topology. Mesh representations are highly efficient for hardware-accelerated rendering and provide explicit surface information. However, generating high-quality, flawless meshes—such as those that are watertight and free of self-intersections—is often a complex process.</span>


[[code]](https://github.com/fogleman/sdf.git) [[paper]](https://graphics.stanford.edu/papers/volrange/volrange.pdf) **AVolumetric Method for Building Complex Models from Range Images** [TOG 1996]  
[[code]](https://github.com/facebookresearch/DeepSDF.git) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf) **DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation** [CVPR 2019]  
[[code]](https://github.com/Kitsunetic/SDF-Diffusion.git) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Shim_Diffusion-Based_Signed_Distance_Fields_for_3D_Shape_Generation_CVPR_2023_paper.html) **Diffusion-Based Signed Distance Fields for 3D Shape Generation** [CVPR 2023]  
[[code]](https://github.com/ranahanocka/MeshCNN.git) [[paper]](https://www.semanticscholar.org/reader/d47959e1f541d09f31dca8317b2f13c844379c4a) **MeshCNN: A Network with an Edge** [TOG 2019]  
[[code]](https://github.com/sw-gong/spiralnet_plus.git) [[paper]](Gong_SpiralNet_A_Fast_and_Highly_Efficient_Mesh_Convolution_Operator_ICCVW_2019_paper) **SpiralNet++: A Fast and Highly Efficient Mesh Convolution Operator** [ICCV ]2019  
[[code]](https://github.com/facebookresearch/meshrcnn.git) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gkioxari_Mesh_R-CNN_ICCV_2019_paper.pdf) **Mesh R-CNN** [ICCV 2019]  
[[code]](https://github.com/HTDerekLiu/neuralSubdiv.git) [[paper]](https://arxiv.org/abs/2005.01819) **Neural Subdivision** [TOG 2020]  
[[code]](https://github.com/cvlab-epfl/MeshSDF.git) [[paper]](https://proceedings.neurips.cc/paper/2020/file/fe40fb944ee700392ed51bfe84dd4e3d-Paper.pdf) **MeshSDF: Differentiable Iso-Surface Extraction** [NIPS 2020]  
[[code]](https://github.com/nywang16/Pixel2Mesh.git) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wen_Pixel2Mesh_Multi-View_3D_Mesh_Generation_via_Deformation_ICCV_2019_paper.pdf) **Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation** [ICCV 2021]  
[[code]](https://github.com/czq142857/NMC.git) [[paper]](https://arxiv.org/abs/2106.11272) **Neural Marching Cubes** [TOG 2021]  
[[code]](https://github.com/lzzcd001/MeshDiffusion.git) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_DPMesh_Exploiting_Diffusion_Prior_for_Occluded_Human_Mesh_Recovery_CVPR_2024_paper.pdf) **DPMesh: Exploiting Diffusion Prior for Occluded Human Mesh Recovery**  [CVPR 2024]  
[[code]](https://github.com/syb7573330/im2avatar.git) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Gerogiannis_Arc2Avatar_Generating_Expressive_3D_Avatars_from_a_Single_Image_via_CVPR_2025_paper.pdf) **Im2Avatar: Colorful 3D Reconstruction from a Single Image** [CVPR 2025]  
[[code]](https://github.com/MeshFormer/MeshFormer.git) [[paper]](https://neurips.cc/virtual/2024/oral/97945) **MeshFormer: High-Quality Mesh Generation with 3D-Guided Reconstruction Model** [NIPS 2024]  


>#### Geometry Processing Library
| Library        | Primary Languages   | Core Features                                             | Differentiable Rendering | Target Use Cases                  |
|----------------|----------------------|-----------------------------------------------------------|---------------------------|----------------------------------|
| PyTorch3D      | Python / C++ / CUDA  | Heterogeneous mesh processing, 3D operators               | Yes                       | 3D deep learning research        |
| NVIDIA Kaolin  | Python / C++ / CUDA  | GPU-accelerated operations, data loading                  | Yes (DIB-R)               | 3D deep learning research        |
| libigl         | C++ / Python         | Classical geometry processing algorithms, header-only lib | No                        | Traditional algorithms, fast prototyping |
| OpenMesh       | C++ / Python         | Efficient mesh data structures                            | No                        | Mesh editing and geometry processing |





### NeRF(Neural Radiance Fields)
> #### <span style="color:lightblue;">💡💡 NeRF represents scene light fields through neural networks, constituting an implicit volumetric representation. It employs one or more multilayer perceptrons (MLPs) to map spatial coordinates $(x,y,z)$ and viewpoint directions to color and volumetric density, thereby defining a continuous, differentiable volumetric field. This representation does not utilize explicit geometry (such as point clouds or meshes), instead training network parameters (typically several megabytes in size) to encode the scene. Common data formats include: input data (images and camera poses) are often stored as .npz (used by Mildenhall et al.'s TinyNeRF examples) or .json (e.g., InstantNGP/NeRFStudio's transforms.json stores camera intrinsic and extrinsic parameters); trained models save weights in formats like .pth/.pt or .npz; some frameworks also use serialization formats such as .msgpack or custom .nerf. NeRF's representation principle is based on volumetric rendering: rays intersect voxels, and the density and color predicted by the network are synthesized into final pixels via volumetric rendering equations. Unlike traditional voxels, NeRF uses continuous functions for implicit representation, efficiently expressing view-dependent lighting and geometric details.</span>  
[[code]](https://github.com/bmild/nerf.git) [[paper]](https://arxiv.org/abs/2003.08934)  **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** [ECCV 2020]  



### 3DGS(3D Gaussian Splatting)
> #### <span style="color:lightblue;">💡💡 3D Gaussian Point Clouds explicitly represent scenes using a set of colored spatial Gaussian volumetric elements (“points”), each possessing attributes such as position, shape (covariance/rotation), color, and transparency. This approach combines point cloud and volumetric rendering concepts: during rendering, each Gaussian volumetric element is projected onto the image for light summation. Common data formats include: .ply or custom formats exportable from raw point cloud models, while Gaussian tiling models themselves can be saved as .ply, .npz, or specialized formats like the newly introduced .spz (SPlatZip) for compressing Gaussian parameters. Its representation principle involves optimizing Gaussian position, size, and color to match the target view, enabling extremely fast rendering with photorealistic quality. Unlike NeRF, Gaussian Tiling employs an explicit sparse representation (each Gaussian voxel can be visualized as a colored, non-uniform point cloud). The trained model features fewer parameters, and rendering involves an explicit stitching/compositing process, making it well-suited for real-time applications. </span>  
[[code]](https://github.com/graphdeco-inria/gaussian-splatting.git) [[paper]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) **3D Gaussian Splatting for Real-Time Radiance Field Rendering** [TOG 2023]  











## Scene Reconstruction
#### Concept
> #### <span style="color:lightblue;">💡 Scene reconstruction aims to capture geometric and optical properties from the real world to create digital 3D models. Its essence lies in inferring the three-dimensional structure of a scene from a series of 2D images, depth maps, or laser scan data. Scene reconstruction is the process of converting real-world scenes into digital 3D models, serving as a foundational technology for fields such as VR/AR and autonomous driving.</span>  
#### Data Acquisition
> <span style="color:lightblue;">💡💡 Passive Methods (Image-Based): These techniques rely on ambient light, using cameras to capture 2D images and infer 3D structure. Their core principle involves analyzing pixel correspondences between images taken from different angles to recover depth information through triangulation. This forms the basis of photogrammetry, offering low cost and ease of deployment, though it remains sensitive to lighting conditions and surface texture.</span>  
> <span style="color:lightblue;">💡💡 Active Methods (Sensor-Based): These techniques directly acquire distance information by actively emitting energy (e.g., laser or structured light) and measuring reflected signals. Laser scanning and LiDAR are representative technologies within this category. They rapidly capture millions of precise 3D coordinate points, generating high-density, high-precision point cloud data that directly reflects the surface geometry of a scene.</span>  



### Traditional Geometric Methods Phase (1980s–Early 2010s)
><span style="color:lightblue;">💡💡 In the early developmental stages of 3D scene reconstruction, research centered on building robust systems from scratch based on fundamental principles of the physical world. This era was dominated by classical theories such as projective geometry and beam-based registration, with its core challenge being the solution to a massive, non-convex optimization problem: how to simultaneously infer both 3D structure and camera motion from only 2D images.</span> 

#### Multi-View Geometry
>> <span style="color:lightblue;">💡💡 All work from the classical era rests upon a solid mathematical foundation, systematically expounded in Richard Hartley and Andrew Zisserman's seminal work, Multi-View Geometry in Computer Vision. These fundamental principles form the theoretical bedrock for all subsequent algorithms.
</span>  

##### Projective Geometry
>>> <span style="color:lightblue;">💡💡 This mathematical framework describes how points in the 3D world are mapped onto the 2D image plane. It provides the theoretical language for understanding perspective projection, camera distortion, and the geometric relationships between multiple views.
</span>  

##### Camera Model
>>> <span style="color:lightblue;">💡💡 The pinhole camera model is the core abstraction, describing the imaging process through intrinsic parameters (focal length, principal points) and extrinsic parameters (rotations, translations). Accurate camera calibration is a prerequisite for reconstruction precision.
</span>  

##### Polarity Constraint
>>> <span style="color:lightblue;">💡💡 This fundamental geometric relationship links two views of the same scene, described by the Fundamental Matrix and Essential Matrix. This constraint is not only central to two-view reconstruction but also a powerful tool for validating feature matching accuracy, effectively eliminating erroneous matches.   
</span>  

##### Bundle Adjustment (BA)
>>> <span style="color:lightblue;">💡💡 As the gold standard for 3D reconstruction, BA is a joint nonlinear optimization process that simultaneously adjusts all camera parameters and 3D point coordinates to minimize the reprojection error of 3D points across all views. Nearly all classical reconstruction methods rely on BA as the final refinement step to achieve globally consistent and high-precision results.   
</span>  



#### Structure from Motion (SFM)
>><span style="color:lightblue;">💡💡 Structure from Motion (SfM) is the process of simultaneously recovering the three-dimensional structure of a scene and the camera pose from a sequence of two-dimensional images. Its primary outputs are a sparse point cloud and a set of calibrated camera parameters. The development of SfM has primarily followed two major technical approaches: Incremental SfM and Global SfM, representing different trade-offs between robustness and efficiency.
</span>  

[[paper]](https://graphics.stanford.edu/papers/volrange/volrange.pdf) **AVolumetric Method for Building Complex Models from Range Images** [TOG 1996]  

##### Incremental SfM
>>> <span style="color:lightblue;">💡💡 The incremental (or sequential) SfM workflow begins with a robust two-view reconstruction, then iteratively adds new images one by one. With each added image, the system performs new triangulation measurements and local or global beam-based adjustments to progressively refine the model. This sequential refinement process grants the algorithm strong tolerance to errors, but it incurs high computational costs and is prone to cumulative drift when processing long sequences.
</span>  

[[paper]](https://dl.acm.org/doi/10.1145/1141911.1141964) **Photo tourism: exploring photo collections in 3D** [TOG 2006]










### Deep Learning-Driven Reconstruction (2014–2020)










## Scene Understanding

































