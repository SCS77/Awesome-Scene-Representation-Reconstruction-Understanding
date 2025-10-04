<h1 align="center">
  <b>Awesome-Scene-Representation-Reconstruction-Understanding</b>
</h1>

📦 A curated collection of papers, datasets, codebases, and resources on Scene Representation, Reconstruction, and Understanding. This list covers both classical and modern approaches, including NeRF, 3D Gaussian Splatting (3DGS), multi-view geometry, SLAM, semantic parsing, neural scene rendering, and embodied perception.

## 😄😄 <span style="color:red;">Under Construction</span>  😄😄


## [👍 Scene Representation](#scene-representation)  

### • [Point cloud](#point-cloud)  
### • [Voxel](#voxel)  
### • [VoluPolygon Mesh](#polygon-mesh)  
### • [NeRF(Neural Radiance Fields)](#nerfneural-radiance-fields)  
### • [3DGS(3D Gaussian Splatting)](#3dgs3d-gaussian-splatting)  
### • [Comparison of Different 3D Data Formats](#comparison-of-different-3d-data-formats)  


## [🔥 Scene Reconstruction](#scene-reconstruction)  

### • [Traditional Geometric Method](#traditional-geometric-method)  
### • [3D-Scene-Generation](#3d-scene-generation)  
### • [NeRF(Neural Radiance Fields)](#nerf) 
### • [3DGS(3D Gaussian Splatting)](#3dgs)  


## [🚀 Scene Understanding](#scene-understanding)  
### • [Scenario geometry understanding](#scenario-geometry-understanding)  
### • [Camera pose estimation](#-camera-pose-estimation)  
### • [Scene Segmentation](#scene-segmentation)  
### • [Scene Reasoning and Understanding](#-scene-reasoning-and-understanding)  




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




#### Viewers & Game Engine Support

##### Game Engines

- [Unity Plugin](https://github.com/aras-p/UnityGaussianSplatting)  
- [Unity Plugin (gsplat-unity)](https://github.com/wuyize25/gsplat-unity)  
- [Unity Plugin (DynGsplat-unity)](https://github.com/HiFi-Human/DynGsplat-unity) - For dynamic splattings  
- [Unreal Plugin](https://github.com/xverse-engine/XV3DGS-UEPlugin)  
- [PlayCanvas Engine](https://github.com/playcanvas/engine)  

##### Web Viewers  
**WebGL**  
- [Splat Viewer](https://github.com/antimatter15/splat)  
- [Gauzilla](https://github.com/BladeTransformerLLC/gauzilla)  
- [Interactive Viewer](https://github.com/kishimisu/Gaussian-Splatting-WebGL)  
- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)  
- [PlayCanvas Model Viewer](https://github.com/playcanvas/model-viewer)  
- [SuperSplat Viewer](https://github.com/playcanvas/supersplat-viewer)  

**WebGPU**

- [EPFL Viewer](https://github.com/cvlab-epfl/gaussian-splatting-web)  
- [WebGPU Splat](https://github.com/KeKsBoTer/web-splat)  

##### Desktop Viewers

**Linux**

- [DearGaussianGUI](https://github.com/leviome/DearGaussianGUI)  
- [LiteViz-GS](https://github.com/panxkun/liteviz-gs)  

##### Native Applications

- [Blender Add-on](https://github.com/ReshotAI/gaussian-splatting-blender-addon)  
- [Blender Add-on (KIRI)](https://github.com/Kiri-Innovation/3dgs-render-blender-addon)  
- [Blender Add-on (404—GEN)](https://github.com/404-Repo/three-gen-blender-plugin)  
- [iOS Metal Viewer](https://github.com/laanlabs/metal-splats)  
- [OpenGL Viewer](https://github.com/limacv/GaussianSplattingViewer)  
- [VR Support (OpenXR)](https://github.com/hyperlogic/splatapult)  
- [ROS2 Support](https://github.com/shadygm/ROSplat)  




### Comparison of Different 3D Data Formats
#### Mesh Formats
| Format (Ext.)     | Type               | Geometry         | Color / Normals / UV                               | Material / Texture        | Animation       | Binary / Compression | Pros                                              | Cons                                                  | Typical Use / Tools                      |
| ----------------- | ------------------ | ---------------- | -------------------------------------------------- | ------------------------- | --------------- | -------------------- | ------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------- |
| OBJ (.obj + .mtl) | Polygon mesh       | Vertices / faces | Normals & UV supported; vertex colors not standard | Basic materials via .mtl  | No (not native) | ASCII (common)       | Widely supported, simple format                   | No native compression; limited material/color support | Blender, MeshLab, Unity (via conversion) |
| STL (.stl)        | Triangle mesh      | Triangles only   | Not supported (except rare extensions)             | Not supported             | No              | ASCII / Binary       | Simple, widely used in 3D printing                | No color/texture/unit info                            | 3D printing, slicers                     |
| PLY (.ply)        | Mesh / Point cloud | Vertices / faces | Vertex colors, normals, per-vertex attributes      | Limited (extensions)      | No              | ASCII / Binary       | Rich attribute support, good for scans            | No standardized material/animation                    | PCL, Meshlab, CloudCompare               |
| OFF (.off)        | Mesh (research)    | Vertices / faces | Very basic                                         | No                        | No              | ASCII                | Very simple, good for algorithms                  | Extremely limited metadata                            | Research, algorithm prototyping          |
| 3MF (.3mf)        | Mesh (3D printing) | Mesh + objects   | Colors supported                                   | Textures & print metadata | No (typically)  | ZIP-based container  | Modern printing standard, unit & material support | More complex, less universally supported than STL     | Modern slicers, 3D printing              |


#### Scene/Animation/Exchange
| Format (Ext.)                       | Type                         | Geometry                     | Color / Normals / UV       | Materials                | Animation / Rigging                 | Binary / Compression                        | Pros                                    | Cons                                        | Typical Use / Tools                    |
| ----------------------------------- | ---------------------------- | ---------------------------- | -------------------------- | ------------------------ | ----------------------------------- | ------------------------------------------- | --------------------------------------- | ------------------------------------------- | -------------------------------------- |
| FBX (.fbx)                          | Scene + animation (Autodesk) | Mesh, curves                 | Supported                  | Supported                | Complex rigging, skeletons, morphs  | Binary / ASCII                              | Industry pipeline, powerful             | Proprietary, import/export inconsistencies  | Maya, 3ds Max, Unity, Unreal           |
| COLLADA (.dae)                      | Scene exchange (XML)         | Mesh, skeleton               | Supported                  | Supported                | Supported                           | XML (zipped possible)                       | Open standard, scene-level info         | Implementation differences cause issues     | Early pipelines, some tools            |
| glTF / GLB (.gltf / .glb)           | Modern runtime format        | Mesh (indexed + buffers)     | Normals, UV, vertex colors | Native PBR               | Animation, morph targets, skeletons | JSON + binary buffer (GLB is single binary) | “JPEG of 3D” — compact, real-time ready | Not CAD-precise; limited advanced materials | Web, real-time rendering, Unity/Unreal |
| USD / USDZ (.usd/.usda/.usdc/.usdz) | Scene description (Pixar)    | Mesh, instances, references  | Supported                  | Full PBR, custom schemas | Animation, variants, layering       | Binary / ASCII                              | Large-scale scene mgmt, versioning      | Complex, steep learning curve               | VFX, Omniverse, film pipelines         |
| Alembic (.abc)                      | Geometry cache               | Animated meshes/point caches | Geometry attributes        | Material refs only       | Frame-by-frame geometry animation   | Binary                                      | High-perf animation caching             | Not editable parameters                     | Houdini, Maya, Cinema4D                |


#### Point Cloud / LiDAR
| Format (Ext.)         | Type                    | Geometry                        | Attributes (intensity, class, time, color) | Binary / Compression                 | Pros                             | Cons                            | Typical Use / Tools                        |
| --------------------- | ----------------------- | ------------------------------- | ------------------------------------------ | ------------------------------------ | -------------------------------- | ------------------------------- | ------------------------------------------ |
| LAS / LAZ (.las/.laz) | LiDAR point cloud       | XYZ + point records             | Rich (intensity, GPS time, classification) | LAS = uncompressed, LAZ = compressed | Industry standard, metadata-rich | Points only, LAZ needs decoding | Remote sensing, surveying (PDAL, LAStools) |
| PCD (.pcd)            | PCL native point cloud  | XYZ                             | Arbitrary per-point attributes             | ASCII / Binary                       | Native to PCL                    | Less common outside PCL         | Robotics, perception pipelines             |
| E57 (.e57)            | Scanner archive         | Point cloud + scanner images    | Metadata, coordinates, images              | Binary (compressed)                  | Designed for scanner exchange    | Fewer tools than LAS            | Laser scanning, engineering                |
| PTS / XYZ (.pts/.xyz) | Simple ASCII point list | XYZ (+intensity/color optional) | Minimal                                    | ASCII (large size)                   | Simple, easy to parse            | Huge file sizes, no metadata    | Debug, quick exports                       |
| PLY (.ply)            | Mesh or point cloud     | XYZ                             | Vertex colors, normals, attributes         | ASCII / Binary                       | Flexible, research-friendly      | Not LiDAR-specialized           | Scans, reconstruction                      |



#### Volume / Voxel / Medical
| Format (Ext.)        | Type                  | Data                     | Binary / Compression           | Pros                                        | Cons                                 | Typical Use / Tools               |
| -------------------- | --------------------- | ------------------------ | ------------------------------ | ------------------------------------------- | ------------------------------------ | --------------------------------- |
| OpenVDB (.vdb)       | Sparse volume grid    | Density / SDF            | Binary                         | Very efficient sparse storage, VFX standard | Not direct mesh, requires conversion | Smoke, fluids, volumetric effects |
| DICOM (folder)       | Medical imaging       | CT/MRI slices + metadata | Binary (optionally compressed) | Standard medical format, metadata-rich      | Complex, privacy-sensitive           | Clinical imaging, radiology       |
| NIfTI (.nii/.nii.gz) | Medical/brain imaging | 3D voxel arrays          | Supports gzip                  | Popular in neuroscience                     | Not used in visualization pipelines  | MRI/CT research                   |
| RAW / VOL / MHD      | Generic volume data   | Raw voxel arrays         | Implementation-specific        | Simple, flexible                            | Requires external metadata           | Simulation, scientific data       |



#### CAD / Engineering (High Precision)
| Format (Ext.)                    | Type           | Content                  | Binary / Compression      | Pros                                   | Cons                      | Typical Use / Tools                 |
| -------------------------------- | -------------- | ------------------------ | ------------------------- | -------------------------------------- | ------------------------- | ----------------------------------- |
| STEP / IGES (.stp/.step/.iges)   | CAD parametric | NURBS, B-rep, assemblies | Text-based (standardized) | Engineering precision, units preserved | Complex, conversion-heavy | CAD/CAM, SolidWorks, CATIA, FreeCAD |
| Parasolid / SAT (.x_t/.x_b/.sat) | CAD kernel     | Precise solids           | Binary / ASCII            | High fidelity, industry kernel         | Proprietary restrictions  | Engineering, CAE/CAM pipelines      |


#### Implicit / Neural Representations
| Format (Ext.)                      | Type                     | Data                           | Compression / Storage  | Pros                               | Cons                                    | Typical Use / Tools              |
| ---------------------------------- | ------------------------ | ------------------------------ | ---------------------- | ---------------------------------- | --------------------------------------- | -------------------------------- |
| NeRF checkpoints (.ckpt/.pth/.npz) | Neural radiance fields   | Network weights + config       | Compact (weights only) | High-quality view synthesis        | Non-standard, requires inference engine | NeRF implementations, NeRFStudio |
| DeepSDF / Occupancy (.npz/.pth)    | Implicit SDF / occupancy | Network weights / latent codes | Small                  | Continuous, compact                | Not directly editable                   | Shape completion, retrieval      |
| 3D Gaussian Splatting (custom)     | Point-based Gaussians    | Position, covariance, color    | Usually binary arrays  | Real-time rendering, high fidelity | Experimental, no standard format        | Research, novel view synthesis   |





## Scene Reconstruction
#### Concept
> #### <span style="color:lightblue;">💡 Scene reconstruction aims to capture geometric and optical properties from the real world to create digital 3D models. Its essence lies in inferring the three-dimensional structure of a scene from a series of 2D images, depth maps, or laser scan data. Scene reconstruction is the process of converting real-world scenes into digital 3D models, serving as a foundational technology for fields such as VR/AR and autonomous driving.</span>  
#### Data Acquisition
> <span style="color:lightblue;">💡💡 Passive Methods (Image-Based): These techniques rely on ambient light, using cameras to capture 2D images and infer 3D structure. Their core principle involves analyzing pixel correspondences between images taken from different angles to recover depth information through triangulation. This forms the basis of photogrammetry, offering low cost and ease of deployment, though it remains sensitive to lighting conditions and surface texture.</span>  
> <span style="color:lightblue;">💡💡 Active Methods (Sensor-Based): These techniques directly acquire distance information by actively emitting energy (e.g., laser or structured light) and measuring reflected signals. Laser scanning and LiDAR are representative technologies within this category. They rapidly capture millions of precise 3D coordinate points, generating high-density, high-precision point cloud data that directly reflects the surface geometry of a scene.</span>  



### Traditional Geometric Method
><span style="color:lightblue;">💡💡 In the early developmental stages of 3D scene reconstruction, research centered on building robust systems from scratch based on fundamental principles of the physical world. This era was dominated by classical theories such as projective geometry and beam-based registration, with its core challenge being the solution to a massive, non-convex optimization problem: how to simultaneously infer both 3D structure and camera motion from only 2D images.</span> 

#### Multi-View Geometry
>> <span style="color:lightblue;">💡💡 All work from the classical era rests upon a solid mathematical foundation, systematically expounded in Richard Hartley and Andrew Zisserman's seminal work, Multi-View Geometry in Computer Vision. These fundamental principles form the theoretical bedrock for all subsequent algorithms.
</span>  

[[code]](https://github.com/openMVG/openMVG.git) **OpenMVG**  

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
The system constructs a complete processing pipeline, including SIFT feature extraction, approximate nearest neighbor matching, geometric validation using RANSAC, and a carefully designed incremental reconstruction process that meticulously selects initial image pairs and the order of subsequent image additions.  

[[paper]](https://ieeexplore.ieee.org/document/6599068) **Towards Linear-Time Incremental Structure from Motion** [ICCV 2013]   
Core innovations include a “preemptive” feature matching strategy to reduce the number of image pair comparisons, and a novel optimization strategy that combines global and local BA. This work also introduces a re-triangulation step to enhance accuracy and mitigate drift.    

[[paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf) **Structure-from-Motion Revisited** [CVPR 2016]  
COLMAP refines nearly every step of the incremental pipeline. Its key contributions include a more robust initialization process, a “next best view” selection strategy to guide reconstruction, an improved triangulation method, and an iterative cycle involving BA, retriangulation, and outlier filtering—all generating highly complete and accurate models.  


##### Global SfM
>>> <span style="color:lightblue;">💡💡 TUnlike incremental methods, global SfM attempts to solve all camera parameters in one go. This typically involves three steps: 1) estimating the relative rotation between all image pairs; 2) averaging these relative rotations to obtain a globally consistent absolute camera orientation; 3) Solving for the position (translation) and 3D structure of all cameras. Theoretically, this approach is more efficient and avoids drift, but it is more sensitive to outliers in the paired geometry estimation. 
</span>  

[[paper]](https://shaharkov.github.io/projects/GlobalMotionEstimation.pdf) **Global Motion Estimation from Point Matches** [CVPR 2001]  
This is a seminal work in the field of rotation averaging. It proposes a robust linear method for solving global rotations from a sequence of pairwise relative rotations. By formulating the problem within a Lie algebra framework, it effectively addresses the non-convexity of rotation averaging. Although existing literature does not explicitly elaborate on this paper, its influence is evident in subsequent global SfM pipelines.  

[[paper]](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Moulon_Global_Fusion_of_2013_ICCV_paper.pdf) **Global Fusion of Relative Motions for Robust, Accurate and Scalable Structure from Motion** [ICCV 2013]  
Robust removal of anomalous relative rotations in polar images using Bayesian inference and cyclic consistency checks; an efficient “a contrario” tri-focal tensor estimation algorithm for stable translation direction acquisition; and a novel translation registration method based on $L_{\infty}$ norm optimization.  

[[paper]](https://ieeexplore.ieee.org/document/1315098) **Lie-algebraic averaging for globally consistent motion estimation** [CVPR 2004]  

[[paper]](https://ieeexplore.ieee.org/document/4270140) **Robust rotation and translation estimation in multiview reconstruction** [CVPR 2007]  

[[paper]](https://scispace.com/pdf/non-sequential-structure-from-motion-kvfu7kd2yw.pdf) **Non-sequential structure from motion** [ICCV OMNIVIS Workshops 2011]  

[[paper]](https://ieeexplore.ieee.org/document/6374980) **Global motion estimation from point matches** [3DIMPVT 2012]  

[[paper]](https://ieeexplore.ieee.org/document/6751169) **A Global Linear Method for Camera Pose Registration** [ICCV 2013]  

[[paper]](https://ieeexplore.ieee.org/document/7410462) **Global Structure-from-Motion by Similarity Averaging** [ICCV 2015]  

[[paper]](https://bmva-archive.org.uk/bmvc/2015/papers/paper046/index.html#:~:text=This%20paper%20derives%20a%20novel%20linear%20position%20constraint,and%20weak%20image%20association%20at%20the%20same%20time.) **Linear Global Translation Estimation from Feature Tracks** [BMVC 2015]  


##### Hierarchical SfM

[[paper]](https://ieeexplore.ieee.org/document/5457435) **Structure-and-Motion Pipeline on a Hierarchical Cluster Tree** [ICCVW 2009]

[[paper]](https://ieeexplore.ieee.org/document/5206677) **Randomized Structure from Motion Based on Atomic 3D Models from Camera Triplets** [CVPR 2009]  

[[paper]](https://link.springer.com/chapter/10.1007/978-3-642-15552-9_8) **Efficient Structure from Motion by Graph Optimization** [ECCV 2010]  


##### Multi-Stage SfM

[[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Zhu_Very_Large-Scale_Global_CVPR_2018_paper.pdf#:~:text=This%20work%20proposes%20a%20divide-and-conquer%20framework%20to%20solve,association%20for%20well-posed%20and%20parallel%20local%20motion%20averaging.) **Parallel Structure from Motion from Local Increment to Global Averaging** [CVPR 2018]  

[[paper]](https://ieeexplore.ieee.org/document/7035853) **Multistage SFM: A Coarse-to-Fine Approach for 3D Reconstruction** [3DV 2014]  

[[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Cui_HSfM_Hybrid_Structure-from-Motion_CVPR_2017_paper.pdf) **HSfM: Hybrid Structure-from-Motion** [ICCV 2017]   


##### Non Rigid SfM

[[paper]](https://ieeexplore.ieee.org/document/6569198) **Robust Structure from Motion in the Presence of Outliers and Missing Data** [CRV 2013]  

##### Viewing graph optimization

[[code]](https://ieeexplore.ieee.org/document/4587678) **Skeletal graphs for efficient structure from motion** [CVPR 2008]  

[[paper]](https://openaccess.thecvf.com/content_iccv_2015/papers/Sweeney_Optimizing_the_Viewing_ICCV_2015_paper.pdf) **Optimizing the Viewing Graph for Structure-from-Motion** [ICCV 2015]  

[[paper]](https://link.springer.com/chapter/10.1007/978-3-319-46487-9_9) **Graph-Based Consistent Matching for Structure-from-Motion** [ECCV 2016]  


##### Unordered feature tracking

[[paper]](https://imagine.enpc.fr/~moulonp/publis/featureTracking_CVMP12.pdf#:~:text=We%20present%20an%20efficient%20algorithm%20to%20fuse%20two-view,a%20lower%20computational%20complexity%20than%20other%20available%20methods.) **Unordered feature tracking made fast and easy** [CVMP 2012]  

[[paper]](https://ieeexplore.ieee.org/document/6460579) **Point Track Creation in Unordered Image Collections Using Gomory-Hu Trees** [ICPR 2012]  


##### Large scale image matching for SfM

[[paper]](https://ieeexplore.ieee.org/document/1238663) **Video Google: A Text Retrieval Approach to Object Matching in Video** [ICCV 2003]  

[[paper]](https://ieeexplore.ieee.org/document/1641018) **Scalable Recognition with a Vocabulary Tree** [CVPR 2006]  

[[paper]](https://ieeexplore.ieee.org/document/5459148) **Building Rome in a Day** [ICCV 2009]  

[[paper]](https://ieeexplore.ieee.org/document/5432202) **Product quantization for nearest neighbor search** [TPAMI 2011]  

[[paper]](https://openaccess.thecvf.com/content_cvpr_2014/papers/Cheng_Fast_and_Accurate_2014_CVPR_paper.pdf) **Fast and Accurate Image Matching with Cascade Hashing for 3D Reconstruction** [CVPR 2014]  





#### Multi-View Stereo(MVS) 
>><span style="color:lightblue;">💡💡 After SfM provides sparse point clouds and camera poses, the MVS algorithm utilizes this information to compute a dense 3D model of the scene. Its core principle leverages photometric consistency: the color (appearance) of a 3D point in space should be similar across all images where it is visible.
</span>  

##### Voxel and Surface Methods
[[paper]](https://ieeexplore.ieee.org/document/609462) **Photorealistic scene reconstruction by voxel coloring** [CVPR 1997]  
The scene is discretized into a three-dimensional voxel grid. The algorithm traverses these voxels in a specific order (such as plane-by-plane scanning), enabling unambiguous determination of each voxel's visibility. If a voxel's projected color differs across all visible images, it is “carved out” (i.e., marked as transparent). The final model consists of all retained, shaded voxels. While conceptually elegant, this approach is constrained by voxel resolution and camera layout limitations.  

[[paper]](https://ieeexplore.ieee.org/document/1467469) **Multi-view stereo via volumetric graph-cuts** [CVPR 2005]  
This work formulates the MVS problem as a model amenable to global optimization using graph cuts, a powerful technique in combinatorial optimization. This approach enables the reconstruction of topologically complex objects without requiring a well-informed initial guess. 

[[paper]](https://ieeexplore.ieee.org/document/661183) **Variational principles, surface evolution, PDEs, level set methods, and the stereo problem** [TIP 1998]  
This work introduces a complex mathematical framework for MVS based on variational principles and partial differential equations (PDEs). It implicitly represents scene surfaces as the level zero of a high-dimensional function, enabling surfaces to naturally alter their topology during evolution.   The authors define an energy functional based on photometric consistency. The surface evolves along the gradient of this energy, effectively moving toward the true geometric shape of the scene. The level set approach is highly effective for handling complex shapes and topological changes, but it is computationally intensive.  


##### The “Match, Expand, Filter” Paradigm  
[[paper]](https://ieeexplore.ieee.org/document/5226635) **Accurate, Dense, and Robust Multiview Stereopsis** [TPAMI 2009]  
Match: Start with sparse, high-quality feature match points.  
Expand: Iteratively propagate these good matches to neighboring pixels to densify the reconstruction.  
Filter: Remove incorrect matches and outliers using visibility constraints (occlusion checks). This cycle is repeated to grow a dense mesh covering the object's surface.

[[paper]](https://ieeexplore.ieee.org/document/5539802) **Towards Internet-scale multi-view stereo** [CVPR 2010]  
The core idea is a “divide and conquer” strategy. First, the large input image set is decomposed into multiple controllable-sized, overlapping image clusters. Then, PMVS is run independently and in parallel on each cluster. Finally, all partial reconstruction results are fused into a single, consistent 3D model through a robust filtering strategy.  

> ##### <span style="color:lightblue;">💡💡 First, the shift from global implicit representations to local explicit representations is key to the maturity of MVS technology. Early voxelization methods, such as Voxel Coloring and Volumetric Graph-cuts, while conceptually elegant and capable of handling arbitrary topologies, faced a fundamental bottleneck: their memory and computational costs grew cubically with output resolution. This made high-detail reconstruction of large-scale scenes impractical. PMVS's patch-based approach ingeniously circumvents this issue by modeling and storing only the surface itself. This shift from a global, implicit representation (“Is this voxel occupied?”) to a local, explicit representation (“There is a small surface patch here”) represents a key breakthrough enabling both density and high detail.
> ##### <span style="color:lightblue;">💡💡 Second, a symbiotic relationship exists between SfM and MVS. Advances in MVS directly benefit from and, in turn, drive the development of SfM. Approaches like PMVS and level set-based methods all require precisely calibrated cameras as input. When SfM systems (such as Bundler) successfully provided these poses from disorganized web images, it directly spurred demand for MVS algorithms capable of handling similarly challenging “field” images. This spurred the emergence of CMVS, explicitly designed to bridge the gap between large-scale SfM outputs and PMVS input requirements. This reveals a clear causal chain: robust sparse reconstruction (SfM) is a necessary prerequisite for robust dense reconstruction (MVS).  
</span>  




#### Simultaneous Localization and Mapping (SLAM)
>><span style="color:lightblue;">💡💡 SLAM aims to solve the problem of a mobile sensor (such as a camera) simultaneously constructing a map in real time within an unknown environment and tracking its own position within that map. This is the classic “chicken-and-egg” problem in robotics.
</span>  

[[paper]](https://d1wqtxts1xzle7.cloudfront.net/44432225/A_stochastic_map_for_uncertain_spatial_r20160405-20234-wf2gi6-libre.pdf?1459858158=&response-content-disposition=inline%3B+filename%3DA_Stochastic_Map_For_Uncertain_Spatial_R.pdf&Expires=1759038853&Signature=GzEnV5oRMiOe3SLX9vYyV0ahr8A6eB~uy-FwRuCVxjFtKw8fOqkdGwFB6HSQJHBl09KQ35RgtGhUPIV7JbCdpB8F~uykz6hcPdPV1bXFXoKlcv-9kvJtt9Tcwo0zRbT5euTy1H~MbyyueNSUGgtQVn57BGRbBA7cbtUu9QnMKc-xGI4F6xxXAlPGnOSpXqWV5m3NVtSH5Z4sb16~rikTfb33wqfpgbbLSTTTphvHAt5zXACLCykUU2Z8HWvy0Y4jxpOmEk8~FNPNuERBRg7nrIShAzd8hKryI9Eh4Nsw99Ib5WFuWlz72aAn~eM78-37g8o05Frb4Qc8~T5yNnNsQw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) **A stochastic map for uncertain spatial relationships.** [Proceedings of the 4th international symposium on Robotics Research 1988]   

[[paper]](https://ieeexplore.ieee.org/document/1638022) **Simultaneous localization and mapping: part I** [MRA 2006]   

[[paper]](https://ieeexplore.ieee.org/document/1638023) **Simultaneous localization and mapping: part II** [MRA 2007]  



##### Filter-Based SLAM: A Probabilistic Approach  
>This approach models SLAM as a sequential Bayesian inference problem. The system state—including camera pose and map feature locations—along with its uncertainties is represented by a probability distribution that updates with each new frame. The Extended Kalman Filter (EKF) serves as the primary tool, propagating a single Gaussian distribution (mean and covariance matrix) throughout the state space.  

[[paper]](https://ieeexplore.ieee.org/document/4160954) **MonoSLAM: Real-Time Single Camera SLAM** [TPAMI 2007]  
This pioneering work demonstrates real-time monocular SLAM on a standard laptop using only a single camera. It employs EKF to jointly estimate the camera's 6-DOF pose and a sparse map of 3D points, achieving real-time performance through efficient feature tracking and map management strategies.  

##### Keyframe-Based SLAM: A Graph Optimization Approach  
>This approach constructs a pose graph where nodes represent camera poses (keyframes) and edges represent relative transformations (constraints) between them. The entire graph is optimized globally to minimize the overall error, typically using nonlinear least squares methods. This framework is more flexible and scalable than filter-based methods, allowing for efficient loop closure and map refinement.  

[[paper]](https://ieeexplore.ieee.org/document/6126544) **ORB: An efficient alternative to SIFT or SURF** [iccv 2011]  
This work introduces ORB-SLAM, a robust monocular SLAM system that utilizes ORB features for real-time tracking and mapping. It employs a keyframe-based approach with a pose graph optimization framework, enabling efficient loop closure detection and global map optimization. The system demonstrates high accuracy and robustness across various challenging environments.  

[[paper]](https://link.springer.com/chapter/10.1007/978-3-319-10605-2_54) **LSD-SLAM: Large-Scale Direct Monocular SLAM** [ECCV 2014]  
This pioneering work introduces LSD-SLAM, a direct monocular SLAM system that operates without feature extraction. It utilizes image intensities directly for pose estimation and mapping, allowing for dense 3D reconstruction. The system employs a semi-dense approach, focusing on high-gradient regions to balance accuracy and computational efficiency. LSD-SLAM demonstrates robust performance in large-scale environments, showcasing the potential of direct methods in SLAM.  

[[paper]](https://ieeexplore.ieee.org/document/7898369) **DSO: Direct Sparse Odometry** [TPAMI 2017]  
This work presents DSO, a direct monocular visual odometry system that optimizes camera poses and sparse 3D points directly from image intensities. It employs a photometric error minimization framework, allowing for accurate pose estimation even in low-texture environments. DSO introduces a novel keyframe selection strategy and a robust optimization scheme, achieving state-of-the-art performance on several benchmark datasets. 

##### RGB-D SLAM
[[paper]](https://www.roboticsproceedings.org/rss11/p01.pdf) **ElasticFusion: Real-Time Dense SLAM and Light Source Estimation** [IJRR 2016]  
This work presents ElasticFusion, a real-time dense SLAM system that utilizes an RGB-D camera to create a globally consistent 3D map. The system employs a surfel-based representation, allowing for efficient fusion of new observations and dynamic updates to the map. ElasticFusion introduces a novel non-rigid deformation model to handle loop closures and maintain global consistency, achieving high-quality dense reconstructions in real time.  




### 3D-Scene-Generation

#### Text to 3D Generation
[[code]](https://github.com/usama2762/Furniture-Optimization) [[paper]](https://peterkan.com/download/ieeevr2018.pdf) **Automatic Furniture Arrangement Using Greedy Cost Minimization** [VR 2018]  
[[code]](https://github.com/Shao-Kui/3DScenePlatform#mageadd) [[paper]](https://cg.cs.tsinghua.edu.cn/course/vis/Shao-Kui/MageAdd.pdf) **MageAdd: Real-Time Interaction Simulation for Scene Synthesis** [MM 2021]  
[[code]](https://ieeexplore.ieee.org/document/9321177) [[paper]](https://cg.cs.tsinghua.edu.cn/course/vis/Shao-Kui/MageAdd.pdf) **Fast 3D Indoor Scene Synthesis by Learning Spatial Relation Priors of Objects** [TVCG 2021]  
[[code]](https://github.com/allenai/procthor) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/27c546ab1e4f1d7d638e6a8dfbad9a07-Abstract-Conference.html) **ProcTHOR: Large-Scale Embodied AI Using Procedural Generation** [NeurIPS 2022]  
[[code]](https://github.com/princeton-vl/infinigen) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Raistrick_Infinigen_Indoors_Photorealistic_Indoor_Scenes_using_Procedural_Generation_CVPR_2024_paper.pdf) **Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation** [CVPR 2024]    
[[code]](https://github.com/weixi-feng/LayoutGPT) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html) **LayoutGPT: Compositional Visual Planning and Generation with Large Language Models** [NeurIPS 2023]  
[[code]](https://github.com/GGGHSL/GraphDreamer) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_GraphDreamer_Compositional_3D_Scene_Synthesis_from_Scene_Graphs_CVPR_2024_paper.pdf) **GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs** [CVPR 2024]  
[[code]](https://github.com/FreddieRao/anyhome_github) [[paper]](https://eccv.ecva.net/virtual/2024/poster/196) **AnyHome: Open-Vocabulary Generation of Structured and Textured 3D Homes** [ECCV 2024]  
[[code]](https://github.com/sceneteller/SceneTeller) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11481_ECCV_2024_paper.php) **SceneTeller: Language-to-3D Scene Generation** [ECCV 2024]   




#### LLM-based Generation

><span style="color:lightblue;">💡💡 LLM-based Generation in the field of 3D generation typically refers to a generative paradigm that relies on Large Language Models (LLMs) as the core driver or semantic interface. Unlike traditional 3D generation methods that rely solely on visual or geometric priors, LLM-based Generation leverages LLMs' strengths in cross-modal semantic understanding, knowledge transfer, and reasoning capabilities. It maps natural language inputs into high-level semantic constraints, thereby driving the construction of three-dimensional objects, scenes, and even dynamic content.
</span> 


[[code]](https://github.com/weixi-feng/LayoutGPT) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html) **LayoutGPT: Compositional Visual Planning and Generation with Large Language Models** [NeurIPS 2023]  
[[code]](https://github.com/GGGHSL/GraphDreamer) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_GraphDreamer_Compositional_3D_Scene_Synthesis_from_Scene_Graphs_CVPR_2024_paper.pdf) **GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs** [CVPR 2024]  
[[code]](https://github.com/FreddieRao/anyhome_github) [[paper]](https://eccv.ecva.net/virtual/2024/poster/196) **AnyHome: Open-Vocabulary Generation of Structured and Textured 3D Homes** [ECCV 2024]  
[[code]](https://github.com/sceneteller/SceneTeller) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11481_ECCV_2024_paper.php) **SceneTeller: Language-to-3D Scene Generation** [ECCV 2024]  
[[paper]](https://acbull.github.io/pdf/SceneCraft_compressed.pdf) **SceneCraft: An LLM Agent for Synthesizing 3D Scenes as Blender Code** [ICML 2024]  
[[code]](https://github.com/omegafantasy/ControllableLandscape) [[paper]](https://cg.cs.tsinghua.edu.cn/Shao-Kui/Papers/Landscape.pdf#:~:text=By%20converting%20plain%20text%20inputs%20into%20parameters%20through,leverage%20optimization%20tech-%20niques%20and%20employ%20rule-based%20refinements.) **Controllable Procedural Generation of Landscapes** [MM 2024]  
[[paper]](https://dl.acm.org/doi/full/10.1145/3680528.3687589) **DIScene: Object Decoupling and Interaction Modeling for Complex Scene Generation** [SIGGRAPH Asia 2024]   
[[code]](https://github.com/atcelen/IDesign/) [[paper]](https://arxiv.org/abs/2404.02838) **I-Design: Personalized LLM Interior Designer** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2406.03866) **LLplace: The 3D Indoor Scene Layout Generation and Editing via Large Language Model** [arXiv 2024]  
[[code]](https://github.com/djFatNerd/CityCraft) [[paper]](https://arxiv.org/abs/2406.04983) **CityCraft: A Real Crafter for 3D City Generation** [arXiv 2024]  
[[code]](https://github.com/cityx-lab/CityX-Lab) [[paper]](https://arxiv.org/abs/2407.17572) **CityX: Controllable Procedural Content Generation for Unbounded 3D Cities** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2412.00091) **Graph Canvas for Controllable 3D Scene Generation** [arXiv 2024]  
[[code]](https://github.com/Urban-World/UrbanWorld) [[paper]](https://arxiv.org/abs/2407.11965) **UrbanWorld: An Urban World Model for 3D City Generation** [arXiv 2024]  
[[code]](https://github.com/Chuny1/3DGPT) [[paper]](https://arxiv.org/abs/2310.12945) **3D-GPT: Procedural 3D Modeling with Large Language Models** [3DV 2025]   
[[code]](https://github.com/zhouzq1/SceneX) [[paper]](https://arxiv.org/abs/2403.15698) **SceneX: Procedural Controllable Large-scale Scene Generation** [AAAI 2025]  
[[code]](https://github.com/SunWeiLin-Lynne/Hierarchically-Structured-Open-Vocabulary-Indoor-Scene-Synthesis) [[paper]](https://arxiv.org/abs/2502.10675) **Hierarchically-Structured Open-Vocabulary Indoor Scene Synthesis with Pre-trained Large Language Model** [AAAI 2025]  
[[code]](https://github.com/dw-dengwei/TreeSearchGen) [[paper]](https://arxiv.org/abs/2503.18476) **Global-Local Tree Search in VLMs for 3D Indoor Scene Generation** [CVPR 2025]  
[[code]](https://github.com/sunfanyunn/LayoutVLM) [[paper]](https://arxiv.org/abs/2412.02193) **LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models** [CVPR 2025]  
[[code]](https://github.com/zzyunzhi/scene-language) [[paper]](https://arxiv.org/abs/2410.16770) **The Scene Language: Representing Scenes with Programs, Words, and Embeddings** [CVPR 2025]  
[[paper]](https://aclanthology.org/2025.findings-acl.994/) **UnrealLLM: Towards Highly Controllable and Interactable 3D Scene Generation by LLM-powered Procedural Content Generation** [ACL Findings 2025]  
[[paper]](https://arxiv.org/abs/2502.15601) **WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM Agents** [arXiv 2025]  
[[code]](https://github.com/Roblox/cube) [[paper]](https://arxiv.org/abs/2503.15475) **Cube: A Roblox View of 3D Intelligence** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2505.02836) **Scenethesis: A Language and Vision Agentic Framework for 3D Scene Generation** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2505.20129) **Agentic 3D Scene Generation with Spatially Contextualized VLMs** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2506.22291) **RoomCraft: Controllable and Complete 3D Indoor Scene Generation** [arXiv 2025]  
[[code]](https://github.com/GradientSpaces/respace) [[paper]](https://arxiv.org/abs/2506.02459) **ReSpace: Text-Driven 3D Scene Synthesis and Editing with Preference Alignment** [arXiv 2025]  
[[code]](https://github.com/rxjfighting/DirectLayout) [[paper]](https://arxiv.org/abs/2506.05341) **Direct Numerical Layout Generation for 3D Indoor Scene Synthesis via Spatial Reasoning** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2508.17832) **HLG: Comprehensive 3D Room Construction via Hierarchical Layout Generation** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2508.05899) **HOLODECK 2.0: Vision-Language-Guided 3D World Generation with Editing** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2509.05263) **LatticeWorld: A Multimodal Large Language Model-Empowered Framework for Interactive Complex World Generation** [arXiv 2025]  
[[code]](https://github.com/gokucs/causalstruct) [[paper]](https://arxiv.org/abs/2509.05263) **Causal Reasoning Elicits Controllable 3D Scene Generation** [arXiv 2025]  



#### Neural-3D Generation

><span style="color:lightblue;">💡💡 Neural-3D Generation is a generative paradigm that directly learns or indirectly drives three-dimensional representations using deep neural networks. Its core concept involves modeling 3D geometry and appearance through high-dimensional implicit functions or learnable parameterized structures, generating renderable 3D content under multimodal conditions (text, images, or semantic labels). Unlike traditional explicit geometry-based modeling approaches, Neural-3D Generation emphasizes the continuity and differentiable rendering properties of implicit representations. This enables end-to-end training of the generation process through gradient optimization.
</span> 

##### Scene Parameters
[[code]](https://github.com/brownvc/deep-synth) [[paper]](https://doi.org/10.1145/3197517.3201362) **Deep Convolutional Priors for Indoor Scene Synthesis** [SIGGRAPH 2018]  
[[code]](https://github.com/brownvc/fast-synth) [[paper]](https://arxiv.org/abs/1811.12463) **Fast and Flexible Indoor Scene Synthesis via Deep Convolutional Generative Models** [CVPR 2019]  
[[paper]](https://arxiv.org/abs/1808.02084) **Deep Generative Modeling for Scene Synthesis via Hybrid Representations** [SIGGRAPH 2020]  
[[code]](https://github.com/cy94/sceneformer) [[paper]](https://arxiv.org/abs/2012.09793) **SceneFormer: Indoor Scene Generation with Transformers** [3DV 2021]  
[[code]](https://github.com/yanghtr/Sync2Gen) [[paper]](https://arxiv.org/abs/2108.13499) **Scene Synthesis via Uncertainty-Driven Attribute Synchronization** [ICCV 2021]  
[[code]](https://github.com/nv-tlabs/atiss) [[paper]](https://arxiv.org/abs/2110.03675) **ATISS: Autoregressive Transformers for Indoor Scene Synthesis** [NeurIPS 2021]  
[[code]](https://github.com/yinyunie/pose2room) [[paper]](https://arxiv.org/abs/2112.03030) **Pose2Room: Understanding 3D Scenes from Human Activities** [ECCV 2022]  
[[code]](https://github.com/onestarYX/summon) [[paper]](https://arxiv.org/abs/2301.01424) **Scene Synthesis from Human Motion** [SIGGRAPH Asia 2022]  
[[code]](https://github.com/yinyunie/ScenePriors) [[paper]](https://arxiv.org/abs/2211.14157) **Learning 3D Scene Priors with 2D Supervision** [CVPR 2023]  
[[code]](https://github.com/yhw-yhw/MIME) [[paper]](https://arxiv.org/abs/2212.04360) **MIME: Human-Aware 3D Scene Generation** [CVPR 2023]  
[[paper]](https://doi.org/10.1145/3588432.3591561) **COFS: COntrollable Furniture layout Synthesis** [SIGGRAPH 2023]  
[[code]](https://github.com/andvg3/LSDM) [[paper]](https://arxiv.org/abs/2310.15948) **Language-driven Scene Synthesis using Multi-conditional Diffusion Model** [NeurIPS 2023]  
[[code]](https://github.com/zhao-yiqun/RoomDesigner) [[paper]](https://arxiv.org/abs/2310.10027) **RoomDesigner: Encoding Anchor-latents for Style-consistent and Shape-compatible Indoor Scene Generation** [3DV 2024]  
[[code]](https://github.com/tangjiapeng/DiffuScene) [[paper]](https://arxiv.org/abs/2303.14207) **DiffuScene: Denoising Diffusion Models for Generative Indoor Scene Synthesis** [CVPR 2024]  
[[code]](https://github.com/zqh0253/SceneWiz3D) [[paper]](https://arxiv.org/abs/2312.08885) **SceneWiz3D: Towards Text-guided 3D Scene Composition** [CVPR 2024]  
[[code]](https://github.com/PhyScene/PhyScene) [[paper]](https://arxiv.org/abs/2404.09465) **PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI** [CVPR 2024]  
[[code]](https://github.com/DreamScene-Project/DreamScene) [[paper]](https://arxiv.org/abs/2404.03575) **DreamScene: 3D Gaussian-Based Text-to-3D Scene Generation via Formation Pattern Sampling** [ECCV 2024]  
[[code]](https://github.com/VDIGPKU/GALA3D) [[paper]](https://arxiv.org/abs/2402.07207) **GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting** [ICML 2024]  
[[paper]](https://arxiv.org/abs/2402.16936) **Disentangled 3D Scene Generation with Layout Learning** [ICML 2024]  
[[paper]](https://openreview.net/forum?id=GIw7pmMPPX) **RelScene: A Benchmark and baseline for Spatial Relations in text-driven 3D Scene Generation** [MM 2024]  
[[paper]](https://arxiv.org/abs/2409.18336) **DeBaRA: Denoising-Based 3D Room Arrangement Generation** [NeurIPS 2024]  
[[paper]](https://arxiv.org/abs/2405.12460) **Physics-based Scene Layout Generation From Human Motion** [SIGGRAPH 2024]  
[[code]](https://github.com/ohad204/Lay-A-Scene) [[paper]](https://arxiv.org/abs/2406.00687) **Lay-A-Scene: Personalized 3D Object Arrangement Using Text-to-Image Priors** [arXiv 2024]  
[[code]](https://github.com/fangchuan/Ctrl-Room) [[paper]](https://arxiv.org/abs/2310.03602) **Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints** [3DV 2025]  
[[code]](https://github.com/alexeybokhovkin/SceneFactor) [[paper]](https://arxiv.org/abs/2412.01801) **SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation** [CVPR 2025]  
[[code]](https://github.com/CASAGPT/CASA-GPT) [[paper]](https://arxiv.org/abs/2504.19478) **CASAGPT: Cuboid Arrangement and Scene Assembly for Interior Design** [CVPR 2025]  
[[code]](https://github.com/nepfaff/steerable-scene-generation) [[paper]](https://arxiv.org/abs/2505.04831) **Steerable Scene Generation with Post Training and Inference-Time Search** [CoRL 2025]  


##### Scene Graph
[[paper]](https://doi.org/10.3115/v1/D14-1217) **Learning Spatial Knowledge for Text to 3D Scene Generation** [EMNLP 2014]  
[[paper]](https://doi.org/10.1111/cgf.12976) **Learning 3D Scene Synthesis from Annotated RGB-D Images** [CGF 2016]  
[[paper]](https://doi.org/10.1145/3130800.3130805) **Adaptive synthesis of indoor scenes via activity-associated object relation graphs** [TOG 2017]  
[[paper]](https://doi.org/10.1145/3272127.3275035) **Language-Driven Synthesis of 3D Scenes from Scene Databases** [TOG 2018]  
[[code]](https://github.com/nv-tlabs/meta-sim) [[paper]](https://arxiv.org/abs/1904.11621) **Meta-Sim: Learning to Generate Synthetic Datasets** [ICCV 2019]  
[[code]](https://github.com/ManyiLi12345/GRAINS) [[paper]](https://arxiv.org/abs/1807.09193) **GRAINS: Generative Recursive Autoencoders for INdoor Scenes** [SIGGRAPH 2019]  
[[code]](https://github.com/brownvc/planit) [[paper]](https://doi.org/10.1145/3306346.3322941) **PlanIT: Planning and Instantiating Indoor Scenes with Relation Graph and Spatial Prior Networks** [SIGGRAPH 2019]  
[[code]](https://github.com/aluo-x/3D_SLN) [[paper]](https://arxiv.org/abs/2007.11744) **End-to-End Optimization of Scene Layout** [CVPR 2020]  
[[code]](https://github.com/nv-tlabs/meta-sim-structure) [[paper]](https://arxiv.org/abs/2008.09092) **Meta-Sim 2 Unsupervised Learning of Scene Structure for Synthetic Data Generation** [ECCV 2020]  
[[code]](https://github.com/he-dhamo/graphto3d) [[paper]](https://arxiv.org/abs/2108.08841) **Graph-to-3D: End-to-End Generation and Manipulation of 3D Scenes Using Scene Graphs** [ICCV 2021]  
[[code]](https://github.com/ymxlzgy/commonscenes) [[paper]](https://arxiv.org/abs/2305.16283) **CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graph Diffusion** [NeurIPS 2023]  
[[code]](https://github.com/tommaoer/SceneHGN) [[paper]](https://arxiv.org/abs/2302.10237) **SceneHGN: Hierarchical Graph Networks for 3D Indoor Scene Generation With Fine-Grained Geometry** [TPAMI 2023]  
[[paper]](https://arxiv.org/abs/2403.14121) **SEK: External Knowledge Enhanced 3D Scene Generation from Sketch** [ECCV 2024]  
[[paper]](https://arxiv.org/abs/2407.05388) **Forest2Seq: Revitalizing Order Prior for Sequential Indoor Scene Synthesis** [ECCV 2024]  
[[code]](https://github.com/ymxlzgy/echoscene) [[paper]](https://arxiv.org/abs/2405.00915) **EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion** [ECCV 2024]  
[[code]](https://github.com/chenguolin/InstructScene) [[paper]](https://arxiv.org/abs/2402.04717) **InstructScene: Instruction-Driven 3D Indoor Scene Synthesis with Semantic Graph Prior** [ICLR 2024]  
[[code]](https://github.com/yangzhifeio/MMGDreamer) [[paper]](https://arxiv.org/abs/2502.05874) **MMGDreamer: Mixed-Modality Graph for Geometry-Controllable 3D Indoor Scene Generation** [AAAI 2025]  
[[code]](https://github.com/cangmushui/FreeScene) [[paper]](https://arxiv.org/abs/2506.02781) **FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts** [CVPR 2025]  
[[code]](https://github.com/yuhengliu02/control-3d-scene) [[paper]](https://arxiv.org/abs/2503.07152) **Controllable 3D Outdoor Scene Generation via Scene Graphs** [ICCV 2025]  
[[paper]](https://arxiv.org/abs/2504.13072) **HiScene: Creating Hierarchical 3D Scenes with Isometric View Generation** [arXiv 2025]  



#### Implicit Layout
><span style="color:lightblue;">💡💡 Implicit Layout in 3D generation and scene representation research typically refers to encoding the spatial structure and object layout of a scene through implicit functions or continuous fields, rather than explicitly specifying geometric boundaries or semantic partitions. This approach emphasizes translating “object presence, occupancy relationships, and relative layouts” within a scene into differentiable continuous function representations. This enables neural networks to learn globally consistent scene layout constraints within high-dimensional latent spaces.
</span> 

[[code]](https://github.com/autonomousvision/giraffe) [[paper]](https://arxiv.org/abs/2011.12100) **GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields** [CVPR 2021]  
[[code]](https://github.com/apple/ml-gsn) [[paper]](https://arxiv.org/abs/2104.00670) **Unconstrained Scene Generation With Locally Conditioned Radiance Fields** [ICCV 2021]  
[[paper]](https://arxiv.org/abs/2104.00587) **NeRF-VAE: A geometry aware 3d scene generative model** [ICML 2021]  
[[code]](https://github.com/apple/ml-gaudi) [[paper]](https://arxiv.org/abs/2207.13751) **GAUDI: A Neural Architect for Immersive 3D Scene Generation** [NeurIPS 2022]  
[[code]](https://github.com/google-research/google-research/tree/master/persistent-nature) [[paper]](https://arxiv.org/abs/2303.13515) **Persistent Nature: A generative model of unbounded 3D worlds** [CVPR 2023]  
[[paper]](https://arxiv.org/abs/2304.09787) **NeuralField-LDM: Scene Generation with Hierarchical Latent Diffusion Models** [CVPR 2023]  
[[code]](https://github.com/zoomin-lee/scene-scale-diffusion) [[paper]](https://arxiv.org/abs/2301.00527) **Diffusion Probabilistic Models for Scene-Scale 3D Categorical Data** [arXiv 2023]  
[[code]](https://github.com/AkiraHero/diffindscene) [[paper]](https://arxiv.org/abs/2306.00519) **DiffInDScene: Diffusion-based High-Quality 3D Indoor Scene Generation** [CVPR 2024]  
[[code]](https://github.com/nv-tlabs/XCube) [[paper]](https://arxiv.org/abs/2312.03806) **XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies** [CVPR 2024]  
[[code]](https://github.com/zoomin-lee/SemCity) [[paper]](https://arxiv.org/abs/2403.07773) **SemCity: Semantic Scene Generation with Triplane Diffusion** [CVPR 2024]  
[[code]](https://github.com/yuhengliu02/pyramid-discrete-diffusion) [[paper]](https://arxiv.org/abs/2311.12085) **Pyramid Diffusion for Fine 3D Large Scene Generation** [ECCV 2024]  
[[code]](https://github.com/imlixinyang/director3d) [[paper]](https://arxiv.org/abs/2406.17601) **Director3D: Real-world Camera Trajectory and 3D Scene Generation from Text** [NeurIPS 2024]  
[[code]](https://github.com/quan-meng/lt3sd) [[paper]](https://arxiv.org/abs/2409.08215) **LT3SD: Latent Trees for 3D Scene Diffusion** [CVPR 2025]  
[[code]](https://github.com/gohyojun15/SplatFlow/) [[paper]](https://arxiv.org/abs/2411.16443) **SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis** [CVPR 2025]  
[[code]](https://github.com/XDimLab/Prometheus) [[paper]](https://arxiv.org/abs/2412.21117) **Prometheus: 3D-Aware Latent Diffusion Models for Feed-Forward Text-to-3D Scene Generation** [CVPR 2025]  
[[code]](https://github.com/3DTopia/DynamicCity) [[paper]](https://arxiv.org/abs/2410.18084) **DynamicCity: Large-Scale Occupancy Generation from Dynamic Scenes** [ICLR 2025]  
[[code]](https://github.com/3dlg-hcvc/NuiScene) [[paper]](https://arxiv.org/abs/2503.16375) **NuiScene: Exploring Efficient Generation of Unbounded Outdoor Scenes** [ICCV 2025]  
[[code]](https://github.com/gohyojun15/VideoRFSplat) [[paper]](https://arxiv.org/abs/2503.15855) **VideoRFSplat: Direct Scene-Level Text-to-3D Gaussian Splatting Generation with Flexible Pose and Multi-View Joint Modeling** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2508.19204) **LSD-3D: Large-Scale 3D Driving Scene Generation with Geometry Grounding** [arXiv 2025]   
[[code]](https://github.com/fudan-zvg/UniUGG) [[paper]](https://arxiv.org/abs/2508.11952) **UniUGG: Unified 3D Understanding and Generation via Geometric-Semantic Encoding** [arXiv 2025]  



#### Image-based Generation
><span style="color:lightblue;">💡💡 Image-based Generation in the field of 3D content creation typically refers to a generative paradigm that synthesizes 3D geometry and appearance by learning cross-dimensional mapping relationships, using 2D images as primary input or constraints. The core of such methods lies in leveraging images as conditional signals to overcome the limitations of sparse and difficult-to-acquire 3D data, enabling high-quality 3D reconstruction and generation through rich 2D prior knowledge.
</span> 

##### Holistic Generation
><span style="color:lightblue;">💡💡 Holistic Generation in the field of 3D generation typically refers to an integrated generation paradigm that adopts a global perspective to unify the modeling of geometric structures, semantic layouts, appearance textures, and even dynamic properties within a scene. Its core philosophy lies in overcoming the limitations of previous methods that focused solely on individual objects, local geometry, or specific modalities. By leveraging end-to-end multimodal fusion and global consistency modeling, it achieves holistic and controllable 3D content generation.
</span> 

[[paper]](https://doi.org/10.1109/ICIP.2019.8803435) **360-Degree Image Completion by Two-Stage Conditional Gans** [ICIP 2019]  
[[code]](https://github.com/lizuoyue/sate_to_ground) [[paper]](https://doi.org/10.1109/CVPR42600.2020.00094) **Geometry-Aware Satellite-to-Ground Image Synthesis for Urban Areas** [CVPR 2020]   
[[paper]](https://arxiv.org/abs/1904.03326) **360 Panorama Synthesis from a Sparse Set of Images with Unknown Field of View** [WACV 2020]  
[[code]](https://github.com/hara012/sig-ss) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16242) **Spherical Image Generation from a Single Image by Considering Scene Symmetry** [AAAI 2021]  
[[code]](https://github.com/apple/ml-envmapnet) [[paper]](https://arxiv.org/abs/2011.10687) **HDR Environment Map Estimation for Real-Time Augmented Reality** [CVPR 2021]  
[[paper]](https://arxiv.org/abs/2012.06628) **Sat2vid: Street-view panoramic video synthesis from a single satellite image** [ICCV 2021]  
[[paper]](https://arxiv.org/abs/2104.00587) **NeRF-VAE: A geometry aware 3d scene generative model** [ICML 2021]  
[[paper]](https://arxiv.org/abs/2204.07286) **Guided Co-Modulated GAN for 360° Field of View Extrapolation** [3DV 2022]  
[[code]](https://github.com/akmtn/OmniDreamer) [[paper]](https://arxiv.org/abs/2203.14668) **Diverse Plausible 360-Degree Image Outpainting for Efficient 3DCG Background Creation** [CVPR 2022]  
[[code]](https://github.com/chang9711/BIPS) [[paper]](https://arxiv.org/abs/2112.06179) **BIPS: Bi-modal Indoor Panorama Synthesis via Residual Depth-aided Adversarial Learning** [ECCV 2022]  
[[code]](https://github.com/FrozenBurning/Text2Light) [[paper]](https://arxiv.org/abs/2209.09898) **Text2Light: Zero-Shot Text-Driven HDR Panorama Generation** [SIGGRAPH Asia 2022]  
[[code]](https://github.com/sswuai/PanoGAN) [[paper]](https://arxiv.org/abs/2203.11832) **Cross-View Panorama Image Synthesis** [TMM 2022]  
[[code]](https://github.com/YujiaoShi/Sat2StrPanoramaSynthesis) [[paper]](https://arxiv.org/abs/2103.01623) **Geometry-Guided Street-View Panorama Synthesis from Satellite Imagery** [TPAMI 2022]  
[[paper]](https://arxiv.org/abs/2303.17076) **DiffCollage: Parallel Generation of Large Content with Diffusion Models** [CVPR 2023]  
[[code]](https://github.com/qianmingduowan/Sat2Density) [[paper]](https://arxiv.org/abs/2303.14672) **Sat2Density: Faithful Density Learning from Satellite-Ground Image Pairs** [ICCV 2023]  
[[code]](https://github.com/shanemankiw/Panodiff) [[paper]](https://arxiv.org/abs/2308.14686) **360-Degree Panorama Generation from Few Unregistered NFoV Images** [MM 2023]  
[[code]](https://github.com/Tangshitao/MVDiffusion) [[paper]](https://arxiv.org/abs/2307.01097) **MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion** [NeurIPS 2023]  
[[code]](https://github.com/hara012/sig-ss) [[paper]](https://doi.org/10.1109/TPAMI.2022.3215933) **Spherical Image Generation From a Few Normal-Field-of-View Images by Considering Scene Symmetry** [TPAMI 2023]  
[[paper]](https://arxiv.org/abs/2305.10853) **LDM3D: Latent Diffusion Model for 3D** [arXiv 2023]  
[[code]](https://github.com/ArcherFMY/SD-T2I-360PanoImage) [[paper]](https://arxiv.org/abs/2311.13141) **Diffusion360: Seamless 360 Degree Panoramic Image Generation based on Diffusion Models** [arXiv 2023]  
[[code]](https://github.com/PanoDiffusion/PanoDiffusion) [[paper]](https://arxiv.org/abs/2307.03177) **PanoDiffusion: 360-degree Panorama Outpainting via Diffusion** [ICLR 2024]  
[[paper]](https://arxiv.org/abs/2312.05208) **ControlRoom3D 🤖Room Generation using Semantic Proxy Rooms** [CVPR 2024]  
[[code]](https://github.com/lizuoyue/sat2scene) [[paper]](https://arxiv.org/abs/2401.10786) **Sat2Scene: 3D Urban Scene Generation from Satellite Images with Diffusion** [CVPR 2024]  
[[code]](https://github.com/chengzhag/PanFusion) [[paper]](https://arxiv.org/abs/2404.07949) **PanFusion: Taming stable diffusion for text to 360◦ panorama image generation** [CVPR 2024]  
[[code]](https://github.com/ShijieZhou-UCLA/DreamScene360) [[paper]](https://arxiv.org/abs/2404.06903) **DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting** [ECCV 2024]  
[[paper]](https://arxiv.org/abs/2407.08061) **Geospecific View Generation - Geometry-Context Aware High-resolution Ground View Inference from Satellite Views** [ECCV 2024]  
[[code]](https://github.com/Mr-Ma-yikun/FastScene) [[paper]](https://arxiv.org/abs/2405.05768) **FastScene: Text-Driven Fast Indoor 3D Scene Generation via Panoramic Gaussian Splatting** [IJCAI 2024]  
[[code]](https://github.com/zju3dv/DiffPano) [[paper]](https://arxiv.org/abs/2410.24203) **DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion** [NeurIPS 2024]  
[[code]](https://github.com/perf-project/PeRF) [[paper]](https://arxiv.org/abs/2310.16831) **PERF: Panoramic Neural Radiance Field from a Single Panorama** [TPAMI 2024]  
[[paper]](https://arxiv.org/abs/2401.10564) **Dream360: Diverse and Immersive Outdoor Virtual Scene Creation via Transformer-Based 360° Image Outpainting** [TVCG 2024]  
[[code]](https://github.com/littlewhitesea/StitchDiffusion) [[paper]](https://arxiv.org/abs/2310.18840) **StitchDiffusion: Customizing 360-Degree Panoramas through Text-to-Image Diffusion Models** [WACV 2024]  
[[code]](https://github.com/zhouhyOcean/HoloDreamer) [[paper]](https://arxiv.org/abs/2407.15187) **HoloDreamer: Holistic 3D Panoramic World Generation from Text Descriptions** [arXiv 2024]  
[[code]](https://github.com/liwrui/SceneDreamer360) [[paper]](https://arxiv.org/abs/2408.13711) **SceneDreamer360: Text-Driven 3D-Consistent Scene Generation with Panoramic Gaussian Splatting** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2501.17162) **CubeDiff: Repurposing Diffusion-Based Image Models for Panorama Generation** [ICLR 2025]  
[[code]](https://github.com/3DTopia/LayerPano3D) [[paper]](https://arxiv.org/abs/2408.13252) **LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation** [SIGGRAPH 2025]  
[[paper]](https://arxiv.org/abs/2503.16611) **A Recipe for Generating 3D Worlds From a Single Image** [arXiv 2025]  
[[code]](https://github.com/HorizonRobotics/EmbodiedGen) [[paper]](https://arxiv.org/abs/2506.10600) **EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence** [arXiv 2025]  
[[paper]](https://immersegen.github.io/) [[paper]](https://arxiv.org/abs/2506.14315) **ImmerseGen: Agent-Guided Immersive World Generation with Alpha-Textured Proxies** [arXiv 2025]  
[[code]](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) [[paper]](https://arxiv.org/abs/2507.21809) **HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels** [arXiv 2025]  
[[code]](https://github.com/SkyworkAI/Matrix-3D) [[paper]](https://arxiv.org/abs/2508.08086) **Matrix-3D: Omnidirectional Explorable 3D World Generation** [arXiv 2025]  



##### Iterative Generation
><span style="color:lightblue;">💡💡 Iterative Generation in the field of 3D generation typically refers to a step-by-step progressive generation paradigm. Rather than producing a complete 3D structure in a single pass, the model undergoes multiple rounds of iterative optimization or hierarchical refinement to progressively enhance geometric accuracy, visual realism, and global consistency. The core principle of this approach is to decompose complex 3D synthesis into a series of incremental steps. Each step refines and enhances the results from the preceding stage, thereby effectively addressing issues such as scale ambiguity, geometric artifacts, and texture inconsistencies.
</span> 

[[code]](https://github.com/sniklaus/3d-ken-burns) [[paper]](https://arxiv.org/abs/1909.05483) **3D Ken Burns Effect from a Single Image** [TOG 2019]  
[[code]](https://github.com/facebookresearch/synsin) [[paper]](https://arxiv.org/abs/1912.08804) **SynSin: End-to-end view synthesis from a single image** [CVPR 2020]  
[[code]](https://github.com/vt-vl-lab/3d-photo-inpainting) [[paper]](https://arxiv.org/abs/2004.04727) **3D Photography Using Context-Aware Layered Depth Inpainting** [CVPR 2020]  
[[code]](https://github.com/google-research/google-research/tree/master/single_view_mpi) [[paper]](https://arxiv.org/abs/2004.11364) **Single-View View Synthesis with Multiplane Images** [CVPR 2020]  
[[code]](https://github.com/tedyhabtegebrial/gvsnet) [[paper]](https://arxiv.org/abs/2008.09106) **Generative View Synthesis: From Single-view Semantics to Novel-view Images** [NeurIPS 2020]  
[[code]](https://github.com/facebookresearch/worldsheet) [[paper]](https://arxiv.org/abs/2012.09854) **Worldsheet: Wrapping the World in a 3D Sheet for View Synthesis from a Single Image** [ICCV 2021]  
[[code]](https://github.com/google-research/google-research/tree/master/infinite_nature) [[paper]](https://arxiv.org/abs/2012.09855) **Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image** [ICCV 2021]  
[[code]](https://github.com/CompVis/geometry-free-view-synthesis) [[paper]](https://arxiv.org/abs/2104.07652) **Geometry-free view synthesis: Transformers and no 3d priors** [ICCV 2021]  
[[code]](https://github.com/google-research/pathdreamer) [[paper]](https://arxiv.org/abs/2105.08756) **Pathdreamer: A World Model for Indoor Navigation** [ICCV 2021]  
[[code]](https://github.com/crockwell/pixelsynth) [[paper]](https://arxiv.org/abs/2108.05892) **PixelSynth: Generating a 3D-Consistent Experience from a Single Image** [ICCV 2021]  
[[code]](https://github.com/xrenaa/Look-Outside-Room) [[paper]](https://arxiv.org/abs/2203.09457) **LOTR: Synthesizing a consistent long-term 3D scene video from a single image** [CVPR 2022]  
[[code]](https://github.com/google-research/google-research/tree/master/infinite_nature_zero) [[paper]](https://arxiv.org/abs/2207.11148) **InfiniteNature-Zero: Learning Perpetual View Generation of Natural Scenes from Single Images** [ECCV 2022]  
[[code]](https://github.com/yshen47/SGAM_NeurIPS22) [[paper]](https://openreview.net/forum?id=17KCLTbRymw) **SGAM: Building a Virtual 3D World through Simultaneous Generation and Mapping** [NeurIPS 2022]  
[[code]](https://github.com/google-research/se3ds) [[paper]](https://arxiv.org/abs/2204.02960) **SE3DS: Simple and Effective Synthesis of Indoor 3D Scenes** [AAAI 2023]  
[[code]](https://github.com/xingyi-li/3d-cinemagraphy) [[paper]](https://arxiv.org/abs/2303.05724) **3D Cinemagraphy from a Single Image** [CVPR 2023]  
[[paper]](https://arxiv.org/abs/2303.17598) **Consistent View Synthesis with Pose-Guided Diffusion Models** [CVPR 2023]  
[[code]](https://github.com/primecai/DiffDreamer) [[paper]](https://arxiv.org/abs/2211.12131) **DiffDreamer: Towards Consistent Unsupervised Single-view Scene Extrapolation with Conditional Diffusion Models** [ICCV 2023]  
[[code]](https://github.com/lukasHoel/text2room) [[paper]](https://arxiv.org/abs/2303.11989) **Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models** [ICCV 2023]  
[[code]](https://github.com/YorkUCVIL/Photoconsistent-NVS) [[paper]](https://arxiv.org/abs/2304.10700) **Long-Term Photometric Consistent Novel View Synthesis with Diffusion Models** [ICCV 2023]  
[[code]](https://github.com/leoShen917/Make-It-4D) [[paper]](https://arxiv.org/abs/2308.10257) **Make-It-4D: Synthesizing a Consistent Long-Term Dynamic Scene Video from a Single Image** [MM 2023]  
[[code]](https://github.com/RafailFridman/SceneScape) [[paper]](https://arxiv.org/abs/2302.01133) **SceneScape: Text-Driven Consistent Scene Generation** [NeurIPS 2023]  
[[code]](https://github.com/jialuli-luka/PanoGen) [[paper]](https://arxiv.org/abs/2305.19195) **PanoGen: Text-Conditioned Panoramic Environment Generation for Vision-and-Language Navigation** [NeurIPS 2023]  
[[code]](https://github.com/luciddreamer-cvlab/LucidDreamer) [[paper]](https://arxiv.org/abs/2311.13384) **LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes** [arXiv 2023]  
[[paper]](https://ken-ouyang.github.io/text2immersion/index.html) [[paper]](https://arxiv.org/abs/2312.09242) **Text2Immersion: Generative Immersive Scene with 3D Gaussians** [arXiv 2023]  
[[code]](https://github.com/zhuqiangLu/AOG-NET-360) [[paper]](https://arxiv.org/abs/2309.03467) **AOG-Net: Autoregressive Omni-Aware Outpainting for Open-Vocabulary 360-Degree Image Generation** [AAAI 2024]  
[[code]](https://github.com/KovenYu/WonderJourney) [[paper]](https://arxiv.org/abs/2312.03884) **WonderJourney: Going from Anywhere to Everywhere** [CVPR 2024]  
[[paper]](https://microtreei.github.io/) [[paper]](https://arxiv.org/abs/2403.09439) **3D-SceneDreamer: Text-Driven 3D-Consistent Scene Generation** [CVPR 2024]  
[[code]](https://github.com/zxcvfd13502/PanoFree) [[paper]](https://arxiv.org/abs/2408.02157) **PanoFree: Tuning-Free Holistic Multi-view Image Generation with Cross-view Self-Guidance** [ECCV 2024]  
[[code]](https://github.com/xingyi-li/iControl3D) [[paper]](https://arxiv.org/abs/2408.01678) **iControl3D: An Interactive System for Controllable 3D Scene Generation** [MM 2024]  
[[code]](https://github.com/MattWallingford/360-1M) [[paper]](https://arxiv.org/abs/2412.07770) **ODIN: Learning to Imagine the World from a Million 360° Videos** [NeurIPS 2024]  
[[paper]](https://cat3d.github.io/) [[paper]](https://arxiv.org/abs/2405.10314) **CAT3D: Create Anything in 3D with Multi-View Diffusion Models** [NeurIPS 2024]  
[[code]](https://github.com/eckertzhang/Text2NeRF) [[paper]](https://arxiv.org/abs/2305.11588) **Text2NeRF: Text-Driven 3D Scene Generation with Neural Radiance Fields** [TVCG 2024]  
[[code]](https://github.com/PengleiGao/OPaMa) [[paper]](https://arxiv.org/abs/2407.10923) **OPa-Ma: Text Guided Mamba for 360-degree Image Out-painting** [arXiv 2024]  
[[code]](https://github.com/YiyingYang12/Scene123) [[paper]](https://arxiv.org/abs/2408.05477) **Scene123: One Prompt to 3D Scene Generation via Video-Assisted and Consistency-Enhanced MAE** [arXiv 2024]  
[[code]](https://github.com/jaidevshriram/realmdreamer) [[paper]](https://arxiv.org/abs/2404.07199) **RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion** [3DV 2025]  
[[code]](https://github.com/paulengstler/invisible-stitch) [[paper]](https://arxiv.org/abs/2404.19758) **Invisible Stitch: Generating Smooth 3D Scenes with Depth Inpainting** [3DV 2025]  
[[code]](https://github.com/SparklingH/BloomScene) [[paper]](https://arxiv.org/abs/2501.10462) **BloomScene: Lightweight Structured 3D Gaussian Splatting for Crossmodal Scene Generation** [AAAI 2025]  
[[code]](https://github.com/KovenYu/WonderWorld) [[paper]](https://arxiv.org/abs/2406.09394) **WonderWorld: Interactive 3D Scene Generation from a Single Image** [CVPR 2025]  
[[code]](https://github.com/jaclyngu/artiscene) [[paper]](https://arxiv.org/abs/2506.00742) **ArtiScene: Language-Driven Artistic 3D Scene Generation Through Image Intermediary** [CVPR 2025]  
[[code]](https://github.com/cvsp-lab/ICLR2025_3D-MOM) [[paper]](https://arxiv.org/abs/2504.05458) **3D-MOM: Optimizing 4D Gaussians for Dynamic Scene Video from Single Landscape Images** [ICLR 2025]  
[[code]](https://github.com/GigaAI-research/WonderTurbo) [[paper]](https://arxiv.org/abs/2504.02261) **WonderTurbo: Generating Interactive 3D World in 0.72 Seconds** [arXiv 2025]  
[[paper]](https://szymanowiczs.github.io/bolt3d) [[paper]](https://arxiv.org/abs/2503.14445) **Bolt3D: Generating 3D Scenes in Seconds** [arXiv 2025]  
[[code]](https://github.com/paulengstler/syncity) [[paper]](https://arxiv.org/abs/2503.16420) **SynCity: Training-Free Generation of 3D Worlds** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2508.15169) **MeSS: City Mesh-Guided Outdoor Scene Generation with Cross-View Consistent Diffusion** [arXiv 2025]  
[[paper]](https://kxhit.github.io/CausNVS.html) [[paper]](https://arxiv.org/abs/2509.06579) **CausNVS: Autoregressive Multi-view Diffusion for Flexible 3D Novel View Synthesis** [arXiv 2025]  
[[code]](https://github.com/xiac20/ScenePainter) [[paper]](https://arxiv.org/abs/2507.19058) **ScenePainter: Semantically Consistent Perpetual 3D Scene Generation with Concept Relation Alignment** [ICCV 2025]  




#### Video-based Generation
><span style="color:lightblue;">💡💡 Video-based Generation in the field of generative AI modeling primarily refers to the use of spatiotemporal diffusion models to progressively reverse the forward diffusion process from a random noise distribution, enabling the conditional synthesis of high-dimensional video sequences. Specifically, this paradigm relies on the Denoising Diffusion Probabilistic Models (DDPM) framework. It applies Gaussian perturbations to video frame sequences through a Markov chain-based noise injection mechanism. During the reverse sampling phase, variational inference optimizes the denoising step. using text, image, or video conditional embeddings as guiding signals to ensure the generated spatio-temporal data aligns with physical priors in terms of pixel-level spatial consistency and inter-frame temporal coherence. The core architecture typically extends the U-Net backbone, incorporating 3D convolutional kernels or factorized Transformer attention modules to capture long-range motion dependencies. Simultaneously, it employs a Variational Autoencoder (VAE) to map raw RGB frames into a low-dimensional latent space, thereby reducing computational overhead and enhancing generation efficiency—as demonstrated by the latent diffusion paradigm used in models like Sora or Veo. Further refining the iterative process from pure noise to high-fidelity dynamic content involves enhancing conditional alignment through classifier-free guidance.
</span> 

##### Two-stage Generation
><span style="color:lightblue;">💡💡 Two-stage Generation, within the generative AI framework, specifically refers to a cascaded diffusion modeling strategy designed for efficient conditional synthesis of high-dimensional video sequences. The first stage deploys a low-resolution spatiotemporal diffusion model. Through a Markov chain noise injection mechanism, it reverse-samples coarse-grained video frame sequences from a Gaussian noise distribution to capture global scene semantics, motion trajectories, and structural priors. Simultaneously, it employs a variational autoencoder (VAE) to compress latent representations, alleviating computational burden. The second stage introduces a high-resolution refinement diffuser, using the first stage's output as a conditional embedding. It employs either a 3D U-Net or a factorized Transformer architecture for iterative denoising optimization, enhancing pixel-level detail filling, texture consistency, and inter-frame optical flow coherence. further aligning text or image prompts through a classifier-free guidance mechanism to achieve progressive enhancement from low-fidelity sketches to high-fidelity dynamic content. For instance, in novel view synthesis tasks, the first stage employs a panorama diffusion model to infer 360-degree scene priors from single-view inputs, generating anchor frame keypoints. followed by a second stage employing a video diffusion model for spatial noise diffusion interpolation, ensuring global consistency and robust loop closure under long-range trajectories. This paradigm significantly reduces the computational overhead of single-stage high-dimensional diffusion and manifests as latent space cascading in model variants like Sora or Veo, thereby balancing generation quality and efficiency.
</span> 

[[paper]](https://arxiv.org/abs/2407.13759) [[website]](https://boyangdeng.com/streetscapes/) **Streetscapes: Large-scale Consistent Street View Generation Using Autoregressive Video Diffusion** [SIGGRAPH 2024]  
[[code]](https://github.com/snap-research/4Real) [[paper]](https://arxiv.org/abs/2406.07472) [[website]](https://snap-research.github.io/4Real/) **4Real: Towards Photorealistic 4D Scene Generation via Video Diffusion Models** [NeurIPS 2024]  
[[paper]](https://arxiv.org/abs/2405.20334) [[website]](https://vivid-dream-4d.github.io/) **VividDream: Generating 3D Scene with Ambient Dynamics** [arXiv 2024]  
[[code]](https://github.com/paintscene4d/paintscene4d.github.io) [[paper]](https://arxiv.org/abs/2412.04471) [[website]](https://paintscene4d.github.io/) **PaintScene4D: Consistent 4D Scene Generation from Text Prompts** [arXiv 2024]  
[[code]](https://github.com/HeliosZhao/GenXD) [[paper]](https://arxiv.org/abs/2411.02319) [[website]](https://gen-x-d.github.io/) **GenXD: Generating Any 3D and 4D Scenes** [ICLR 2025]  
[[code]](https://github.com/zju3dv/StarGen) [[paper]](https://arxiv.org/abs/2501.05763) [[website]](https://zju3dv.github.io/StarGen/) **StarGen: A Spatiotemporal Autoregression Framework with Video Diffusion Model for Scalable and Controllable Scene Generation** [CVPR 2025]  
[[code]](https://github.com/HiDream-ai/DreamJourney) [[paper]](https://arxiv.org/abs/2506.17705) [[website]](https://dream-journey.vercel.app/) **DreamJourney: Perpetual View Generation with Video Diffusion Models** [TMM 2025]  
[[code]](https://github.com/wenqsun/DimensionX) [[paper]](https://arxiv.org/abs/2411.04928) [[website]](https://chenshuo20.github.io/DimensionX/) **DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion** [ICCV 2025]  
[[code]](https://github.com/TQTQliu/Free4D) [[paper]](https://arxiv.org/abs/2503.20785) [[website]](https://free4d.github.io/) **Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency** [ICCV 2025]  


##### One-stage Generation
><span style="color:lightblue;">💡💡 One-stage Generation specifically refers to end-to-end single-stage diffusion frameworks within generative video modeling paradigms. By deploying unified U-Net or Transformer-based denoising networks directly on high-dimensional spatiotemporal latent spaces, it performs multi-step Markovian reverse sampling from a pure Gaussian noise distribution to achieve instantaneous synthesis of conditional video sequences, bypassing cascaded refinement strategies. This approach leverages the spatiotemporal extension of pre-trained image diffusers (e.g., Stable Diffusion), incorporating 3D causal attention or factorized convolutions to capture inter-frame motion consistency and global semantic dependencies. while compressing the multi-step denoising process into a single-step forward pass via knowledge distillation or adversarial post-training, significantly boosting inference speed and reducing memory overhead. For instance, the SF-V model achieves single-forward-pass video generation by adversarially fine-tuning pre-trained video diffusers, ensuring high-fidelity alignment between dynamic content and conditional embeddings in text-to-video tasks without intermediate low-resolution anchor frames. This demonstrates superior efficiency and robustness in interactive applications like real-time portrait animation. While computational bottlenecks may arise in extreme high-resolution scenarios, optimizations through classifier-free guidance and variational inference this paradigm has emerged as an efficient, unified benchmark path in the evolution from VDM to Sora.
</span> 


[[paper]](https://arxiv.org/abs/2309.17080) [[website]](https://anthonyhu.github.io/gaia1) **GAIA-1: A Generative World Model for Autonomous Driving** [arXiv 2023]  
[[paper]](https://arxiv.org/abs/2311.13549) **ADriver-I: A General World Model for Autonomous Driving** [arXiv 2023]  
[[code]](https://github.com/cure-lab/MagicDrive) [[paper]](https://arxiv.org/abs/2310.02601) [[website]](https://gaoruiyuan.com/magicdrive/) **MagicDrive: Street View Generation with Diverse 3D Geometry Control** [ICLR 2024]  
[[code]](https://github.com/wenyuqing/panacea) [[paper]](https://arxiv.org/abs/2311.16813) [[website]](https://panacea-ad.github.io/) **Panacea: Panoramic and Controllable Video Generation for Autonomous Driving** [CVPR 2024]  
[[code]](https://github.com/BraveGroup/Drive-WM) [[paper]](https://arxiv.org/abs/2311.17918) [[website]](https://drive-wm.github.io/) **Drive-WM: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving** [CVPR 2024]  
[[code]](https://github.com/Akaneqwq/360DVD) [[paper]](https://arxiv.org/abs/2401.06578) [[website]](https://akaneqwq.github.io/360DVD/) **360DVD: Controllable Panorama Video Generation with 360-Degree Video Diffusion Model** [CVPR 2024]  
[[code]](https://github.com/JeffWang987/DriveDreamer) [[paper]](https://arxiv.org/abs/2309.09777) [[website]](https://drivedreamer.github.io/) **DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving** [ECCV 2024]  
[[code]](https://github.com/shalfun/DrivingDiffusion) [[paper]](https://arxiv.org/abs/2310.07771) [[website]](https://drivingdiffusion.github.io/) **DrivingDiffusion: Layout-Guided Multi-View Driving Scenarios Video Generation with Latent Diffusion Model** [ECCV 2024]  
[[code]](https://github.com/fudan-zvg/WoVoGen) [[paper]](https://arxiv.org/abs/2312.02934) **WoVoGen: World Volume-Aware Diffusion for Controllable Multi-camera Driving Scene Generation** [ECCV 2024]  
[[code]](https://github.com/OpenDriveLab/Vista) [[paper]](https://arxiv.org/abs/2405.17398) [[website]](https://opendrivelab.com/Vista/) **Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability** [NeurIPS 2024]  
[[code]](https://github.com/eloialonso/diamond) [[paper]](https://arxiv.org/abs/2405.12399) [[website]](https://diamond-wm.github.io/) **DIAMOND: Diffusion for World Modeling: Visual Details Matter in Atari** [NeurIPS 2024]  
[[code]](https://github.com/flymin/MagicDrive3D) [[paper]](https://arxiv.org/abs/2405.14475) [[website]](https://gaoruiyuan.com/magicdrive3d/) **MagicDrive3D: Controllable 3D Generation for Any-View Rendering in Street Scenes** [arXiv 2024]  
[[code]](https://github.com/westlake-autolab/Delphi) [[paper]](https://arxiv.org/abs/2406.01349) [[website]](https://westlake-autolab.github.io/delphi.github.io/) **Delphi: Unleashing Generalization of End-to-End Autonomous Driving with Controllable Long Video Generation** [arXiv 2024]  
[[code]](https://github.com/zympsyche/BevWorld) [[paper]](https://arxiv.org/abs/2407.05679) **BEVWorld: A Multimodal World Model for Autonomous Driving via Unified BEV Latent Space** [arXiv 2024]  
[[code]](https://github.com/PJLab-ADG/DriveArena) [[paper]](https://arxiv.org/abs/2408.00415) [[website]](https://pjlab-adg.github.io/DriveArena/) **DriveArena: A Closed-loop Generative Simulation Platform for Autonomous Driving** [arXiv 2024]  
[[code]](https://github.com/LiAutoAD/DIVE) [[paper]](https://arxiv.org/abs/2409.01595) [[website]](https://liautoad.github.io/DIVE/) **DiVE: DiT-based Video Generation with Enhanced Control** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2409.04003v3) [[website]](https://pjlab-adg.github.io/DriveArena/dreamforge/) **DreamForge: Motion-Aware Autoregressive Video Generation for Multi-View Driving Scenes** [arXiv 2024]  
[[code]](https://github.com/EnVision-Research/SyntheOcc) [[paper]](https://arxiv.org/abs/2410.00337v1) [[website]](https://len-li.github.io/syntheocc-web/) **SyntheOcc: Synthesize Geometric-Controlled Street View Images through 3D Semantic MPIs** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2412.01407) **HoloDrive: Holistic 2D-3D Multi-Modal Street Scene Generation for Autonomous Driving** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2412.03520) [[website]](https://luhannan.github.io/CogDrivingPage/) **CogDriving: Multi-View Driving Scene Video Generation with Holistic Attention** [arXiv 2024]  
[[code]](https://github.com/YS-IMTech/Imagine360) [[paper]](https://arxiv.org/abs/2412.03552) [[website]](https://ys-imtech.github.io/projects/Imagine360/) **Imagine360: Immersive 360 Video Generation from Perspective Anchor** [arXiv 2024]  
[[code]](https://github.com/YvanYin/DrivingWorld) [[paper]](https://arxiv.org/abs/2412.19505) [[website]](https://huxiaotaostasy.github.io/DrivingWorld/) **DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT** [arXiv 2024]  
[[code]](https://github.com/Drexubery/ViewCrafter) [[paper]](https://arxiv.org/abs/2409.02048) [[website]](https://drexubery.github.io/ViewCrafter/) **ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis** [arXiv 2024]  
[[code]](https://github.com/Kunhao-Liu/ViewExtrapolator) [[paper]](https://arxiv.org/abs/2411.14208) [[website]](https://kunhao-liu.github.io/ViewExtrapolator/) **ViewExtrapolator: Novel View Extrapolation with Video Diffusion Priors** [arXiv 2024]  
[[code]](https://github.com/f1yfisher/DriveDreamer2) [[paper]](https://arxiv.org/abs/2403.06845) [[website]](https://drivedreamer2.github.io/) **DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation** [AAAI 2025]  
[[paper]](https://arxiv.org/abs/2406.13527) [[website]](https://4k4dgen.github.io/) **4K4DGen: Panoramic 4D Generation at 4K Resolution** [ICLR 2025]  
[[code]](https://github.com/GameGen-X/GameGen-X) [[paper]](https://arxiv.org/abs/2411.00769v3) [[website]](https://gamegen-x.github.io/) **GameGen-X: Interactive Open-world Game Video Generation** [ICLR 2025]  
[[paper]](https://arxiv.org/abs/2408.14837) [[website]](https://gamengen.github.io/) **GameNGen: Diffusion Models Are Real-Time Game Engines** [ICLR 2025]  
[[code]](https://github.com/Beckschen/genEx) [[paper]](https://arxiv.org/abs/2411.11844) [[website]](https://generative-world-explorer.github.io/) **Genex: Generative World Explorer** [ICLR 2025]  
[[paper]](https://arxiv.org/abs/2503.00045) **GLAD: A Streaming Scene Generator for Autonomous Driving** [ICLR 2025]  
[[code]](https://github.com/yanty123/DrivingSphere) [[paper]](https://arxiv.org/abs/2411.11252) [[website]](https://yanty123.github.io/DrivingSphere/) **DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation** [CVPR 2025]  
[[code]](https://github.com/zju3dv/street_crafter) [[paper]](https://arxiv.org/abs/2412.13188) [[website]](https://zju3dv.github.io/street_crafter/) **StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models** [CVPR 2025]  
[[paper]](https://arxiv.org/abs/2409.05463) [[website]](https://metadrivescape.github.io/papers_project/drivescapev1/index.html) **DriveScape: Towards High-Resolution Controllable Multi-View Driving Video Generation** [CVPR 2025]  
[[code]](https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation) [[paper]](https://arxiv.org/abs/2412.05435) [[website]](https://arlo0o.github.io/uniscene/) **UniScene: Unified Occupancy-centric Driving Scene Generation** [CVPR 2025]  
[[code]](https://github.com/vita-epfl/GEM) [[paper]](https://arxiv.org/abs/2412.11198) [[website]](https://vita-epfl.github.io/GEM.github.io/) **GEM: A Generalizable Ego-Vision Multimodal World Model for Fine-Grained Ego-Motion, Object Dynamics, and Scene Composition Control** [CVPR 2025]  
[[code]](https://github.com/YanhaoWu/UMGen) [[paper]](https://arxiv.org/abs/2503.14945) [[website]](https://yanhaowu.github.io/UMGen/) **UMGen: Generating Multimodal Driving Scenes via Next-Scene Prediction** [CVPR 2025]  
[[paper]](https://arxiv.org/abs/2411.18613) [[website]](https://cat-4d.github.io/) **CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models** [CVPR 2025]  
[[code]](https://github.com/snap-research/wonderland/) [[paper]](https://arxiv.org/abs/2412.12091) [[website]](https://snap-research.github.io/wonderland/) **Wonderland: Navigating 3D Scenes from a Single Image** [CVPR 2025]  
[[code]](https://github.com/hanyang-21/VideoScene) [[paper]](https://arxiv.org/abs/2504.01956) [[website]](https://hanyang-21.github.io/VideoScene/) **VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step** [CVPR 2025]  
[[paper]](https://arxiv.org/abs/2504.02764) [[website]](https://shengjun-zhang.github.io/SceneSplatter/) **Scene Splatter: Momentum 3D Scene Generation from Single Image with Video Diffusion Model** [CVPR 2025]  
[[paper]](https://arxiv.org/abs/2412.11100) [[website]](https://dynamic-scaler.pages.dev/) **DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes** [CVPR 2025]  
[[code]](https://github.com/Little-Podi/AdaWorld) [[paper]](https://arxiv.org/abs/2503.18938) [[website]](https://adaptable-world-model.github.io/) **AdaWorld: Learning Adaptable World Models with Latent Actions** [ICML 2025]  
[[paper]](https://www.nature.com/articles/s41586-025-08600-3) [[website]](https://huggingface.co/microsoft/wham) **WHAM: World and Human Action Models towards gameplay ideation** [Nature 2025]  
[[paper]](https://arxiv.org/abs/2501.00601) [[website]](https://pointscoder.github.io/DreamDrive/) **DreamDrive: Generative 4D Scene Modeling from Street View Images** [arXiv 2025]  
[[code]](https://github.com/SenseTime-FVG/OpenDWM) [[paper]](https://arxiv.org/abs/2502.11663) [[website]](https://sensetime-fvg.github.io/MaskGWM/) **MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction** [arXiv 2025]  
[[code]](https://github.com/dk-liang/UniFuture) [[paper]](https://arxiv.org/abs/2503.13587) [[website]](https://dk-liang.github.io/UniFuture/) **UniFuture: A Unified Driving World Model for Future Generation and Perception** [arXiv 2025]  
[[code]](https://github.com/Li-Zn-H/SimWorld) [[paper]](https://arxiv.org/abs/2503.13952) **SimWorld: A Unified Benchmark for Simulator-Conditioned Scene Generation via World Model** [arXiv 2025]  
[[code]](https://github.com/royalmelon0505/dist4d) [[paper]](https://arxiv.org/abs/2503.15208) [[website]](https://royalmelon0505.github.io/DiST-4D/) **DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2503.20523) [[website]](https://wayve.ai/thinking/gaia-2/) **GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving** [arXiv 2025]  
[[code]](https://github.com/byeongjun-park/SteerX) [[paper]](https://arxiv.org/abs/2503.12024) [[website]](https://byeongjun-park.github.io/SteerX/) **SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2503.09160) **WonderVerse: Extendable 3D Scene Generation with Video Generative Models** [arXiv 2025]  
[[code]](https://github.com/ML-GSAI/FlexWorld) [[paper]](https://arxiv.org/abs/2503.13265) [[website]](https://ml-gsai.github.io/FlexWorld/) **FlexWorld: Progressively Expanding 3D Scenes for Flexible-View Synthesis** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2504.10001) **GaussVideoDreamer: 3D Scene Generation with Video Diffusion and Inconsistency-Aware Gaussian Splatting** [arXiv 2025]  
[[code]](https://github.com/xizaoqu/WorldMem) [[paper]](https://arxiv.org/abs/2504.12369) [[website]](https://xizaoqu.github.io/worldmem/) **WORLDMEM: Long-term Consistent World Simulation with Memory** [arXiv 2025]  
[[code]](https://github.com/PKU-YuanGroup/HoloTime) [[paper]](https://arxiv.org/abs/2504.21650) [[website]](https://zhouhyocean.github.io/holotime/) **HoloTime: Taming Video Diffusion Models for Panoramic 4D Scene Generation** [arXiv 2025]  
[[code]](https://github.com/microsoft/MineWorld) [[paper]](https://arxiv.org/abs/2504.08388) **MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft** [arXiv 2025]  
[[code]](https://github.com/KwaiVGI/GameFactory) [[paper]](https://arxiv.org/abs/2501.08325) [[website]](https://yujiwen.github.io/gamefactory/) **GameFactory: Creating New Games with Generative Interactive Videos** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2503.22231) [[website]](https://xiaomi-research.github.io/cogen/) **CoGen: 3D Consistent Video Generation via Adaptive Conditioning for Autonomous Driving** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2506.08006) [[website]](https://metadriverse.github.io/dreamland/) **Dreamland: Controllable World Creation with Simulator and Generative Models** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2506.04225) **Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation** [arXiv 2025]  
[[code]](https://github.com/SkyworkAI/Matrix-Game) [[paper]](https://matrix-game-v2.github.io/static/pdf/report.pdf) [[website]](https://matrix-game-v2.github.io/) **Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2506.17201) [[website]](https://hunyuan-gamecraft.github.io/) **Hunyuan-GameCraft: High-dynamic Interactive Game Video Generation with Hybrid History Condition** [arXiv 2025]  
[[code]](https://github.com/Colezwhy/CoCo4D-Gen) [[paper]](https://arxiv.org/abs/2506.19798) [[website]](https://colezwhy.github.io/coco4d/) **CoCo4D: Comprehensive and Complex 4D Scene Generation** [arXiv 2025]  
[[code]](https://github.com/GigaAI-research/WonderFree) [[paper]](https://arxiv.org/abs/2506.20590) [[website]](https://wonder-free.github.io/) **WonderFree: Enhancing Novel View Quality and Cross-View Consistency for 3D Scene Exploration** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2508.04467) [[website]](https://4dvd.github.io/) **4DVD: Cascaded Dense-view Video Diffusion Model for High-quality 4D Content Generation** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2508.04147) [[website]](https://idcnet-scene.github.io/) **IDCNet: Guided Video Diffusion for Metric-Consistent RGBD Scene Generation with Precise Camera Control** [arXiv 2025]  
[[code]](https://github.com/3DTopia/4DNeX) [[paper]](https://arxiv.org/abs/2508.13154) [[website]](https://4dnex.github.io/) **4DNeX: Feed-Forward 4D Generative Modeling Made Easy** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2505.18151) [[website]](https://kyleleey.github.io/WonderPlay/) **WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions** [ICCV 2025]  
[[code]](https://github.com/flymin/MagicDrive-V2) [[paper]](https://arxiv.org/abs/2411.13807v3) [[website]](https://gaoruiyuan.com/magicdrivedit/) **MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control** [ICCV 2025]  
[[code]](https://github.com/tianfr/DynamicVoyager) [[paper]](https://arxiv.org/abs/2507.04183) [[website]](https://tianfr.github.io/project/DynamicVoyager/) **DynamicVoyager: Voyaging into Unbounded Dynamic Scenes from a Single View** [ICCV 2025]  
[[code]](https://github.com/nv-tlabs/InfiniCube) [[paper]](https://arxiv.org/abs/2412.03934) [[website]](https://research.nvidia.com/labs/toronto-ai/infinicube/) **InfiniCube: Unbounded and Controllable Dynamic 3D Driving Scene Generation with World-Guided Video Models** [ICCV 2025]  
[[code]](https://github.com/runjiali-rl/vmem) [[paper]](https://arxiv.org/abs/2506.18903) [[website]](https://v-mem.github.io/) **VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory** [ICCV 2025]  
[[code]](https://github.com/KIMGEONUNG/VideoFrom3D) [[paper]](https://arxiv.org/abs/2509.17985) [[website]](https://kimgeonung.github.io/VideoFrom3D/) **VideoFrom3D: 3D Scene Video Generation via Complementary Image and Video Diffusion Models** [SIGGRAPH Asia 2025]  



##### 3D Scene Editing

[[paper]](https://arxiv.org/abs/2112.01530) [[project]](https://lukashoel.github.io/stylemesh/) [[code]](https://github.com/lukasHoel/stylemesh) **StyleMesh: Style Transfer for Indoor 3D Scene Reconstructions** [CVPR 2022]  
[[paper]](https://arxiv.org/abs/2212.11984) [[project]](https://snap-research.github.io/discoscene/) [[code]](https://github.com/snap-research/discoscene) **DisCoScene: Spatially Disentangled Generative Radiance Fields for Controllable 3D-aware Scene Synthesis** [CVPR 2023]  
[[paper]](https://arxiv.org/abs/2301.09629) [[project]](https://ivl.cs.brown.edu/research/lego-net.html) [[code]](https://github.com/QiuhongAnnaWei/LEGO-Net) **LEGO-Net: Learning Regular Rearrangements of Objects in Rooms** [CVPR 2023]  
[[paper]](https://arxiv.org/abs/2304.03526) [[project]](https://len-li.github.io/lift3d-web/) [[code]](https://github.com/EnVision-Research/Lift3D) **Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative Radiance Field** [CVPR 2023]  
[[paper]](https://arxiv.org/abs/2308.16880) **Text2Scene: Text-driven Indoor Scene Stylization with Part-aware Details** [CVPR 2023]  
[[paper]](https://arxiv.org/abs/2304.09302) [[project]](https://cabinet-object-rearrangement.github.io/) [[code]](https://github.com/NVlabs/cabi_net) **CabiNet: Scaling Neural Collision Detection for Object Rearrangement with Procedural Scene Generation** [ICRA 2023]  
[[paper]](https://arxiv.org/abs/2305.11337) **RoomDreamer: Text-Driven 3D Indoor Scene Synthesis with Coherent Geometry and Texture** [MM 2023]  
[[paper]](https://arxiv.org/abs/2311.17261) [[project]](https://daveredrum.github.io/SceneTex/) [[code]](https://github.com/daveredrum/SceneTex) **SceneTex: High-Quality Texture Synthesis for Indoor Scenes via Diffusion Priors** [CVPR 2024]  
[[paper]](https://arxiv.org/abs/2312.05208) [[project]](https://jonasschult.github.io/ControlRoom3D/) **ControlRoom3D 🤖Room Generation using Semantic Proxy Rooms** [CVPR 2024]  
[[paper]](https://arxiv.org/abs/2404.10681) [[project]](https://www.chenyingshu.com/stylecity3d/) [[code]](https://github.com/chenyingshu/stylecity3d) **StyleCity: Large-Scale 3D Urban Scenes Stylization** [ECCV 2024]  
[[paper]](https://arxiv.org/abs/2406.02461) [[project]](https://qwang666.github.io/RoomTex/) [[code]](https://github.com/qwang666/RoomTex-) **RoomTex: Texturing Compositional Indoor Scenes via Iterative Inpainting** [ECCV 2024]  
[[paper]](https://arxiv.org/abs/2311.12050) [[project]](https://3d-goi.github.io/) **3D-GOI: 3D GAN Omni-Inversion for Multifaceted and Multi-object Editing** [ECCV 2024]  
[[paper]](https://openreview.net/forum?id=V5HU1OvHnx) [[code]](https://github.com/Shao-Kui/3DScenePlatform?tab=readme-ov-file#sceneexpander) **SceneExpander: Real-Time Scene Synthesis for Interactive Floor Plan Editing** [MM 2024]  
[[paper]](https://arxiv.org/abs/2406.09292) [[project]](https://neural-assets.github.io/) **Neural Assets: 3D-Aware Multi-Object Scene Synthesis with Image Diffusion Models** [NeurIPS 2024]  
[[paper]](https://arxiv.org/abs/2409.18336) **DeBaRA: Denoising-Based 3D Room Arrangement Generation** [NeurIPS 2024]  
[[paper]](https://doi.org/10.1145/3680528.3687633) [[project]](https://vcc.tech/research/2024/InstanceTex) **InstanceTex: Instance-level Controllable Texture Synthesis for 3D Scenes via Diffusion Priors** [SIGGRAPH Asia 2024]  
[[paper]](https://doi.org/10.1109/TVCG.2023.3268115) [[code]](https://github.com/Shao-Kui/3DScenePlatform#scenedirector) **SceneDirector: Interactive Scene Synthesis by Simultaneously Editing Multiple Objects in Real-Time** [TVCG 2024]  
[[paper]](https://arxiv.org/abs/2310.13119) [[project]](https://ybbbbt.com/publication/dreamspace/) [[code]](https://github.com/ybbbbt/dreamspace) **DreamSpace: Dreaming Your Room Space with Text-Driven Panoramic Texture Propagation** [VR 2024]  
[[paper]](https://arxiv.org/abs/2310.03602) [[project]](https://fangchuan.github.io/ctrl-room.github.io/) [[code]](https://github.com/fangchuan/Ctrl-Room) **Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints** [3DV 2025]  
[[paper]](https://arxiv.org/abs/2412.16778) **RoomPainter: View-Integrated Diffusion for Consistent Indoor Scene Texturing** [CVPR 2025]  
[[paper]](https://arxiv.org/abs/2502.10377) [[project]](https://restyle3d.github.io/) [[code]](https://github.com/GradientSpaces/ReStyle3D) **ReStyle3D: Scene-Level Appearance Transfer with Semantic Correspondences** [SIGGRAPH 2025]  


##### Human-Scene Interaction
[[paper]](https://arxiv.org/abs/2205.13001) **Towards Diverse and Natural Scene-aware 3D Human Motion Synthesis** [CVPR 2022]  
[[paper]](https://arxiv.org/abs/2207.12824) [[project]](https://zkf1997.github.io/COINS/index.html) [[code]](https://github.com/zkf1997/COINS) **COINS: Compositional Human-Scene Interaction Synthesis with Semantic Control** [ECCV 2022]  
[[paper]](https://arxiv.org/abs/2301.06015) [[project]](https://scenediffuser.github.io/) [[code]](https://github.com/scenediffuser/Scene-Diffuser) **SceneDiffuser: Diffusion-based Generation, Optimization, and Planning in 3D Scenes** [CVPR 2023]  
[[paper]](https://arxiv.org/abs/2305.12411) [[project]](https://zkf1997.github.io/DIMOS/) [[code]](https://github.com/zkf1997/DIMOS) **DIMOS: Synthesizing Diverse Human Motions in 3D Indoor Scenes** [ICCV 2023]  
[[paper]](https://arxiv.org/abs/2302.00883) [[project]](https://xbpeng.github.io/projects/InterPhys/index.html) **InterPhys: Synthesizing Physical Character-Scene Interactions** [SIGGRAPH 2023]  
[[paper]](https://arxiv.org/abs/2308.09036) [[project]](https://liangpan99.github.io/InterScene/) [[code]](https://github.com/liangpan99/InterScene) **InterScene: Synthesizing Physically Plausible Human Motions in 3D Scenes** [3DV 2024]  
[[paper]](https://arxiv.org/abs/2311.17737) [[project]](https://craigleili.github.io/projects/genzi/) **GenZI: Zero-Shot 3D Human-Scene Interaction Generation** [CVPR 2024]  
[[paper]](https://arxiv.org/abs/2309.07918) [[project]](https://xizaoqu.github.io/unihsi/) [[code]](https://github.com/OpenRobotLab/UniHSI) **UniHSI: Unified Human-Scene Interaction via Prompted Chain-of-Contacts** [ICLR 2024]  
[[paper]](https://arxiv.org/abs/2411.19921) [[project]](https://wenjiawang0312.github.io/projects/sims/) [[code]](https://github.com/WenjiaWang0312/sims-stylized_hsi) **SIMS: Simulating Stylized Human-Scene Interactions with Retrieval-Augmented Script Generation** [ICCV 2025]  
[[paper]](https://arxiv.org/abs/2503.19901) [[project]](https://liangpan99.github.io/TokenHSI/) [[code]](https://github.com/liangpan99/TokenHSI) **TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization** [CVPR 2025]  


##### Embodied AI

[[paper]](https://arxiv.org/abs/2206.06994) [[project]](https://procthor.allenai.org/) [[code]](https://github.com/allenai/procthor) **ProcTHOR: Large-Scale Embodied AI Using Procedural Generation** [NeurIPS 2022]  
[[paper]](https://arxiv.org/abs/2312.09067) [[project]](https://yueyang1996.github.io/holodeck/) [[code]](https://github.com/allenai/Holodeck) **Holodeck: Language Guided Generation of 3D Embodied AI Environments** [CVPR 2024]  
[[paper]](https://arxiv.org/abs/2404.09465) [[project]](https://physcene.github.io/) [[code]](https://github.com/PhyScene/PhyScene) **PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI** [CVPR 2024]  
[[paper]](https://arxiv.org/abs/2411.09823) [[project]](https://wangyian-me.github.io/Architect/) [[code]](https://github.com/wangyian-me/architect_official_code) **Architect: Generating Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting** [NeurIPS 2024]  
[[paper]](https://arxiv.org/abs/2407.10943) [[project]](https://grutopia.github.io/) [[code]](https://github.com/OpenRobotLab/GRUtopia) **GRUtopia: Dream General Robots in a City at Scale** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2410.09604) [[project]](https://embodied-city.fiblab.net/) [[code]](https://github.com/tsinghua-fib-lab/EmbodiedCity) **EmbodiedCity: A Benchmark Platform for Embodied Agent in Real-world City Environment** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2412.05789) [[code]](https://github.com/pzhren/InfiniteWorld) **InfiniteWorld: A Unified Scalable Simulation Framework for General Visual-Language Robot Interaction** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2407.08725) [[project]](https://metadriverse.github.io/metaurban/) [[code]](https://github.com/metadriverse/metaurban) **MetaUrban: An Embodied AI Simulation Platform for Urban Micromobility** [ICLR 2025]  



##### Robotics

[[paper]](https://arxiv.org/abs/2302.00111) [[project]](https://universal-policy.github.io/unipi/) **UniPi: Learning Universal Policies via Text-Guided Video Generation** [NeurIPS 2023]  
[[paper]](https://arxiv.org/abs/2309.08587) [[project]](https://hierarchical-planning-foundation-model.github.io/) [[code]](https://github.com/anuragajay/hip) **HiP: Compositional Foundation Models for Hierarchical Planning** [NeurIPS 2023]  
[[paper]](https://arxiv.org/abs/2406.11740) [[project]](https://haojhuang.github.io/imagine_page/) [[code]](https://github.com/HaojHuang/imagination-policy-cor24) **Imagination Policy: Using Generative Point Cloud Models for Learning Manipulation Policies** [CoRL 2024]  
[[paper]](https://arxiv.org/abs/2411.01775) [[project]](https://eureka-research.github.io/eurekaverse/) [[code]](https://github.com/eureka-research/eurekaverse) **Eurekaverse: Environment Curriculum Generation via Large Language Models** [CoRL 2024]  
[[paper]](https://arxiv.org/abs/2312.13139) [[project]](https://gr1-manipulation.github.io/) [[code]](https://github.com/bytedance/GR-1) **GR-1: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation** [ICLR 2024]  
[[paper]](https://arxiv.org/abs/2311.01455) [[project]](https://robogen-ai.github.io/) [[code]](https://github.com/Genesis-Embodied-AI/RoboGen) **RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation** [ICML 2024]  
[[paper]](https://arxiv.org/abs/2402.10534) **VLP: Using Left and Right Brains Together: Towards Vision and Language Planning** [ICML 2024]  
[[paper]](https://arxiv.org/abs/2404.01812) [[project]](https://actnerf.github.io/) [[code]](https://github.com/ActNeRF/ActNeRF) **ActNeRF: Uncertainty-aware Active Learning of NeRF-based Object Models for Robot Manipulators using Visual and Re-orientation Actions** [IROS 2024]  
[[paper]](https://arxiv.org/abs/2409.09016) [[code]](https://github.com/OpenDriveLab/CLOVER) **CLOVER: Closed-Loop Visuomotor Control with Generative Expectation for Robotic Manipulation** [NeurIPS 2024]  
[[paper]](https://arxiv.org/abs/2410.06158) [[project]](https://gr2-manipulation.github.io/) **GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2410.23277) [[project]](https://slowfast-vgen.github.io/) [[code]](https://github.com/slowfast-vgen/slowfast-vgen) **SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation** [ICLR 2025]  
[[paper]](https://arxiv.org/abs/2412.14803) [[project]](https://video-prediction-policy.github.io/) [[code]](https://github.com/roboterax/video-prediction-policy) **Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations** [ICML 2025]  
[[paper]](https://arxiv.org/abs/2501.09781) [[project]](https://maverickren.github.io/VideoWorld.github.io/) [[code]](https://github.com/ByteDance-Seed/VideoWorld) **VideoWorld: Exploring Knowledge Learning from Unlabeled Videos** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2503.14492) [[project]](https://research.nvidia.com/labs/dir/cosmos-transfer1/) [[code]](https://github.com/nvidia-cosmos/cosmos-transfer1) **Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2504.20995) [[project]](https://tesseractworld.github.io/) [[code]](https://github.com/UMass-Embodied-AGI/TesserAct) **TesserAct: Learning 4D Embodied World Models** [arXiv 2025]  


###### Autonomous Driving

[[paper]](https://arxiv.org/abs/2309.17080) [[project]](https://anthonyhu.github.io/gaia1) **GAIA-1: A Generative World Model for Autonomous Driving** [arXiv 2023]  
[[paper]](https://arxiv.org/abs/2311.17663) [[code]](https://github.com/haomo-ai/Cam4DOcc) **Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications** [arXiv 2023]  
[[paper]](https://arxiv.org/abs/2311.17918) [[project]](https://drive-wm.github.io/) [[code]](https://github.com/BraveGroup/Drive-WM) **Drive-WM: Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving** [CVPR 2024]  
[[paper]](https://arxiv.org/abs/2309.09777) [[project]](https://drivedreamer.github.io/) [[code]](https://github.com/JeffWang987/DriveDreamer) **DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving** [ECCV 2024]  
[[paper]](https://arxiv.org/abs/2311.16038) [[project]](https://wzzheng.net/OccWorld/) [[code]](https://github.com/wzzheng/OccWorld) **OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving** [ECCV 2024]  
[[paper]](https://arxiv.org/abs/2312.02934) [[code]](https://github.com/fudan-zvg/WoVoGen) **WoVoGen: World Volume-Aware Diffusion for Controllable Multi-camera Driving Scene Generation** [ECCV 2024]  
[[paper]](https://arxiv.org/abs/2310.02601) [[project]](https://gaoruiyuan.com/magicdrive/) [[code]](https://github.com/cure-lab/MagicDrive) **MagicDrive: Street View Generation with Diverse 3D Geometry Control** [ICLR 2024]  
[[paper]](https://arxiv.org/abs/2405.17398) [[project]](https://opendrivelab.com/Vista/) [[code]](https://github.com/OpenDriveLab/Vista) **Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability** [NeurIPS 2024]  
[[paper]](https://arxiv.org/abs/2405.20337) [[project]](https://wzzheng.net/OccSora/) [[code]](https://github.com/wzzheng/OccSora) **OccSora: 4D Occupancy Generation Models as World Simulators for Autonomous Driving** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2406.01349) [[project]](https://westlake-autolab.github.io/delphi.github.io/) [[code]](https://github.com/westlake-autolab/Delphi) **Delphi: Unleashing Generalization of End-to-End Autonomous Driving with Controllable Long Video Generation** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2408.00415) [[project]](https://pjlab-adg.github.io/DriveArena/) [[code]](https://github.com/PJLab-ADG/DriveArena) **DriveArena: A Closed-loop Generative Simulation Platform for Autonomous Driving** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2409.01595) [[project]](https://liautoad.github.io/DIVE/) [[code]](https://github.com/LiAutoAD/DIVE) **DiVE: DiT-based Video Generation with Enhanced Control** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2409.04003) [[project]](https://pjlab-adg.github.io/DriveArena/dreamforge/) **DreamForge: Motion-Aware Autoregressive Video Generation for Multi-View Driving Scenes** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2412.19505) [[project]](https://huxiaotaostasy.github.io/DrivingWorld/) [[code]](https://github.com/YvanYin/DrivingWorld) **DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2408.14197) [[project]](https://drive-occworld.github.io/) [[code]](https://github.com/yuyang-cloud/Drive-OccWorld) **Drive-OccWorld: Driving in the Occupancy World: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving** [AAAI 2025]  
[[paper]](https://arxiv.org/abs/2411.11252) [[project]](https://yanty123.github.io/DrivingSphere/) [[code]](https://github.com/yanty123/DrivingSphere) **DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation** [CVPR 2025]  
[[paper]](https://arxiv.org/abs/2503.00045) **GLAD: A Streaming Scene Generator for Autonomous Driving** [ICLR 2025]  
[[paper]](https://arxiv.org/abs/2501.00601) [[project]](https://pointscoder.github.io/DreamDrive/) **DreamDrive: Generative 4D Scene Modeling from Street View Images** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2503.14492) [[project]](https://research.nvidia.com/labs/dir/cosmos-transfer1/) [[code]](https://github.com/nvidia-cosmos/cosmos-transfer1) **Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control** [arXiv 2025]  







### NeRF

#### Acceleration
[[code]](https://github.com/bmild/nerf) [[paper]](https://arxiv.org/abs/2003.08934) [[project]](https://www.matthewtancik.com/nerf) **NeRF:Representing Scenes as Neural Radiance Fields for View Synthesis** [ECCV 2020]  
[[code]](https://github.com/NVlabs/instant-ngp) [[paper]](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf) [[project]](https://nvlabs.github.io/instant-ngp/) **Instant Neural Graphics Primitives with a Multiresolution Hash Encoding** [TOG 2022]  
[[code]](https://github.com/snap-research/R2L) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/626_ECCV_2022_paper.php) [[project]](https://snap-research.github.io/R2L/) **R2L: Distilling Neural Radiance Field to Neural Light Field for Efficient Novel View Synthesis** [ECCV 2022]  
[[code]](https://github.com/thomasneff/AdaNeRF) [[paper]](https://thomasneff.github.io/adanerf/adanerf_supplementary.pdf) [[project]](https://thomasneff.github.io/adanerf/) **AdaNeRF: Adaptive Sampling for Real-time Rendering of Neural Radiance Fields** [ECCV 2022]  
[[code]](https://github.com/Xharlie/pointnerf) [[paper]](https://arxiv.org/pdf/2201.08845) [[project]](https://xharlie.github.io/projects/project_sites/pointnerf/index.html) **Point-NeRF: Point-based Neural Radiance Fields** [CVPR 2020]  
[[code]](https://github.com/lwwu2/diver) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_DIVeR_Real-Time_and_Accurate_Neural_Radiance_Fields_With_Deterministic_Integration_CVPR_2022_paper.pdf) **DIVeR: Real-time and Accurate Neural Radiance Fields with Deterministic Integration for Volume Rendering** [CVPR 2022]  
[[code]](https://github.com/sunset1995/DirectVoxGO) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Direct_Voxel_Grid_Optimization_Super-Fast_Convergence_for_Radiance_Fields_Reconstruction_CVPR_2022_paper.pdf) [[project]](https://sunset1995.github.io/dvgo/) **Direct Voxel Grid Optimization: Super-Fast Convergence for Radiance Fields Reconstruction** [CVPR 2022]  
[[code]](https://github.com/dvlab-research/EfficientNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_EfficientNeRF__Efficient_Neural_Radiance_Fields_CVPR_2022_paper.pdf) **EfficientNeRF – Efficient Neural Radiance Fields** [CVPR 2022]  
[[code]](https://github.com/google-research/multinerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Barron_Mip-NeRF_360_Unbounded_Anti-Aliased_Neural_Radiance_Fields_CVPR_2022_paper.html) [[project]](https://jonbarron.info/mipnerf360/) **Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields** [CVPR 2022]  
[[code]](https://github.com/sxyu/plenoctree) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_PlenOctrees_for_Real-Time_Rendering_of_Neural_Radiance_Fields_ICCV_2021_paper.pdf) [[project]](https://alexyu.net/plenoctrees/) **PlenOctrees for Real-time Rendering of Neural Radiance Fields** [ICCV 2021]  
[[code]](https://github.com/creiser/kilonerf) [[paper]](https://thomasneff.github.io/adanerf/adanerf_supplementary.pdf) [[project]](https://creiser.github.io/kilonerf/) **KiloNeRF: Speeding Up Neural Radiance Fields with Thousands of Tiny MLPs** [ICCV 2021]  
[[code]](https://github.com/simicvm/snerg) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Hedman_Baking_Neural_Radiance_Fields_for_Real-Time_View_Synthesis_ICCV_2021_paper.pdf) **Baking Neural Radiance Fields for Real-Time View-Synthesis** [ICCV 2021]  
[[code]](https://github.com/snap-research/MobileR2L) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Real-Time_Neural_Light_Field_on_Mobile_Devices_CVPR_2023_paper.pdf) [[project]](https://snap-research.github.io/MobileR2L/) **Real-time neural light field on mobile devices** [CVPR 2023]  
[[code]](https://github.com/wolfball/PlenVDB) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_PlenVDB_Memory_Efficient_VDB-Based_Radiance_Fields_for_Fast_Training_and_CVPR_2023_paper.pdf) [[project]](https://plenvdb.github.io/) **PlenVDB: Memory Efficient VDB-Based Radiance Fields for Fast Training and Rendering** [CVPR 2023]  
[[code]](https://github.com/apchenstu/TensoRF) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920332.pdf) [[project]](https://apchenstu.github.io/TensoRF/) **TensoRF: Tensorial Radiance Fields** [ECCV 2022]  
[[code]](https://github.com/google/hypernerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Turki_HybridNeRF_Efficient_Neural_Rendering_via_Adaptive_Volumetric_Surfaces_CVPR_2024_paper.pdf) [[project]](https://haithemturki.com/hybrid-nerf/) **HybridNeRF: Efficient Neural Rendering via Adaptive Volumetric Surfaces** [CVPR 2024]  
[[code]](https://github.com/facebookresearch/NSVF) [[paper]](https://arxiv.org/pdf/2007.11571) [[project]](https://lingjie0206.github.io/papers/NSVF/) **NSVF: Neural Sparse Voxel Fields** [NeurIPS 2020]  
[[code]](https://github.com/computational-imaging/automatic-integration) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Lindell_AutoInt_Automatic_Integration_for_Fast_Neural_Volume_Rendering_CVPR_2021_paper.pdf) [[project]](http://www.computationalimaging.org/publications/automatic-integration/) **AutoInt: Automatic Integration for Fast Neural Volume Rendering** [CVPR 2021]  
[[code]](https://github.com/ubc-vision/derf) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Rebain_DeRF_Decomposed_Radiance_Fields_CVPR_2021_paper.pdf) [[project]](https://ubc-vision.github.io/derf/) **DeRF: Decomposed Radiance Fields** [CVPR 2021]  
[[code]](https://github.com/facebookresearch/DONERF) [[paper]](https://arxiv.org/abs/2103.03231) [[project]](https://diglib.eg.org/items/cf4a7108-7130-469e-8886-48a767fd54e5) **DONeRF: Towards Real-Time Rendering of Compact Neural Radiance Fields using Depth Oracle Networks** [CGO 2021]  
[[code]](https://github.com/mrcabellom/fastNerf) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Garbin_FastNeRF_High-Fidelity_Neural_Rendering_at_200FPS_ICCV_2021_paper.pdf) **FastNeRF: High-Fidelity Neural Rendering at 200FPS** [ICCV 2021]  
[[code]](https://github.com/vsitzmann/light-field-networks.git) [[paper]](https://arxiv.org/abs/2106.02634) [[project]](https://www.vincentsitzmann.com/lfns/) **Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering** [NeurIPS 2021]  
[[paper]](https://licj15.github.io/rt-nerf/assets/2022ICCAD_RT_NeRF_31Oct2022.pdf) [[project]](https://licj15.github.io/rt-nerf/) **RT-NeRF: Real-Time On-Device Neural Radiance Fields Towards Immersive AR/VR Rendering** [ICCAD 2022]  
[[code]](https://github.com/zju3dv/ENeRF) [[paper]](https://arxiv.org/abs/2112.01517) [[project]](https://zju3dv.github.io/enerf/) **ENeRF: Efficient Neural Radiance Fields for Interactive Free-viewpoint Video** [SIGGRAPH Asia 2022]  
[[code]](https://github.com/Heng14/DyLiN) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_DyLiN_Making_Light_Field_Networks_Dynamic_CVPR_2023_paper.pdf) [[project]](https://dylin2023.github.io/) **DyLiN: Making Light Field Networks Dynamic** [CVPR 2023]  
[[code]](https://github.com/dunbar12138/DSNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_Depth-Supervised_NeRF_Fewer_Views_and_Faster_Training_for_Free_CVPR_2022_paper.pdf) [[project]](https://www.cs.cmu.edu/~dsnerf/) **Depth-supervised NeRF: Fewer Views and Faster Training for Free** [CVPR 2022]  
[[code]](https://github.com/sxyu/svox2) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Fridovich-Keil_Plenoxels_Radiance_Fields_Without_Neural_Networks_CVPR_2022_paper.pdf) [[project]](https://github.com/sxyu/svox2.git) **Plenoxels: Radiance Fields without Neural Networks** [CVPR 2022]  
[[paper]](https://arxiv.org/abs/2302.14859) [[project]](https://bakedsdf.github.io/) **BakedSDF: Meshing Neural SDFs for Real-Time View Synthesis** [SIGGRAPH 2023]  
[[code]](https://github.com/VISION-SJTU/Lightning-NeRF) [[paper]](https://arxiv.org/abs/2403.05907) **Lightning NeRF: Efficient Hybrid Scene Representation for Autonomous Driving** [ICRA 2024]  
[[code]](https://github.com/NVIDIAGameWorks/kaolin-wisp) [[paper]](https://arxiv.org/abs/2206.07707) [[project]](https://nv-tlabs.github.io/vqad/) **Variable Bitrate Neural Fields** [SIGGRAPH 2022]  





#### Quality & Realism Improvement

[[code]](https://github.com/google-research/multinerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Verbin_Ref-NeRF_Structured_View-Dependent_Appearance_for_Neural_Radiance_Fields_CVPR_2022_paper.pdf?_hsenc=p2ANqtz-8tadeadAJeGMwdMK0dCQgL4tcspDr7QP-jHu5vlS_dI1xLF0CUSZRiUo_5SCHuLAIP0XSO) [[project]](https://gcl-seminar.github.io/Awesome-Graphics-Papers/papers/CVPR/2022/RefNeRF/) **Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields** [CVPR 2022]   
[[code]](https://github.com/oppo-us-research/NeuRBF) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_NeuRBF_A_Neural_Fields_Representation_with_Adaptive_Radial_Basis_Functions_ICCV_2023_paper.pdf) [[project]](https://oppo-us-research.github.io/NeuRBF-website/) **NeurBF: A Neural Fields Representation with Adaptive Radial Basis Functions** [ICCV 2023]  
[[code]](https://zyqz97.github.io/GP_NeRF/) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_GP-NeRF_Generalized_Perception_NeRF_for_Context-Aware_3D_Scene_Understanding_CVPR_2024_paper.pdf) [[project]](https://zyqz97.github.io/GP_NeRF/) **GP-NeRF: Generalized Perception NeRF for Context-Aware 3D Scene Understanding** [CVPR 2024]  
[[code]](https://github.com/Crishawy/NeXT) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920069.pdf) **NeXT: Towards High Quality Neural Radiance Fields via Multi-Skip Transformer** [ECCV 2022]  
[[code]](https://github.com/cgtuebingen/NeRD-Neural-Reflectance-Decomposition) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Boss_NeRD_Neural_Reflectance_Decomposition_From_Image_Collections_ICCV_2021_paper.pdf) [[project]](https://markboss.me/publication/2021-nerd/) **NeRD: Neural Reflectance Decomposition from Image Collections** [ICCV 2021]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Srinivasan_NeRV_Neural_Reflectance_and_Visibility_Fields_for_Relighting_and_View_CVPR_2021_paper.pdf) [[project]](https://pratulsrinivasan.github.io/nerv/) **NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis** [CVPR 2022]  
[[code]](https://github.com/SuLvXiangXin/zipnerf-pytorch) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Barron_Zip-NeRF_Anti-Aliased_Grid-Based_Neural_Radiance_Fields_ICCV_2023_paper.pdf) **Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields** [ICCV 2023]  
[[paper]](https://github.com/thucz/PanoGRF) [[paper]](https://3dvar.com/Chen2023PanoGRF.pdf) [[project]](https://thucz.github.io/PanoGRF/) **PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline Panoramas** [NeurIPS 2023]  
[[code]](https://github.com/3D-FRONT-FUTURE/NeuDA) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Cai_NeuDA_Neural_Deformable_Anchor_for_High-Fidelity_Implicit_Surface_Reconstruction_CVPR_2023_paper.pdf) [[project]](https://3d-front-future.github.io/neuda/) **NeuDA: Neural deformable anchor for high-fidelity implicit surface reconstruction** [CVPR 2023]  
[[code]](https://github.com/ActiveVisionLab/nope-nerf/tree/main) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Bian_NoPe-NeRF_Optimising_Neural_Radiance_Field_With_No_Pose_Prior_CVPR_2023_paper.pdf) [[project]](https://nope-nerf.active.vision/) **NoPe-NeRF: Optimising neural radiance field with no pose prior** [CVPR 2023]  
[[code]](https://github.com/sony/NeISF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_NeISF_Neural_Incident_Stokes_Field_for_Geometry_and_Material_Estimation_CVPR_2024_paper.pdf) [[project]](https://sony.github.io/NeISF/) **NeISF: Neural Incident Stokes Field for Geometry and Material Estimation** [CVPR 2024]  
[[code]](https://github.com/lyclyc52/SANeRF-HQ) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_SANeRF-HQ_Segment_Anything_for_NeRF_in_High_Quality_CVPR_2024_paper.pdf) [[project]](https://lyclyc52.github.io/SANeRF-HQ/) **SANeRF-HQ: Segment Anything for NeRF in High Quality** [CVPR 2024]  
[[code]](https://github.com/autonomousvision/murf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_MuRF_Multi-Baseline_Radiance_Fields_CVPR_2024_paper.pdf) [[project]](https://haofeixu.github.io/murf/) **MuRF: Multi-Baseline Radiance Fields** [CVPR 2024]  
[[code]](https://github.com/s3anwu/pbrnerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_PBR-NeRF_Inverse_Rendering_with_Physics-Based_Neural_Fields_CVPR_2025_paper.pdf) **PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields** [CVPR 2025]  
[[paper]](https://github.com/baskargroup/SC-NeRF) [[paper]](https://arxiv.org/abs/2503.21958) [[project]](https://baskargroup.github.io/SC-NeRF/) **SC-NeRF: NeRF-based Point Cloud Reconstruction using a Stationary Camera for Agricultural Applications** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025W/CV4Metaverse/papers/Zhang_IL-NeRF_Incremental_Learning_for_Neural_Radiance_Fields_with_Camera_Pose_CVPRW_2025_paper.pdf) [[project]](https://3d-front-future.github.io/neuda/) **IL-NeRF: Incremental Learning for Neural Radiance Fields with Camera Pose Learning** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025W/CV4Metaverse/papers/Seo_ARC-NeRF_Area_Ray_Casting_for_Broader_Unseen_View_Coverage_in_CVPRW_2025_paper.pdf) [[project]](https://shawn615.github.io/arc-nerf/) **ARC-NeRF: Area Ray Casting for Broader Unseen View Coverage in Few-shot Object Insertion** [CVPR 2025]  
[[code]](https://github.com/linjohnss/FrugalNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_FrugalNeRF_Fast_Convergence_for_Extreme_Few-shot_Novel_View_Synthesis_without_CVPR_2025_paper.pdf) [[project]](https://linjohnss.github.io/frugalnerf/) **FrugalNeRF: Fast Convergence for Extreme Few-shot Novel View Synthesis** [CVPR 2025]  
[[code]](https://github.com/wen-yuan-zhang/NeRFPrior) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_NeRFPrior_Learning_Neural_Radiance_Field_as_a_Prior_for_Indoor_CVPR_2025_paper.pdf)**NeRFPrior: Learning Neural Radiance Field as a Prior for Signed Distance Fields** [CVPR 2025]  
[[code]](https://github.com/tancik/fourier-feature-networks) [[paper]](https://proceedings.neurips.cc/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf) [[project]](https://bmild.github.io/fourfeat/) **Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains** [NeurIPS 2020]  
[[code]](https://github.com/rover-xingyu/L2G-NeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Local-to-Global_Registration_for_Bundle-Adjusting_Neural_Radiance_Fields_CVPR_2023_paper.pdf) [[project]](https://rover-xingyu.github.io/L2G-NeRF/) **Local-to-global Registration for Bundle-adjusting Neural Radiance Fields** [CVPR 2023]  
[[paper]](https://arxiv.org/abs/2405.14871) [[project]](https://dorverbin.github.io/nerf-casting/) **NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections** [SIGGRAPH Asia 2024]  
[[code]](https://github.com/syntec-research/LitNeRF) [[paper]](https://drive.google.com/file/d/1fFbioHF6FMirIZPHhrdKwoYXHnv71GHL/view) [[project]](https://syntec-research.github.io/LitNeRF/) **LitNeRF: Intrinsic Radiance Decomposition for High-Quality View Synthesis and Relighting of Faces** [CSIGGRAPH Asia 2023]  
[[code]](https://github.com/cwchenwang/NeRF-SR) [[paper]](https://cg.cs.tsinghua.edu.cn/papers/MM-2022-NeRF-SR.pdf) [[project]](https://cwchenwang.github.io/NeRF-SR/) **NeRF-SR: High-Quality Neural Radiance Fields using Supersampling** [ACM 2022]  
[[code]](https://github.com/cwchenwang/NeRF-SR) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_RefSR-NeRF_Towards_High_Fidelity_and_Super_Resolution_View_Synthesis_CVPR_2023_paper.pdf) [[project]](https://cwchenwang.github.io/NeRF-SR/) **RefSR-NeRF: Towards High-Fidelity and Super-Resolution View Synthesis** [CVPR 2023]  




#### Dynamic & Deformable Scenes

[[code]](https://github.com/seoha-kim/Sync-NeRF) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28057) [[project]](https://seoha-kim.github.io/sync-nerf/) **Sync-NeRF: Generalizing Dynamic NeRFs to Unsynchronized Videos** [AAAI 2024]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Otonari_Entity-NeRF_Detecting_and_Removing_Moving_Entities_in_Urban_Scenes_CVPR_2024_paper.pdf) [[project]](https://otonari726.github.io/entitynerf/) **Entity-NeRF: Detecting and Removing Moving Entities in Urban Scenes** [CVPR 2024]  
[[code]](https://github.com/google/dynibar) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_DynIBaR_Neural_Dynamic_Image-Based_Rendering_CVPR_2023_paper.pdf) [[project]](https://dynibar.github.io/) **DynIBaR: Neural Dynamic Image-Based Rendering** [CVPR 2023]  
[[code]](https://github.com/DSaurus/Tensor4D) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shao_Tensor4D_Efficient_Neural_4D_Decomposition_for_High-Fidelity_Dynamic_Reconstruction_and_CVPR_2023_paper.pdf) [[project]](https://liuyebin.com/tensor4d/tensor4d.html) **Tensor4D: Efficient Neural 4D Decomposition for High-fidelity Dynamic Reconstruction and Rendering** [CVPR 2023]  
[[code]](https://github.com/facebookresearch/hyperreel) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Attal_HyperReel_High-Fidelity_6-DoF_Video_With_Ray-Conditioned_Sampling_CVPR_2023_paper.pdf) [[project]](https://hyperreel.github.io/) **HyperReel: High-Fidelity 6-DoF Video with Ray-Conditioned Samplings** [CVPR 2023]  
[[code]](https://github.com/Caoang327/HexPlane) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_HexPlane_A_Fast_Representation_for_Dynamic_Scenes_CVPR_2023_paper.pdf) **HexPlane: A Fast Representation for Dynamic Scenes** [CVPR 2023]  
[[code]](https://github.com/facebookresearch/robust-dynrf) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Robust_Dynamic_Radiance_Fields_CVPR_2023_paper.pdf) [[project]](https://robust-dynrf.github.io/) **Robust Dynamic Radiance Fields** [CVPR 2023]  
[[code]](https://github.com/YilingQiao/DMRF) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Qiao_Dynamic_Mesh-Aware_Radiance_Fields_ICCV_2023_paper.pdf) [[project]](https://mesh-aware-rf.github.io/) **Dynamic Mesh-Aware Radiance Fields** [ICCV 2023]  
[[code]](https://github.com/fengres/mixvoxels) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Mixed_Neural_Voxels_for_Fast_Multi-view_Video_Synthesis_ICCV_2023_paper.pdf) [[project]](https://fengres.github.io/mixvoxels/) **MixVoxels: Mixed Neural Voxels for Fast Multi-view Video Synthesis** [ICCV 2023]  
[[code]](https://github.com/kaichen-z/DynPoint) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/dbdc7a9779ce0278c6e43b62c7e97759-Paper-Conference.pdf) [[project]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dbdc7a9779ce0278c6e43b62c7e97759-Abstract-Conference.html) **DynPoint: Dynamic Neural Point For View Synthesis** [NeurIPS 2023]  
[[code]](https://github.com/ChikaYan/d2nerf) [[paper]](https://papers.nips.cc/paper_files/paper/2022/file/d2cc447db9e56c13b993c11b45956281-Paper-Conference.pdf) [[project]](https://papers.nips.cc/paper_files/paper/2022/hash/d2cc447db9e56c13b993c11b45956281-Abstract-Conference.html) **D2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video** [NeurIPS 2022]  
[[code]](https://github.com/google/nerfies) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Park_Nerfies_Deformable_Neural_Radiance_Fields_ICCV_2021_paper.pdf) [[project]](https://nerfies.github.io/) **Nerfies: Deformable Neural Radiance Fields** [ICCV 2021]  
[[code]](https://github.com/google/hypernerf) [[paper]](https://arxiv.org/pdf/2106.13228) [[project]](https://hypernerf.github.io/) **HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields** [SIGGRAPH 2021]  
[[code]](https://github.com/facebookresearch/nonrigid_nerf) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Tretschk_Non-Rigid_Neural_Radiance_Fields_Reconstruction_and_Novel_View_Synthesis_of_ICCV_2021_paper.pdf) [[project]](https://vcai.mpi-inf.mpg.de/projects/nonrigid_nerf/) **Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Deforming Scene from Monocular Video** [ICCV 2021]  
[[code]](https://github.com/MightyChaos/fsdnerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Flow_Supervision_for_Deformable_NeRF_CVPR_2023_paper.pdf) [[project]](https://mightychaos.github.io/projects/fsdnerf/) **Flow Supervision for Deformable NeRF** [CVPR 2023]  
[[code]](https://github.com/xingyi-li/s-dyrf/tree/main) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_S-DyRF_Reference-Based_Stylized_Radiance_Fields_for_Dynamic_Scenes_CVPR_2024_paper.pdf) [[project]](https://xingyi-li.github.io/s-dyrf/
) **S-DyRF: Reference-Based Stylized Radiance Fields for Dynamic Scenes** [CVPR 2024]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_DS-NeRV_Implicit_Neural_Video_Representation_with_Decomposed_Static_and_Dynamic_CVPR_2024_paper.pdf) [[project]](https://haoyan14.github.io/DS-NeRV/) **DS-NeRV: Implicit Neural Video Representation with Decomposed Static and Dynamic Codes** [CVPR 2024]  
[[code]](https://github.com/FYTalon/pienerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Feng_PIE-NeRF_Physics-based_Interactive_Elastodynamics_with_NeRF_CVPR_2024_paper.pdf) [[project]](https://fytalon.github.io/pienerf/?ref=aiartweekly) **PIE-NeRF: Physics-based Interactive Elastodynamics with NeRF** [CVPR 2024]  
[[code]](https://github.com/huiqiang-sun/DyBluRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kumar_DynaMoDe-NeRF_Motion-aware_Deblurring_Neural_Radiance_Field_for_Dynamic_Scenes_CVPR_2025_paper.pdf) [[project]](https://huiqiang-sun.github.io/dyblurf/) **Motion-aware Deblurring Neural Radiance Field for Dynamic Scenes** [ICCV 2021]  
[[code]](https://github.com/albertpumarola/D-NeRF) [[paper]](https://arxiv.org/abs/2011.13961) [[project]](https://www.albertpumarola.com/research/D-NeRF/index.html) **D-NeRF: Neural Radiance Fields for Dynamic Scenes** [CVPR 2021]  
[[code]](https://github.com/zhengqili/Neural-Scene-Flow-Fields) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Neural_Scene_Flow_Fields_for_Space-Time_View_Synthesis_of_Dynamic_CVPR_2021_paper.pdf) [[project]](https://www.cs.cornell.edu/~zl548/NSFF/) **Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes** [CVPR 2021]  
[[code]](https://github.com/jefftan969/dasr) [[paper]](https://jefftan969.github.io/dasr/paper.pdf) [[project]](https://jefftan969.github.io/dasr/) **Distilling Neural Fields for Real-Time Articulated Shape Reconstruction** [CVPR 2023]  
[[code]](https://github.com/huiqiang-sun/DyBluRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_DyBluRF_Dynamic_Neural_Radiance_Fields_from_Blurry_Monocular_Video_CVPR_2024_paper.pdf) [[project]](https://kaist-viclab.github.io/dyblurf-site/) **Dynamic Deblurring Neural Radiance Fields for Blurry Monocular Video** [CVPR 2024]  
[[code]](https://github.com/merlresearch/Gear-NeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Gear-NeRF_Free-Viewpoint_Rendering_and_Tracking_with_Motion-aware_Spatio-Temporal_Sampling_CVPR_2024_paper.pdf) [[project]](https://merl.com/research/highlights/gear-nerf) **Gear-NeRF: Free-Viewpoint Rendering and Tracking with Motion-aware Spatio-Temporal Sampling** [CVPR 2024]  
[[code]](https://github.com/gafniguy/4D-Facial-Avatars) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Gafni_Dynamic_Neural_Radiance_Fields_for_Monocular_4D_Facial_Avatar_Reconstruction_CVPR_2021_paper.pdf) [[project]](https://gafniguy.github.io/4D-Facial-Avatars/) **Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction** [CVPR 2021]  
[[code]](https://github.com/nogu-atsu/NARF) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Noguchi_Neural_Articulated_Radiance_Field_ICCV_2021_paper.pdf) **Neural Articulated Radiance Field** [ICCV 2021]  
[[code]](https://github.com/JanaldoChen/Anim-NeRF?tab=readme-ov-file) [[paper]](https://arxiv.org/abs/2106.13629) **Animatable Neural Radiance Fields from Monocular RGB Videos** [arXiv 2021]  
[[code]](https://github.com/googleinterns/IBRNet/tree/master) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_IBRNet_Learning_Multi-View_Image-Based_Rendering_CVPR_2021_paper.pdf) [[project]](https://ibrnet.github.io/) **IBRNet: Learning Multi-View Image-Based Rendering** [CVPR 2021]  
[[code]](https://github.com/lingjie0206/Neural_Actor_Main_Code) [[paper]](https://dl.acm.org/doi/epdf/10.1145/3478513.3480528) [[project]](https://vcai.mpi-inf.mpg.de/projects/NeuralActor/) **Neural Actor: Neural Free-view Synthesis of Human Actors with Pose Control** [SIGGRAPH Asia 2021]  
[[code]](https://github.com/hustvl/TiNeuVox) [[paper]](https://arxiv.org/abs/2205.15285) [[project]](https://jaminfong.cn/tineuvox/) **TiNeuVox: Fast Dynamic Radiance Fields with Time-Aware Neural Voxels** [SIGGRAPH Asia 2021]  
[[code]](https://yifanjiang19.github.io/alignerf/demo/compare.html) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_AligNeRF_High-Fidelity_Neural_Radiance_Fields_via_Alignment-Aware_Training_CVPR_2023_paper.pdf) [[project]](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_AligNeRF_High-Fidelity_Neural_Radiance_Fields_via_Alignment-Aware_Training_CVPR_2023_paper.pdf) **AligNeRF: High-Fidelity Neural Radiance Fields via Alignment-Aware Training** [CVPR 2023]  
[[code]](https://github.com/NVlabs/BundleSDF) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.pdf) [[project]](https://bundlesdf.github.io/) **BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown** [CVPR 2023]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Xian_Space-Time_Neural_Irradiance_Fields_for_Free-Viewpoint_Video_CVPR_2021_paper.pdf) [[project]](https://video-nerf.github.io/) **Space-time Neural Irradiance Fields for Free-Viewpoint Video** [CVPR 2021]  
[[code]](https://github.com/yilundu/nerflow) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Du_Neural_Radiance_Flow_for_4D_View_Synthesis_and_Video_Processing_ICCV_2021_paper.pdf) **Neural Radiance Flow for 4D View Synthesis and Video Processing** [ICCV 2021]  
[[code]](https://github.com/AlgoHunt/StreamRF) [[paper]](https://papers.nips.cc/paper_files/paper/2022/file/57c2cc952f388f6185db98f441351c96-Paper-Conference.pdf) [[project]](https://papers.nips.cc/paper_files/paper/2022/hash/57c2cc952f388f6185db98f441351c96-Abstract-Conference.html) **Streaming Radiance Fields for 3D Video Synthesis** [NeurIPS 2022]  
[[code]](https://github.com/gaochen315/DynamicNeRF) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Dynamic_View_Synthesis_From_Dynamic_Monocular_Video_ICCV_2021_paper.pdf) [[project]](https://free-view-video.github.io/) **Dynamic View Synthesis from Dynamic Monocular Video** [ICCV 2021]  
[[code]](https://github.com/fengres/mixvoxels) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Neural_3D_Video_Synthesis_From_Multi-View_Video_CVPR_2022_paper.pdf) [[project]](https://fengres.github.io/mixvoxels/) **Mixed Neural Voxels for Fast Multi-view Video Synthesis** [CVPR 2022]  
[[code]](https://github.com/zju3dv/neuralbody) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Peng_Neural_Body_Implicit_Neural_Representations_With_Structured_Latent_Codes_for_CVPR_2021_paper.pdf) [[project]](https://zju3dv.github.io/neuralbody/) **Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans** [CVPR 2021]  




#### Large-Scale & Unbounded Scenes

[[code]](https://github.com/google-research/multinerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Barron_Mip-NeRF_360_Unbounded_Anti-Aliased_Neural_Radiance_Fields_CVPR_2022_paper.html) [[project]](https://jonbarron.info/mipnerf360/) **Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields** [CVPR 2022]  
[[paper]](https://github.com/thucz/PanoGRF) [[paper]](https://3dvar.com/Chen2023PanoGRF.pdf) [[project]](https://thucz.github.io/PanoGRF/) **PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline Panoramas** [NeurIPS 2023]  
[[code]](https://github.com/wolfball/PlenVDB) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_PlenVDB_Memory_Efficient_VDB-Based_Radiance_Fields_for_Fast_Training_and_CVPR_2023_paper.pdf) [[project]](https://plenvdb.github.io/) **PlenVDB: Memory Efficient VDB-Based Radiance Fields for Fast Training and Rendering** [CVPR 2023]  
[[SLIDES]](https://cvpr.thecvf.com/media/cvpr-2024/Slides/30207_zHbzPLg.pdf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Grounding_and_Enhancing_Grid-based_Models_for_Neural_Fields_CVPR_2024_paper.pdf) [[project]](https://sites.google.com/view/cvpr24-2034-submission/home) **Grounding and Enhancing Grid-Based Models for Neural Fields** [CVPR 2024]  
[[code]](https://github.com/autonomousvision/murf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_MuRF_Multi-Baseline_Radiance_Fields_CVPR_2024_paper.pdf) [[project]](https://haofeixu.github.io/murf/) **MuRF: Multi-Baseline Radiance Fields** [CVPR 2024]  
[[code]](https://github.com/showlab/DynVideo-E) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_DynVideo-E_Harnessing_Dynamic_NeRF_for_Large-Scale_Motion-_and_View-Change_Human-Centric_CVPR_2024_paper.pdf) **DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing** [CVPR 2024]  
[[code]](https://github.com/google/hypernerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Turki_HybridNeRF_Efficient_Neural_Rendering_via_Adaptive_Volumetric_Surfaces_CVPR_2024_paper.pdf) [[project]](https://haithemturki.com/hybrid-nerf/) **HybridNeRF: Efficient Neural Rendering via Adaptive Volumetric Surfaces** [CVPR 2024]  
[[code]](https://github.com/chobao/) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Bao_Free360_Layered_Gaussian_Splatting_for_Unbounded_360-Degree_View_Synthesis_from_CVPR_2025_paper.pdf) [[project]](https://zju3dv.github.io/free360/) **Free360: Layered Gaussian Splatting for Unbounded 360-Degree View Synthesis from Extremely Sparse and Unposed Views** [CVPR 2025]  
[[code]](https://github.com/zyqz97/Aerial_lifting) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Aerial_Lifting_Neural_Urban_Semantic_and_Building_Instance_Lifting_from_CVPR_2024_paper.pdf) [[project]](https://zyqz97.github.io/Aerial_Lifting/) **Aerial Lifting: Neural Urban Semantic and Building Instance Lifting from Aerial Imagery** [CVPR 2024]  
[[code]](https://github.com/sunset1995/DirectVoxGO) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Direct_Voxel_Grid_Optimization_Super-Fast_Convergence_for_Radiance_Fields_Reconstruction_CVPR_2022_paper.pdf) [[project]](https://sunset1995.github.io/dvgo/) **Direct Voxel Grid Optimization: Super-Fast Convergence for Radiance Fields Reconstruction** [CVPR 2022]  
[[code]](https://github.com/wuminye/TeTriRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_TeTriRF_Temporal_Tri-Plane_Radiance_Fields_for_Efficient_Free-Viewpoint_Video_CVPR_2024_paper.pdf) [[project]](https://wuminye.github.io/projects/TeTriRF/) **TeTriRF: Temporal Tri-Plane Radiance Fields for Efficient Free-Viewpoint Video** [CVPR 2023]  
[[code]](https://github.com/facebookresearch/NSVF) [[paper]](https://arxiv.org/pdf/2007.11571) [[project]](https://lingjie0206.github.io/papers/NSVF/) **NSVF: Neural Sparse Voxel Fields** [NeurIPS 2020]  
[[code]](https://github.com/creiser/kilonerf) [[paper]](https://thomasneff.github.io/adanerf/adanerf_supplementary.pdf) [[project]](https://creiser.github.io/kilonerf/) **KiloNeRF: Speeding Up Neural Radiance Fields with Thousands of Tiny MLPs** [ICCV 2021]  
[[code]](https://github.com/simicvm/snerg) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Hedman_Baking_Neural_Radiance_Fields_for_Real-Time_View_Synthesis_ICCV_2021_paper.pdf) **SNeRG: Baking Neural Radiance Fields for Real-Time View Synthesis** [SIGGRAPH 2021]  
[[paper]](https://github.com/JiahuiLei/MoSca) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lei_MoSca_Dynamic_Gaussian_Fusion_from_Casual_Videos_via_4D_Motion_CVPR_2025_paper.pdf) [[project]](https://jiahuilei.com/projects/mosca/) **MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds** [CVPR 2024]  
[[code]](https://github.com/MiZhenxing/Switch-NeRF) [[paper]](https://openreview.net/forum?id=PQ2zoIZqvm) [[project]](https://mizhenxing.github.io/switchnerf/) **Switch-NeRF: Learning Scene Decomposition with Mixture of Experts for Large-scale Neural Radiance Fields** [ICLR 2023]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Tancik_Block-NeRF_Scalable_Large_Scene_Neural_View_Synthesis_CVPR_2022_paper.pdf) [[project]](https://waymo.com/research/block-nerf/) **Block-NeRF: Scalable Large Scene Neural View Synthesis** [CVPR 2022]  
[[code]](https://github.com/sail-sg/InfNeRF) [[paper]](https://arxiv.org/abs/2403.14376) [[project]](https://jiabinliang.github.io/InfNeRF.io/) **Inf-NeRF: Towards Infinite Scale NeRF Rendering with O(log n) Space Complexity** [SIGGRAPH Asia 2024]  
[[code]](https://github.com/cmusatyalab/mega-nerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Turki_Mega-NERF_Scalable_Construction_of_Large-Scale_NeRFs_for_Virtual_Fly-Throughs_CVPR_2022_paper.pdf) [[project]](https://meganerf.cmusatyalab.org/) **Mega-NeRF: Scalable Radiance Fields for Large Outdoor Scenes** [CVPR 2023]  
[[code]](https://github.com/Kai-46/nerfplusplus) [[paper]](https://arxiv.org/abs/2010.07492) **NeRF++: Analyzing and Improving Neural Radiance Fields for Unbounded 3D Scenes** [CVPR 2021]  
[[code]](https://github.com/city-super/BungeeNeRF) [[paper]](https://city-super.github.io/citynerf/img/1947.pdf) [[project]](https://city-super.github.io/citynerf/ **CityNeRF: Building NeRF at City Scale** [ECCV 2021]  





#### Sparse Inputs & Generalization

[[code]](https://github.com/barbararoessle/dense_depth_priors_nerf) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Roessle_Dense_Depth_Priors_for_Neural_Radiance_Fields_From_Sparse_Input_CVPR_2022_paper.pdf) [[project]](https://barbararoessle.github.io/dense_depth_priors_nerf/) **Dense Depth Priors for Neural Radiance Fields from Sparse Input Views** [CVPR 2022]  
[[code]](https://github.com/xxlong0/SparseNeuS) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920210.pdf) [[project]](https://www.xxlong.site/SparseNeuS/) **SparseNeuS: Fast Generalizable Neural Surface Reconstruction from Sparse Views** [ECCV 2022]  
[[code]](https://github.com/mjmjeong/InfoNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_InfoNeRF_Ray_Entropy_Minimization_for_Few-Shot_Neural_Volume_Rendering_CVPR_2022_paper.pdf) [[project]](https://cv.snu.ac.kr/research/InfoNeRF/) **InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering** [CVPR 2022]  
[[code]](https://github.com/linjohnss/FrugalNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_FrugalNeRF_Fast_Convergence_for_Extreme_Few-shot_Novel_View_Synthesis_without_CVPR_2025_paper.pdf) [[project]](https://linjohnss.github.io/frugalnerf/) **FrugalNeRF: Fast Convergence for Extreme Few-shot Novel View Synthesis** [CVPR 2025]  
[[code]](https://github.com/ajayjain/DietNeRF) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Jain_Putting_NeRF_on_a_Diet_Semantically_Consistent_Few-Shot_View_Synthesis_ICCV_2021_paper.pdf) [[project]](https://www.ajayj.com/dietnerf) **Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis** [ICCV 2021]  
[[code]](https://github.com/vincentfung13/MINE) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_MINE_Towards_Continuous_Depth_MPI_With_NeRF_for_Novel_View_ICCV_2021_paper.pdf) [[project]](https://vincentfung13.github.io/projects/mine/) **MINE: Towards Continuous Depth MPI with NeRF for Novel View Synthesis** [ICCV 2021]  
[[code]](https://github.com/NIRVANALAN/LN3Diff) [[paper]](https://arxiv.org/pdf/2403.12019) [[project]](https://nirvanalan.github.io/projects/ln3diff/) **Ln3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation** [ECCV 2024]  
[[code]](https://github.com/simicvm/snerg) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Single-Stage_Diffusion_NeRF_A_Unified_Approach_to_3D_Generation_and_ICCV_2023_paper.pdf) [[project]](https://hanshengchen.com/ssdnerf/) **Single-stage Diffusion NeRF: A Unified Approach to 3D Generation and Reconstruction** [ICCV 2023]  
[[paper]](https://github.com/JiahuiLei/MoSca) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Chou_GSNeRF_Generalizable_Semantic_Neural_Radiance_Fields_with_Enhanced_3D_Scene_CVPR_2024_paper.pdf) [[project]](https://timchou-ntu.github.io/gsnerf/) **GSNeRF: Generalizable Semantic Neural Radiance Fields with Enhanced 3D Scene Understanding** [CVPR 2024]  
[[code]](https://github.com/stelzner/srt) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sajjadi_Scene_Representation_Transformer_Geometry-Free_Novel_View_Synthesis_Through_Set-Latent_Scene_CVPR_2022_paper.pdf) [[project]](https://srt-paper.github.io/) **Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representation** [CVPR 2022]  
[[code]](https://github.com/autonomousvision/graf) [[paper]](https://proceedings.neurips.cc/paper/2020/file/e92e1b476bb5262d793fd40931e0ed53-Paper.pdf) **GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis** [NeurIPS 2020]  
[[code]](https://github.com/autonomousvision/giraffe) [[paper]](https://www.cvlibs.net/publications/Niemeyer2021CVPR.pdf) [[project]](https://m-niemeyer.github.io/project-pages/giraffe/index.html) **GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields** [CVPR 2021]  
[[code]](https://github.com/marcoamonteiro/pi-GAN) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chan_Pi-GAN_Periodic_Implicit_Generative_Adversarial_Networks_for_3D-Aware_Image_Synthesis_CVPR_2021_paper.pdf) [[project]](https://marcoamonteiro.github.io/pi-GAN-website/) **pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis** [CVPR 2021]  
[[code]](https://github.com/Kitsunetic/SDF-Diffusion) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shim_Diffusion-Based_Signed_Distance_Fields_for_3D_Shape_Generation_CVPR_2023_paper.pdf) [[project]](https://kitsunetic.github.io/sdf-diffusion/) **Diffusion-Based Signed Distance Fields for 3D Shape Generation** [CVPR 2023]  
[[code]](https://github.com/leejielong/DiSR-NeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_DiSR-NeRF_Diffusion-Guided_View-Consistent_Super-Resolution_NeRF_CVPR_2024_paper.pdf) **DiSR-NeRF: Diffusion-Guided View-Consistent Super-Resolution NeRF** [CVPR 2024]  
[[code]](https://github.com/ActiveVisionLab/nope-nerf/tree/main) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Bian_NoPe-NeRF_Optimising_Neural_Radiance_Field_With_No_Pose_Prior_CVPR_2023_paper.pdf) [[project]](https://nope-nerf.active.vision/) **NoPe-NeRF: Optimising neural radiance field with no pose prior** [CVPR 2023]  
[[code]](https://github.com/baskargroup/SC-NeRF) [[paper]](https://arxiv.org/abs/2503.21958) [[project]](https://baskargroup.github.io/SC-NeRF/) **SC-NeRF: NeRF-based Point Cloud Reconstruction using a Stationary Camera for Agricultural Applications** [CVPR 2025]  
[[code]](https://github.com/munshisanowarraihan/nerf-meta) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Tancik_Learned_Initializations_for_Optimizing_Coordinate-Based_Neural_Representations_CVPR_2021_paper.pdf) [[project]](https://www.matthewtancik.com/learnit) **Learned Initializations for Optimizing Coordinate-Based Neural Representations (MetaNeRF)** [CVPR 2021]  
[[code]](https://github.com/apchenstu/mvsnerf) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_MVSNeRF_Fast_Generalizable_Radiance_Field_Reconstruction_From_Multi-View_Stereo_ICCV_2021_paper.pdf) [[project]](https://apchenstu.github.io/mvsnerf/) **MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo** [ICCV 2021]  
[[code]](https://github.com/jchibane/srf) [[paper]](https://ieeexplore.ieee.org/document/9578451) [[project]](https://virtualhumans.mpi-inf.mpg.de/srf/) **Stereo Radiance Fields (SRF): Learning View Synthesis from Sparse Views of Novel Scenes** [CVPR 2021]  
[[code]](https://github.com/ajayjain/dietnerf) [[paper]](https://arxiv.org/abs/2104.00677) [[project]](https://ajayj.com/dietnerf) **DietNeRF: Reducing Spatial Bias in NeRF for Unsupervised Material Transfer** [arXiv 2021]  
[[code]](https://github.com/facebookresearch/StyleNeRF) [[paper]](https://jiataogu.me/style_nerf/files/StyleNeRF.pdf) [[project]](https://jiataogu.me/style_nerf/) **StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis** [ICLR 2022]  
[[code]](https://github.com/SheldonTsui/GOF_NeurIPS2021) [[paper]](https://arxiv.org/abs/2111.00969) [[project]](https://sheldontsui.github.io/projects/GOF) **Generative Occupancy Fields for 3D Surface-Aware Image Synthesis** [NeurIPS 2021]  
[[code]](https://github.com/VITA-Group/SinNeRF) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820712.pdf) [[project]](https://vita-group.github.io/SinNeRF/) **SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single Imag** [ECCV 2022]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_RODIN_A_Generative_Model_for_Sculpting_3D_Digital_Avatars_Using_CVPR_2023_paper.pdf)) [[project]](https://3d-avatar-diffusion.microsoft.com/) **Rodin: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion** [CVPR 2023]  
[[code]](https://github.com/zubair-irshad/NeO-360) [[paper]](https://arxiv.org/pdf/2308.12967) [[project]](https://zubair-irshad.github.io/projects/neo360.html) **NeO360: Neural Fields for Sparse View Synthesis of Outdoor Scenes** [ICCV 2023]  
[[code]](https://github.com/Youngju-Na/UFORecon) [[paper]](https://arxiv.org/abs/2403.05086) [[project]](https://youngju-na.github.io/uforecon.github.io/) **UFORecon: Generalizable Sparse-View Surface Reconstruction from Arbitrary and Unfavorable Set** [CVPR 2024]   
[[code]](https://github.com/ruili3/Know-Your-Neighbors) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Know_Your_Neighbors_Improving_Single-View_Reconstruction_via_Spatial_Vision-Language_Reasoning_CVPR_2024_paper.pdf) [[project]](https://ruili3.github.io/kyn/) **Know Your Neighbors: Improving Single-View Reconstruction via Spatial Vision-Language Reasoning** [CVPR 2024]




#### Generative Models & Editing

[[code]](https://github.com/autonomousvision/giraffe) [[paper]](https://www.cvlibs.net/publications/Niemeyer2021CVPR.pdf) [[project]](https://m-niemeyer.github.io/project-pages/giraffe/index.html) **GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields** [CVPR 2021]  
[[code]](https://github.com/marcoamonteiro/pi-GAN) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chan_Pi-GAN_Periodic_Implicit_Generative_Adversarial_Networks_for_3D-Aware_Image_Synthesis_CVPR_2021_paper.pdf) [[project]](https://marcoamonteiro.github.io/pi-GAN-website/) **pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis** [CVPR 2021]  
[[code]](https://github.com/Kitsunetic/SDF-Diffusion.git) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Shim_Diffusion-Based_Signed_Distance_Fields_for_3D_Shape_Generation_CVPR_2023_paper.html) **Diffusion-Based Signed Distance Fields for 3D Shape Generation** [CVPR 2023]  
[[code]](https://github.com/DP-Recon/DP-Recon) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Ni_Decompositional_Neural_Scene_Reconstruction_with_Generative_Diffusion_Prior_CVPR_2025_paper.pdf) [[project]](https://dp-recon.github.io/) **Decompositional Neural Scene Reconstruction with Generative Diffusion Prior** [CVPR 2025]  
[[code]](https://github.com/stevliu/editnerf) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Editing_Conditional_Radiance_Fields_ICCV_2021_paper.pdf) [[project]](http://editnerf.csail.mit.edu/) **Editing Conditional Radiance Fields** [ICCV 2021]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kaneko_AR-NeRF_Unsupervised_Learning_of_Depth_and_Defocus_Effects_From_Natural_CVPR_2022_paper.pdf) [[project]](https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/ar-nerf/) **AR-NeRF: Unsupervised Learning of Depth and Defocus Effects from Natural Images** [CVPR 2022]  
[[code]](https://github.com/r4dl/LAENeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Radl_LAENeRF_Local_Appearance_Editing_for_Neural_Radiance_Fields_CVPR_2024_paper.pdf) [[project]](https://r4dl.github.io/LAENeRF/) **LAENeRF: Local Appearance Editing for Neural Radiance Fields** [CVPR 2024]  
[[code]](https://github.com/SamsungLabs/SPIn-NeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Mirzaei_SPIn-NeRF_Multiview_Segmentation_and_Perceptual_Inpainting_With_Neural_Radiance_Fields_CVPR_2023_paper.pdf) [[project]](https://spinnerf3d.github.io/) **SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting with Neural Radiance Fields** [CVPR 2023]   
[[code]](https://github.com/zju3dv/SINE) [[paper]](http://www.cad.zju.edu.cn/home/gfzhang/papers/sine/sine.pdf) [[project]](https://zju3dv.github.io/sine/) **SINE: Semantic-driven image-based NeRF editing with prior-guided editing field** [CVPR 2023]  
[[code]](https://github.com/nerfdeformer/nerfdeformer) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_NeRFDeformer_NeRF_Transformation_from_a_Single_View_via_3D_Scene_CVPR_2024_paper.pdf) **NeRFDeformer: NeRF Transformation from a Single View via 3D Scene Flows** [CVPR 2024]  
[[code]](https://github.com/kerrj/lerf) [[paper]](https://arxiv.org/abs/2303.09553) [[project]](https://www.lerf.io/) **LERF: Language Embedded Radiance Fields** [ICCV 2023]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Mazzucchelli_IReNe_Instant_Recoloring_of_Neural_Radiance_Fields_CVPR_2024_paper.pdf) [[project]](https://iviazz97.github.io/irene/) **IReNe: Instant Recoloring of Neural Radiance Fields** [CVPR 2024]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Kaneko_Improving_Physics-Augmented_Continuum_Neural_Radiance_Field-Based_Geometry-Agnostic_System_Identification_with_CVPR_2024_paper.pdf) [[project]](https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/lpo/) **Improving Physics-Augmented Continuum Neural Radiance Field-Based Geometry-Agnostic System Identification with Lagrangian Particle Optimization** [CVPR 2024]  
[[code]](https://github.com/cgtuebingen/SIGNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Dihlmann_SIGNeRF_Scene_Integrated_Generation_for_Neural_Radiance_Fields_CVPR_2024_paper.pdf) [[project]](https://signerf.jdihlmann.com/) **SIGNeRF: Scene Integrated Generation for Neural Radiance Fields** [CVPR 2024]  
[[code]](https://github.com/JasonLSC/NeRFCodec_public) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_NeRFCodec_Neural_Feature_Compression_Meets_Neural_Radiance_Fields_for_Memory-Efficient_CVPR_2024_paper.pdf) **NeRFCodec: Neural Feature Compression Meets Neural Radiance Fields for Memory-Efficient Scene Representation** [CVPR 2024]  
[[code]](https://github.com/cnhaox/NeRF-HuGS) [[paper]](https://arxiv.org/pdf/2403.17537) [[project]](https://cnhaox.com/NeRF-HuGS/) **NeRF-HuGS: Improved Neural Radiance Fields in Non-static Scenes Using Heuristics-Guided Segmentation** [CVPR 2024]  
[[code]](https://github.com/kwanyun/FFaceNeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yun_FFaceNeRF_Few-shot_Face_Editing_in_Neural_Radiance_Fields_CVPR_2025_paper.pdf) [[project]](https://kwanyun.github.io/FFaceNeRF_page/) **FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields** [CVPR 2025]  
[[code]](https://github.com/graphdeco-inria/nerfshop) [[paper]](http://www-sop.inria.fr/reves/Basilic/2023/JKKDLD23/nerfshop.pdf) [[project]](https://repo-sam.inria.fr/fungraph/nerfshop/) **NeRFshop: Interactive Editing of Neural Radiance Fields** [I3D 2023]  
[[code]](https://github.com/jhq1234/ED-NeRF) [[paper]](https://arxiv.org/pdf/2310.02712) [[project]](https://jhq1234.github.io/ed-nerf.github.io/) **ED-NeRF: Efficient Text-Guided Editing of 3D Scene With Latent Space NeRF** [ICLR 2024]   
[[paper]](https://arxiv.org/abs/2402.08622) [[project]](https://mfischer-ucl.github.io/nerf_analogies/) **NeRF Analogies: Example-Based Visual Attribute Transfer for NeRFs,** [CVPR 2024]  
[[code]](https://github.com/ethanweber/nerfiller) [[paper]](https://arxiv.org/abs/2312.04560) [[project]](https://ethanweber.me/nerfiller/) **NeRFiller: Completing Scenes via Generative 3D Inpainting** [CVPR 2024]  
[[code]](https://github.com/kcshum/pose-conditioned-NeRF-object-fusion) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Shum_Language-driven_Object_Fusion_into_Neural_Radiance_Fields_with_Pose-Conditioned_Dataset_CVPR_2024_paper.pdf) **Language-driven Object Fusion into Neural Radiance Fields with Pose-Conditioned Dataset Updates** [CVPR 2024]   
[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_DynVideo-E_Harnessing_Dynamic_NeRF_for_Large-Scale_Motion-_and_View-Change_Human-Centric_CVPR_2024_paper.pdf) [[project]](https://zju3dv.github.io/sine/) **DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing** [CVPR 2024]  
[[code]](https://github.com/sinoyou/nelf-pro) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/You_NeLF-Pro_Neural_Light_Field_Probes_for_Multi-Scale_Novel_View_Synthesis_CVPR_2024_paper.pdf) [[project]](https://sinoyou.github.io/nelf-pro/) **NeLF-Pro: Neural Light Field Probes for Multi-Scale Novel View Synthesis** [CVPR 2024]  
[[code]](https://github.com/autonomousvision/graf) [[paper]](https://proceedings.neurips.cc/paper/2020/file/e92e1b476bb5262d793fd40931e0ed53-Paper.pdf) **GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis** [NeurIPS 2020]  
[[code]](https://github.com/simicvm/snerg) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Single-Stage_Diffusion_NeRF_A_Unified_Approach_to_3D_Generation_and_ICCV_2023_paper.pdf) [[project]](https://hanshengchen.com/ssdnerf/) **Single-stage Diffusion NeRF: A Unified Approach to 3D Generation and Reconstruction** [ICCV 2023]  
[[code]](https://github.com/ayaanzhaque/instruct-nerf2nerf) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Haque_Instruct-NeRF2NeRF_Editing_3D_Scenes_with_Instructions_ICCV_2023_paper.pdf) [[project]](https://instruct-nerf2nerf.github.io/) **Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions** [ICCV 2023]  
[[code]](h[ttps://github.com/ethanweber/nerfiller](https://github.com/hbai98/Componerf)) [[paper]](https://vlislab22.github.io/componerf/static/videos/componerf.pdf) [[project]](https://vlislab22.github.io/componerf/) **CompoNeRF: Text-Guided Multi-object Compositional NeRF with Editable 3D Scene Layout** [ICCV 2023]  
[[paper]](https://openreview.net/forum?id=8HwI6UavYc) [[project]](https://replaceanything3d.github.io/) **ReplaceAnything3D: Text-Guided Object Replacement in 3D Scenes** [NeurIPS 2024]   
[[code]](https://github.com/BillyXYB/FaceDNeRF) [[paper]](https://arxiv.org/abs/2306.00783) **FaceDNeRF: Semantics-Driven Face Reconstruction with Diffusion Priors** [NeurIPS 2023]  
[[code]](https://github.com/Dongjiahua/VICA-NeRF) [[paper]](https://openreview.net/pdf?id=Pk49a9snPe) [[project]](https://dongjiahua.github.io/VICA-NeRF/) **ViCA-NeRF: View-Consistency-Aware 3D Editing of Neural Radiance Fields** [Neurips 2023]  








### 3DGS

[[code]](https://github.com/graphdeco-inria/gaussian-splatting) [[paper]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) [[project]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) **3D Gaussian Splatting for Real-Time Radiance Field Rendering** [TOG 2023]  

#### Acceleration - Optimization Algorithms, Hardware Acceleration, Adaptive Rendering

><span style="color:lightblue;">💡💡 Optimization Algorithms: Progressive frequency and sampling dominate early 2025.</span>  
><span style="color:lightblue;">💡💡 Hardware Acceleration: Mid-year shifts to GPU-specific rasterization.</span>  
><span style="color:lightblue;">💡💡 Adaptive Rendering: Late 2025 emphasizes mobile and real-time adjustments.</span>  
><span style="color:lightblue;">💡💡 Challenges and Projects: Memory bottlenecks addressed through quantization</span>  


[[code]](https://github.com/YouyuChen0207/DashGaussian) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_DashGaussian_Optimizing_3D_Gaussian_Splatting_in_200_Seconds_CVPR_2025_paper.pdf) [[project]](https://dashgaussian.github.io/) **VDashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds** [CVPR 2025]  
[[code]](https://github.com/Sharath-girish/efficientgaussian) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07976.pdf)  **EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS** [ECCV 2024]  
[[code]](https://github.com/Accelsnow/gaussian-splatting-distwar) [[paper]](https://arxiv.org/abs/2401.05345) **DISTWAR atomic reduction optimization on 3D Gaussian Splatting** [arXiv 2025]  
[[code]](https://github.com/r4dl/StopThePop) [[paper]](https://dl.acm.org/doi/10.1145/3658187) [[project]](https://r4dl.github.io/StopThePop/) **StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering** [TOG 2024]  
[[code]](https://github.com/fatPeter/mini-splatting) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09866.pdf) **Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians** [ECCV 2024]  
[[code]](https://github.com/fatPeter/mini-splatting2) [[paper]](https://arxiv.org/pdf/2411.12788) **Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification** [arXiv 2024]  
[[code]](https://github.com/city-super/Octree-GS) [[paper]](https://ieeexplore.ieee.org/document/10993308) [[project]](https://city-super.github.io/octree-gs/) **Octree-GS: Towards Consistent Real-time Rendering** [TPAMI 2025]  
[[code]](https://github.com/cvlab-kaist/PF3plat) [[paper]](https://openreview.net/pdf?id=VjI1NnsW4t) [[project]](https://cvlab-kaist.github.io/PF3plat/) **PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting** [ICML 2025]  
[[code]](https://github.com/SaiJoshika/FreGS-3D-Gaussian-Splatting) [[paper]](https://arxiv.org/abs/2403.06908) [[project]](https://rogeraigc.github.io/FreGS-Page/) **FreGS: 3D Gaussian Splatting with Progressive Frequency Regularizations** [CVPR 2024]  
[[code]](https://github.com/tuallen/speede3dgs) [[paper]](https://arxiv.org/pdf/2506.07917) [[project]](https://speede3dgs.github.io/) **Speedy Deformable 3D Gaussian Splatting: Fast Rendering and Compression of Dynamic Scenes** [arXiv 2025]  
[[code]](https://github.com/hjhyunjinkim/MH-3DGS) [[paper]](https://arxiv.org/abs/2506.12945) [[project]](https://hjhyunjinkim.github.io/MH-3DGS/) **Metropolis-Hastings Sampling for 3D Gaussian Reconstruction** [NeurIPS 2025]  
[[code]](https://github.com/donydchen/mvsplat) [[paper]](https://arxiv.org/abs/2403.14627) [[project]](https://donydchen.github.io/mvsplat/) **MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images** [ECCV 2024]  
[[code]](https://github.com/WeiPhil/nbvh) [[paper]](https://weiphil.s3.eu-central-1.amazonaws.com/neural_bvh.pdf) [[project]](https://weiphil.github.io/portfolio/neural_bvh) **N-BVH: Neural ray queries with bounding volume hierarchies** [ SIGGRAPH 2024]  
[[code]](https://github.com/NVlabs/LongSplat) [[paper]](https://arxiv.org/abs/2508.14041) [[project]](https://linjohnss.github.io/longsplat/) **LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos** [ICCV 2025]  
[[code]](https://github.com/hiroxzwang/adrgaussian) [[paper]](https://dl.acm.org/doi/10.1145/3680528.3687675) [[project]](https://hiroxzwang.github.io/publications/adrgaussian/) **AdR-Gaussian: Adaptive Radius** [SIGGRAPH Asia 2024]  
[[code]](https://github.com/j-alex-hanson/speedy-splatg) [[paper]](https://arxiv.org/pdf/2412.00578) [[project]](https://speedysplat.github.io/) **Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives** [CVPR 2025]  
[[code]](https://github.com/ant-research/PlanarSplatting) [[paper]](https://arxiv.org/abs/2412.03451) [[project]](https://icetttb.github.io/PlanarSplatting/) **PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes** [CVPR 2025]  
[[code]](https://github.com/InternLandMark/FlashGS) [[paper]](https://arxiv.org/pdf/2408.07967) [[project]](https://maxwellf1.github.io/flashgs_page/) **FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-resolution Rendering** [arXiv 2024]  
[[code]](https://github.com/humansensinglab/taming-3dgs) [[paper]](https://humansensinglab.github.io/taming-3dgs/docs/paper_lite.pdf) [[project]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) **Taming 3DGS: High-Quality Radiance Fields with Limited Resources** [SIGGRAPH Asia 2024]  
[[code]](https://drive.google.com/file/d/1J9Hl7KPMOkBmrt6UTFf_u9sQcjbblILA/view) [[paper]](https://zju3dv.github.io/longvolcap/) [[project]](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) **Representing Long Volumetric Video with Temporal Gaussian Hierarchy** [TOG 2024]  
[[code]](https://arxiv.org/pdf/2411.11363) [[paper]](https://arxiv.org/pdf/2411.11363) [[project]](https://yaourtb.github.io/GPS-Gaussian+) **GPS-Gaussian+: Generalizable Pixel-Wise 3D Gaussian Splatting for Real-Time Human-Scene Rendering from Sparse Views** [T-PAMI 2025]  
[[code]](https://github.com/Yuhuoo/EasySplat) [[paper]](https://arxiv.org/abs/2501.01003) **EasySplat: View-Adaptive Learning** [ICME 2025]  
[[code]](https://github.com/VITA-Group/VideoLifter) [[paper]](https://arxiv.org/abs/2501.01949) [[project]](https://videolifter.github.io/) **VideoLifter: Lifting Videos to 3D with Fast Hierarchical Stereo Alignment** [CVPR 2025]  
[[code]](https://github.com/SJTU-MVCLab/SeeLe) [[paper]](https://arxiv.org/abs/2503.05168) [[project]](https://seele-project.netlify.app/) **SEELE: A Unified Acceleration Framework for Real-Time Gaussian Splatting** [arXiv 2023]  
[[code]](https://github.com/fraunhoferhhi/Improving-ADC-3DGS) [[paper]](https://arxiv.org/pdf/2503.14274) **3D Gaussian Splatting for Real-Time Radiance Field Rendering** [ICPR 2025]  
[[paper]](https://arxiv.org/pdf/2412.07293) [[project]](https://openreview.net/forum?id=EJZfcKXdiT) **Event-3DGS: Event-based 3D Reconstruction Using 3D Gaussian Splatting** [NeurIPS 2024]  
[[code]](https://github.com/lukasHoel/3DGS-LM) [[paper]](https://arxiv.org/abs/2409.12892) [[project]](https://lukashoel.github.io/3DGS-LM/) **3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt** [ICCV 2025]  
[[paper]](https://m-niemeyer.github.io/radsplat/static/pdf/niemeyer2024radsplat.pdf) [[project]](https://lukashoel.github.io/3DGS-LM/) **RadSplat: Radiance Field-Informed Gaussian Splatting for Robust Real-Time Rendering with 900+ FPS** [3DV 2025]  
[[paper]](https://arxiv.org/abs/2409.12892) **TC-GS: A Faster and Flexible 3DGS Module Utilizing Tensor Cores** [arXiv 2025]  
[[paper]](https://arxiv.org/pdf/2505.08510) **3DGS$^2$: Near Second-order Converging 3D Gaussian Splatting** [arxiv 2025]  
[[paper]](https://arxiv.org/pdf/2505.08510) **FOCI: Trajectory Optimization on Gaussian Splats** [arxiv 2025]  
[[paper]](https://arxiv.org/pdf/2501.14534) **Trick-GS: A Balanced Bag of Tricks for Efficient Gaussian Splatting** [arxiv 2025]  
[[paper]](https://arxiv.org/pdf/2501.00342) **SG-Splatting: Accelerating 3D Gaussian Splatting with Spherical Gaussians** [arxiv 2025]  
[[paper]](https://arxiv.org/pdf/2412.07608) **Faster and Better 3D Splatting via Group Training** [arxiv 2024]  





#### Quality & Realism Improvement - Geometric Precision, Materials & Lighting, Special Conditions

><span style="color:lightblue;">💡💡 Geometric Precision: Multi-scale alignments.</span>  
><span style="color:lightblue;">💡💡 Materials & Lighting: Residual tone mappers.</span>  
><span style="color:lightblue;">💡💡 Special Conditions: Blur-agnostic and watermarking.</span>  


[[code]](https://github.com/LiuJF1226/GaussHDR) [[paper]](https://arxiv.org/abs/2503.10143) [[project]](https://liujf1226.github.io/GaussHDR/) **GaussHDR: High Dynamic Range Gaussian Splatting** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wan_S2Gaussian_Sparse-View_Super-Resolution_3D_Gaussian_Splatting_CVPR_2025_paper.pdf) [[project]](https://jeasco.github.io/S2Gaussian/) **S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting** [CVPR 2025]  
[[code]](https://github.com/LetianHuang/op43dgs) [[paper]](https://arxiv.org/abs/2402.00752) [[project]](https://letianhuang.github.io/op43dgs/) **On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy** [ECCV 2024]  
[[code]](https://github.com/benhenryL/Deblurring-3D-Gaussian-Splatting) [[paper]](https://arxiv.org/abs/2401.00834) [[project]](https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/) **Deblurring-3D-Gaussian-Splatting** [ECCV 2024]  
[[paper]](https://arxiv.org/abs/2501.14231) **Micro-macro Wavelet-based Gaussian Splatting for 3D Reconstruction from Unconstrained Images** [AAAI 2025]  
[[code]](https://github.com/kuai-lab/cvpr25_3D-GSW) [[paper]](https://arxiv.org/abs/2409.13222) **3D-GSW: 3D Gaussian Splatting for Robust Watermarking** [CVPR 2025]  
[[code]](https://github.com/hbb1/2d-gaussian-splatting) [[paper]](https://arxiv.org/abs/2403.17888) [[project]](https://surfsplatting.github.io/) **2D Gaussian Splatting for Geometrically Accurate Radiance Fields** [SIGGRAPH 2024]  
[[code]](https://github.com/Anttwo/SuGaR) [[paper]](https://arxiv.org/abs/2311.12775) [[project]](https://anttwo.github.io/sugar/) **SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering** [CVPR 2024]  
[[code]](https://github.com/Asparagus15/GaussianShader) [[paper]](https://arxiv.org/abs/2311.17977) [[project]](https://asparagus15.github.io/GaussianShader.github.io/) **GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces** [CVPR 2024]   
[[code]](https://github.com/HanzhiChang/MeshSplat) [[paper]](https://arxiv.org/pdf/2508.17811) [[project]](https://hanzhichang.github.io/meshsplat_web/) **MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splattingt** [arXiv 2025]  
[[code]](https://github.com/garrisonz/LabelGS) [[paper]](https://arxiv.org/pdf/2508.19699) **LabelGS: Label-Aware 3D Gaussian Splatting for 3D Scene Segmentation** [PRCV 2025]  
[[code]](https://github.com/XiaoBin2001/Improved-GS) [[paper]](https://arxiv.org/pdf/2508.12313) [[project]](https://xiaobin2001.github.io/improved-gs-web/) **Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering** [arXiv 2025]  
[[paper]](https://arxiv.org/abs/2509.02141) [[project]](https://mohitm1994.github.io/GRMM/) **GRMM: Real-Time High-Fidelity Gaussian Morphable Head Model with Learned Residuals** [arXiv 2025]  
[[code]](https://github.com/kcheng1021/GaussianPro) [[paper]](https://arxiv.org/abs/2402.14650) [[project]](https://kcheng1021.github.io/gaussianpro.github.io/) **GaussianPro: 3D Gaussian Splatting with Progressive Propagation** [ICML 2024]  
[[code]](https://github.com/NJU-3DV/Relightable3DGaussian) [[paper]](https://arxiv.org/abs/2311.16043) [[project]](https://nju-3dv.github.io/projects/Relightable3DGaussian/) **Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing** [ECCV 2024]  
[[paper]](https://arxiv.org/abs/2403.10814) [[project]](https://lukashoel.github.io/3DGS-LM/) **DarkGS: Learning Neural Illumination and 3D Gaussians Relighting for Robotic Exploration in the Dark** [ICRA 2024]  
[[code]](https://github.com/lzhnb/Analytic-Splatting) [[paper]](https://arxiv.org/abs/2403.11056) [[project]](https://lzhnb.github.io/project-pages/analytic-splatting/) **Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration** [ECCV 2024]  
[[code]](https://github.com/snldmt/BAGS) [[paper]](https://arxiv.org/pdf/2403.04926) [[project]](https://nwang43jhu.github.io/BAGS/) **BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Modeling** [ECCV 2024]  
[[code]](https://github.com/yanyan-li/GeoGaussian) [[paper]](https://arxiv.org/abs/2403.11324) [[project]](https://yanyan-li.github.io/project/gs/geogaussian.html) **GeoGaussian: Geometry-aware Gaussian Splatting for Scene Rendering** [ECCV 2024]  
[[code]](https://github.com/turandai/gaussian_surfels) [[paper]](https://arxiv.org/pdf/2404.17774) [[project]](https://turandai.github.io/projects/gaussian_surfels/) **High-quality Surface Reconstruction using Gaussian Surfels** [ACM SIGGRAPH 2024]  
[[code]](https://github.com/lzhnb/GS-IR) [[paper]](https://arxiv.org/abs/2311.16473) [[project]](https://lzhnb.github.io/project-pages/gs-ir.html) **GS-IR: 3D Gaussian Splatting for Inverse Rendering** [arXiv 2025]  
[[code]](https://github.com/cuiziteng/Luminance-GS) [[paper]](https://arxiv.org/abs/2504.01503v2) [[project]](https://cuiziteng.github.io/Luminance_GS_web/) **Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment** [CVPR 2025]  
[[code]](https://github.com/Jumponthemoon/WeatherGS) [[paper]](https://arxiv.org/pdf/2412.18862) [[project]](https://jumponthemoon.github.io/weather-gs/) **WeatherGS: 3D Scene Reconstruction in Adverse Weather Conditions via Gaussian Splatting** [arXiv 2025]  




#### Dynamic & Deformable Scenes - 4D Representations, Motion Decomposition, Long Videos

><span style="color:lightblue;">💡💡 4D Representations: Flow-guided extensions.</span>  
><span style="color:lightblue;">💡💡 Motion Decomposition: Progressive segmentation.</span>  
><span style="color:lightblue;">💡💡 Long Videos: Robust unposed methods.</span>  


[[code]](https://github.com/JonathonLuiten/Dynamic3DGaussians) [[paper]](https://arxiv.org/pdf/2308.09713) [[project]](https://dynamic3dgaussians.github.io/) **Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis** [3DV 2024]  
[[code]](https://github.com/mikeqzy/3dgs-avatar-release) [[paper]](https://arxiv.org/abs/2312.09228) [[project]](https://neuralbodies.github.io/3DGS-Avatar/index.html) **3DGS-Avatar: Animatable Avatars via Deformable 3D Gaussian Splatting** [CVPR 2024]  
[[code]](https://github.com/longxiang-ai/Human101) [[paper]](https://arxiv.org/pdf/2312.15258) [[project]](https://longxiang-ai.github.io/Human101/) **Human101: Training 100+FPS Human Gaussians in 100s from 1 View** [Arxiv 2023]  
[[code]](https://github.com/ingra14m/Deformable-3D-Gaussians) [[paper]](https://arxiv.org/abs/2309.13101) [[project]](https://ingra14m.github.io/Deformable-Gaussians/) **Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction** [CVPR 2024]  
[[code]](https://github.com/jiawei-ren/dreamgaussian4d) [[paper]](https://arxiv.org/abs/2312.17142) [[project]](https://jiawei-ren.github.io/projects/dreamgaussian4d/) **DreamGaussian4D:Generative 4D Gaussian Splatting** [Arxiv 2023]  
[[code]](https://github.com/VITA-Group/4DGen) [[paper]](https://vita-group.github.io/4DGen/) [[project]](https://vita-group.github.io/4DGen/) **4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency** [Arxiv 2024]  
[[code]](https://github.com/oppo-us-research/SpacetimeGaussians) [[paper]](https://arxiv.org/pdf/2312.16812) [[project]](https://oppo-us-research.github.io/SpacetimeGaussians-website/) **Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis** [CVPR 2024]  
[[code]](https://github.com/hustvl/4DGaussians) [[paper]](https://arxiv.org/pdf/2310.08528v2) [[project]](https://guanjunwu.github.io/4dgs/index.html) **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering** [CVPR 2024]  
[[code]](https://github.com/zhichengLuxx/GaGS) [[paper]](https://arxiv.org/pdf/2404.06270) [[project]](https://npucvr.github.io/GaGS/) **3D Geometry-aware Deformable Gaussian Splatting for Dynamic View Synthesis** [CVPR 2024]  
[[code]](https://github.com/lizhe00/AnimatableGaussians) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Animatable_Gaussians_Learning_Pose-dependent_Gaussian_Maps_for_High-fidelity_Human_Avatar_CVPR_2024_paper.pdf) [[project]](https://animatable-gaussians.github.io/) **Animatable Gaussians: Learning Pose-Dependent Gaussian Maps for High-Fidelity Human Avatar Modeling** [CVPR 2024]  
[[code]](https://github.com/moqiyinlun/HiFi4G_Dataset) [[paper]](https://arxiv.org/abs/2312.03461) [[project]](https://nowheretrix.github.io/HiFi4G/) **HiFi4G: High-Fidelity Human Performance Rendering via Compact Gaussian Splatting** [CVPR 2024]  
[[code]](https://github.com/skhu101/GauHuman) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_GauHuman_Articulated_Gaussian_Splatting_from_Monocular_Human_Videos_CVPR_2024_paper.pdf) [[project]](https://skhu101.github.io/GauHuman/) **GauHuman: Articulated Gaussian Splatting from Monocular Human Videos** [CVPR 2024]  
[[code]](https://github.com/YuelangX/Gaussian-Head-Avatar) [[paper]](https://arxiv.org/abs/2312.03029) [[project]](https://yuelangx.github.io/gaussianheadavatar/) **Gaussian Head Avatar:Ultra High-fidelity Head Avatar via Dynamic Gaussians** [CVPR 2024]  
[[code]](https://github.com/xwx0924/SurgicalGaussian) [[paper]](https://arxiv.org/abs/2407.05023) [[project]](https://surgicalgaussian.github.io/) **SurgicalGaussian: Deformable 3D Gaussians for High-Fidelity Surgical Scene Reconstruction** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2312.00112) [[project]](https://agelosk.github.io/dynmf/) **DynMF: Neural Motion Factorization for Real-time Dynamic View Synthesis with 3D Gaussian Splatting** [ECCV 2024]  
[[code]](https://github.com/markomih/SplatFields) [[paper]](https://github.com/markomih/SplatFields) [[project]](https://markomih.github.io/SplatFields/) **SplatFields:Neural Gaussian Splats for Sparse 3D and 4D Reconstruction** [ECCV 2024]  
[[code]](https://github.com/RuijieZhu94/MotionGS) [[paper]](https://openreview.net/pdf?id=6FTlHaxCpR) [[project]](https://github.com/RuijieZhu94/MotionGS) **MotionGS: Exploring Explicit Motion Guidance for Deformable 3D Gaussian Splatting** [NeurIPS 2024]  
[[code]](https://github.com/juno181/Ex4DGS) [[paper]](https://arxiv.org/abs/2410.15629) [[project]](https://leejunoh.com/Ex4DGS/) **Fully Explicit Dynamic Gaussian Splatting** [NeurIPS 2024]  
[[paper]](https://papers.neurips.cc/paper_files/paper/2024/hash/e95da8078ec8389533c802e368da5298-Abstract-Conference.html) [[project]](https://deep-diver.github.io/neurips2024/posters/0syctgl4in/) **4D Gaussian Splatting in the Wild with Uncertainty-Aware Regularization** [NeurIPS 2024]  
[[code]](https://github.com/fudan-zvg/4d-gaussian-splatting) [[paper]](https://arxiv.org/abs/2310.10642) [[project]](https://fudan-zvg.github.io/4d-gaussian-splatting/) **Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting** [ICLR 2024]  
[[code]](https://github.com/waczjoan/D-MiSo) [[paper]](https://arxiv.org/abs/2405.14276) **D-MISO: Editing Dynamic 3D Scenes Using Multi-Gaussians Soup** [NeurIPS 2024]  
[[code]](https://github.com/facebookresearch/D3GA) [[paper]](https://arxiv.org/pdf/2311.08581) [[project]](https://zielon.github.io/d3ga/) **D3GA - Drivable 3D Gaussian Avatars** [3DV 2025]  
[[code]](https://github.com/xg-chu/GPAvatar) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Feng_GPAvatar_High-fidelity_Head_Avatars_by_Learning_Efficient_Gaussian_Projections_CVPR_2025_paper.pdf) [[project]](https://xg-chu.site/project_gpavatar/) **GPAvatar: High-Fidelity Head Avatars by Learning Efficient Gaussian Projections** [ICLR 2024]  
[[code]](https://github.com/gqk/HiCoM) [[paper]](https://arxiv.org/pdf/2411.07541) **HiCoM: Hierarchical Coherent Motion for Dynamic Streamable Scenes with 3D Gaussian Splatting** [NeurIPS 2024]  
[[code]](https://github.com/mlzxy/motion-blender-gs) [[paper]](https://arxiv.org/abs/2503.09040) [[project]](https://mlzxy.github.io/motion-blender-gs/) **Motion Blender Gaussian Splatting** [CoRL 2025]  
[[code]](https://github.com/wgsxm/OmniPhysGS) [[paper]](https://arxiv.org/abs/2501.18982) [[project]](https://wgsxm.github.io/projects/omniphysgs/) **OmniPhysGS: 3D Constitutive Gaussians for General Physics-based Dynamics Generation** [ICLR 2025]  
[[code]](https://github.com/brownvc/GauFRe) [[paper]](https://lynl7130.github.io/gaufre/static/pdfs/WACV_2025___GauFRe%20(1).pdf) [[project]](https://lynl7130.github.io/gaufre/index.html) **GauFRe🧇: Gaussian Deformation Fields for Real-time Dynamic Novel View Synthesis** [WACV 2025]  
[[code]](https://github.com/weify627/4D-Rotor-Gaussians) [[paper]](https://arxiv.org/abs/2402.03307) [[project]](https://weify627.github.io/4drotorgs/) **4D-Rotor Gaussian Splatting: Towards Efficieant Novel View Synthesis for Dynamic Scenes** [SIGGRAPH 2024]  
[[code]](https://github.com/GuanxingLu/ManiGaussian) [[paper]](https://arxiv.org/abs/2403.08321) [[project]](https://guanxinglu.github.io/ManiGaussian/) **ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation** [ECCV 2024]  






#### Large-Scale & Unbounded Scenes

><span style="color:lightblue;">💡💡 City-Level Reconstructions: Progressive propagation.</span>  
><span style="color:lightblue;">💡💡 Driving Scenes: Moving object resistance.</span>  
><span style="color:lightblue;">💡💡 Parallel Optimization: Scaling parameters.</span>  


[[code]](https://github.com/kangpeilun/VastGaussian) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Lin_VastGaussian_Vast_3D_Gaussians_for_Large_Scene_Reconstruction_CVPR_2024_paper.pdf) [[project]](https://vastgaussian.github.io/) **VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction** [CVPR 2024]  
[[code]](https://github.com/nyu-systems/Grendel-GS) [[paper]](https://arxiv.org/abs/2406.18533) [[project]](https://daohanlu.github.io/scaling-up-3dgs/) **Gaussian Splatting at Scale with Distributed Training System** [ICLR 2025]  
[[code]](https://github.com/Linketic/CityGaussian) [[paper]](https://arxiv.org/pdf/2404.01133) [[project]](https://dekuliutesla.github.io/citygs/) **CityGaussian Series for High-quality Large-Scale Scene Reconstruction with Gaussians** [ECCV 2024]  
[[code]](https://github.com/Linketic/CityGaussian/tree/CityGaussian_V2.0) [[paper]](https://dekuliutesla.github.io/CityGaussianV2/static/paper/CityGaussianV2.pdf) [[project]](https://dekuliutesla.github.io/CityGaussianV2/) **CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes** [ICLR 2025]  
[[code]](https://arxiv.org/abs/2412.17612) [[paper]](https://arxiv.org/abs/2412.17612) [[project]](https://gyy456.github.io/CoSurfGS/#:~:text=To%20address%20these%20issues%2C%20we%20propose%20CoSurfGS%2C%20a,based%20on%20distributed%20learning%20for%20large-scale%20surface%20reconstruction.) **CoSurfGS: Collaborative 3D Surface Gaussian Splatting with Distributed Learning for Large Scene Reconstruction** [ECCV 2024]  
[[code]](https://github.com/chengweialan/DeSiRe-GS) [[paper]](https://arxiv.org/abs/2411.11921) **DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction for Urban Driving Scenes** [Arxiv 2024]  
[[code]](https://github.com/3DV-Coder/FGS-SLAM) [[paper]](https://arxiv.org/pdf/2503.01109) **FGS-SLAM: Fourier-based Gaussian Splatting for Real-time SLAM with Sparse and Dense Map Fusion** [IROS 2025]  
[[code]](https://github.com/EastbeanZhang/Gaussian-Wild) [[paper]](https://arxiv.org/abs/2403.08321) [[project]](https://eastbeanzhang.github.io/GS-W/) **Gaussian in the Wild: 3D Gaussian Splatting for Unconstrained Image Collections** [ECCV 2024]  
[[code]](https://github.com/autonomousvision/gaussian-opacity-fields) [[paper]](https://drive.google.com/file/d/1_IEpaSqDP4DzQ3TbhKyjhXo6SKscpaeq/view) [[project]](https://niujinshuchong.github.io/gaussian-opacity-fields/) **Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes** [SIGGRAPH ASIA 2024]  
[[code]](https://github.com/zhaofuq/LOD-3DGS) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3687762) [[project]](https://zhaofuq.github.io/LetsGo/) **LetsGo: Large-Scale Garage Modeling and Rendering via LiDAR-Assisted Gaussian Primitives** [SIGGRAPH Asia 2024]  
[[code]](https://github.com/AIBluefisher/DOGS) [[paper]](https://aibluefisher.github.io/DOGS/) [[project]](https://aibluefisher.github.io/DOGS/) **DOGS: Distributed-Oriented Gaussian Splatting for Large-Scale 3D Reconstruction Via Gaussian Consensus** [NeurIPS 2024]  
[[code]](https://github.com/VITA-Group/LightGaussian) [[paper]](https://arxiv.org/pdf/2311.17245) [[project]](https://lightgaussian.github.io/) **LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS** [NeurIPS 2024]  
[[code]](https://github.com/xiliu8006/3DGS-Enhancer) [[paper]](3DGS-Enhancer: Enhancing Unbounded 3D Gaussian Splatting with View-consistent 2D Diffusion Priors) [[project]](https://xiliu8006.github.io/3DGS-Enhancer-project/) **3DGS-Enhancer: Enhancing Unbounded 3D Gaussian Splatting with View-consistent 2D Diffusion Priors** [NeurIPS 2024]  
[[code]](https://github.com/Jixuan-Fan/Momentum-GS) [[paper]](https://arxiv.org/abs/2412.04887) [[project]](https://jixuan-fan.github.io/Momentum-GS_Page/) **Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction** [ICCV 2025]  
[[code]](https://github.com/graphdeco-inria/hierarchical-3d-gaussians) [[paper]](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/hierarchical-3d-gaussians_low.pdf) [[project]](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/) **A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets** [ECCV 2024]  
[[code]](https://github.com/AIBluefisher/DOGS) [[paper]](https://aibluefisher.github.io/DOGS/) [[project]](https://aibluefisher.github.io/DOGS/) **DOGS: Distributed-Oriented Gaussian Splatting for Large-Scale 3D Reconstruction Via Gaussian Consensus** [NeurIPS 2024]  




#### Sparse Inputs & Generalization

><span style="color:lightblue;">💡💡 Single/Few Views: Sparse synthesis.</span>  
><span style="color:lightblue;">💡💡 DFeed-Forward: Instant models.</span>  
><span style="color:lightblue;">💡💡 Generalization: Neural splats.</span>  


[[code]](https://github.com/ForMyCat/SparseGS) [[paper]](https://arxiv.org/abs/2312.00206) [[project]](https://formycat.github.io/SparseGS-Real-Time-360-Sparse-View-Synthesis-using-Gaussian-Splatting/) **SparseGS: Real-Time 360° Sparse View Synthesis using Gaussian Splatting** [arXiv 2023]  
[[paper]](https://arxiv.org/abs/2503.04314) [[project]](https://jeasco.github.io/S2Gaussian/) **S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting** [CVPR 2025]  
[[code]](https://github.com/VITA-Group/FSGS) [[paper]](https://arxiv.org/abs/2312.00451) [[project]](https://zehaozhu.github.io/FSGS/) **FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting** [ECCV 2024]  
[[code]](https://github.com/dcharatan/pixelsplat) [[paper]](https://davidcharatan.com/pixelsplat/) [[project]](https://davidcharatan.com/pixelsplat/) **pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction** [ECCV 2024]  
[[code]](https://github.com/szymanowiczs/splatter-image) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Szymanowicz_Splatter_Image_Ultra-Fast_Single-View_3D_Reconstruction_CVPR_2024_paper.pdf)  **Splatter Image: Ultra-Fast Single-View 3D Reconstruction** [CVPR 2024]  
[[code]](https://github.com/Fictionarry/DNGaussian) [[paper]](https://arxiv.org/abs/2403.06912) [[project]](https://guanxinglu.github.io/ManiGaussian/) **DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization** [CVPR 2024]  
[[code]](https://github.com/Chrixtar/latentsplat) [[paper]](https://geometric-rl.mpi-inf.mpg.de/latentsplat/static/assets/latentSplat.pdf) [[project]](https://geometric-rl.mpi-inf.mpg.de/latentsplat/) ** latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction** [ECCV 2024]  
[[code]](https://github.com/SkyworkAI/Gamba) [[paper]](https://arxiv.org/abs/2403.18795) [[project]](https://florinshen.github.io/gamba-project/) **Gamba: Marry Gaussian Splatting with Mamba for single view 3D reconstruction** [TPAMI 2024]  
[[code]](https://github.com/jiaw-z/CoR-GS) [[paper]](https://arxiv.org/pdf/2405.12110) [[project]](https://jiaw-z.github.io/CoR-GS/) **CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization** [ECCV 2024]  
[[code]](https://github.com/Open3DVLab/HiSplat) [[paper]](https://arxiv.org/pdf/2410.06245) [[project]](https://open3dvlab.github.io/HiSplat/) **HiSplat: Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction** [ICLR 2025]  
[[code]](https://github.com/TencentARC/FreeSplatter) [[paper]](https://arxiv.org/abs/2412.09573) **FreeSplatter: Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction** [ICCV 2025]  
[[code]](https://github.com/gurutvapatle/AD-GS) [[paper]](https://arxiv.org/abs/2509.11003) [[project]](https://gurutvapatle.github.io/publications/2025/ADGS.html) **AD-GS: Alternating Densification for Sparse-Input 3D Gaussian Splatting** [ACM SIGGRAPH Asia 2025]  
[[code]](https://github.com/VAST-AI-Research/TriplaneGaussian) [[paper]](https://arxiv.org/abs/2312.09147) [[project]](https://zouzx.github.io/TriplaneGaussian/) **Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers** [Arxiv 2023]  
[[code]](https://github.com/markomih/SplatFields) [[paper]](https://github.com/markomih/SplatFields) [[project]](https://markomih.github.io/SplatFields/) **SplatFields:Neural Gaussian Splats for Sparse 3D and 4D Reconstruction** [ECCV 2024]  
[[code]](https://github.com/Gynjn/selfsplat) [[paper]](https://arxiv.org/abs/2411.17190) [[project]](https://gynjn.github.io/selfsplat/) **SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting** [CVPR 2025]  
[[code]](https://github.com/ranrhuang/SPFSplat) [[paper]](https://arxiv.org/abs/2403.08321) [[project]](https://ranrhuang.github.io/spfsplat/) **No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views** [ICCV 2025]  
[[code]](https://github.com/ueoo/DropGaussian) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Park_DropGaussian_Structural_Regularization_for_Sparse-view_Gaussian_Splatting_CVPR_2025_paper.pdf) **DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting** [CVPR 2025]  
[[code]](https://github.com/donydchen/mvsplat) [[paper]](https://arxiv.org/abs/2403.14627) [[project]](https://donydchen.github.io/mvsplat/) **MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images** [ECCV 2024]  
[[code]](https://github.com/xingyoujun/transplat) [[paper]](https://xingyoujun.github.io/transplat/) [[project]](https://xingyoujun.github.io/transplat/) **TranSplat: Generalizable 3D Gaussian Splatting from Sparse Multi-View Images with Transformers** [AAAI 2025]  
[[code]](https://github.com/chobao/) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Bao_Free360_Layered_Gaussian_Splatting_for_Unbounded_360-Degree_View_Synthesis_from_CVPR_2025_paper.pdf) [[project]](https://zju3dv.github.io/free360/) **Free360: Layered Gaussian Splatting for Unbounded 360-Degree View Synthesis from Extremely Sparse and Unposed Views** [CVPR 2025] 





#### Generative Models

><span style="color:lightblue;">💡💡 Single/Few Views: Sparse synthesis.</span>  
><span style="color:lightblue;">💡💡 DFeed-Forward: Instant models.</span>  
><span style="color:lightblue;">💡💡 Generalization: Neural splats.</span>  


[[code]](https://github.com/hzxie/GaussianCity) [[paper]](https://arxiv.org/abs/2406.06526) **Generative Gaussian Splatting for Unbounded 3D City Generation** [CVPR 2025]  
[[code]](https://github.com/dreamgaussian/dreamgaussian) [[paper]](https://arxiv.org/abs/2309.16653) [[project]](https://dreamgaussian.github.io/) **DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation** [ICLR 2024]  
[[code]](https://github.com/weiqi-zhang/DiffGS) [[paper]](https://arxiv.org/abs/2410.19657) [[project]](https://junshengzhou.github.io/DiffGS/) **DiffGS: Functional Gaussian Splatting Diffusion** [NeurIPS 2024]  
[[code]](https://github.com/hustvl/GaussianDreamer) [[paper]](https://arxiv.org/abs/2310.08529) [[project]](https://taoranyi.com/gaussiandreamer/) **GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models** [CVPR 2024]  
[[code]](https://github.com/gohyojun15/SplatFlow) [[paper]](https://arxiv.org/abs/2411.16443) [[project]](https://gohyojun15.github.io/SplatFlow/) **SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis** [CVPR 2025]  
[[code]](https://github.com/chenguolin/DiffSplat) [[paper]](https://arxiv.org/abs/2501.16764) [[project]](https://chenguolin.github.io/projects/DiffSplat/) **DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation** [ICLR 2025]  




#### Style Transfer & Scene Editing

><span style="color:lightblue;">💡💡 Style Transfer: Neural transfers.</span>  
><span style="color:lightblue;">💡💡 Scene Editing: Language-aware.</span>  


[[code]](https://github.com/Kunhao-Liu/StyleGaussian) [[paper]](https://arxiv.org/abs/2403.07807) [[project]](https://kunhao-liu.github.io/StyleGaussian/) **StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting** [SIGGRAPH Asia 2024]  
[[code]](https://github.com/Kristen-Z/StylizedGS) [[paper]](https://arxiv.org/abs/2404.05220) [[project]](https://kristen-z.github.io/stylizedgs/) **StylizedGS: Controllable Stylization for 3D Gaussian Splatting** [TPAMI 2025]  
[[code]](https://github.com/bernard0047/style-splat) [[paper]](https://arxiv.org/abs/2407.09473) [[project]](https://bernard0047.github.io/stylesplat/) **StyleSplat: 3D Object Style Transfer with Gaussian Splatting** [arXiv 2024]  
[[code]](https://github.com/HarukiYqM/Reference-based-Scene-Stylization) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/076c1fa639a7190e216e734f0a1b3e7b-Paper-Conference.pdf) **Reference-based Controllable Scene Stylization** [NeurIPS 2024]  
[[code]](https://github.com/ActiveVisionLab/gaussctrl) [[paper]](https://arxiv.org/abs/2403.08733) [[project]](https://gaussctrl.active.vision/) **GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing** [ECCV 2024]  
[[code]](https://github.com/nianticlabs/morpheus) [[paper]](https://nianticlabs.github.io/morpheus/resources/Morpheus.pdf) [[project]](https://nianticlabs.github.io/morpheus/) **Text-Driven 3D Gaussian Splat Shape and Color Stylization** [CVPR 2025]  
[[code]](https://github.com/minghanqin/LangSplat) [[paper]](https://arxiv.org/pdf/2312.16084) [[project]](https://langsplat.github.io/) **LangSplat: 3D Language Gaussian Splatting** [CVPR 2024]  
[[paper]](https://arxiv.org/pdf/2506.09565) [[project]](https://semanticsplat.github.io/) **SemanticSplat: Feed-Forward 3D Scene Understanding with Language-Aware Gaussian Fields** [Arxiv 2025]  
[[code]](https://github.com/HaroldChen19/gaussianvton) [[paper]](https://arxiv.org/abs/2405.07472) [[project]](https://haroldchen19.github.io/gsvton/) **GaussianVTON: 3D Human Virtual Try-ON via Multi-Stage Gaussian Splatting Editing with Image Prompting** [ arXiv 2024]  
[[code]](https://github.com/umangi-jain/gaussiancut) [[paper]](https://openreview.net/pdf?id=Ns0LQokxa5) [[project]](https://umangi-jain.github.io/gaussiancut/) **GaussianCut: Interactive segmentation via graph cut for 3D Gaussian Splatting** [NeurIPS 2024]  
[[code]](https://github.com/vpx-ecnu/ABC-GS) [[paper]](https://arxiv.org/pdf/2503.22218) [[project]](https://vpx-ecnu.github.io/ABC-GS-website/) **ABC-GS: Alignment-Based Controllable Style Transfer for 3D Gaussian Splatting** [ICME 2025]  
[[code]](https://github.com/juhyeon-kwon/Instruct-4DGS) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Kwon_Efficient_Dynamic_Scene_Editing_via_4D_Gaussian-based_Static-Dynamic_Separation_CVPR_2025_paper.html) [[project]](https://hanbyelcho.info/instruct-4dgs/) **Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation** [CVPR 2025]  




#### Robotics & SLAM

><span style="color:lightblue;">💡💡 Visual SLAM: Dense and semantic.</span>  
><span style="color:lightblue;">💡💡 Localization Navigation: Hierarchical planning.</span>  
><span style="color:lightblue;">💡💡 Active Reconstruction: Gaussian herding.</span>  


[[code]](https://github.com/muskie82/MonoGS) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Matsuki_Gaussian_Splatting_SLAM_CVPR_2024_paper.pdf) [[project]](https://rmurai.co.uk/projects/GaussianSplattingSLAM/) **Gaussian Splatting SLAM** [CVPR 2024]  
[[code]](https://github.com/yanchi-3dv/diff-gaussian-rasterization-for-gsslam) [[paper]](https://arxiv.org/pdf/2311.11700) [[project]](https://gs-slam.github.io/) **GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting** [CVPR 2024]  
[[code]](https://github.com/JohannaXie/GauSS-MI) [[paper]](https://www.roboticsproceedings.org/rss21/p030.pdf) **GauSS-MI:Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction** [RSS 2025]  
[[code]](https://github.com/XiaohanLei/GaussNav) [[paper]](https://arxiv.org/abs/2403.11625) [[project]](https://xiaohanlei.github.io/projects/GaussNav/) **GaussNav: Gaussian Splatting for Visual Navigation** [TPAMI 2025]  
[[code]](https://github.com/ShuhongLL/SGS-SLAM) [[paper]](https://arxiv.org/pdf/2402.03246) [[project]](https://www.youtube.com/watch?v=y83yw1E-oUo) **SGS-SLAM: Semantic Gaussian Splatting For Neural Dense SLAM** [ECCV 2024]  
[[code]](https://github.com/armlabstanford/Touch-GS) [[paper]](https://arxiv.org/pdf/2403.09875) [[project]](https://arm.stanford.edu/touch-gs) **Touch-GS: Visual-Tactile Supervised 3D Gaussian Splatting** [IROS 2024]  
[[paper]](https://openreview.net/forum?id=EyEE7547vy) [[project]](https://tyxiong23.github.io/event3dgs) **Event3DGS: Event-Based 3D Gaussian Splatting for High-Speed Robot Egomotion** [CoRL 2024]  
[[code]](https://github.com/MrSecant/GaussianGrasper) [[paper]](https://arxiv.org/abs/2403.09637) [[project]](https://mrsecant.github.io/GaussianGrasper/) **GaussianGrasper: 3D Language Gaussian Splatting for Open-vocabulary Robotic Grasping** [LRA 2024]  
[[paper]](https://rffr.leggedrobotics.com/works/teleoperation/RadianceFieldsForTeleoperation.pdf) [[project]](https://rffr.leggedrobotics.com/works/teleoperation/) **Radiance Fields for Robotic Teleoperation** [IROS 2024]  
[[code]](https://github.com/jimazeyu/GraspSplats) [[paper]](https://arxiv.org/pdf/2409.02084) [[project]](https://graspsplats.github.io/) **GraspSplats: Efficient Manipulation with 3D Feature Splatting** [CoRL 2024]  
[[paper]](https://arxiv.org/pdf/2409.17624) [[project]](https://zijunfdu.github.io/HGS-Planner/) **HGS-Planner: Hierarchical Planning Framework for Active Scene Reconstruction Using 3D Gaussian Splatting** [ECCV 2024]  
[[code]](https://github.com/hjr37/CG-SLAM) [[paper]](https://arxiv.org/abs/2403.16095) [[project]](https://zju3dv.github.io/cg-slam/) **Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field** [ECCV 2024]  
[[code]](https://github.com/rmurai0610/MASt3R-SLAM) [[paper]](https://edexheim.github.io/mast3r-slam/) [[project]](https://edexheim.github.io/mast3r-slam/) **MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors** [CVPR 2025]  
[[code]](https://github.com/PRBonn/PIN_SLAM) [[paper]](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2024tro.pdf)  **PIN-SLAM: LiDAR SLAM using a Point-Based Implicit Neural Representation** [T-RO 2024]  



#### Semantic Understanding & Segmentation

><span style="color:lightblue;">💡💡 Open-Vocabulary: Language-embedded.</span>  
><span style="color:lightblue;">💡💡 Instance Segmentation: Boundary-enhanced.</span>  


[[paper]](https://openaccess.thecvf.com/content/CVPR2025W/OpenSUN3D/papers/Wiedmann_DCSEG_Decoupled_3D_Open-Set_Segmentation_using_Gaussian_Splatting_CVPRW_2025_paper.pdf) **DCSEG:Decoupled 3D Open-Set Segmentation using Gaussian Splatting** [CVPR 2025]  
[[code]](https://github.com/XuHu0529/SAGS) [[paper]](https://arxiv.org/pdf/2401.17857) **SAGD: Boundary-Enhanced Segment Anything in 3D Gaussian via Gaussian Decomposition** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2404.14249) [[project]](https://gbliao.github.io/CLIP-GS.github.io/) **CLIP-GS: CLIP-Informed Gaussian Splatting for Real-time and View-consistent 3D Semantic Understanding** [ECCV 2024]  
[[code]](https://github.com/sharinka0715/semantic-gaussians) [[paper]](https://arxiv.org/pdf/2403.15624) [[project]](https://sharinka0715.github.io/semantic-gaussians/) **Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting** [ECCV 2024]  
[[paper]](https://arxiv.org/pdf/2401.05925) [[project]](https://david-dou.github.io/CoSSegGaussians/) **CoSSegGaussians: Compact and Swift Scene Segmenting 3D Gaussians with Dual Feature Fusion** [arXiv 2024]  
[[code]](https://github.com/lifuguan/LangSurf) [[paper]](https://arxiv.org/pdf/2412.17635) [[project]](https://langsurf.github.io/) **LangSurf: Language-Embedded Surface Gaussians for 3D Scene Understanding** [arXiv 2025]  
[[code]](https://github.com/HorizonRobotics/GLS) [[paper]](https://arxiv.org/pdf/2411.18066) [[project]](https://jiaxiongq.github.io/GLS_ProjectPage/) **GLS: Geometry-aware 3D Language Gaussian Splatting** [arXiv 2025]  
[[code]](https://github.com/weijielyu/Gaga) [[paper]](https://arxiv.org/abs/2404.07977) [[project]](https://www.gaga.gallery/) **Gaga: Group Any Gaussians via 3D-aware Memory Bank** [arXiv 2025]  
[[code]](https://github.com/THU-luvision/OmniSeg3D) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Ying_OmniSeg3D_Omniversal_3D_Segmentation_via_Hierarchical_Contrastive_Learning_CVPR_2024_paper.pdf) [[project]](https://oceanying.github.io/OmniSeg3D/) **OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning** [CVPR 2024]  
[[code]](https://github.com/ShijieZhou-UCLA/feature-3dgs) [[paper]](https://arxiv.org/abs/2312.03203) [[project]](https://feature-3dgs.github.io/) **Feature 3DGS:Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields** [CVPR 2024]  
[[code]](https://github.com/buaavrcg/LEGaussians) [[paper]](https://arxiv.org/abs/2403.08321) [[project]](https://guanxinglu.github.io/ManiGaussian/) **LEGaussians: Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding** [CVPR 2024]  
[[code]](https://github.com/GuanxingLu/ManiGaussian) [[paper]](https://arxiv.org/abs/2311.18482) [[project]](https://buaavrcg.github.io/LEGaussians/) **ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation** [ECCV 2024]  
[[code]](https://github.com/Quyans/GOI-Hyperplane) [[paper]](https://arxiv.org/pdf/2405.17596) [[project]](https://quyans.github.io/GOI-Hyperplane) **GOI: Find 3D Gaussians of Interest with an Optimizable Open-vocabulary Semantic-space Hyperplane** [ACM MM 2025]  
[[code]](https://github.com/google-research/foundation-model-embedded-3dgs) [[paper]](https://arxiv.org/abs/2401.01970) [[project]](https://xingxingzuo.github.io/fmgs/) **FMGS: Foundation Model Embedded 3D Gaussian Splatting for Holistic 3D Scene Understanding** [IJCV 2025]  
[[code]](https://mbjurca.github.io/rt-gs2/) [[paper]](https://mbjurca.github.io/rt-gs2/) [[project]](https://mbjurca.github.io/rt-gs2/) **RT-GS2: Real-Time Generalizable Semantic Segmentation for 3D Gaussian Representations of Radiance Fields** [BMVC 2024]  
[[code]](https://github.com/minghanqin/LangSplat) [[paper]](https://arxiv.org/pdf/2312.16084) [[project]](https://langsplat.github.io/) **LangSplat: 3D Language Gaussian Splatting** [CVPR 2024]  
[[paper]](https://arxiv.org/abs/2503.22204) [[project]](https://vulab-ai.github.io/Segment-then-Splat/) **Segment then Splat: A Unified Approach for 3D Open-Vocabulary Segmentation based on Gaussian Splatting** [arXiv 2024]  
[[code]](https://github.com/kaist-ami/Dr-Splat) [[paper]](https://arxiv.org/abs/2502.16652) [[project]](https://drsplat.github.io/) **Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration** [CVPR 2025]  
[[code]](https://github.com/lhj-git/InstanceGasuusian_code) [[paper]](https://arxiv.org/pdf/2411.19235) [[project]](https://lhj-git.github.io/InstanceGaussian/) **InstanceGaussian: Appearance-Semantic Joint Gaussian Representation for 3D Instance-Level Perception** [CVPR 2025]  




#### Physics Simulation & Interaction

><span style="color:lightblue;">💡💡 Open-Vocabulary: Language-embedded.</span>  
><span style="color:lightblue;">💡💡 Instance Segmentation: Boundary-enhanced.</span>  


[[code]](https://github.com/XPandora/PhysGaussian) [[paper]](https://arxiv.org/abs/2311.12198) [[project]](https://xpandora.github.io/PhysGaussian/) **PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics** [CVPR 2024]  
[[code]](https://github.com/waczjoan/GASP) [[paper]](https://arxiv.org/abs/2409.05819) [[project]](https://waczjoan.github.io/GASP/) **GASP: Gaussian Splatting for Physic-Based Simulations** [arXiv 2024]  
[[code]](https://github.com/wgsxm/OmniPhysGS) [[paper]](https://arxiv.org/abs/2501.18982) [[project]](https://wgsxm.github.io/projects/omniphysgs/) **OmniPhysGS: 3D Constitutive Gaussians for General Physics-based Dynamics Generation** [ICLR 2025]  
[[code]](https://github.com/wangmiaowei/DecoupledGaussian) [[paper]](https://arxiv.org/abs/2503.05484v1) [[project]](https://wangmiaowei.github.io/DecoupledGaussian.github.io/) **DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction** [CVPR 2025]  
[[code]](https://ucla.app.box.com/s/yt4i3wm2i5m2ubace4l1cj3untr13ud1) [[paper]](https://arxiv.org/abs/2401.16663) [[project]](https://yingjiang96.github.io/VR-GS) **VR-GS: A Physical Dynamics-Aware Interactive Gaussian Splatting System in Virtual Reality** [SIGGRAPH 2024]  
[[code]](https://arxiv.org/pdf/2502.19459) [[paper]](https://arxiv.org/pdf/2502.19459) [[project]](https://articulate-gs.github.io/) **ArtGS: Building Interactable Replicas of Complex Articulated Objects via Gaussian Splatting** [ICLR 2025]  
[[code]](https://github.com/EnVision-Research/Gaussian-Property) [[paper]](https://gaussian-property.github.io/) [[project]](https://gaussian-property.github.io/) **GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs** [ICCV 2025]  
[[code]](https://github.com/jiawei-ren/dreamgaussian4d) [[paper]](https://arxiv.org/abs/2312.17142) [[project]](https://jiawei-ren.github.io/projects/dreamgaussian4d/) **DreamGaussian4D:Generative 4D Gaussian Splatting** [Arxiv 2023]  
[[code]](https://github.com/Colmar-zlicheng/Spring-Gaus) [[paper]](https://arxiv.org/abs/2403.09434) [[project]](https://zlicheng.com/spring_gaus/) **Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians** [ECCV 2024]  
[[code]](https://github.com/liuff19/Physics3D) [[paper]](https://arxiv.org/abs/2406.04338) [[project]](https://guanxinglu.github.io/ManiGaussian/) **Physics3D: Learning Physical Properties of 3D Gaussians via Video Diffusion** [Arxiv 2024]  
[[code]](https://github.com/a1600012888/PhysDreamer) [[paper]](https://arxiv.org/abs/2404.13026) [[project]](https://physdreamer.github.io/) **PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation** [ECCV 2024]  
[[code]](https://github.com/bdaiinstitute/embodied_gaussians) [[paper]](https://openreview.net/forum?id=AEq0onGrN2) [[project]](https://embodied-gaussians.github.io/) **Physically Embodied Gaussian Splatting: A Visually Learnt and Physically Grounded 3D Representation for Robotics** [CRL 2024]  




#### Compression & Storage Optimization


[[code]](https://github.com/YihangChen-ee/HAC) [[paper]](https://arxiv.org/abs/2403.14530) [[project]](https://yihangchen-ee.github.io/project_hac/) **HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression** [ECCV 2024]  
[[code]](https://github.com/YihangChen-ee/HAC-plus) [[paper]](https://arxiv.org/abs/2501.12255) [[project]](https://yihangchen-ee.github.io/project_hac++/) **HAC++: Towards 100X Compression of 3D Gaussian Splatting** [TPAMI 2025]  
[[code]](https://github.com/maincold2/OMG) [[paper]](https://arxiv.org/abs/2503.16924) [[project]](https://maincold2.github.io/omg/) **Optimized Minimal 3D Gaussian Splatting** [NeurIPS 2025]  
[[code]](https://github.com/KeKsBoTer/c3dgs) [[paper]](https://arxiv.org/abs/2401.02436) [[project]](https://niedermayr.dev/c3dgs/) **Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis** [arXiv 2024]  
[[code]](https://github.com/wyf0912/ContextGS) [[paper]](https://arxiv.org/pdf/2405.20721) **ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Model** [ECCV 2024]  
[[code]](https://github.com/dai647/DF_3DGS) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Dai_Efficient_Decoupled_Feature_3D_Gaussian_Splatting_via_Hierarchical_Compression_CVPR_2025_paper.pdf) **Efficient Decoupled Feature 3D Gaussian Splatting via Hierarchical Compression** [CVPR 2025]  
[[code]](https://github.com/w-m/3dgs-compression-survey) [[paper]](https://github.com/w-m/3dgs-compression-survey) **3DGS.zip: A survey on 3D Gaussian Splatting Compression Methods** [CGF 2025]  
[[code]](https://github.com/DrunkenPoet/GHAP) [[paper]](https://arxiv.org/abs/2506.09534) **GHAP: Gaussian Herding Across Pens** [arXiv 2024]  
[[code1]](https://github.com/maincold2/Compact-3DGS) [[code2]](https://github.com/maincold2/Dynamic_C3DGS/) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_Compact_3D_Gaussian_Representation_for_Radiance_Field_CVPR_2024_paper.pdf) [[project]](https://maincold2.github.io/c3dgs/) **Compact 3D Gaussian Representation for Radiance Field** [CVPR 2024]  




#### Special Scenes & Applications  
><span style="color:lightblue;">💡💡 Panoramic: Gaussian panoramas.</span>  
><span style="color:lightblue;">💡💡 Underwater: Dynamic adaptations.</span>  
><span style="color:lightblue;">💡💡 Medical: Color enhancements.</span>  
><span style="color:lightblue;">💡💡 Transparent Objects: Geometric guidance.</span>  


[[code]](https://github.com/ShijieZhou-UCLA/dreamscene360) [[paper]](https://arxiv.org/abs/2404.06903) [[project]](https://dreamscene360.github.io/) **DreamScene360 Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting** [ECCV 2024]  
[[code]](https://github.com/water-splatting/water-splatting) [[paper]](https://water-splatting.github.io/paper.pdf) [[project]](https://water-splatting.github.io/) **WaterSplatting Fast Underwater 3D Scene Reconstruction Using Gaussian Splatting** [3DV 2025]  
[[code]](https://github.com/dxyang/seasplat/) [[paper]](https://arxiv.org/abs/2409.17345) [[project]](https://seasplat.github.io/) **SeaSplat: Representing Underwater Scenes with 3D Gaussian Splatting and a Physically Grounded Image Formation Model** [ICRA 2025]  
[[code]](https://github.com/PKU-VCL-Geometry/GeoSplatting) [[paper]](https://arxiv.org/abs/2410.24204) [[project]](https://pku-vcl-geometry.github.io/GeoSplatting/) **GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering** [ICCV 2025]  


[[code]](https://github.com/horizon-research/Fov-3DGS) [[paper]](https://linwk20.github.io/assets/pdf/asplos25_vr.pdf) [[project]](https://horizon-lab.org/metasapiens/) **Official Implementation of MetaSapiens: Real-Time Neural Rendering with Efficiency-Aware Pruning and Accelerated Foveated Rendering** [ASPLOS 2025]  
[[code]](https://github.com/zju3dv/street_gaussians/) [[paper]](https://arxiv.org/abs/2401.01339) [[project]](https://zju3dv.github.io/street_gaussians/) **Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting** [ECCV 2024]  
[[code]](https://github.com/Linketic/CityGaussian) [[paper]](https://arxiv.org/pdf/2404.01133) [[project]](https://dekuliutesla.github.io/citygs/) **CityGaussian Series for High-quality Large-Scale Scene Reconstruction with Gaussians** [ECCV 2024]  
[[code]](https://github.com/Linketic/CityGaussian/tree/CityGaussian_V2.0) [[paper]](https://dekuliutesla.github.io/CityGaussianV2/static/paper/CityGaussianV2.pdf) [[project]](https://dekuliutesla.github.io/CityGaussianV2/) **CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes** [ICLR 2025]  
[[code]](https://github.com/xhd0612/GaussianRoom) [[paper]](https://arxiv.org/abs/2405.19671) [[project]](https://xhd0612.github.io/GaussianRoom.github.io/) **GaussianRoom: Improving 3D Gaussian Splatting with SDF Guidance and Monocular Cues for Indoor Scene Reconstruction** [ICRA 2025]  
[[]](https://github.com/alibaba/MNN) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_TaoAvatar_Real-Time_Lifelike_Full-Body_Talking_Avatars_for_Augmented_Reality_via_CVPR_2025_paper.pdf) [[project]](https://pixelai-team.github.io/TaoAvatar/) **TaoAvatar: Real-Time Lifelike Full-Body Talking Avatars for Augmented Reality via 3D Gaussian Splatting** [CVPR 2025]  
[[code]](https://github.com/YuQiao0303/Fancy123) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_Fancy123_One_Image_to_High-Quality_3D_Mesh_Generation_via_Plug-and-Play_CVPR_2025_paper.pdf)  **Fancy123: One Image to High-Quality 3D Mesh Generation via Plug-and-Play Deformation** [CVPR 2025]  





## Scene Understanding

### Scenario geometry understanding

> <span style="color:lightblue;">💡💡 Scenario geometry understanding in the domain of computer vision and generative modeling encompasses the inference of explicit 3D spatial hierarchies and relational invariances within dynamic environmental contexts, leveraging geometric deep learning paradigms such as equivariant graph neural networks (EGNNs) or transformer-based multi-view fusion to distill multi-modal sensor data—ranging from RGB-D imagery and LiDAR point clouds—into semantically enriched representations that capture volumetric occupancy, surface normals, and affine transformations via differentiable rendering pipelines like neural radiance fields (NeRF) or 3D Gaussian splatting, thereby facilitating holistic scene parsing that integrates low-level geometric primitives with high-level affordance predictions for tasks in autonomous driving, robotic manipulation, and conditional video synthesis. This process typically employs contrastive self-supervision or diffusion-based priors to enforce scale-ambiguity resolution and temporal coherence, as exemplified in frameworks that jointly optimize geometric consistency and semantic segmentation through variational autoencoders (VAEs) mapping latent scene graphs, enabling robust extrapolation to novel viewpoints and occlusion handling while mitigating artifacts in reconstructed manifolds, ultimately bridging pixel-wise observations to Euclidean embeddings that underpin scalable 3D world models in embodied AI systems.</span>  


#### Depth estimation

[[project]](https://github.com/topics/depth-estimation)

> <span style="color:lightblue;">💡💡 Depth estimation refers to inferring pixel-level relative or absolute depth maps from monocular, stereo, or multi-view image inputs. This is achieved through geometric deep learning paradigms such as equivariant convolutional neural networks or Transformer-based multi-scale fusion architectures, Scale-invariant features are extracted from RGB or RGB-D data, enabling inverse perspective transformation mapping from 2D pixel projections to 3D Euclidean embeddings. Self-supervised contrastive learning or diffusion priors are integrated to mitigate data scarcity issues, ensuring robustness against occlusion and illumination variations. Specifically, the monocular depth estimation paradigm relies on encoder-decoder backbones like ResNet or Swin Transformer. It captures disparity cues and texture gradients through attention-guided edge-aware attention modules, then employs variational inference in the latent space to optimize absolute scale recovery, as demonstrated by models like MiDaS or Depth Anything. Multi-source depth fusion integrates real, synthetic, and zero-shot data, enhancing generalization through cross-domain knowledge distillation. For autonomous driving and robot navigation tasks, it further employs Neural Radiance Fields (NeRF) or 3D Gaussian splatting for differentiable rendering validation, enabling progressive refinement from coarse-grained occupancy prediction to fine-grained surface normals. Despite challenges like scale ambiguity and domain transfer, recent trends evolve toward depth foundation models that jointly optimize multimodal supervision to bridge 2D observations with 3D world models, thereby supporting scene geometry understanding and conditional video synthesis in embodied AI systems.</span>  

##### Traditional and Supervised Monocular Depth Estimation
 
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11078760) **Event-Based Stereo Depth Estimation: A Survey** [TPAMI 2025]  
[[paper]](https://ieeexplore.ieee.org/document/988771) **A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms** [SMBV 2001]  
[[paper]](https://ieeexplore.ieee.org/document/4359315) **Stereo Processing by Semiglobal Matching and Mutual Information** [TPAMI 2008]  
[[paper]](https://proceedings.neurips.cc/paper_files/paper/2014/file/91c56ce4a249fae5419b90cba831e303-Paper.pdf) **Depth Map Prediction from a Single Image Using a Multi-Scale Deep Network** [NeurIPS 2014]  
[[paper]](https://openaccess.thecvf.com/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf) **Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture** [ICCV 2015]   
[[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Deep_Convolutional_Neural_2015_CVPR_paper.pdf) **Deep Convolutional Neural Fields for Depth Estimation from a Single Image** [CVPR 2015]  
[[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Multi-Scale_Continuous_CRFs_CVPR_2017_paper.pdf) **Multi-Scale Continuous CRFs as Sequential Deep Networks for Monocular Depth Estimation** [CVPR 2017]  
[[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Godard_Unsupervised_Monocular_Depth_CVPR_2017_paper.pdf) **Unsupervised Monocular Depth Estimation with Left-Right Consistency** [CVPR 2017]  
[[paper]](https://openaccess.thecvf.com/content_cvpr_2017/html/Kuznietsov_Semi-Supervised_Deep_Learning_CVPR_2017_paper.html) **Semi-Supervised Deep Learning for Monocular Depth Map Prediction** [CVPR 2017]  
[[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Fu_Deep_Ordinal_Regression_CVPR_2018_paper.pdf) **Deep Ordinal Regression Network for Monocular Depth Estimation** [CVPR 2017]  
[[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yin_GeoNet_Unsupervised_Learning_CVPR_2018_paper.pdf) **GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose** [CVPR 2018]  


##### Self-Supervised and Zero-Shot Monocular Depth Estimation

[[code]](https://github.com/Ravi-Garg/Unsupervised_Depth_Estimation) [[paper]](https://arxiv.org/pdf/1603.04992) **Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue** [ECCV 2016]  
[[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.pdf) **Unsupervised Learning of Depth and Ego-Motion from Video** [CVPR 2017]  
[[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Niu_Im2Struct_Recovering_3D_CVPR_2018_paper.pdf) **3D Shape Reconstruction from a Single RGB Image Using a Deep Convolutional Neural Network** [NeurIPS 2018]  
[[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yin_GeoNet_Unsupervised_Learning_CVPR_2018_paper.pdf) **GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose** [CVPR 2018]  
[[code]](https://github.com/mrharicot/monodepth) **Unsupervised Monocular Depth Estimation with Left-Right Consistency** [CVPR 2017]
[[code]](https://github.com/nianticlabs/monodepth2) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.pdf) **Digging Into Self-Supervised Monocular Depth Estimation** [ICCV 2019]  
[[code]](https://github.com/mattpoggi/mono-uncertainty) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Poggi_On_the_Uncertainty_of_Self-Supervised_Monocular_Depth_Estimation_CVPR_2020_paper.pdf) **On the Uncertainty of Self-Supervised Monocular Depth Estimation** [CVPR 2020]  
[[code]](https://github.com/TRI-ML/packnet-sfm) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guizilini_3D_Packing_for_Self-Supervised_Monocular_Depth_Estimation_CVPR_2020_paper.pdf) **3D Packing for Self-Supervised Monocular Depth Estimation** [ECCV 2020]  
[[code]](https://github.com/shariqfarooq123/AdaBins) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhat_AdaBins_Depth_Estimation_Using_Adaptive_Bins_CVPR_2021_paper.pdf) **AdaBins: Depth Estimation Using Adaptive Bins** [CVPR 2021]  
[[code]](https://github.com/isl-org/MiDaS) [[paper]](https://ieeexplore.ieee.org/abstract/document/9178977) **Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer** [TPAMI 2020]  
[[code]](https://github.com/AutoAILab/FusionDepth) [[paper]](https://arxiv.org/abs/2109.09628) **Learning Self-Supervised Monocular Depth with Pseudo-LiDAR** [CVPRW 2021]  
[[code]](https://github.com/aliyun/NeWCRFs) [[paper]](https://arxiv.org/abs/2203.01502) **NeW CRFs: Neural Window Fully-connected CRFs for Monocular Depth Estimation** [CVPR 2022]  
[[code]](https://github.com/tri-ml/vidar) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Guizilini_Towards_Zero-Shot_Scale-Aware_Monocular_Depth_Estimation_ICCV_2023_paper.html) **Towards Zero-Shot Scale-Aware Monocular Depth Estimation** [ICCV 2023]  
[[code]](https://github.com/LiheYoung/Depth-Anything) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Depth_Anything_Unleashing_the_Power_of_Large-Scale_Unlabeled_Data_CVPR_2024_paper.pdf) **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data** [CVPR 2024]  
[[code]](https://github.com/DepthAnything/Depth-Anything-V2) [[paper]](https://arxiv.org/abs/2406.09414) **Depth Anything V2** [NeurIPS 2024]  
[[paper]](https://arxiv.org/abs/2109.09628) **Vision-Language Embodiment for Monocular Depth Estimation** [CVPR 2025]  
[[code]](https://github.com/DepthAnything/PromptDA) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_Prompting_Depth_Anything_for_4K_Resolution_Accurate_Metric_Depth_Estimation_CVPR_2025_paper.pdf) **Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation** [CVPR 2025]  
[[code]](https://github.com/CompVis/depth-fm) [[paper]](https://arxiv.org/abs/2403.13788) **DepthFM: Fast Monocular Depth Estimation with Flow Matching** [AAAI 2025]  
[[code]](https://github.com/isl-org/ZoeDepth) [[paper]](https://arxiv.org/pdf/2302.12288) **ZoeDepth: Combining relative and metric depth** [CVPR 2025]  
[[code]](https://github.com/DepthAnything/Video-Depth-Anything) [[paper]](https://arxiv.org/abs/2501.12375) **Video Depth Anything** [CVPR 2025]  



##### Stereo and Multi-View Depth Estimation
[[paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Luo_Efficient_Deep_Learning_CVPR_2016_paper.pdf) **Efficient Deep Learning for Stereo Matching** [CVPR 2016]  
[[code]](https://github.com/kelkelcheng/GC-Net-Tensorflow) [[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Kendall_End-To-End_Learning_of_ICCV_2017_paper.pdf) **End-to-End Learning of Geometry and Context for Deep Stereo Regression** [CVPR 2017]  
[[code]](https://github.com/JiaRenChang/PSMNet) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chang_Pyramid_Stereo_Matching_CVPR_2018_paper.pdf) **Pyramid Stereo Matching Network** [CVPR 2018]  
[[code]](https://github.com/feihuzhang/GANet) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_GA-Net_Guided_Aggregation_Net_for_End-To-End_Stereo_Matching_CVPR_2019_paper.pdf) **GA-Net: Guided Aggregation Net for End-to-End Stereo Matching** [CVPR 2019]  
[[code]](https://github.com/gengshan-y/high-res-stereo) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Hierarchical_Deep_Stereo_Matching_on_High-Resolution_Images_CVPR_2019_paper.pdf) **Hierarchical Deep Stereo Matching on High-Resolution Images** [CVPR 2020]  
[[code]](https://github.com/haofeixu/aanet) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_AANet_Adaptive_Aggregation_Network_for_Efficient_Stereo_Matching_CVPR_2020_paper.pdf) **AANet: Adaptive Aggregation Network for Efficient Stereo Matching** [CVPR 2020]  
[[code]](https://github.com/gallenszl/CFNet) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_CFNet_Cascade_and_Fused_Cost_Volume_for_Robust_Stereo_Matching_CVPR_2021_paper.pdf) **CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching** [CVPR 2021]  
[[code]](https://github.com/XuelianCheng/LEAStereo) [[paper]](https://proceedings.neurips.cc/paper/2020/file/fc146be0b230d7e0a92e66a6114b840d-Paper.pdf) **Hierarchical Neural Architecture Search for Deep Stereo Matching** [NeurIPS 2020]
[[code]](https://github.com/gallenszl/PCWNet) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920280.pdf) **PCW-Net: Pyramid Combination and Warping Cost Volume for Stereo Matching** [ECCV 2022]  
[[code]](https://github.com/megvii-research/CREStereo) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf) **Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation** [CVPR 2022]  
[[code]](https://github.com/gangweiX/ACVNet) [[paper]](https://arxiv.org/pdf/2209.12699) **Attention Concatenation Volume for Accurate and Efficient Stereo Matching** [CVPR 2022]  
[[code]](https://github.com/gangweiX/IGEV) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Iterative_Geometry_Encoding_Volume_for_Stereo_Matching_CVPR_2023_paper.pdf) **IGEV-Stereo: Iterative Geometry Encoding Volume for Stereo Matching** [CVPR 2023]  
[[code]](https://github.com/lly00412/SEDNet) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Learning_the_Distribution_of_Errors_in_Stereo_Matching_for_Joint_CVPR_2023_paper.pdf) **Learning the Distribution of Errors in Stereo Matching for Joint Disparity and Uncertainty Estimation** [CVPR 2023]  
[[code]](https://github.com/NVlabs/FoundationStereo) [[paper]](https://arxiv.org/abs/2501.09898) **FoundationStereo: Zero-Shot Stereo Matching** [CVPR 2025]  
[[code]](https://github.com/UCI-ISA-Lab/MultiHeadDepth-HomoDepth) [[paper]](https://arxiv.org/abs/2411.10013) **Efficient Depth Estimation for Unstable Stereo Camera Systems on AR** [CVPR 2025]  
[[code]](https://github.com/bartn8/stereoanywhere) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Iterative_Geometry_Encoding_Volume_for_Stereo_Matching_CVPR_2023_paper.pdf) **Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail** [CVPR 2025]  
[[code]](https://github.com/Insta360-Research-Team/DEFOM-Stereo) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_DEFOM-Stereo_Depth_Foundation_Model_Based_Stereo_Matching_CVPR_2025_paper.pdf) **DEFOM-Stereo: Depth Foundation Model Based Stereo Matching** [CVPR 2025]  
[[code]](https://github.com/Junda24/MonSter) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Cheng_MonSter_Marry_Monodepth_to_Stereo_Unleashes_Power_CVPR_2025_paper.pdf) **MonSter: Marry Monodepth to Stereo Unleashes Power** [CVPR 2025]  




#### Camera pose estimation

> <span style="color:lightblue;">💡💡 Camera pose estimation in computer vision and embodied AI frameworks refers to the inference of extrinsic camera parameters—encompassing 6-DoF rotation and translation vectors—from monocular, stereo, or multi-view image sequences, typically posed as a non-linear least-squares optimization over reprojection residuals in the Perspective-n-Point (PnP) paradigm, augmented by RANSAC for outlier rejection and bundle adjustment via Levenberg-Marquardt solvers to jointly refine sparse 3D landmarks and sequential poses within structure-from-motion (SfM) or simultaneous localization and mapping (SLAM) pipelines; deep learning extensions, such as PoseNet or MapNet, deploy convolutional or transformer-based regressors to directly map RGB inputs to SE(3) Lie group embeddings through self-supervised photometric or geometric consistency losses, incorporating equivariant attention mechanisms to enforce rotational invariance and mitigate scale ambiguities in monocular visual odometry (VO), as exemplified in DROID-SLAM or TartanVO models that fuse learned optical flow with probabilistic Kalman filtering for drift-resistant trajectory estimation, while recent advancements leverage diffusion priors or neural radiance fields (NeRF) for differentiable pose optimization in novel view synthesis tasks, enabling robust relocalization under dynamic occlusions and illumination variances by distilling multi-modal sensor fusion into compact Gaussian splatting representations that bridge 2D observations to Euclidean scene graphs, thereby underpinning scalable world models for autonomous navigation and conditional video generation despite persistent challenges in textureless environments and long-tail generalization.<span> 


[[paper]](https://ieeexplore.ieee.org/document/1315094) **Visual Odometry** [CVPR 2004]  
[[paper]](MonoSLAM: Real-Time Single Camera SLAM) **MonoSLAM: Real-Time Single Camera SLAM** [TPAMI 2007]  
[[paper]](https://ieeexplore.ieee.org/document/4538852) **Parallel Tracking and Mapping for Small AR Workspaces** [ISMAR 2007]  
[[code]](https://github.com/raulmur/ORB_SLAM) [[paper]](https://webdiis.unizar.es/~raulmur/MurMontielTardosTRO15.pdf) **ORB-SLAM** [TPAMI 2025]  
[[code]](https://github.com/raulmur/ORB_SLAM2) [[paper]](https://arxiv.org/pdf/1610.06475) **ORB-SLAM2** [TPAMI 2016]  
[[code]](https://github.com/UZ-SLAMLab/ORB_SLAM3) [[paper]](https://ieeexplore.ieee.org/document/9440682) **ORB-SLAM3** [TRO 2021]  
[[code]](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3) [[paper]](https://openaccess.thecvf.com/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf) **MonSter: Marry Monodepth to Stereo Unleashes Power** [ICCV 2025]  
[[code]](https://github.com/princeton-vl/RAFT) [[paper]](https://arxiv.org/pdf/2003.12039) **RAFT: Recurrent All-Pairs Field Transforms for Optical Flow** [ECCV 2020]  
[[code]](https://github.com/arthurchen0518/DirectionNet) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Wide-Baseline_Relative_Camera_Pose_Estimation_With_Directional_Learning_CVPR_2021_paper.pdf) **Wide-Baseline Relative Camera Pose Estimation with Directional Learning** [CVPR 2021]  
[[code]](https://github.com/princeton-vl/DROID-SLAM) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/89fcd07f20b6785b92134bd6c1d0fa42-Paper.pdf) **DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras** [NeurIPS 2021]  
[[code]](https://github.com/PruneTruong/DenseMatching) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Truong_Learning_Accurate_Dense_Correspondences_and_When_To_Trust_Them_CVPR_2021_paper.pdf) **Learning Accurate Dense Correspondences and When to Trust Them** [CVPR 2021]  
[[code]](https://github.com/ubc-vision/COTR) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_COTR_Correspondence_Transformer_for_Matching_Across_Images_ICCV_2021_paper.pdf) **COTR: Correspondence Transformer for Matching Across Images** [ICCV 2021]  
[[code]](https://github.com/memmelma/VO-Transformer) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Memmel_Modality-Invariant_Visual_Odometry_for_Embodied_Vision_CVPR_2023_paper.pdf) **Modality-Invariant Visual Odometry for Embodied Vision** [CVPR 2023]  
[[code]](https://github.com/wrchen530/leapvo) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_LEAP-VO_Long-term_Effective_Any_Point_Tracking_for_Visual_Odometry_CVPR_2024_paper.pdf) **LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry** [CVPR 2024]  
[[code]](https://github.com/h2xlab/ZeroVO) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lai_ZeroVO_Visual_Odometry_with_Minimal_Assumptions_CVPR_2025_paper.pdf) **ZeroVO: Visual Odometry with Minimal Assumptions** [CVPR 2025]  
[[code]](https://openaccess.thecvf.com/content/CVPR2025/papers/Dong_Reloc3r_Large-Scale_Training_of_Relative_Camera_Pose_Regression_for_Generalizable_CVPR_2025_paper.pdf) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Dong_Reloc3r_Large-Scale_Training_of_Relative_Camera_Pose_Regression_for_Generalizable_CVPR_2025_paper.pdf) **Reloc3r: Large-scale Training of Relative Camera Pose Regression** [CVPR 2025]  



#### SLAM and Global Pose Estimation

[[paper]](https://ieeexplore.ieee.org/document/1238654) **Real-Time Simultaneous Localisation and Mapping with a Single Camera** [ICCV 2001]  
[[code]](https://github.com/raulmur/ORB_SLAM) [[paper]](https://webdiis.unizar.es/~raulmur/MurMontielTardosTRO15.pdf) **ORB-SLAM** [TPAMI 2025]   
[[code]](https://github.com/raulmur/ORB_SLAM2) [[paper]](https://arxiv.org/pdf/1610.06475) **ORB-SLAM2** [TPAMI 2016]  
[[code]](https://github.com/UZ-SLAMLab/ORB_SLAM3) [[paper]](https://ieeexplore.ieee.org/document/9440682) **ORB-SLAM3** [TRO 2021]  
[[code]](https://github.com/princeton-vl/DROID-SLAM) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/89fcd07f20b6785b92134bd6c1d0fa42-Paper.pdf) **DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras** [NeurIPS 2021]  
[[code]](https://github.com/InternLandMark/LandMark) [[paper]](https://arxiv.org/abs/2303.14001) **Grid-guided Neural Radiance Fields for Large Urban Scenes** [CVPR 2023]  
[[code]](https://github.com/facebookresearch/vggsfm) [[paper]](https://arxiv.org/pdf/2312.04563) **VGGSfM: Visual Geometry Grounded Deep Structure From Motion** [CVPR 2024]  
[[code]](https://github.com/frickyinn/SRPose) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10919.pdf) **SRPose: Two-view Relative Pose Estimation with Sparse Keypoints** [ECCV 2024]  
[[code]](https://github.com/rmurai0610/MASt3R-SLAM) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.pdf) **MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors** [CVPR 2025]  
[[paper]](Rockwell_Dynamic_Camera_Poses_and_Where_to_Find_Them_CVPR_2025_paper) **Dynamic Camera Poses and Where to Find Them** [CVPR 2025]  
[[code]](https://github.com/m-kruse98/SplatPose) [[paper]](https://arxiv.org/pdf/2404.06832) **SplatPose & Detect: Pose-Agnostic 3D Anomaly Detection** [IROS 2025]  
[[code]](https://github.com/wenhuiwei-ustc/BotVIO) [[paper]](https://ieeexplore.ieee.org/document/11024235) **BotVIO: A Lightweight Transformer-Based Visual-Inertial Odometry for Robotics** [TRO 2025]  




#### SFM

[[paper]](https://dl.acm.org/doi/10.1145/1141911.1141964) **Photo Tourism: Exploring Photo Collections in 3D** [SIGGRAPH 2006]  
[[paper]](https://ieeexplore.ieee.org/document/5459148) **Building Rome in a Day** [ICCV 2009] 
[[paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf) **Structure-from-Motion Revisited** [CVPR 2016]  
[[paper]](https://openaccess.thecvf.com/content_iccv_2013/papers/Moulon_Global_Fusion_of_2013_ICCV_paper.pdf) **Global Fusion of Relative Motions for Robust, Accurate and Scalable Structure from Motion** [ICCV 2013]  
[[paper]](https://ieeexplore.ieee.org/document/10378404) **Generalized Differentiable RANSAC** [TPAMI 2022]  
[[code]](https://github.com/barbararoessle/e2e_multi_view_matching) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Roessle_End2End_Multi-View_Feature_Matching_with_Differentiable_Pose_Optimization_ICCV_2023_paper.pdf) **End2End Multi-View Feature Matching with Differentiable Pose OptimizationI** [CVPR 2023]  
[[code]](https://github.com/facebookresearch/vggsfm) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_VGGSfM_Visual_Geometry_Grounded_Deep_Structure_From_Motion_CVPR_2024_paper.pdf) **VGGSfM: Visual Geometry Grounded Deep Structure From Motion** [CVPR 2024]  
[[code]](https://github.com/RobustFieldAutonomyLab/CVD-SfM) [[paper]](https://arxiv.org/abs/2508.01936) **CVD-SfM: A Cross-View Deep Front-end Structure-from-Motion** [IROS 2025]  
[[code]](https://selflein.github.io/Light3R/) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Elflein_Light3R-SfM_Towards_Feed-forward_Structure-from-Motion_CVPR_2025_paper.pdf) **Light3R-SfM: Towards Feed-forward Structure-from-Motion** [CVPR 2025]  
[[code]](https://github.com/naver/mast3r) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html) **Grounding Image Matching in 3D with MASt3R** [CVPR 2024]  
[[code]](https://github.com/FadiKhatib/resfm) [[paper]](https://openreview.net/pdf?id=wldwEhQ7cl) **RESfM: Robust Deep Equivariant Structure from Motion** [ICLR 2025]  
[[code]](https://github.com/Ivonne320/GenSfM) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Structure-from-Motion_with_a_Non-Parametric_Camera_Model_CVPR_2025_paper.pdf) **Structure-from-Motion with a Non-Parametric Camera Model** [CVPR 2025]  







### Scene Segmentation
> <span style="color:lightblue;">💡💡 Scene Segmentation in computer vision and generative modeling paradigms delineates the pixel- or voxel-wise partitioning of complex environmental contexts into semantically coherent regions, leveraging deep convolutional or transformer-based architectures such as U-Net variants, Mask R-CNN, or Segment Anything Model (SAM) extensions to extract multi-scale hierarchical features via encoder-decoder backbones augmented with attention-gated skip connections and deformable convolutions that enforce boundary refinement and contextual aggregation, as comprehensively surveyed in recent advancements integrating self-supervised pretraining on large-scale datasets like ADE20K or Cityscapes to mitigate domain gaps and enhance zero-shot generalization across indoor-outdoor scenes. This process encompasses semantic segmentation for categorical labeling of holistic scene components, instance segmentation for individualized object delineation through mask-aware clustering and non-maximum suppression over oriented bounding frustums, and panoptic segmentation unifying both via hybrid losses optimizing cross-entropy for stuff classes and Dice-IoU for thing instances, while emerging video scene parsing frameworks extend temporal consistency through 3D convolutions or recurrent transformers that propagate labels across frame sequences via optical flow-guided message passing and diffusion priors for occluded region imputation, addressing challenges in dynamic motion blur and long-range dependencies as highlighted in holistic reviews of vision tasks encompassing video semantic segmentation, instance segmentation, panoptic segmentation, tracking, and open-vocabulary variants with transformer architectures capturing spatio-temporal contexts. Contemporary innovations further incorporate vision-language models for open-vocabulary prompting, enabling query-adaptive segmentation in unstructured environments through cross-modal alignment and knowledge distillation from foundational models like CLIP or DINO, thereby facilitating downstream applications in autonomous navigation, augmented reality, and conditional scene synthesis by bridging low-level geometric primitives with high-level affordance reasoning despite persistent hurdles in real-time efficiency and rare-class imbalance, with recent surveys underscoring the evolution toward unified multimodal pipelines that fuse RGB-D inputs for robust depth-aware partitioning and deep learning-driven 3D point cloud analysis evaluating traditional-to-modern methods via performance metrics to guide application-specific selections.<span> 


#### Semantic segmentation, instance segmentation, panoramic segmentation

[[code]](https://github.com/charlesq34/pointnet.git) [[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)  **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation** [CVPR 2017]  
[[code]](https://github.com/charlesq34/pointnet2.git) [[paper]](https://dl.acm.org/doi/abs/10.5555/3295222.3295263) **PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space**  [NIPS 2017]  
[[code]](https://github.com/collector-m/VoxelNet_CVPR_2018_PointCloud) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.pdf)  **VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection** [NeurIPS 2018]  
[[code]](https://github.com/yangyanli/PointCNN) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2018/file/f5f8590cd58a54e94377e6ae2eded4d9-Paper.pdf) **PointCNN: Convolution On X-Transformed Points**  [NIPS 2018]  
[[code]](https://github.com/WangYueFt/dgcnn.git) [[paper]](https://dl.acm.org/doi/abs/10.1145/3326362) **Dynamic Graph CNN for Learning on Point Clouds** [TOG 2019]  
[[code]](https://github.com/HuguesTHOMAS/KPConv.git) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Thomas_KPConv_Flexible_and_Deformable_Convolution_for_Point_Clouds_ICCV_2019_paper.pdf) **KPConv: Flexible and Deformable Convolution for Point Clouds** [ICCV 2019]  
[[code]](https://github.com/Sekunde/3D-SIS) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_3D-SIS_3D_Semantic_Instance_Segmentation_of_RGB-D_Scans_CVPR_2019_paper.pdf)  **3D-SIS: 3D Semantic Instance Segmentation of RGB-D Scans** [CVPR 2019]  
[[code]](https://github.com/QingyongHu/RandLA-Net.git) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_RandLA-Net_Efficient_Semantic_Segmentation_of_Large-Scale_Point_Clouds_CVPR_2020_paper.pdf) **RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds** [CVPR 2020]  
[[code]](https://github.com/hszhao/PointWeb) [[paper]](https://llijiang.github.io/papers/cvpr19_pointweb.pdf) **ointWeb: Enhancing Local Neighborhood Features for Point Cloud Processing**  [CVPR 2020]  
[[code]](https://github.com/POSTECH-CVLab/point-transformer.git) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf) **Point Transformer** [ICCV 2021]  
[[code]](https://github.com/Pointcept/PointTransformerV2.git) [[paper]](https://papers.nips.cc/paper_files/paper/2022/hash/d78ece6613953f46501b958b7bb4582f-Abstract-Conference.html) **Point Transformer V2: Grouped Vector Attention and Partition-based Pooling** [NIPS 2022]  
[[code]](https://github.com/Pointcept/PointTransformerV3.git) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf) **Point Transformer V3: Simpler, Faster, Stronger** [CVPR 2024]  
[[code]](https://github.com/valeoai/rangevit) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Ando_RangeViT_Towards_Vision_Transformers_for_3D_Semantic_Segmentation_in_Autonomous_CVPR_2023_paper.pdf)  **RangeViT: Towards Vision Transformers for 3D Semantic Segmentation** [CVPR 2023]  
[[code]](https://github.com/laughtervv/SGPN) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_SGPN_Similarity_Group_CVPR_2018_paper.pdf)  **SGPN:Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation** [CVPR 2018]  
[[code]](https://github.com/Sekunde/3D-SIS) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_3D-SIS_3D_Semantic_Instance_Segmentation_of_RGB-D_Scans_CVPR_2019_paper.pdf)  **3D-SIS: 3D Semantic Instance Segmentation of RGB-D Scans** [CVPR 2019]  
[[code]](https://github.com/Yang7879/3D-BoNet) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2019/file/d0aa518d4d3bfc721aa0b8ab4ef32269-Paper.pdf)  **Unified-Lift: Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting** [NeurIPS 2019]  
[[code]](https://github.com/OpenMask3D/openmask3d) [[paper]](https://papers.nips.cc/paper_files/paper/2023/file/d77b5482e38339a8068791d939126be2-Paper-Conference.pdf)  **Unified-Lift: Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting** [NeurIPS 2023]  
[[code]](https://github.com/yashbhalgat/Contrastive-Lift) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1cb5b3d64bdf3c6642c8d9a8fbecd019-Abstract-Conference.html)  **Contrastive Lift: 3D Object Instance Segmentation by Slow-Fast Contrastive Fusion** [NeurIPS 2023]  
[[code]](https://github.com/aminebdj/3D-OWIS) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/801750bc49fdc3d498e9ee63479f315e-Abstract-Conference.html)  **3D Indoor Instance Segmentation in an Open-World** [NeurIPS 2023]  
[[code]](https://github.com/Jumpat/SegmentAnythingin3D) [[paper]](https://arxiv.org/abs/2304.12308)  **SA3D: Segment Anything in 3D with Radiance Fields** [NeurIPS 2023]  
[[code]](https://github.com/IRMVLab/SNI-SLAM) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_SNI-SLAM_Semantic_Neural_Implicit_SLAM_CVPR_2024_paper.pdf)  **SNI-SLAM: Semantic Neural Implicit SLAM** [CVPR 2024]  
[[code]](https://github.com/Pointcept/Pointcept) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Peng_OA-CNNs_Omni-Adaptive_Sparse_CNNs_for_3D_Semantic_Segmentation_CVPR_2024_paper.pdf)  **OA-CNNs: Omni-Adaptive Sparse CNNs for 3D Semantic Segmentation** [CVPR 2024]  
[[code]](https://github.com/rozdavid/unscene3d) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Rozenberszki_UnScene3D_Unsupervised_3D_Instance_Segmentation_for_Indoor_Scenes_CVPR_2024_paper.pdf)  **UnScene3D: Unsupervised 3D Instance Segmentation for Indoor Scenes** [CVPR 2024]  
[[code]](https://github.com/SooLab/Part2Object) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02657.pdf)  **Part2Object: Hierarchical Unsupervised 3D Instance Segmentation** [ECCV 2024]  
[[code]](https://github.com/RyanG41/SA3DIP) [[paper]](https://arxiv.org/abs/2411.03819)  **SA3DIP: Segment Any 3D Instance with Potential 3D Priors** [NeurIPS 2024]  
[[code]](https://github.com/apple/ml-kpconvx) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Thomas_KPConvX_Modernizing_Kernel_Point_Convolution_with_Kernel_Attention_CVPR_2024_paper.pdf)  **KPConvX: Modernizing Kernel Point Convolution with Kernel Attention** [CVPR 2024]  
[[code]](https://github.com/weiguangzhao/BFANet) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_BFANet_Revisiting_3D_Semantic_Segmentation_with_Boundary_Feature_Analysis_CVPR_2025_paper.pdf)  **BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis** [CVPR 2025]  
[[code]](https://github.com/DP-Recon/DP-Recon) [[paper]](https://arxiv.org/abs/2503.14830)  **Decompositional Neural Scene Reconstruction with Generative Diffusion Prior** [CVPR 2025]  
[[code]](https://kuai-lab.github.io/cvpr2025protoocc/) [[paper]](https://arxiv.org/abs/2503.15185)  **3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation** [CVPR 2025]  
[[code]](https://github.com/TyroneLi/CUA_O3D) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Cross-Modal_and_Uncertainty-Aware_Agglomeration_for_Open-Vocabulary_3D_Scene_Understanding_CVPR_2025_paper.pdf)  **Cross-Modal and Uncertainty-Aware Agglomeration for Open-Vocabulary 3D Scene Understanding** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf)  **Panoptic Segmentation** [CVPR 2019]  
[[code]](https://github.com/bowenc0221/panoptic-deeplab) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Panoptic-DeepLab_A_Simple_Strong_and_Fast_Baseline_for_Bottom-Up_Panoptic_CVPR_2020_paper.pdf)  **Panoptic-DeepLab: A Simple, Strong Baseline** [CVPR 2020]  
[[code]](https://github.com/DeepSceneSeg/EfficientPS) [[paper]](https://arxiv.org/abs/2004.02307)  **EfficientPS: Efficient Panoptic Segmentation** [IJCV 2021]  
[[code]](https://github.com/drprojects/superpoint_transformer) [[paper]](https://arxiv.org/abs/2306.08045)  **Efficient 3D Semantic Segmentation with Superpoint Transformer** [ICCV 2023]  
[[code]](https://github.com/astra-vision/PaSCo) [[paper]](https://arxiv.org/abs/2312.02158)  **PaSCo: Urban 3D Panoptic Scene Completion with Uncertainty Awareness** [CVPR 2024]  
[[code]](https://github.com/visinf/cups) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Hahn_Scene-Centric_Unsupervised_Panoptic_Segmentation_CVPR_2025_paper.pdf)  **Scene-Centric Unsupervised Panoptic Segmentation** [CVPR 2025]  
[[paper]](https://arxiv.org/pdf/2506.21348)  **PanSt3R: Multi-view Consistent Panoptic Segmentation** [ICCV 2025]  

[[code]](https://github.com/Harry-Zhi/semantic_nerf) [[paper]](https://arxiv.org/abs/2103.15875)  **Semantic-NeRF: Semantic Neural Radiance Fields** [ICCV 2021]  
[[code]](https://github.com/zyqz97/GP-NeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_GP-NeRF_Generalized_Perception_NeRF_for_Context-Aware_3D_Scene_Understanding_CVPR_2024_paper.pdf)  **GP-NeRF: Generalized Perception NeRF for Context-Aware 3D Scene Understanding** [CVPR 2023]  
[[code]](https://github.com/kerrj/lerf) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Kerr_LERF_Language_Embedded_Radiance_Fields_ICCV_2023_paper.pdf)  **LERF: Language Embedded Radiance Fields** [ICCV 2023]  
[[code]](https://github.com/oppo-us-research/NeuRBF) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_NeuRBF_A_Neural_Fields_Representation_with_Adaptive_Radial_Basis_Functions_ICCV_2023_paper.pdf)  **NeurBF: A Neural Fields Representation with Adaptive Radial Basis Functions** [ICCV 2023]  
[[code]](https://github.com/Jumpat/SegmentAnythingin3D) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/525d24400247f884c3419b0b7b1c4829-Paper-Conference.pdf)  **Segment Anything in 3D with NeRFs** [NeurIPS 2023]  
[[code]](https://github.com/pcl3dv/OV-NeRF) [[paper]](https://ieeexplore.ieee.org/document/10630553)  **OV-NeRF: Open-Vocabulary Neural Radiance Fields with Vision and Language Foundation Models for 3D Semantic Understanding** [IEEE TCSVT 2023]  
[[code]](https://github.com/ZechuanLi/GO-N3RDet) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_GO-N3RDet_Geometry_Optimized_NeRF-enhanced_3D_Object_Detector_CVPR_2025_paper.pdf)  **GO-N3RDet: Geometry Optimized NeRF-enhanced 3D Object Detector** [CVPR 2025]  
[[code]](https://github.com/IRMVLab/SNI-SLAM) [[paper]](https://arxiv.org/pdf/2311.11016)  **SNI-SLAM: Semantic Neural Implicit SLAM** [CVPR 2024]  
[[code]](https://github.com/DP-Recon/DP-Recon) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Ni_Decompositional_Neural_Scene_Reconstruction_with_Generative_Diffusion_Prior_CVPR_2025_paper.pdf) [[project]](https://dp-recon.github.io/) **Decompositional Neural Scene Reconstruction with Generative Diffusion Prior** [CVPR 2025]  
[[code]](https://github.com/mRobotit/Cues3D) [[paper]](https://www.sciencedirect.com/science/article/pii/S1566253525002374)  **Cues3D: Unleashing the power of sole NeRF for consistent and unique 3D instance segmentation** [Information Fusion 2025]  


[[code]](https://github.com/minghanqin/LangSplat) [[paper]](https://arxiv.org/pdf/2312.16084) [[project]](https://langsplat.github.io/) **LangSplat: 3D Language Gaussian Splatting** [CVPR 2024]  
[[code]](https://github.com/THU-luvision/OmniSeg3D) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Ying_OmniSeg3D_Omniversal_3D_Segmentation_via_Hierarchical_Contrastive_Learning_CVPR_2024_paper.pdf) **OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning** [CVPR 2024]  
[[code]](https://github.com/ShijieZhou-UCLA/feature-3dgs) [[paper]](https://arxiv.org/abs/2312.03203) [[project]](https://feature-3dgs.github.io/) **Feature 3DGS:Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields** [CVPR 2024]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025W/OpenSUN3D/papers/Wiedmann_DCSEG_Decoupled_3D_Open-Set_Segmentation_using_Gaussian_Splatting_CVPRW_2025_paper.pdf) **DCSEG:Decoupled 3D Open-Set Segmentation using Gaussian Splatting** [CVPR 2025]  
[[code]](https://github.com/XuHu0529/SAGS) [[paper]](https://arxiv.org/pdf/2401.17857) **SAGD: Boundary-Enhanced Segment Anything in 3D Gaussian via Gaussian Decomposition** [arXiv 2024]  
[[paper]](https://arxiv.org/abs/2404.14249) **CLIP-GS: CLIP-Informed Gaussian Splatting for Real-time and View-consistent 3D Semantic Understanding** [ECCV 2024]  
[[code]](https://github.com/wxrui182/GSemSplat) [[paper]](https://arxiv.org/abs/2412.16932)  **GSemSplat: Generalizable Semantic 3D Gaussian Splatting from Uncalibrated Image Pairs** [arXiv 2024]  
[[code]](https://github.com/sharinka0715/semantic-gaussians) [[paper]](https://arxiv.org/pdf/2403.15624)  **Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting** [ECCV 2024]  
[[code]](https://github.com/lifuguan/LangSurf) [[paper]](https://arxiv.org/pdf/2412.17635) [[project]](https://langsurf.github.io/) **LangSurf: Language-Embedded Surface Gaussians for 3D Scene Understanding** [arXiv 2025]  
[[paper]](https://arxiv.org/pdf/2412.00392)  **GradiSeg: Gradient-Guided Gaussian Segmentation with Enhanced 3D Boundary Precision** [arXiv 2024]  
[[code]](https://github.com/HorizonRobotics/GLS) [[paper]](https://arxiv.org/pdf/2411.18066)  **GLS: Geometry-aware 3D Language Gaussian Splatting** [arXiv 2024]  
[[code]](https://github.com/weijielyu/Gaga) [[paper]](https://arxiv.org/abs/2404.07977)  **Gaga: Group Any Gaussians via 3D-aware Memory Bank** [arXiv 2024]  
[[code]](https://github.com/WHU-USI3DV/GAGS) [[paper]](https://arxiv.org/abs/2412.13654)  **GAGS: Granularity-Aware Feature Distillation for Language Gaussian Splatting** [arXiv 2024]  
[[code]](https://github.com/insait-institute/OccamLGS) [[paper]](https://arxiv.org/abs/2412.01807)  **Occam's LGS: A Simple Approach for Language Gaussian Splatting** [arXiv 2024]  
[[code]](https://github.com/buaavrcg/LEGaussians) [[paper]](https://openaccess.thecvf.com/content/WACV2025W/LLVMAD/papers/Chahe_Query3D_LLM-Powered_Open-Vocabulary_Scene_Segmentation_with_Language_Embedded_3D_Gaussians_WACVW_2025_paper.pdf)  **Query3D: LLM-Powered Open-Vocabulary Scene Segmentation with Language Embedded 3D Gaussian** [CVPR 2024]  
[[paper]](https://arxiv.org/abs/2506.09565)  **SemanticSplat: Feed-Forward 3D Scene Understanding with Semantic Gaussians** [Arxiv 2025]  
[[code]](https://github.com/zhaihongjia/PanoGS) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhai_PanoGS_Gaussian-based_Panoptic_Segmentation_for_3D_Open_Vocabulary_Scene_Understanding_CVPR_2025_paper.pdf)  **PanoGS: Gaussian-based Panoptic Segmentation for 3D Open Vocabulary Scene Understanding** [CVPR 2025]  
[[code]](https://github.com/BITyia/DroneSplat) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_DroneSplat_3D_Gaussian_Splatting_for_Robust_3D_Reconstruction_from_In-the-Wild_CVPR_2025_paper.pdf)  **DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery** [CVPR 2025]  
[[code]](https://github.com/Zhao-Yian/iSegMan) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_iSegMan_Interactive_Segment-and-Manipulate_3D_Gaussians_CVPR_2025_paper.html)  **iSegMan: Interactive Segment-and-Manipulate 3D Gaussians** [CVPR 2025]  
[[code]](https://github.com/JojiJoseph/3dgs-gradient-segmentation) [[paper]](https://arxiv.org/abs/2409.11681)  **Gradient-Driven 3D Segmentation and Affordance Transfer in Gaussian Splatting Using 2D Masks** [arXiv 2024]  
[[code]](https://github.com/Runsong123/Unified-Lift) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Rethinking_End-to-End_2D_to_3D_Scene_Segmentation_in_Gaussian_Splatting_CVPR_2025_paper.pdf)  **Unified-Lift: Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting** [CVPR 2025]  
[[code]](https://github.com/RuijieZhu94/ObjectGS) [[paper]](https://arxiv.org/pdf/2507.15454)  **ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting** [ICCV 2025]  
[[code]](https://github.com/mlzxy/motion-blender-gs) [[paper]](https://arxiv.org/abs/2503.09040)  **Motion Blender Gaussian Splatting for Dynamic Scene Reconstruction** [CoRL 2025]  
[[code]](https://github.com/uynitsuj/pogs) [[paper]](https://arxiv.org/abs/2503.05189)  **Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot Manipulation of Irregularly Shaped Objects** [arXiv 2025]  
[[code]](https://github.com/umangi-jain/gaussiancut) [[paper]](https://openreview.net/pdf?id=Ns0LQokxa5) [[project]](https://umangi-jain.github.io/gaussiancut/) **GaussianCut: Interactive segmentation via graph cut for 3D Gaussian Splatting** [NeurIPS 2024]  
[[paper]](https://arxiv.org/pdf/2412.10231)  **SuperGSeg: Open-Vocabulary 3D Segmentation with Structured Super-Gaussians** [arXiv 2024]  
[[code]](https://github.com/MyrnaCCS/contrastive-gaussian-clustering) [[paper]](https://arxiv.org/abs/2404.12784)  **Contrastive Gaussian Clustering: Weakly Supervised 3D Scene Segmentation** [ICPR 2024]  
[[code]](https://any3dis.github.io/) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_Any3DIS_Class-Agnostic_3D_Instance_Segmentation_by_2D_Mask_Tracking_CVPR_2025_paper.pdf)  **Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking** [CVPR 2025]  
[[code]](https://github.com/GAP-LAB-CUHK-SZ/SAMPro3D) [[paper]](https://ieeexplore.ieee.org/document/11125552)  **Locating SAM Prompts in 3D for Zero-Shot Instance Segmentation** [3DV 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Rethinking_End-to-End_2D_to_3D_Scene_Segmentation_in_Gaussian_Splatting_CVPR_2025_paper.pdf)  **Lifting by Gaussians** [WACV 2025]  
[[code]](https://github.com/zhaihongjia/PanoGS) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhai_PanoGS_Gaussian-based_Panoptic_Segmentation_for_3D_Open_Vocabulary_Scene_Understanding_CVPR_2025_paper.pdf)  **PanoGS: Gaussian-based Panoptic Segmentation for 3D Open Vocabulary Scene Understanding** [CVPR 2025]  
[[code]](https://github.com/wangyuyy/PLGS) [[paper]](https://arxiv.org/pdf/2410.17505v2)  **PLGS: Robust Panoptic Lifting with 3D Gaussian Splatting** [TIP 2025]  



#### 3D Object Detection & Recognition

> <span style="color:lightblue;">💡💡 3D Object Detection & Recognition in cluttered scenes involves the localization and categorical identification of 3D bounding boxes or oriented frustums from multi-modal inputs like fused LiDAR-camera point-voxel hybrids, deploying anchor-free regressors or center-point transformers pretrained on datasets such as nuScenes or KITTI to predict yaw-rotated cuboids via heatmap regression and offset refinement, with vision-language models (VLMs) enabling zero-shot recognition through cross-modal alignment of textual queries to geometric descriptors in BEV (bird's-eye-view) or voxelized embeddings. This paradigm leverages synergistic sensor fusion via attention-gated bilinear pooling and uncertainty-aware Kalman filtering to counter noise in adverse weather, as reviewed in deep learning surveys emphasizing end-to-end differentiability for joint detection-recognition pipelines that incorporate NeRF-based priors for novel view extrapolation and long-tail class balancing through generative augmentation, fostering advancements in autonomous driving perception where multi-task losses over classification, regression, and segmentation unify low-level feature extraction with high-level affordance reasoning, despite challenges in computational efficiency and domain adaptation addressed by federated learning in edge-deployed systems. <span>  

[[paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Chen_Monocular_3D_Object_CVPR_2016_paper.pdf)  **Monocular 3D Object Detection for Autonomous Driving** [CVPR 2016]  
[[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liang_Multi-Task_Multi-Sensor_Fusion_for_3D_Object_Detection_CVPR_2019_paper.pdf)  **Multi-Fusion: Robust Real-Time 3D Object Detection** [CVPR 2028]  
[[code]](https://github.com/mileyan/pseudo_lidar) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Pseudo-LiDAR_From_Visual_Depth_Estimation_Bridging_the_Gap_in_3D_CVPR_2019_paper.pdf)  **Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving** [CVPR 2019]  
[[code]](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Stereo_R-CNN_Based_3D_Object_Detection_for_Autonomous_Driving_CVPR_2019_paper.pdf)  **Stereo R-CNN: Two-Stage 3D Object Detection** [CVPR 2019]  
[[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_MonoPair_Monocular_3D_Object_Detection_Using_Pairwise_Spatial_Relationships_CVPR_2020_paper.pdf)  **MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships** [CVPR 2020]  
[[code]](https://github.com/TRAILab/CaDDN) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Reading_Categorical_Depth_Distribution_Network_for_Monocular_3D_Object_Detection_CVPR_2021_paper.pdf)  **CaDDN: Categorical Depth Distribution Network for Monocular 3D Object Detection** [CVPR 2021]  
[[code]](https://github.com/zion-king/Center-based-3D-Object-Detection-and-Tracking) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf)  **Center-Based 3D Object Detection and Tracking** [CVPR 2021]  
[[code]](https://github.com/lifuguan/GP-NeRF) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_GP-NeRF_Generalized_Perception_NeRF_for_Context-Aware_3D_Scene_Understanding_CVPR_2024_paper.pdf)  **GP-NeRF: Generalized Perception NeRF for Context-Aware 3D Scene Understanding** [CVPR 2023]  
[[code]](https://github.com/yangtiming/ImOV3D) [[paper]](https://arxiv.org/abs/2410.24001)  **ImOV3D: Learning Open Vocabulary Point Clouds 3D Object Detection** [NeurIPS 2024]  
[[code]](https://github.com/RuiyuM/STONE) [[paper]](https://arxiv.org/abs/2410.03918)  **STONE: A Submodular Optimization Framework for Active 3D Object Detection** [NeurIPS 2024]  
[[code]](https://github.com/ZechuanLi/GO-N3RDet) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_GO-N3RDet_Geometry_Optimized_NeRF-enhanced_3D_Object_Detector_CVPR_2025_paper.pdf)  **GO-N3RDet: Geometry Optimized NeRF-enhanced 3D Object Detector** [CVPR 2025]  
[[code]](https://github.com/suhaisheng/UniMamba) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Jin_UniMamba_Unified_Spatial-Channel_Representation_Learning_with_Group-Efficient_Mamba_for_LiDAR-based_CVPR_2025_paper.pdf)  **UniMamba: Unified Spatial-Channel Representation Learning with Group-Efficient Mamba for LiDAR-based 3D Object Detection** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_ROD-MLLM_Towards_More_Reliable_Object_Detection_in_Multimodal_Large_Language_CVPR_2025_paper.pdf)  **ROD-MLLM: Towards More Reliable Object Detection in Multimodal Large Language Models** [CVPR 2025]  
[[code]](https://github.com/Say2L/FSHNet) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_FSHNet_Fully_Sparse_Hybrid_Network_for_3D_Object_Detection_CVPR_2025_paper.pdf)  **FSHNet: Fully Sparse Hybrid Network for 3D Object Detection** [CVPR 2025]  
[[code]](https://github.com/yjnanan/PO3AD) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_PO3AD_Predicting_Point_Offsets_toward_Better_3D_Point_Cloud_Anomaly_CVPR_2025_paper.pdf)  **PO3AD: Predicting Point Offsets toward Better 3D Point Cloud Anomaly Detection** [CVPR 2025]  
[[code]](https://github.com/xmuqimingxia/DOtAv2) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xia_Learning_to_Detect_Objects_from__Multi-Agent_LiDAR_Scans_without_CVPR_2025_paper.pdf)  **Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels** [CVPR 2025]  








### Scene Reasoning and Understanding  

> <span style="color:lightblue;">💡💡 Scene Reasoning and Understanding in computer vision, embodied AI, and generative modeling paradigms encompasses the higher-order probabilistic inference of causal dependencies, affordance relations, and counterfactual dynamics within spatio-temporally structured environments, leveraging neuro-symbolic architectures that synergistically fuse vision-language transformers (e.g., CLIP or Flamingo variants) with differentiable inductive logic programming (ILP) or graph neural networks (GNNs) to distill multi-modal sensor data—spanning RGB-D point clouds, LiDAR voxels, and egocentric video streams—into executable scene graphs encoding hierarchical object interactions, temporal event chains, and physical priors via variational message passing on factor graphs that optimize joint likelihoods over observed trajectories and latent counterfactuals, as surveyed in recent advancements integrating self-supervised contrastive learning on datasets like CLEVRER or ARC for emergent reasoning capabilities in compositional scenes. This process extends beyond pixel-level segmentation to relational grounding through open-vocabulary prompting mechanisms that align textual commonsense queries with geometric embeddings in SE(3)-equivariant latent spaces, employing diffusion-based trajectory forecasting or reinforcement learning from human feedback (RLHF) to simulate multi-agent interactions and anticipate occlusion-resolved outcomes, while Bayesian network approximations enforce causal invariance under viewpoint shifts and domain shifts, mitigating hallucinations in generative priors like those in World Models or Sora extensions. Contemporary frameworks further incorporate meta-learning for few-shot adaptation, distilling relational inductive biases from large-scale video corpora to enable zero-shot planning in novel scenarios, addressing challenges in long-horizon reasoning and ambiguity resolution through hybrid symbolic-neural verification pipelines that bridge low-level perceptual primitives with high-level deliberative cognition, thereby underpinning scalable applications in autonomous robotics, interactive virtual agents, and conditional scene synthesis despite persistent hurdles in computational scalability and ethical alignment of inferred world models. <span>  



#### Scene relationship modeling  

> <span style="color:lightblue;">💡💡 Scene relationship modeling in computer vision and generative AI paradigms primarily refers to the extraction of spatial topology, semantic associations, and causal dependencies among objects within a scene from multimodal sensor data such as point clouds and RGB-D images. This is achieved through equivariant graph neural networks (EGNNs) or Transformer-based relational inference modules. and causal dependencies within scenes. This achieves optimized mapping from pixel-level geometric embeddings to higher-order relational graphs. By employing variational message passing to jointly maximize observed trajectories and latent interaction likelihoods on factor graphs, while incorporating contrastive self-supervised pretraining (e.g., Sceneverse framework) to capture long-range dependencies and scale invariance. This supports progressive refinement of global consistency in 3D scene generation and novel view synthesis tasks. Despite challenges posed by relational ambiguity in sparse data, recent trends evolve toward knowledge distillation-driven unified world models, bridging low-level perceptual primitives with high-level behavioral planning. <span>  


[[code]](https://github.com/evelinehong/3D-CLR-Official) [[paper]](https://arxiv.org/abs/2303.11327)  **3D Concept Learning and Reasoning From Multi-View Images** [CVPR 2023]  
[[code]](https://github.com/UMass-Embodied-AGI/3D-LLM) [[paper]](https://arxiv.org/abs/2307.12981)  **3D-LLM: Injecting the 3D World into Large Language Models** [NeurIPS 2023]  
[[code]](https://github.com/embodied-generalist/embodied-generalist) [[paper]](https://arxiv.org/abs/2311.12871)  **LEO: An Embodied Generalist Agent in 3D World. Authors** [ICML 2024]  
[[code]](https://github.com/3d-vista/3D-VisTA) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_3D-VisTA_Pre-trained_Transformer_for_3D_Vision_and_Text_Alignment_ICCV_2023_paper.pdf)  **3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment** [ ICCV 2023]  
[[code]](https://github.com/yuhuan-wu/P2T) [[paper]](https://mmcheng.net/wp-content/uploads/2022/09/22TPAMI-P2T.pdf)  **Pyramid Pooling Transformer for Scene Understanding** [TPAMI 2025]  
[[code]](https://github.com/visinf/veto) [[paper]](Sudhakaran_Vision_Relation_Transformer_for_Unbiased_Scene_Graph_Generation_ICCV_2023_paper)  **Vision Relation Transformer for Unbiased Scene Graph Generation.** [ICCV 2023]  
[[code]](https://github.com/guikunchen/SDSGG) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/ee74a6ade401e200985e2421b20bbae4-Paper-Conference.pdf)  **Scene Graph Generation with Role-Playing Large Language Models.** [NeurIPS 2024]  
[[code]](https://github.com/xmuqimingxia/DOtAv2) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xia_Learning_to_Detect_Objects_from__Multi-Agent_LiDAR_Scans_without_CVPR_2025_paper.pdf)  **DOtA++ & DOtA : Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels** [CVPR 2024]  
[[code]](https://github.com/naver-ai/egtr) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Im_EGTR_Extracting_Graph_from_Transformer_for_Scene_Graph_Generation_CVPR_2024_paper.pdf)  **EGTR: Extracting Graph from Transformer for Scene Graph Generation** [CVPR 2024]  
[[code]](https://github.com/bagh2178/SG-Nav) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html)  **SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation** [NeurIPS 2024]  
[[code]](https://github.com/MSR3D/MSR3D) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/feaeec8ec2d3cb131fe18517ff14ec1f-Paper-Datasets_and_Benchmarks_Track.pdf)  **Multi-modal Situated Reasoning in 3D Scenes.** [NeurIPS 2024]  
[[code]](https://github.com/ai4ce/MSG) [[paper]](https://arxiv.org/abs/2410.11187)  **Multiview Scene Graph.** [NeurIPS 2024]  
[[code]](https://github.com/ZhangLab-DeepNeuroCogLab/CSEGG) [[paper]](https://arxiv.org/pdf/2310.01636)  **Adaptive Visual Scene Understanding: Incremental Learning in Scene Graph Generation** [NeurIPS 2024]  
[[code]](https://github.com/ChocoWu/USG) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Universal_Scene_Graph_Generation_CVPR_2025_paper.pdf)  **Universal Scene Graph Generation.** [CVPR 2024]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xia_Learning_to_Detect_Objects_from__Multi-Agent_LiDAR_Scans_without_CVPR_2025_paper.pdf)  **Multi-Modal Data-Efficient 3D Scene Understanding for Autonomous Driving.** [TPAMI 2025]  
[[code]](https://github.com/naver-ai/egtr) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Im_EGTR_Extracting_Graph_from_Transformer_for_Scene_Graph_Generation_CVPR_2024_paper.pdf)  **EGTR:Extracting Graph from Transformer for Scene Graph Generation** [CVPR 2024]  
[[code]](https://github.com/kagawa588/DiffVsgg) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_DiffVsgg_Diffusion-Driven_Online_Video_Scene_Graph_Generation_CVPR_2025_paper.pdf)  **DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation.** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Unbiased_Video_Scene_Graph_Generation_via_Visual_and_Semantic_Dual_CVPR_2025_paper.pdf)  **Unbiased Video Scene Graph Generation via Visual and Semantic Dual.** [CVPR 2025]  
[[code]](https://github.com/ZhangCYG/OpenFunGraph) [[paper]](https://arxiv.org/abs/2503.19199)  **Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces.** [CVPR 2025]  
[[code]](https://github.com/NVlabs/ArtiScene) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Gu_ArtiScene_Language-Driven_Artistic_3D_Scene_Generation_Through_Image_Intermediary_CVPR_2025_paper.pdf)  **ArtiScene: Language-Driven Artistic 3D Scene Generation Through Image Intermediary.** [CVPR 2025]  
[[code]](https://github.com/DP-Recon/DP-Recon) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Ni_Decompositional_Neural_Scene_Reconstruction_with_Generative_Diffusion_Prior_CVPR_2025_paper.pdf)  **Decompositional Neural Scene Reconstruction with Generative Diffusion Prior.** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Fan_Scene_Map-based_Prompt_Tuning_for_Navigation_Instruction_Generation_CVPR_2025_paper.pdf)  **Scene Map-based Prompt Tuning for Navigation Instruction Generation.** [CVPR 2025]  
[[code]](https://diagramagent.github.io/) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wei_From_Words_to_Structured_Visuals_A_Benchmark_and_Framework_for_CVPR_2025_paper.pdf)  **From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation.** [CVPR 2025]  
[[code]](https://motion-prompting.github.io/) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Geng_Motion_Prompting_Controlling_Video_Generation_with_Motion_Trajectories_CVPR_2025_paper.pdf)  **Motion Prompting: Controlling Video Generation with Motion Trajectories.** [CVPR 2025]  
[[code]](https://github.com/LeiyiHU/mona) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_5100_Breaking_Performance_Shackles_of_Full_Fine-Tuning_on_Visual_Recognition_CVPR_2025_paper.pdf)  **5%>100%: Breaking Performance Shackles of Full Fine-Tuning on Visual Recognition Tasks** [CVPR 2025]  
[[code]](https://github.com/hanyang-21/VideoScene) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VideoScene_Distilling_Video_Diffusion_Model_to_Generate_3D_Scenes_in_CVPR_2025_paper.pdf)  **VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step.** [CVPR 2025]  
[[code]](https://github.com/kagawa588/DiffVsgg) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_DiffVsgg_Diffusion-Driven_Online_Video_Scene_Graph_Generation_CVPR_2025_paper.pdf)  **DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation.** [CVPR 2025]  
[[code]](https://github.com/Hoyyyaard/LSceneLLM) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhi_LSceneLLM_Enhancing_Large_3D_Scene_Understanding_Using_Adaptive_Visual_Preferences_CVPR_2025_paper.pdf)  **LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preservations.** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Unbiased_Video_Scene_Graph_Generation_via_Visual_and_Semantic_Dual_CVPR_2025_paper.pdf)  **Unbiased Video Scene Graph Generation via Visual and Semantic Dual.** [CVPR 2025]  




#### Behavior interaction recognition  

> <span style="color:lightblue;">💡💡 Behavioral interaction recognition, within the framework of behavioral AI and computer vision, specifically refers to inferring atomic action sequences, collaborative patterns, and intent alignment from video sequences or multi-agent trajectories through spatio-temporal graph embeddings. This is achieved by extracting dynamic features from RGB frames using multi-scale large kernel convolution modules (MLKCM) or attention-guided optical flow networks, enabling end-to-end classification and localization optimization. Examples include the real-time deployment of YOLO variants for classroom behavior detection. MLKCM or attention-guided optical flow networks to extract dynamic features from RGB frames, enabling end-to-end classification and localization optimization. This is exemplified by the real-time deployment of YOLO variants in classroom behavior detection. This paradigm leverages contrastive learning losses and knowledge distillation to mitigate labeled data scarcity and domain transfer challenges, ensuring robustness and low-latency inference in crowded scenes. It further extends to behavioral AI impact assessment by bridging object detection to contextual prediction through meaningful event pattern recognition. In autonomous robotics and surveillance applications, it integrates uncertainty-aware Kalman filtering to handle occlusions and noise variations. While computational overhead remains a bottleneck, the 2025 trend emphasizes federated learning and edge deployment to enhance interaction generalization. <span>  


[[paper]](https://www.nature.com/articles/s41467-023-43156-8)  **Relational Visual Representations Underlie Human Social Interaction Recognition.** [Nature Communications 2023]  
[[code]](https://github.com/MCG-NJU/SportsHHI) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_SportsHHI_A_Dataset_for_Human-Human_Interaction_Detection_in_Sports_Videos_CVPR_2024_paper.pdf)  **SportsHHI: A Dataset for Human-Human Interaction Detection in Sports Videos.** [CVPR 2024]  
[[paper]](https://arxiv.org/pdf/2410.20155)  **Human-Object Interaction Detection Collaborated with Large Relation-driven Diffusion Models** [NeurIPS 2024]  
[[code]](https://sirui-xu.github.io/InterDreamer/) [[paper]](https://arxiv.org/pdf/2403.19652)  **InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction.** [NeurIPS 2024]  
[[code]](https://github.com/jyuntins/harmony4d) [[paper]](https://arxiv.org/abs/2410.20294)  **Harmony4D: A Video Dataset for In-The-Wild Close Human... Authors: Not specified.** [NeurIPS 2024]  
[[code]](https://juzezhang.github.io/HOIM3_ProjectPage/) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_HOI-M3_Capture_Multiple_Humans_and_Objects_Interaction_within_Contextual_Environment_CVPR_2024_paper.pdf)  **HOI-M3: Capture Multiple Humans and Objects Interaction within...** [CVPR 2024]  
[[code]](https://github.com/AfterJourney00/IMHD-Dataset) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_IM_HOI_Inertia-aware_Monocular_Capture_of_3D_Human-Object_Interactions_CVPR_2024_paper.pdf)  ** I’M HOI: Inertia-aware Monocular Capture of 3D Human-Object Interactions** [CVPR 2024]  
[[code]](https://github.com/zhenzhiwang/intercontrol) [[paper]](https://arxiv.org/abs/2311.15864)  **Zero-shot Human Interaction Generation by Controlling Every Joint.** [NeurIPS 2024]  
[[code]](https://github.com/xiexh20/HDM) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_Template_Free_Reconstruction_of_Human-object_Interaction_with_Procedural_Interaction_Generation_CVPR_2024_paper.html)  **Template Free Reconstruction of Human-object Interaction with** [CVPR 2024]  
[[code]](https://github.com/ChelsieLei/EZ-HOI) [[paper]](https://arxiv.org/pdf/2410.23904)  **EZ-HOI: VLM Adaptation via Guided Prompt Learning for Zero-Shot HOI Detection** [NeurIPS 2024]  
[[code]](https://github.com/mk-minchul/sapiensid) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_SapiensID_Foundation_for_Human_Recognition_CVPR_2025_paper.pdf)  **SapiensID: Foundation for Human Recognition.** [CVPR 2025]  
[[code]](https://andrewjohngilbert.github.io/HumanvsMachineMinds/) [[paper]](https://openaccess.thecvf.com/content/CVPR2025W/MAR/papers/Rahmaniboldaji_Human_vs._Machine_Minds_Ego-Centric_Action_Recognition_Compared_CVPRW_2025_paper.pdf)  **Human vs. Machine Minds: Ego-Centric Action Recognition Compared.** [CVPR 2025]  
[[code]](https://github.com/JinluZhang1126/InteractAnything) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_InteractAnything_Zero-shot_Human_Object_Interaction_Synthesis_via_LLM_Feedback_and_CVPR_2025_paper.pdf)  **InteractAnything: Zero-shot Human Object Interaction Synthesis via LLM Feedback** [CVPR 2025]  
[[code]](https://github.com/OpenMICG/GAP3DS) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_Vision-Guided_Action_Enhancing_3D_Human_Motion_Prediction_with_Gaze-informed_Affordance_CVPR_2025_paper.pdf)  **Vision-Guided Action: Enhancing 3D Human Motion Prediction with Gaze-informed** [CVPR 2025]  
[[code]](https://github.com/KUCognitiveInformaticsLab/Huperflow-Website) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_HuPerFlow_A_Comprehensive_Benchmark_for_Human_vs._Machine_Motion_Estimation_CVPR_2025_paper.pdf#:~:text=HuPerFlow%20is%20the%20first%20large-scale%20human-perceived%20flow%20benchmark,exploration%20of%20human%20motion%20perception%20in%20natural%20scenes.)  **HuPerFlow: A Comprehensive Benchmark for Human vs. Machine** [CVPR 2025]  
[[code]](https://github.com/wenboran2002/open-3dhoi) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wen_Reconstructing_In-the-Wild_Open-Vocabulary_Human-Object_Interactions_CVPR_2025_paper.pdf)  **Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zeng_ChainHOI_Joint-based_Kinematic_Chain_Modeling_for_Human-Object_Interaction_Generation_CVPR_2025_paper.pdf)  **ChainHOI: Joint-based Kinematic Chain Modeling for Human-Object Interaction Generation.** [CVPR 2025]  
[[code]](https://github.com/snuvclab/ParaHome) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_ParaHome_Parameterizing_Everyday_Home_Activities_Towards_3D_Generative_Modeling_of_CVPR_2025_paper.pdf)  **ParaHome: Parameterizing Everyday Home Activities Towards 3D Generative Modeling of Human-Object Interactions** [CVPR 2025]  
[[code]](https://github.com/boycehbz/CloseApp) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Reconstructing_Close_Human_Interaction_with_Appearance_and_Proxemics_Reasoning_CVPR_2025_paper.pdf)  **Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning.** [CVPR 2025]  
[[code]](https://github.com/alparius/pico) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Cseke_PICO_Reconstructing_3D_People_In_Contact_with_Objects_CVPR_2025_paper.pdf)  **PICO: Reconstructing 3D People In Contact with Objects.** [CVPR 2025]  




#### Scenario reasoning and prediction  

> <span style="color:lightblue;">💡💡 Scene reasoning prediction within embodied AI and world model paradigms encompasses Bayesian optimization processes that perform causal chain inference, counterfactual simulation, and long-term trajectory prediction from multimodal inputs such as first-person video streams. by leveraging large language models (LLMs) and multimodal LLMs (MLLMs) to drive neuro-symbolic architectures that fuse scene graphs with diffusion priors. This achieves joint likelihood maximization for semantic segmentation, relational grounding, and dynamic extrapolation in complex environments, as demonstrated in the evolution from perceptual to behavioral intelligence. This approach adapts to few-shot scenarios through reinforcement learning from human feedback (RLHF) and meta-learning, ensuring zero-shot planning robustness under novel interactions. Simultaneously, it validates geometric consistency via differentiable rendering pipelines like NeRF, bridging low-order perception to high-order deliberative cognition. Despite computational scalability and ethical alignment challenges, the recent Embodied Arena benchmark advances unified evaluation frameworks to support scalable deployment of virtual agents and robotic decision-making. <span> 


[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_Decompose_More_and_Aggregate_Better_Two_Closer_Looks_at_Frequency_CVPR_2023_paper.pdf) **Decompose More and Aggregate Better: Two Closer Looks at Frequency Representation Learning for Human Motion Prediction.** [CVPR 2023]  
[[paper]](Sun_DeFeeNet_Consecutive_3D_Human_Motion_Prediction_With_Deviation_Feedback_CVPR_2023_paper)  **DeFeeNet: Consecutive 3D Human Motion Prediction with Deviation Feedback** [CVPR 2023]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.pdf)  **EqMotion: Equivariant Multi-agent Motion Prediction with Invariant Interaction Reasoning** [CVPR 2023]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_FEND_A_Future_Enhanced_Distribution-Aware_Contrastive_Learning_Framework_for_Long-Tail_CVPR_2023_paper.pdf)  **FEND: A Future Enhanced Distribution-Aware Contrastive Learning Framework for Long-tail Trajectory Prediction** [CVPR 2023]  
[[code]](https://github.com/RLuke22/FJMP) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Rowe_FJMP_Factorized_Joint_Multi-Agent_Motion_Prediction_Over_Learned_Directed_Acyclic_CVPR_2023_paper.pdf)  **FJMP: Factorized Joint Multi-Agent Motion Prediction over Learned Directed Acyclic Interaction Graphs.** [CVPR 2023]  
[[code]](https://github.com/Cram3r95/argo2goalmp) [[paper]](https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Conde_Improving_Multi-Agent_Motion_Prediction_With_Heuristic_Goals_and_Motion_Refinement_CVPRW_2023_paper.pdf)  **Improving Multi-Agent Motion Prediction with Heuristic Goals and Motion Refinement** [CVPR 2023]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_IPCC-TP_Utilizing_Incremental_Pearson_Correlation_Coefficient_for_Joint_Multi-Agent_Trajectory_CVPR_2023_paper.pdf)  **IPCC-TP: Utilizing Incremental Pearson Correlation Coefficient for Joint Multi-Agent Trajectory Prediction** [CVPR 2023]  
[[code]](https://github.com/MediaBrain-SJTU/LED) [[paper]](Mao_Leapfrog_Diffusion_Model_for_Stochastic_Trajectory_Prediction_CVPR_2023_paper)  **Leapfrog Diffusion Model for Stochastic Trajectory Prediction.** [CVPR 2023]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_MotionDiffuser_Controllable_Multi-Agent_Motion_Prediction_Using_Diffusion_CVPR_2023_paper.pdf)  **MotionDiffuser: Controllable Multi-Agent Motion Prediction using Diffusion** [CVPR 2023]  
[[code]](https://drive-wm.github.io/) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Driving_into_the_Future_Multiview_Visual_Forecasting_and_Planning_with_CVPR_2024_paper.pdf)  **Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving** [CVPR 2024]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Generalized_Predictive_Model_for_Autonomous_Driving_CVPR_2024_paper.pdf)  **Generalized Predictive Model for Autonomous Driving.** [CVPR 2024]  
[[code]](https://github.com/fudan-zvg/DeMo) [[paper]](https://arxiv.org/abs/2410.05982)  **DeMo: Decoupling Motion Forecasting into Directional Intentions** [NeurIPS 2024]  
[[code]](https://github.com/neerjathakkar/PAR) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Thakkar_Poly-Autoregressive_Prediction_for_Modeling_Interactions_CVPR_2025_paper.pdf)  **Poly-Autoregressive Prediction for Modeling Interactions.** [CVPR 2024]  
[[code]](https://github.com/AIR-THU/V2X-Graph) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/1812042b83f20707a898ff6f8af7db84-Paper-Conference.pdf)  **Learning Cooperative Trajectory Representations for Motion Prediction.** [NeurIPS 2024]  
[[code]](https://github.com/xmuqimingxia/DOtAv2) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/1812042b83f20707a898ff6f8af7db84-Paper-Conference.pdf)  **Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels** [CVPR 2025]  
[[code]](https://github.com/AdaCompNUS/WhatMatters) [[paper]](https://arxiv.org/pdf/2306.15136)  **What Truly Matters in Trajectory Prediction for Autonomous Driving?** [NeurIPS 2023]  
[[code]](https://github.com/DSL-Lab/MoFlow) [[paper]](https://arxiv.org/abs/2503.09950)  **MoFlow: One-Step Flow Matching for Human Trajectory Forecasting via Implicit Maximum Likelihood Estimation Distillation** [CVPR 2025]  
[[code]](https://github.com/hustvl/DiffusionDrive) [[paper]](https://arxiv.org/abs/2411.15139)  **Truncated Diffusion Model for End-to-End Autonomous Driving** [CVPR 2025]  
[[code]](https://github.com/zju3dv/street_crafter) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_StreetCrafter_Street_View_Synthesis_with_Controllable_Video_Diffusion_Models_CVPR_2025_paper.pdf)  **StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models** [CVPR 2025]  
[[code]](https://github.com/xifen523/OmniTrack) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_Omnidirectional_Multi-Object_Tracking_CVPR_2025_paper.pdf)  **Omnidirectional Multi-Object Tracking** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Tan_SceneDiffuser_City-Scale_Traffic_Simulation_via_a_Generative_World_Model_CVPR_2025_paper.pdf)  **SceneDiffuser: City-Scale Traffic Simulation via a Generative World Model** [CVPR 2025]  
[[code]](https://github.com/lzhyu/SimMotionEdit) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_SimMotionEdit_Text-Based_Human_Motion_Editing_with_Motion_Similarity_Prediction_CVPR_2025_paper.pdf)  **SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction** [CVPR 2025]  
[[code]](https://github.com/gywns6287/SOAP) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_SOAP_Vision-Centric_3D_Semantic_Scene_Completion_with_Scene-Adaptive_Decoder_and_CVPR_2025_paper.pdf#:~:text=To%20address%20these%20issues%2C%20we%20introduce%20a%20novel,oc-cluded%20region-aware%20view%20projection%20and%20a%20scene-adaptive%20decoder)  **SOAP: Vision-Centric 3D Semantic Scene Completion with Scene-Adaptive** [CVPR 2025]  
[[code]](https://github.com/hustvl/GaussTR) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_GaussTR_Foundation_Model-Aligned_Gaussian_Transformer_for_Self-Supervised_3D_Spatial_Understanding_CVPR_2025_paper.html)  **GaussTR: Foundation Model-Aligned Gaussian Transformer for Self-Supervised 3D Spatial Understanding** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zeng_DeepLA-Net_Very_Deep_Local_Aggregation_Networks_for_Point_Cloud_Analysis_CVPR_2025_paper.pdf)  **DeepLA-Net: Very Deep Local Aggregation Networks for Point Cloud Analysis.** [CVPR 2025]  
[[code]](https://github.com/fudan-zvg/BridgeAD) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Bridging_Past_and_Future_End-to-End_Autonomous_Driving_with_Historical_Prediction_CVPR_2025_paper.pdf)  **Bridging Past and Future: End-to-End Autonomous Driving with Historical Prediction** [CVPR 2025]  
[[code]](https://s-attack.github.io/) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Bahari_Certified_Human_Trajectory_Prediction_CVPR_2025_paper.pdf)  **Certified Human Trajectory Prediction** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Taketsugu_Physical_Plausibility-aware_Trajectory_CVPR_2025_supplemental.pdf)  **Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment** [CVPR 2025]  










### Physical Perception  

> <span style="color:lightblue;">💡💡 Physical Perception in embodied AI and computer vision paradigms encompasses the multimodal inference of environmental dynamics and affordances from sensor streams such as RGB-D imagery, LiDAR point clouds, and tactile arrays, leveraging equivariant neural architectures like SE(3)-transformers or diffusion-based world models to distill raw perceptual signals into latent representations that encode rigid body motions, contact forces, and inertial priors via probabilistic filtering over factor graphs, as illuminated in recent CVPR sessions on shifting from passive sensing to context-aware actuation. This process integrates self-supervised contrastive learning on large-scale embodied datasets like RoboNet or Epic-Kitchens to resolve ambiguities in viewpoint-invariant feature extraction, enabling robust extrapolation to novel scenarios through neural radiance fields (NeRF) or Gaussian splatting for differentiable rendering of occluded regions, while Bayesian fusion mechanisms mitigate noise in dynamic lighting and motion blur, thereby bridging pixel-level observations to actionable geometric embeddings that underpin real-time navigation and manipulation in physical world models as surveyed in comprehensive embodied AI overviews. <span> 


#### Physical Property Perception  

> <span style="color:lightblue;">💡💡 Physical Property Perception in robotics and AI frameworks refers to the estimation of intrinsic material attributes—such as elasticity, friction coefficients, density, and thermal conductivity—from visual-tactile observations, employing hybrid vision-language-tactile models pretrained on synthetic datasets via knowledge distillation to predict deformable dynamics through finite element simulations embedded in variational autoencoders (VAEs), as demonstrated in recent breakthroughs enabling safe object handling via inference of physical affordances. This paradigm extends traditional sensor fusion by incorporating large tactile-vision-language models for zero-shot property regression, optimizing cross-modal alignment losses over multimodal inputs to enhance generalization across unseen textures and geometries, while equivariant graph neural networks (EGNNs) capture neighborhood interactions for boundary-aware refinement in cluttered environments, fostering advancements in intelligent robotic design that respond adaptively to surrounding physical cues as reviewed in AI-driven robotics innovations. <span> 

[[code]](https://github.com/KUCognitiveInformaticsLab/Huperflow-Website) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_HuPerFlow_A_Comprehensive_Benchmark_for_Human_vs._Machine_Motion_Estimation_CVPR_2025_paper.pdf#:~:text=We%20introduce%20HuPerFlow%E2%80%94a%20benchmark%20for%20human-perceived%20flow%2C%20measured,%E2%88%BC38%2C400%20response%20vectors%20collected%20through%20online%20psychophysical%20experiments.)  **HuPerFlow: A Comprehensive Benchmark for Human vs. Machine Motion Estimation Comparison** [CVPR 2025]  
[[code]](https://github.com/xjc97/mridc) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Decouple_Distortion_from_Perception_Region_Adaptive_Diffusion_for_Extreme-low_Bitrate_CVPR_2025_paper.pdf)  **Decouple Distortion from Perception: Region Adaptive Diffusion for Extreme-low Bitrate Perception Image Compression** [CVPR 2025]  
[[code]](https://github.com/danier97/depthcues) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Danier_DepthCues_Evaluating_Monocular_Depth_Perception_in_Large_Vision_Models_CVPR_2025_paper.pdf)  **DepthCues: Evaluating Monocular Depth Perception in Large Vision Models** [CVPR 2025]  
[[code]](https://github.com/xrhan/diffmotion) [[paper]](https://arxiv.org/abs/2506.02473)  **Generative Perception of Shape and Material from Differential Motion** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Hyperdimensional_Uncertainty_Quantification_for_Multimodal_Uncertainty_Fusion_in_Autonomous_Vehicles_CVPR_2025_paper.pdf)  **Hyperdimensional Uncertainty Quantification for Multimodal Uncertainty Fusion in Autonomous Vehicles Perception** [CVPR 2025]  
[[paper]](Garcia_Learning_Physics_From_Video_Unsupervised_Physical_Parameter_Estimation_for_Continuous_CVPR_2025_paper)  **Learning Physics From Video: Unsupervised Physical Parameter Estimation for Continuous Dynamical Systems** [CVPR 2025]  
[[code]](https://github.com/haoyu-x/vision-in-action) [[paper]](https://arxiv.org/abs/2506.15666)  **Vision in Action: Learning Active Perception from Human Demonstrations** [CVPR 2025]  
[[code]](https://github.com/facebookresearch/fairchem) [[paper]](https://arxiv.org/pdf/2502.12147)  **Learning Smooth and Expressive Interatomic Potentials for Physical Property Prediction** [ICML 2025]  
[[paper]](https://arxiv.org/pdf/2504.07165)  **Perception in Reflection** [ICML 2025]  
[[code]](https://github.com/phybench-official/phybench) [[paper]](https://arxiv.org/abs/2504.16074)  **PHYBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models** [NeurIPS 2025]  
[[code]](https://github.com/Chopper-233/Physics-AD) [[paper]](https://arxiv.org/abs/2503.03562)  **Towards Visual Discrimination and Reasoning of Real-World Physical Dynamics: Physics-Grounded Anomaly Detection** [CVPR 2025]  
[[code]](https://github.com/by-luckk/PhysGen3D) [[paper]](https://arxiv.org/abs/2503.20746)  **PhysGen3D: Crafting a Miniature Interactive World from a Single Image** [CVPR 2025]  
[[code]](https://github.com/DreamMr/RAP) [[paper]](https://arxiv.org/abs/2503.01222)  **Retrieval-Augmented Perception: High-Resolution Image Perception** [ICML 2025]  
[[code]](https://github.com/sutkarsh/focal) [[paper]](https://arxiv.org/abs/2507.10375)  **Test-Time Canonicalization by Foundation Models for Robust Perception** [ICML 2025]  
[[code]](https://github.com/HimangiM/UniPhy_CVPR2025) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Mittal_UniPhy_Learning_a_Unified_Constitutive_Model_for_Inverse_Physics_Simulation_CVPR_2025_paper.pdf) **UniPhy: Common Latent-Conditioned Neural Constitutive Model for Diverse Materials** [CVPR 2025]  
[[paper]](https://arxiv.org/pdf/2506.20601)  **Video Perception Model for 3D Scene Synthesis** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Vision-Language_Embodiment_for_Monocular_Depth_Estimation_CVPR_2025_paper.pdf)  **Vision-Language Embodiment for Monocular Depth EstimationZ** [CVPR 2025]  
[[code]](https://github.com/ZhangLab-DeepNeuroCogLab/MotionPerceiver) [[paper]](https://arxiv.org/abs/2405.16493)  **Flow Snapshot Neurons in Action: Deep Neural Networks Generalize to Biological Motion Perception** [NeurIPS 2024]  
[[code]](https://github.com/Jukgei/gic) [[paper]](https://arxiv.org/pdf/2406.14927)  **Gaussian-Informed Continuum for Physical Property Identification and Simulation** [NeurIPS 2024]  
[[code]](https://gmargo11.github.io/active-sensing-loco/) [[paper]](https://arxiv.org/abs/2311.01405) **Learning to See Physical Properties with Active Sensing Motor Policies** [ICML 2024]  
[[code]](https://github.com/xherdan76/LIP) [[paper]](https://openreview.net/pdf?id=WZu4gUGN13)  **Latent Intuitive Physics: Learning to Transfer Hidden Physics from A 3D Video** [ICLR 2024]  
[[code]](https://github.com/CognitiveModeling/Loci-Looped) [[paper]](https://arxiv.org/abs/2310.10372)  **LOOPING LOCI: Learning Object Permanence from Videos** [ICML 2024]  
[[code]](https://github.com/sled-group/moh) [[paper]](https://arxiv.org/abs/2407.06192)  **Multi-Object Hallucination in Vision Language Models** [NeurIPS 2024]  
[[code]](https://github.com/facebookresearch/neuralfeels) [[paper]](https://www.science.org/doi/10.1126/scirobotics.adl0628)  **NeuralFeels with neural fields Visuo-tactile perception for in-hand manipulation** [Science Robotics 2024]  
[[code]](https://github.com/mtangemann/motion_energy_segmentation) [[paper]](https://arxiv.org/abs/2411.01505)  **Object segmentation from common fate: Motion energy processing in visual perception** [NeurIPS 2024]  
[[code]](https://github.com/ajzhai/NeRF2Physics) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhai_Physical_Property_Understanding_from_Language-Embedded_Feature_Fields_CVPR_2024_paper.pdf)  **Physical Property Understanding from Language-Embedded Feature Fields** [CVPR 2024]  
[[code]](https://physical-reasoning-project.github.io/) [[paper]](https://arxiv.org/abs/2402.06119)  **ContPhy: Continuum Physical Concept Learning and Reasoning from Videos** [ICML 2024]  
[[code]](https://github.com/stanford-crfm/helm) [[paper]](https://arxiv.org/abs/2410.07112)  **VHELM: A Holistic Evaluation of Vision Language Models** [NeurIPS 2024]   



#### Physical Interaction Modeling  

> <span style="color:lightblue;">💡💡 Physical Interaction Modeling in embodied AI entails the simulation of contact-rich dynamics and multi-body articulations within virtual or hybrid physics engines, utilizing differentiable simulators like MuJoCo or Brax augmented with neural policy networks to learn forward and inverse kinematics mappings that enforce conservation laws and frictional constraints via Lagrangian mechanics in latent spaces, as exemplified in generative approaches for synthesizing interactive 3D scenes with realistic layouts and articulated affordances. This framework leverages world models trained through reinforcement learning from physical simulators to forecast emergent behaviors in long-horizon trajectories, incorporating diffusion priors for stochastic sampling of interaction outcomes under uncertainty, while neuro-symbolic verification pipelines ensure causal consistency across simulated and real-world deployments, addressing scalability challenges in high-dimensional state spaces as systematically reviewed in surveys on learning embodied intelligence.<span> 


[[code]](https://long-horizon-assembly.github.io/) [[paper]](https://long-horizon-assembly.github.io/arch_paper.pdf)  **ARCH: Hierarchical Hybrid Learning for Long-Horizon Contact-Rich Robotic Assembly** [CoRL 2025]  
[[code]](https://www.yimingdou.com/hearing_hands/) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Dou_Hearing_Hands_Generating_Sounds_from_Physical_Interactions_in_3D_Scenes_CVPR_2025_paper.pdf)  **Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes** [CVPR 2025]  
[[code]](https://github.com/wzyabcas/InterAct) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_InterAct_Advancing_Large-Scale_Versatile_3D_Human-Object_Interaction_Generation_CVPR_2025_paper.pdf)  **InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation** [CVPR 2025]  
[[code]](https://github.com/rickakkerman/InterDyn) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Bahari_Certified_Human_Trajectory_Prediction_CVPR_2025_paper.pdf)  **InterDyn: Controllable Interactive Dynamics with Video Diffusion Models** [CVPR 2025]  
[[code]](https://github.com/Sirui-Xu/InterMimic) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_InterMimic_Towards_Universal_Whole-Body_Control_for_Physics-Based_Human-Object_Interactions_CVPR_2025_paper.pdf)  **InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions** [CVPR 2025]  
[[code]](https://github.com/OnlyLoveKFC/Neural_P3M) [[paper]](https://neurips.cc/media/neurips-2024/Slides/93679.pdf)  **Neural P3M: A Long-Range Interaction Modeling Enhancer for Geometric GNNs** [NeurIPS 2025]  
[[code]](https://www.physics-gen.org/) [[paper]](https://arxiv.org/abs/2503.05333)  **PhysicsGen: Can Generative Models Learn from Images to Predict Complex Physical Relations?** [CVPR 2025]  
[[code]](https://github.com/neerjathakkar/PAR) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Thakkar_Poly-Autoregressive_Prediction_for_Modeling_Interactions_CVPR_2025_paper.pdf)  **Poly-Autoregressive Prediction for Modeling Interactions** [CVPR 2025]  
[[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Ji_POMP_Physics-consistent_Motion_Generative_Model_through_Phase_Manifolds_CVPR_2025_paper.html)  **POMP: Physics-consistent Motion Generative Model through Phase Manifolds** [CVPR 2025]  
[[code]](https://github.com/AI4Science-WestlakeU/diffphycon) [[paper]](https://openreview.net/pdf?id=MbZuh8L0Xg)  **A Generative Approach to Control Complex Physical Systems** [NeurIPS 2024]  
[[code]](https://github.com/UMass-Embodied-AGI/CHAIC) [[paper]](https://arxiv.org/abs/2411.01796)  **Constrained Human-AI Cooperation (CHAIC): An Inclusive Embodied Social Intelligence Challenge** [ NeurIPS 2024]  
[[code]](https://github.com/yyvhang/EgoChoir_release) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Bahari_Certified_Human_Trajectory_Prediction_CVPR_2025_paper.pdf)  **EgoChoir: Capturing 3D Human-Object Interaction Regions from Egocentric Views** [CVPR 2025]  





#### Operability and Feasibility Analysis  

> <span style="color:lightblue;">💡💡 Operability and Feasibility Analysis in robotics and AI planning paradigms involves the prospective evaluation of action sequences against kinematic, dynamic, and economic constraints through optimization over feasible action spaces, deploying mixed-integer linear programming (MILP) or sampling-based planners like RRT* integrated with uncertainty-aware Monte Carlo tree search to quantify success probabilities and resource trade-offs in partially observable environments, as applied in autonomous navigation and excavation tasks via robot-centric feasibility metrics. This process incorporates technical-economic assessments via multi-objective genetic algorithms that balance ROI projections with risk mitigation, drawing on feasibility studies for collaborative robot integration to validate process viability prior to deployment, while recent AI feasibility guides emphasize pre-build validation through prototype simulations and gap analysis to guide decision-making in edge cases like adverse weather or hardware limitations, thereby streamlining project launches in industrial automation contexts.<span> 
