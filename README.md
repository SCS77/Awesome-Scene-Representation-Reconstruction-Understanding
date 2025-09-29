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
[[code]](https://github.com/chobao/Free360) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Bao_Free360_Layered_Gaussian_Splatting_for_Unbounded_360-Degree_View_Synthesis_from_CVPR_2025_paper.pdf) [[project]](https://zju3dv.github.io/free360/) **Free360: Layered Gaussian Splatting for Unbounded 360-Degree View Synthesis from Extremely Sparse and Unposed Views** [CVPR 2025]  
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






#### Generative Models & Editing

#### Robotics & SLAM











### 3DGS

























## Scene Understanding

































