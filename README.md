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



### Traditional Geometric Methods Phase (1980s–Early 2010s)
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

[[paper]](http://www.umiacs.umd.edu/users/venu/cvpr04final.pdf) **Lie-algebraic averaging for globally consistent motion estimation** [CVPR 2004]  

[[paper]](http://imagine.enpc.fr/~monasse/Stereo/Projects/MartinecPajdla07.pdf) **Robust rotation and translation estimation in multiview reconstruction** [CVPR 2007]  

[[paper]](http://www.maths.lth.se/vision/publdb/reports/pdf/enqvist-kahl-etal-wovcnnc-11.pdf) **Non-sequential structure from motion** [ICCV OMNIVIS Workshops 2011]  

[[paper]](https://web.math.princeton.edu/~amits/publications/sfm_3dimpvt12.pdf) **Global motion estimation from point matches** [3DIMPVT 2012]  

[[paper]](http://www.cs.sfu.ca/~pingtan/Papers/iccv13_sfm.pdf) **A Global Linear Method for Camera Pose Registration** [ICCV 2013]  

[[paper]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Cui_Global_Structure-From-Motion_by_ICCV_2015_paper.pdf) **Global Structure-from-Motion by Similarity Averaging** [ICCV 2015]  

[[paper]](http://arxiv.org/abs/1503.01832) **Linear Global Translation Estimation from Feature Tracks** [BMVC 2015]  


##### Hierarchical SfM

[[paper]](http://www.diegm.uniud.it/fusiello/papers/3dim09.pdf) **Structure-and-Motion Pipeline on a Hierarchical Cluster Tree** [Workshop on 3-D Digital Imaging and Modeling 2009]

[[paper]](https://www.researchgate.net/publication/224579249_Randomized_structure_from_motion_based_on_atomic_3D_models_from_camera_triplets) **Randomized Structure from Motion Based on Atomic 3D Models from Camera Triplets** [CVPR 2009]

[[paper]](https://dspace.cvut.cz/bitstream/handle/10467/62206/Havlena_stat.pdf?sequence=1&isAllowed=y) **Efficient Structure from Motion by Graph Optimization** [ECCV 2010]

[[paper]](http://www.diegm.uniud.it/fusiello/papers/cviu15.pdf) **Hierarchical structure-and-motion recovery from uncalibrated images** [CVIU 2015]

##### Multi-Stage SfM

[[paper]](https://arxiv.org/abs/1702.08601) **Parallel Structure from Motion from Local Increment to Global Averaging** [arXiv 2017]

[[code]](https://researchweb.iiit.ac.in/~rajvi.shah/projects/multistagesfm/) [[paper]](http://arxiv.org/abs/1512.06235) **Multistage SFM: A Coarse-to-Fine Approach for 3D Reconstruction** [3DV 2014 / arXiv 2016]

[[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Cui_HSfM_Hybrid_Structure-from-Motion_CVPR_2017_paper.pdf) **HSfM: Hybrid Structure-from-Motion** [ICCV 2017]

##### Non Rigid SfM

[[paper]](http://arxiv.org/abs/1609.02638) **Robust Structure from Motion in the Presence of Outliers and Missing Data** [2016]

##### Viewing graph optimization

[[code]](http://www.cs.cornell.edu/~snavely/projects/skeletalset/) **Skeletal graphs for efficient structure from motion** [CVPR 2008]

[[paper]](http://homes.cs.washington.edu/~csweeney/papers/optimizing_the_viewing_graph.pdf) **Optimizing the Viewing Graph for Structure-from-Motion** [ICCV 2015]

[[paper]](https://home.cse.ust.hk/~tshenaa/files/pub/eccv2016_graph_match.pdf) **Graph-Based Consistent Matching for Structure-from-Motion** [ECCV 2016]

##### Unordered feature tracking

[[paper]](http://imagine.enpc.fr/~moulonp/publis/featureTracking_CVMP12.pdf) **Unordered feature tracking made fast and easy** [CVMP 2012]

[[paper]](http://www.maths.lth.se/vision/publdb/reports/pdf/svarm-simayijang-etal-i2-12.pdf) **Point Track Creation in Unordered Image Collections Using Gomory-Hu Trees** [ICPR 2012]

**Fast connected components computation in large graphs by vertex pruning** [IEEE Transactions on Parallel and Distributed Systems 2016]

##### Large scale image matching for SfM

[[code]](http://www.robots.ox.ac.uk/~vgg/research/vgoogle/) **Video Google: A Text Retrieval Approach to Object Matching in Video** [ICCV 2003]

[[paper]](http://www.vis.uky.edu/~stewe/publications/nister_stewenius_cvpr2006.pdf) **Scalable Recognition with a Vocabulary Tree** [CVPR 2006]

[[paper]](https://grail.cs.washington.edu/rome/rome_paper.pdf) **Building Rome in a Day** [ICCV 2009]

[[paper]](https://hal.inria.fr/file/index/docid/825085/filename/jegou_pq_postprint.pdf) **Product quantization for nearest neighbor search** [IEEE Transactions on Pattern Analysis and Machine Intelligence 2011]

[[paper]](http://www.nlpr.ia.ac.cn/jcheng/papers/CameraReady-CasHash.pdf) **Fast and Accurate Image Matching with Cascade Hashing for 3D Reconstruction** [CVPR 2014]

[[paper]](https://www.infona.pl/resource/bwmeta1.element.elsevier-3a6310b2-2ad0-3bdd-839d-8daecaca680d/content/partDownload/8900b0c7-b69c-39dc-8cbd-94217452a25f) **Recent developments in large-scale tie-point matching** [ISPRS 2016]

[[paper]](http://homes.cs.washington.edu/~csweeney/papers/graphmatch.pdf) **Graphmatch: Efficient Large-Scale Graph Construction for Structure from Motion** [3DV 2017]






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




### 3D AIGC

#### Text to 3D Generation

































## Scene Understanding

































