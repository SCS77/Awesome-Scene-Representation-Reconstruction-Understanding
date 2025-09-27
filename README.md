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

[[code]](https://github.com/charlesq34/pointnet.git) [[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)  **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation** [CVPR 2017]  
[[code]](https://github.com/charlesq34/pointnet2.git) [[paper]](https://dl.acm.org/doi/abs/10.5555/3295222.3295263) **PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space**  [NIPS 2017]  
[[code]](https://github.com/WangYueFt/dgcnn.git) [[paper]](https://dl.acm.org/doi/abs/10.1145/3326362) **Dynamic Graph CNN for Learning on Point Clouds** [TOG 2019]  
[[code]](https://github.com/HuguesTHOMAS/KPConv.git)[[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Thomas_KPConv_Flexible_and_Deformable_Convolution_for_Point_Clouds_ICCV_2019_paper.pdf) **KPConv: Flexible and Deformable Convolution for Point Clouds** [ICCV 2019]  
[[code]](https://github.com/DylanWusee/pointconv.git)[[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_PointConv_Deep_Convolutional_Networks_on_3D_Point_Clouds_CVPR_2019_paper.pdf) **PointConv: Deep Convolutional Networks on 3D Point Clouds** [CVPR 2019]  
[[code]](https://github.com/POSTECH-CVLab/point-transformer.git)[[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf) **Point Transformer** [ICCV 2021]  
[[code]](https://github.com/Pointcept/PointTransformerV2.git)[[paper]](https://papers.nips.cc/paper_files/paper/2022/hash/d78ece6613953f46501b958b7bb4582f-Abstract-Conference.html) **Point Transformer V2: Grouped Vector Attention and Partition-based Pooling** [NIPS 2022]  
[[code]](https://github.com/Pointcept/PointTransformerV3.git)[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf) **Point Transformer V3: Simpler, Faster, Stronger** [CVPR 2024]  
[[code]](https://github.com/guochengqian/PointNeXt.git)[[paper]](https://papers.nips.cc/paper_files/paper/2022/hash/9318763d049edf9a1f2779b2a59911d3-Abstract-Conference.html)**PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies** [NIPS 2022]  
[[code]](https://github.com/Gardlin/PCR-CG.git)[[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700439.pdf)**PCR-CG: PCR-CG: Point Cloud Registration via Color and Geometry** [ECCV 2022]  
[[code]](https://github.com/Pointcept/Pointcept.git)[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Sonata_Self-Supervised_Learning_of_Reliable_Point_Representations_CVPR_2025_paper.pdf) **Sonata: Self-Supervised Learning of Reliable Point Representations** [CVPR 2025]  
[[code]](https://github.com/QingyongHu/RandLA-Net.git)[[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_RandLA-Net_Efficient_Semantic_Segmentation_of_Large-Scale_Point_Clouds_CVPR_2020_paper.pdf) **RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds** [CVPR 2020]  
[[code]](https://github.com/jrryzh/pointr.git)[[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_PoinTr_Diverse_Point_Cloud_Completion_With_Geometry-Aware_Transformers_ICCV_2021_paper.pdf) **PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers** [ICCV 2021]  
[[code]](https://github.com/facebookresearch/SparseConvNet.git)[[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Graham_3D_Semantic_Segmentation_CVPR_2018_paper.pdf) **3D Semantic Segmentation with Submanifold Sparse Convolutional Networks** [CVPR 2018]  

#### Point Cloud Processing Tool
[[code]](https://github.com/PointCloudLibrary/pcl.git) PCL  
[[code]](https://github.com/kzampog/cilantro.git) cilantro  
[[code]](https://github.com/isl-org/Open3D.git) Open3D  
[[code]](https://github.com/fwilliams/point-cloud-utils.git) point-cloud-utils  
[[code]](https://github.com/torch-points3d/torch-points3d.git) torch-points3d  
[[code]](https://github.com/pyg-team/pytorch_geometric.git) PyTorch Geometric (PyG)  
[[code]](https://github.com/open-mmlab/OpenPCDet.git) OpenPCDet  
[[code]](https://github.com/google/draco.git) draco  
[[code]](https://github.com/daavoo/pyntcloud.git) pyntcloud  
[[code]](https://github.com/NVIDIA/MinkowskiEngine.git) MinkowskiEngine  


### Voxel
> #### <span style="color:lightblue;">💡 Voxel Grids: As an extension of pixels in three-dimensional space, voxel grids divide space into regular cubic units (voxels). Each voxel can store occupancy information (i.e., whether the space is occupied), color, or other attributes. This structured representation facilitates Boolean operations and volumetric analysis, but its memory consumption increases cubically with resolution and can produce jagged edges due to discretization.</span>








### Polygon Mesh
> #### <span style="color:lightblue;">💡 Polygon Meshes: This has long been the dominant representation method in computer graphics. It consists of vertices, edges, and faces (typically triangles or quadrilaterals), explicitly defining an object's surface topology. Mesh representations are highly efficient for hardware-accelerated rendering and provide explicit surface information. However, generating high-quality, flawless meshes—such as those that are watertight and free of self-intersections—is often a complex process.</span>





### NeRF(Neural Radiance Fields)
> #### <span style="color:lightblue;">💡💡 NeRF represents scene light fields through neural networks, constituting an implicit volumetric representation. It employs one or more multilayer perceptrons (MLPs) to map spatial coordinates $(x,y,z)$ and viewpoint directions to color and volumetric density, thereby defining a continuous, differentiable volumetric field. This representation does not utilize explicit geometry (such as point clouds or meshes), instead training network parameters (typically several megabytes in size) to encode the scene. Common data formats include: input data (images and camera poses) are often stored as .npz (used by Mildenhall et al.'s TinyNeRF examples) or .json (e.g., InstantNGP/NeRFStudio's transforms.json stores camera intrinsic and extrinsic parameters); trained models save weights in formats like .pth/.pt or .npz; some frameworks also use serialization formats such as .msgpack or custom .nerf. NeRF's representation principle is based on volumetric rendering: rays intersect voxels, and the density and color predicted by the network are synthesized into final pixels via volumetric rendering equations. Unlike traditional voxels, NeRF uses continuous functions for implicit representation, efficiently expressing view-dependent lighting and geometric details.</span>  


### 3DGS(3D Gaussian Splatting)
> #### <span style="color:lightblue;">💡💡 3D Gaussian Point Clouds explicitly represent scenes using a set of colored spatial Gaussian volumetric elements (“points”), each possessing attributes such as position, shape (covariance/rotation), color, and transparency. This approach combines point cloud and volumetric rendering concepts: during rendering, each Gaussian volumetric element is projected onto the image for light summation. Common data formats include: .ply or custom formats exportable from raw point cloud models, while Gaussian tiling models themselves can be saved as .ply, .npz, or specialized formats like the newly introduced .spz (SPlatZip) for compressing Gaussian parameters. Its representation principle involves optimizing Gaussian position, size, and color to match the target view, enabling extremely fast rendering with photorealistic quality. Unlike NeRF, Gaussian Tiling employs an explicit sparse representation (each Gaussian voxel can be visualized as a colored, non-uniform point cloud). The trained model features fewer parameters, and rendering involves an explicit stitching/compositing process, making it well-suited for real-time applications.
</span> 






## Scene Reconstruction














## Scene Understanding

































