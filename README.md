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
| 表示方法   | 核心图元            | 数据结构              | 性质 | 关键优势                                         | 关键劣势                                                     |
|------------|----------------------|------------------------|------|--------------------------------------------------|--------------------------------------------------------------|
| 点云       | 三维点 (x, y, z, ...) | 无序点集              | 显式 | 灵活，直接从传感器获取                           | 无序，非结构化，缺乏拓扑信息                                 |
| 体素网格   | 立方体（体素）       | 三维规则/稀疏网格      | 显式 | 规整结构，适用于 3D CNN                          | 内存消耗大，分辨率受限                                       |
| 多边形网格 | 顶点、边、面         | 图/半边结构           | 显式 | 拓扑明确，渲染高效，易于编辑                     | 拓扑固定，难以表示复杂或非流形几何                           |
| NeRF       | -                    | 神经网络（MLP）        | 隐式 | 高真实感，能表示复杂光学效应                     | 训练和渲染速度慢，难以编辑                                   |
| 3DGS       | 三维高斯函数         | 高斯参数列表           | 混合 | 实时渲染，照片级真实感                           | 存储开销大，编辑困难，依赖 SfM                               |


### Point cloud
> #### <span style="color:lightblue;">💡 Point Clouds: This is the most direct output format for many sensors (such as LiDAR) and reconstruction algorithms (such as SfM and MVS). It represents a simple collection of three-dimensional coordinate points (X, Y, Z), often accompanied by attributes like color (RGB) and normals. Point clouds offer the advantages of simple structure and ease of acquisition. However, their drawback lies in the absence of explicit topological connection information, making it challenging to perform high-quality rendering or surface analysis directly.</span>

[[code]](https://github.com/charlesq34/pointnet.git) [[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)  PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation [CVPR 2017]  
[[code]](https://github.com/charlesq34/pointnet2.git) [[paper]](https://dl.acm.org/doi/abs/10.5555/3295222.3295263) PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space [NIPS 2017]




### Voxel
> #### <span style="color:lightblue;">💡 Voxel Grids: As an extension of pixels in three-dimensional space, voxel grids divide space into regular cubic units (voxels). Each voxel can store occupancy information (i.e., whether the space is occupied), color, or other attributes. This structured representation facilitates Boolean operations and volumetric analysis, but its memory consumption increases cubically with resolution and can produce jagged edges due to discretization.</span>








### Polygon Mesh
> #### <span style="color:lightblue;">💡 Polygon Meshes: This has long been the dominant representation method in computer graphics. It consists of vertices, edges, and faces (typically triangles or quadrilaterals), explicitly defining an object's surface topology. Mesh representations are highly efficient for hardware-accelerated rendering and provide explicit surface information. However, generating high-quality, flawless meshes—such as those that are watertight and free of self-intersections—is often a complex process.</span>





### NeRF(Neural Radiance Fields)




### 3DGS(3D Gaussian Splatting)



## Scene Reconstruction














## Scene Understanding

































