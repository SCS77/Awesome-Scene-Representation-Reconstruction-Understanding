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

| 格式 | 核心原理 | 数据结构 | 主要优点 | 主要缺点 | 典型用例 |
| ---- | -------- | -------- | -------- | -------- | -------- |
| 点云 | 离散3D点的集合，直接表示采样表面。 | (X,Y,Z,...) 列表 | 结构简单，直接来自传感器，保真度高 | 无拓扑信息，渲染困难，数据量大，不规则 | 原始数据存储，SLAM，3D感知 |
| 体素网格 | 空间离散化为规则的3D栅格。 | 3D数组 | 结构规整，便于体积运算，拓扑关系隐式 | 内存消耗大，分辨率受限，存在量化误差 | 医疗成像，模拟，实时图形 |
| 多边形网格 | 由顶点、边和面构成的表面表示。 | 顶点列表，面索引列表 | 渲染高效（硬件加速），拓扑明确，内存效率高 | 拓扑复杂，难以处理非流形几何，生成复杂 | 游戏，电影特效，CAD，数字人 |
| 神经隐式表示 (NeRF/SDF) | 用神经网络参数化的连续函数表示场景。 | 神经网络权重 | 连续，与分辨率无关，内存紧凑，能表示复杂拓扑 | 训练和查询速度慢，难以编辑，通常逐场景优化 | 新视角合成，形状补全，高保真重建 |
| 3D高斯渲射 (3DGS) | 用大量可优化的3D高斯体表示场景。 | 高斯参数列表 | 实时渲染，训练速度快，渲染质量高 | 内存占用较大，存在伪影，编辑仍是挑战 | 实时AR/VR，数字孪生，虚拟漫游 |

### Point cloud
> #### <span style="color:lightblue;">💡 Point Clouds: This is the most direct output format for many sensors (such as LiDAR) and reconstruction algorithms (such as SfM and MVS). It represents a simple collection of three-dimensional coordinate points (X, Y, Z), often accompanied by attributes like color (RGB) and normals. Point clouds offer the advantages of simple structure and ease of acquisition. However, their drawback lies in the absence of explicit topological connection information, making it challenging to perform high-quality rendering or surface analysis directly.</span>







### Voxel
> #### <span style="color:lightblue;">💡 Voxel Grids: As an extension of pixels in three-dimensional space, voxel grids divide space into regular cubic units (voxels). Each voxel can store occupancy information (i.e., whether the space is occupied), color, or other attributes. This structured representation facilitates Boolean operations and volumetric analysis, but its memory consumption increases cubically with resolution and can produce jagged edges due to discretization.</span>








### Polygon Mesh
> #### <span style="color:lightblue;">💡 Polygon Meshes: This has long been the dominant representation method in computer graphics. It consists of vertices, edges, and faces (typically triangles or quadrilaterals), explicitly defining an object's surface topology. Mesh representations are highly efficient for hardware-accelerated rendering and provide explicit surface information. However, generating high-quality, flawless meshes—such as those that are watertight and free of self-intersections—is often a complex process.</span>





### NeRF(Neural Radiance Fields)




### 3DGS(3D Gaussian Splatting)



## Scene Reconstruction














## Scene Understanding

































