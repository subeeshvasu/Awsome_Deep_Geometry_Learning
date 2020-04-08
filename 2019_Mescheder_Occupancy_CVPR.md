## Key points from Abstract and Intro

+ Unlike images, in 3D there is no canonical representation which is both computationally and memory efficient yet allows for representing high-resolution geometry of arbitrary topology. 
+ Many of the state-of-the-art learning based 3D reconstruction approaches can hence only represent very coarse 3D geometry or are limited to a restricted domain.
+ Existing work on learning-based 3D reconstruction can be broadly categorized by the output representation they produce as either voxel-based, point-based or mesh-based.

#### Voxel based 
Due to their simplicity and similarity to the 2D image representation, voxels are the most commonly used representation for discriminative and generative 3D tasks. However, they suffer from the maximum resolution that can be achieved at the output. Even with the most recent multi-resolution reconstruction technique, output resolution is still limited to 256^3 voxel grids

#### point-based 
Point cloud based methods lack the connectivity structure of the underlying mesh and hence require additional postprocessing steps to extract 3D geometry from the network output. These methods are also limited in the number of points which can be reliably predicted using a standard feed-forward network.

#### Mesh based
Typically based on deforming a template mesh and hence do not allow arbitrary topologies. Also, only limited number of vertices can be reliably predicted using a standard feed-forward network.

### Occupancy Networks
+ A new representation for learning-based 3D reconstruction methods.
+ Occupancy networks implicitly represent the 3D surface as the continuous decision boundary of a deep neural network classifier. This representation encodes a description of the 3D output at infinite resolution without excessive memory footprint.
+ The resulting neural network can be evaluated at arbitrary resolution. For the inference, a paraellizable multi-resolution isosurface extraction algorithm is employed to efficiiently generate the mesh from network output.
+ Experiments to demonstrate applications for the challenging tasks of 3D reconstruction from single images, noisy point clouds and coarse discrete voxel grids.

## Key points from method

+ Define occupancy function as a continuous function that will map any arbirary point in space to a real value of 1 or , indicating if the points is inside or outside of the surface?.

+ A single batch of training data can include, samples from N different shapes, with M random points sampled from in and out of each shape, with the loss as the cross entropy with respect to the grond truth occupancy values.

+ To generate such point cloud as a function of input data in the form of image or point cloud, one can append features from those along with the input coordinates. (Refer Net arch in supplementary). Features are extracted using different types of encoders depending on the nature of input shape.

+ The performance of the method depends on the sampling scheme that we employ for drawing the random samples. Up on empirical evaluation, they found that sampling uniformly inside the bounding box of the object with an additional small padding yields the best results.

+ They also talk about learning a probabilistic latent variable models using their 3D representation, but I didn't understand that part.

+ For the inference, Multiresolution IsoSurface Extraction (MISE), a hierarchical isosurface extraction algorithm is proposed. MISE enables to extract high resolution meshes from the occupancy network without densely evaluating all points of a high-dimensional occupancy grid.

+ After getting occupancy function an sufficiently finer resolution grid, they apply Marching Cubes algorithm to extract an approximate isosurface. 

+ The mesh extracted by the Marching Cubes algorithm is further refined in two additional steps. In a first step, the mesh is simplified using the Fast-Quadric-Mesh-Simplification algorithm. The output of which os refined using first and second order (i.e., gradient) information derived from the network. They sample random points from each face of the output mesh and tweak their position (?) to satisfy the criterion drived from network. But this will again amount to additional processing steps? Can't they increase the output resolution before marching to cube to yield similar results?



## Key points from experiments


## key points on Future works


# What I need to do

Try to use the post-processing mesh refinement on top of deep sdf output to see if the shape can be improvised. Neverthless, the improvisation that they are doing here is to overcome artifacts stems from Marching cube algorithm. So we will need to use something like a gan to improve the shape deformation that's caused by the nature in which we are currently learning the shapes.



