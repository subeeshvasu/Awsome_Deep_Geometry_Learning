## Key points from Abstract and Intro

+ Unlike images, in 3D there is no canonical representation which is both computationally and memory efficient yet allows for representing high-resolution geometry of arbitrary topology. 
+ Many of the state-of-the-art learning based 3D reconstruction approaches can hence only represent very coarse 3D geometry or are limited to a restricted domain.
+ Existing work on learning-based 3D reconstruction can be broadly categorized by the output representation they produce as either voxel-based, point-based or mesh-based.

#### Voxel based: Due to their simplicity and similarity to the 2D image representation, voxels are the most commonly used representation for discriminative and generative 3D tasks. However, they suffer from the maximum resolution that can be achieved at the output. Even with the most recent multi-resolution reconstruction technique, output resolution is still limited to 256^3 voxel grids

#### point-based: Point cloud based methods lack the connectivity structure of the underlying mesh and hence require additional postprocessing steps to extract 3D geometry from the network output. These methods are also limited in the number of points which can be reliably predicted using a standard feed-forward network.

#### Mesh based: Typically based on deforming a template mesh and hence do not allow arbitrary topologies. Also, only limited number of vertices can be reliably predicted using a standard feed-forward network.

+ Propose Occupancy Networks, a new representation for learning-based 3D reconstruction methods.
+ Occupancy networks implicitly represent the 3D surface as the continuous decision boundary of a deep neural network classifier. This representation encodes a description of the 3D output at infinite resolution without excessive memory footprint.
+ Experiments to demonstrate applications for the challenging tasks of 3D reconstruction from single images, noisy point clouds and coarse discrete voxel grids.

## Key points from method


## Key points from experiments


## key points on Future works

