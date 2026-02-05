## An Overview of the Basics of the GL Transmission Format

> **Version 2.0** — Based on the glTF 2.0 specification by the Khronos Group

glTF was designed and specified by the **Khronos Group** for the efficient transfer of 3D content over networks.

The core of glTF is a **JSON file** that describes the structure and composition of a scene containing 3D models. The top-level elements of this file are:

| Element | Description |
|---|---|
| **scenes, nodes** | Basic structure of the scene |
| **cameras** | View configurations for the scene |
| **meshes** | Geometry of 3D objects |
| **buffers, bufferViews, accessors** | Data references and data layout descriptions |
| **materials** | Definitions of how objects should be rendered |
| **textures, images, samplers** | Surface appearance of objects |
| **skins** | Information for vertex skinning |
| **animations** | Changes of properties over time |

These elements are contained in arrays. References between the objects are established by using their indices to look up, e.g., the objects in the arrays.

It is also possible to store the whole asset in a single binary glTF file. In this case, the JSON data is stored as a string, followed by the binary data of buffers or images.

---

## Concepts

The conceptual relationships between the top-level elements of a glTF asset are shown below:

```
                        scene
                          |
                        node
                       / | \ \
                camera mesh skin animation
                      |
                  material   accessor
                    |          |
                 texture   bufferView
                  / \          |
            sampler  image   buffer
```

---

## Scenes, Nodes

The glTF JSON may contain **scenes** (with an optional default scene). Each scene can contain an array of indices of **nodes**.

Each of the nodes can contain an array of indices of its **children**, building a node hierarchy.

```json
{
  "nodes": [
    { "children": [1, 2, 3] },
    {},
    {},
    {}
  ]
}
```

A node may contain a local **transform**. This can be given as a column-major **matrix** (e.g. `[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]`), or with separate **translation**, **rotation** and **scale** properties, where the rotation is given as a quaternion. The local transformation matrix is then computed as:

```
M = T × R × S
```

where **T**, **R**, and **S** are the matrices that are created from the translation, rotation, and scale. The global transform of a node is given by the product of all local transforms on the path from the root to the respective node.

Each node may also refer to a **camera**, using indices that point into the meshes and cameras arrays. These elements are then attached to these nodes. During rendering, instances of these elements are created and transformed with the global transform of the node.

The translation, rotation, and scale properties of a node may also be the target of an **animation**: the animation then describes how one property changes over time. The attached objects will move accordingly, allowing to model moving objects or camera flights.

Nodes are also used in **vertex skinning**: a node hierarchy can define the skeleton of an animated character. The node then refers to a mesh and to a skin. The skin contains further information about how the mesh is deformed based on the current skeleton pose.

---

## Cameras

Each of the nodes may refer to one of the **cameras** that are defined in the glTF asset.

```json
{
  "cameras": [
    {
      "type": "perspective",
      "perspective": {
        "aspectRatio": 1.5,
        "yfov": 0.65,
        "zfar": 100,
        "znear": 0.01
      }
    }
  ]
}
```

There are two types of cameras: **perspective** and **orthographic**, and they define the projection matrix.

The value for the far clipping plane distance of a perspective camera, `zfar`, is optional. When it is omitted, the camera uses a special projection matrix for infinite projections.

When one of the nodes refers to a camera, then an instance of this camera is created. The camera matrix of this instance is given by the global transform matrix of the node.

---

## Textures, Images, Samplers

The **textures** contain information about textures that may be applied to rendered objects. Textures are referred to by materials to define the basic color of the objects, as well as physical properties that affect the object appearance.

```json
{
  "textures": [
    {
      "source": 0,
      "sampler": 0
    }
  ]
}
```

A texture consists of a reference to the **source**, which is one of the **images** of the asset, and a reference to a **sampler**.

### Images

The **images** define the image data used for the texture. This data can be given via a URI that is the location of an image file, or by a reference to a **bufferView** and a MIME type that defines the type of the image data that is stored in the buffer view.

```json
{
  "images": [
    { "uri": "file0.png" },
    {
      "bufferView": 3,
      "mimeType": "image/jpeg"
    }
  ]
}
```

### Samplers

The **samplers** describe the wrapping and scaling of a texture. The constant values correspond to OpenGL constants that can directly be passed to `glTexParameter`.

```json
{
  "samplers": [
    {
      "magFilter": 9729,
      "minFilter": 9987,
      "wrapS": 33648,
      "wrapT": 10497
    }
  ]
}
```

---

## Binary Data References

The images and buffers of a glTF asset may refer to external files that contain the data that are required for rendering the 3D content.

### Buffers

The **buffers** refer to binary files (`.bin`) that contain geometry or animation data.

```json
{
  "buffers": [
    {
      "uri": "buffer01.bin",
      "byteLength": 102040
    }
  ]
}
```

### Images (binary references)

The images refer to image files (`.png`, `.jpg`) that contain texture data for the models.

```json
{
  "images": [
    { "uri": "image01.png" }
  ]
}
```

The data is referred to via URIs, but can also be included directly in the JSON using data URIs. The data URI defines the MIME type, and contains the data as a base64-encoded string.

### Buffer Data (URI)

```
"data:application/gltf-buffer;base64,AAAABBBB..."
```

### Image Data (PNG)

```
"data:image/png;base64,VVVVXXXXZZZZ..."
```

---

## Meshes

The meshes may contain multiple **mesh primitives**. These refer to the geometry data that is required for rendering.

Each mesh primitive has a **rendering mode**, which is a constant indicating whether it should be rendered as `POINTS`, `LINES`, or `TRIANGLES`.

The primitive also refers to **indices** and the **attributes** of the vertices, using the indices of **accessors** for this data. The material that should be used for rendering is also given, by the index of the material.

```json
{
  "meshes": [
    {
      "primitives": [
        {
          "mode": 4,
          "indices": 0,
          "attributes": {
            "POSITION": 1,
            "NORMAL": 2
          },
          "material": 0
        }
      ]
    }
  ]
}
```

Each attribute is defined by mapping the attribute name to the index of the accessor that contains the attribute data. This data will be used as the vertex attributes when rendering the mesh. The attributes may, for example, define the `POSITION` and the `NORMAL` of the vertices:

| Attribute | Accessor | Type | Component | Description |
|---|---|---|---|---|
| `POSITION` | 1 | `"VEC3"` | `5126 (FLOAT)` | Vertex positions |
| `NORMAL` | 2 | `"VEC3"` | `5126 (FLOAT)` | Vertex normals |

### Morph Targets

A mesh may define multiple **morph targets**. Such a morph target describes a deformation of the original mesh.

Each node may also refer to a mesh or a camera, using indices that point to these elements.

To define a mesh with morph targets, each mesh primitive can contain an array of **targets**. These are dictionaries that map names of attributes to the indices of accessors that contain the displacements of the geometry for the targets.

The mesh may also contain an array of **weights** that define the contribution of each morph target to the final, rendered state of the mesh.

```json
{
  "primitives": [
    {
      "attributes": { "POSITION": 1 },
      "targets": [
        { "POSITION": 10 },
        { "POSITION": 11 }
      ]
    }
  ],
  "weights": [0, 0.5]
}
```

Combining multiple morph targets with different weights allows, for example, modeling different facial expressions of a character. The weights can be modified with an animation, to interpolate between different states of the geometry.

---

## Buffers, BufferViews, Accessors

The **buffers** contain the data that is used for the geometry of 3D models, animations, and skinning. The **bufferViews** add structural information to this data. The **accessors** define the exact type and layout of the data.

### Buffers

Each of the buffers refers to a binary data file, using a **URI**. It is the source or block of raw data with the given **byteLength**.

### BufferViews

Each of the **bufferViews** refers to one buffer. It has a **byteOffset** and a **byteLength**, defining the part of the buffer that the bufferView refers to, and an optional **target** (buffer target).

```json
{
  "bufferViews": [
    {
      "buffer": 0,
      "byteOffset": 0,
      "byteLength": 16,
      "target": 35
    }
  ]
}
```

### Accessors

The **accessors** define how the data of a bufferView is interpreted. They may have a **byteOffset** referring to the start of the data and contain information about the **type** and **componentType** of the bufferView data.

The data may, for example, be defined as 2D vectors of floating point values when the type is `"VEC2"` and the componentType is `5126` (FLOAT). The range of values is stored in the **min** and **max** properties.

```json
{
  "accessors": [
    {
      "bufferView": 0,
      "byteOffset": 0,
      "type": "VEC2",
      "componentType": 5126,
      "count": 2,
      "min": [0.1, 0.2],
      "max": [0.9, 0.8]
    }
  ]
}
```

### Data Layout Example

The data of multiple accessors may be interleaved inside a bufferView. In this case, the bufferView will have a **byteStride** property that says how many bytes are between the start of one element of an accessor, and the start of the next.

The buffer data is read from a file:

```
| Byte 0 ......... Byte 35 |
```

The bufferView defines a segment of the buffer data:

```json
{
  "byteOffset": 4,
  "byteLength": 28,
  "byteStride": 12
}
```

The accessor defines an additional offset:

```json
{
  "bufferView": 0,
  "byteOffset": 4,
  "count": 2,
  "type": "VEC2"
}
```

The bufferView defines a stride between the elements:

```
| 12 bytes | 12 bytes | 12 bytes |
  byteStride = 12
```

The accessor defines that the elements are 2D float vectors:

```
componentType = FLOAT
```

The values are written into the final accessor data, at the positions that are given by the indices.

---

## Sparse Accessors

When only few elements of an **accessor** differ from a default value (which is often the case for morph targets), then the data can be given in a very compact form using a **sparse** accessor description.

The **sparse** data block contains the **count** of sparse data elements.

The **values** refer to the bufferView that contains the sparse data values.

The **target indices** refer to a bufferView that contains the indices. These are defined with a componentType. If no values are given, they default to `0`.

```json
{
  "sparse": {
    "count": 3,
    "values": {
      "bufferView": 2
    },
    "indices": {
      "bufferView": 1,
      "componentType": 5123
    }
  }
}
```

---

## Materials

Each mesh primitive may refer to one of the **materials** that are contained in a glTF asset. The materials describe how an object should be rendered, based on physical material properties. This allows to apply **Physically Based Rendering (PBR)** descriptions, to make sure that the appearance of the rendered object is consistent among all renderers.

### Metallic-Roughness Model

The default material model is the **Metallic-Roughness-Model**. It uses values between `0.0` and `1.0` to describe how much the material characteristics resemble that of a metal, and how rough the surface of the object is. These properties may either be given as individual values that apply to the whole object, or be read from textures.

The properties that define a material in the Metallic-Roughness-Model are summarized in the **pbrMetallicRoughness** object:

- **baseColorFactor** contains scaling factors for the red, green, blue and alpha component of the color. If no texture is given, these values define the color of the whole object.
- The **metallicFactor** and **roughnessFactor** are multiplied with the values of the respective textures (or `1.0` if textures aren't present).

```json
{
  "materials": [
    {
      "pbrMetallicRoughness": {
        "baseColorFactor": [1.0, 0.75, 0.5, 1.0],
        "metallicFactor": 0.1,
        "roughnessFactor": 0.5,
        "baseColorTexture": {
          "index": 1
        },
        "metallicRoughnessTexture": {
          "index": 5
        }
      }
    }
  ]
}
```

### Additional Texture Properties

In addition to the properties that are defined via the Metallic-Roughness-Model, the material may contain other properties that affect the rendering:

- **emissiveTexture**: Refers to a texture that defines areas of the material that emit light. The information is contained in the RGB components, and a scalar factor will be applied to these values.

- **normalTexture**: Refers to a texture that defines areas of the object and its **strength**: a scalar that influences how strongly the normal texture applies to the surface.

- **occlusionTexture**: Refers to a texture that defines areas of the surface that receive less indirect lighting. The information is contained in the "red" channel of the texture. It illuminates parts of the object surface; it defines the color of the light that is reflected from the surface. The global ambient component of the lighting is scaled with this value, and the corresponding attenuation of the texture.

### Material Properties in Textures

Each texture property in the material can contain the data of the respective texture map. The textures may have a `texCoord` value, which is an index that refers to the texture coordinate set index. This determines which `TEXCOORD_n` attribute from the mesh primitive is used to compute the texture coordinates for this texture, with `0` being the default.

---

## Skins

A glTF asset may contain the information that is necessary to perform **vertex skinning**. With vertex skinning, it is possible to let the vertices of a mesh be influenced by the pose of a skeleton, based on its current pose.

A node that refers to a mesh may also refer to a **skin**.

```json
{
  "nodes": [
    {
      "mesh": 0,
      "skin": 0
    }
  ]
}
```

### Skin Properties

The skins contain an array of **joints**, which are the indices of nodes that define the skeleton hierarchy, and the **inverseBindMatrices**, which is a reference to an accessor that contains one matrix for each joint.

```json
{
  "skins": [
    {
      "inverseBindMatrices": 0,
      "joints": [1, 2, 3]
    }
  ]
}
```

The skeleton hierarchy is modeled with nodes, just like the scene structure. Each joint node may have a local transform, and an array of children, and the "bones" of the skeleton are given implicitly, as the connections between the joints.

The mesh primitives of a skinned mesh contain the `POSITION` attribute that refers to the accessor for the vertex positions, and two special attributes that are required for skinning: a `JOINTS_0` and a `WEIGHTS_0` attribute, each referring to an accessor.

The `JOINTS_0` attribute data contains the indices of the joints that should affect the vertex. The `WEIGHTS_0` attribute data defines the weights indicating how strongly the joint should influence the vertex.

From this information, the **skinning matrix** can be computed.

---

## Computing the Skinning Matrix

The skinning matrix describes how the vertices of a mesh are transformed based on the current pose of a selection. The skinning matrix is a weighted combination of joint matrices.

### Computing the Joint Matrices

The skin refers to the **inverseBindMatrices**. This is an accessor which contains one inverse bind matrix for each joint. Each of these matrices transforms the mesh from the local space of the joint, based on the current global transformation of the joint, and is called `globalJointTransform`.

From these matrices, a **jointMatrix** may be computed for each joint:

```
jointMatrix[j] =
    inverse(globalTransformOfNode) *
    globalJointTransform[j] *
    inverseBindMatrices[j]
```

Any global transform of the node that contains the mesh and the skin is cancelled out by pre-multiplying the joint matrix with the inverse of this transform.

For implementations based on OpenGL or WebGL, the `jointMatrices` array will be passed to the vertex shader as a uniform.

### Combining the Joint Matrices to Create the Skinning Matrix

The primitives of a skinned mesh contain the `POSITION`, `JOINT` and `WEIGHT` attributes, referring to accessors. These accessors contain one element for each vertex:

```
vertex 0:  [JOINTS: j0 j1 j2 j3]  [WEIGHTS: w0 w1 w2 w3]
vertex 1:  [JOINTS: j0 j1 j2 j3]  [WEIGHTS: w0 w1 w2 w3]
vertex 2:  [JOINTS: j0 j1 j2 j3]  [WEIGHTS: w0 w1 w2 w3]
...
```

The data of these accessors is passed as attributes to the vertex shader, together with the `jointMatrices` array. In the vertex shader, the **skinMatrix** is computed. It is a linear combination of the joint matrices whose indices are contained in the `JOINTS_0` attribute, weighted with the `WEIGHTS_0` values:

```glsl
// Vertex Shader
skinMatrix =
    weight.x * jointMatrix[joint.x] +
    weight.y * jointMatrix[joint.y] +
    weight.z * jointMatrix[joint.z] +
    weight.w * jointMatrix[joint.w];
```

The **skinMatrix** transforms the vertices based on the skeleton pose, before they are transformed with the model-view-perspective matrix.

---

## Animations

A glTF asset can contain **animations**. An animation can be applied to the properties of a node that define the local translation of the node, or to the weights for the morph targets.

Each animation consists of two elements: an array of **channels** and an array of **samplers**.

Each channel defines the **target** of the animation. This target usually refers to a node, using the index of the node, and to a **path**, which is the name of the property that should be animated (e.g. `"translation"`, `"rotation"`, or `"scale"` — affecting the local transforms of the node, or `"weights"` — in order to animate the weights of the morph targets of the meshes that are referred to by the node). The channel also refers to a **sampler**, which defines the actual animation data.

A **sampler** refers to the **input** and **output** data, using the indices of accessors that provide the data. The input refers to an accessor with scalar floating-point values, which are the times of the key frames of the animation. The output refers to an accessor that contains the values for the animated property at the respective key frames. The sampler also defines an **interpolation** mode for the animation, which may be `"LINEAR"`, `"STEP"`, or `"CUBICSPLINE"`.

```json
{
  "animations": [
    {
      "channels": [
        {
          "sampler": 0,
          "target": {
            "node": 1,
            "path": "rotation"
          }
        }
      ],
      "samplers": [
        {
          "input": 6,
          "output": 7,
          "interpolation": "LINEAR"
        }
      ]
    }
  ]
}
```

### Animation Samplers

During the animation, a "global" animation time (in seconds) is advanced.

The sampler looks up the key frames for the current time, in the input data. The corresponding values of the output data are read, and interpolated based on the interpolation mode.

The resulting interpolated value is forwarded and applied to the properties of a mesh attached to a node.

### Animation Channel Targets

The interpolated value is provided by an animation sampler and may be applied to different animation channel targets:

- **Animating the translation of a node**: e.g. `[1.0, 2.0, 3.0]` → translation vector
- **Animating a rotation of a node (via quaternion)**: e.g. `[1.0, 0.0, 0.0, 0.0]` → quaternion rotation
- **Animating the weights of morph targets**: weights applied to mesh morph targets

Animating the weights for the morph targets allows defining the primitives of a mesh that is attached to a node.

> **Note**: The vertex skinning in glTF is similar to the skinning in COLLADA. See Section 4.7 in the COLLADA specification for further details.

---

## Extensions

The glTF format allows **extensions** to add new functionality, or to simplify the definition of properties.

When an extension is used in a glTF asset, it has to be listed in the **extensionsUsed** property at the top level. Extensions may also be listed as **extensionsRequired** to indicate that the asset can only be loaded correctly with support for the extension.

```json
{
  "extensionsUsed": [
    "KHR_materials_pbrSpecularGlossiness"
  ],
  "extensionsRequired": [
    "KHR_materials_pbrSpecularGlossiness"
  ]
}
```

### Extensions Allow Adding

- The scope of the extension is defined by the place where it appears in the glTF JSON:
  - The whole asset
  - Individual objects
  - Individual properties

### Existing Extensions

Several extensions are developed and maintained by the Khronos Group. The following extensions are official extensions that are ratified by the Khronos Group:

| Extension | Description |
|---|---|
| `KHR_draco_mesh_compression` | Allows glTF geometry to be compressed with the Draco library |
| `KHR_lights_punctual` | Adds light sources to the scene |
| `KHR_materials_clearcoat` | Allows adding a clear coating layer to existing glTF PBR materials |
| `KHR_materials_transmission` | Transparent materials can be extended with an index of refraction |
| `KHR_materials_ior` | Index of refraction for materials |
| `KHR_materials_iridescence` | Models thin-film effects where the hue depends on the viewing angle and the thickness of the thin layer |
| `KHR_materials_sheen` | Adds a color parameter for the backscattering caused by cloth fibers |
| `KHR_materials_specular` | Allows defining the strength and color of specular reflections |
| `KHR_materials_transmission` | More realistic modeling of reflection, refraction, and opacity |
| `KHR_materials_unlit` | Unlit material model |
| `KHR_materials_variants` | Multiple materials for the same geometry, to be selected at runtime |
| `KHR_materials_volume` | Detailed modeling of the thickness and attenuation of translucent objects |
| `KHR_mesh_quantization` | More compact representation of vertex attributes with smaller data types |
| `KHR_texture_basisu` | Allows textures with KTX v2 images and Basis Universal supercompression |
| `KHR_texture_transform` | Adds support for KTX v2 images with Basis Universal |
| `EXT_mesh_gpu_instancing` | Adds support for GPU instancing, rendering many copies of a single mesh |
| `EXT_meshopt_compression` | Adds support for meshoptimizer-based compression |

---

## Binary glTF Files

In the standard format, there are two approaches for including binary resources: the buffer data, and the data for textures. They may be referenced from URIs, or embedded in the JSON data of the glTF using data URIs. When they are referenced via URIs, each buffer and image is stored as a separate file — which means one additional download request. When they are embedded as data URIs, the base64 encoding of the binary data will increase the file size considerably.

To overcome these drawbacks, there is the option to combine the glTF JSON and the binary data into a single binary glTF file. It is a little-endian file with the extension `.glb`. It consists of a header, which gives basic information about the version and the chunks that contain the actual data. The first chunk contains the JSON data formatted as a string. The second (optional) chunk contains the binary data, remaining chunks may contain extra data.

### Binary glTF Structure

```
12-byte header:
┌──────────────────────────────────────────────────┐
│ magic (4 bytes)  │ version (4 bytes) │ length    │
│ 0x46546C67       │ 2                 │ (4 bytes) │
│ ("glTF")         │                   │           │
└──────────────────────────────────────────────────┘

Chunk 0 (JSON):
┌──────────────────────────────────────────────────┐
│ chunkLength │ chunkType              │ chunkData  │
│ (4 bytes)   │ 0x4E4F534A ("JSON")   │ (padded)   │
└──────────────────────────────────────────────────┘

Chunk 1 (Binary Buffer):
┌──────────────────────────────────────────────────┐
│ chunkLength │ chunkType              │ chunkData  │
│ (4 bytes)   │ 0x004E4942 ("BIN\0")  │ (padded)   │
└──────────────────────────────────────────────────┘
```

- **Header** (12 bytes): Contains the magic number `glTF`, the version number, and the total file length.
- **Chunk 0 — JSON** (`0x4E4F534A`): The first chunk type indicates that this is the structured JSON content of the glTF. The data is a JSON string. This must always be the first chunk.
- **Chunk 1 — Binary** (`0x004E4942`): This is used to indicate what type of data is contained in the second chunk — it is the binary data. Which buffer and image data types are available depend on what is stored in the JSON chunk. This data is version 2.

---

## Further Resources

- **The Khronos glTF landing page**: [https://www.khronos.org/gltf/](https://www.khronos.org/gltf/)
- **The Khronos glTF GitHub repository**: [https://github.com/KhronosGroup/glTF](https://github.com/KhronosGroup/glTF)

---

*glTF and the glTF logo are trademarks of the Khronos Group Inc.*
*© 2016–2022 Marco Hutter — www.marco-hutter.de*
