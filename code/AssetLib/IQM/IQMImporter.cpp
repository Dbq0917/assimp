/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2026, assimp team

All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
copyright notice, this list of conditions and the
following disclaimer.

* Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the
following disclaimer in the documentation and/or other
materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
contributors may be used to endorse or promote products
derived from this software without specific prior
written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/

#ifndef ASSIMP_BUILD_NO_IQM_IMPORTER

#include <assimp/DefaultIOSystem.h>
#include <assimp/IOStreamBuffer.h>
#include <assimp/ai_assert.h>
#include <assimp/importerdesc.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/ByteSwapper.h>
#include <assimp/quaternion.h>
#include <assimp/matrix4x4.h>
#include <memory>
#include <numeric>

#include "IQMImporter.h"
#include "iqm.h"

// RESOURCES:
// http://sauerbraten.org/iqm/
// https://github.com/lsalzman/iqm

inline void swap_block( uint32_t *block, size_t size ){
    (void)block; // suppress 'unreferenced formal parameter' MSVC warning
    size >>= 2;
    for ( size_t i = 0; i < size; ++i )
        AI_SWAP4( block[ i ] );
}

static constexpr aiImporterDesc desc = {
    "Inter-Quake Model Importer",
    "",
    "",
    "",
    aiImporterFlags_SupportBinaryFlavour,
    0,
    0,
    0,
    0,
    "iqm"
};

namespace Assimp {

// ------------------------------------------------------------------------------------------------
//  Default constructor
IQMImporter::IQMImporter() :
        mScene(nullptr) {
    // empty
}

// ------------------------------------------------------------------------------------------------
//  Returns true, if file is a binary Inter-Quake Model file.
bool IQMImporter::CanRead(const std::string &pFile, IOSystem *pIOHandler, bool checkSig) const {
    const std::string extension = GetExtension(pFile);

    if (extension == "iqm") {
        return true;
    }
        
    if (!extension.length() || checkSig) {
        if (!pIOHandler) {
            return true;
        }
        std::unique_ptr<IOStream> pStream(pIOHandler->Open(pFile, "rb"));
        unsigned char data[15];
        if (!pStream || 15 != pStream->Read(data, 1, 15)) {
            return false;
        }
        return !memcmp(data, "INTERQUAKEMODEL", 15);
    }
    
    return false;
}

// ------------------------------------------------------------------------------------------------
const aiImporterDesc *IQMImporter::GetInfo() const {
    return &desc;
}

// IQM stores ten animation channels per joint pose:
// 0..2 translation, 3..6 quaternion rotation (x, y, z, w), 7..9 scale.
constexpr unsigned int IQM_NUM_CHANNELS = 10u;
constexpr unsigned int IQM_CHANNEL_MASK_ALL = (1u << IQM_NUM_CHANNELS) - 1u;

struct IQMJointData {
    std::string name;
    int parent;
    aiVector3D translation;
    aiQuaternion rotation;
    aiVector3D scale;
    aiNode *node;
    aiMatrix4x4 localMatrix;
    aiMatrix4x4 globalMatrix;
    aiMatrix4x4 inverseGlobalMatrix;
};

struct IQMPoseData {
    int parent;
    unsigned int mask;
    float channelOffset[IQM_NUM_CHANNELS];
    float channelScale[IQM_NUM_CHANNELS];
};

struct IQMAnimData {
    std::string name;
    unsigned int firstFrame;
    unsigned int numFrames;
    float frameRate;
    unsigned int flags;
};

namespace {

aiMatrix4x4 ComposeLocalMatrix(const aiVector3D &translation, const aiQuaternion &rotation,
        const aiVector3D &scale) {
    return aiMatrix4x4(scale, rotation, translation);
}

template <typename T>
void SwapPod(T &value) {
    swap_block(reinterpret_cast<uint32_t *>(&value), sizeof(T));
}

unsigned int PopCount(unsigned int value) {
    unsigned int count = 0;
    while (value) {
        value &= value - 1;
        ++count;
    }
    return count;
}

// Creates one node per joint, keeps the file's parent indices and names, and
// stores the parent-local base pose as the node transformation.
void BuildSkeletonNodes(aiScene *pScene, std::vector<IQMJointData> &joints) {
    if (joints.empty()) {
        return;
    }

    for (IQMJointData &joint : joints) {
        joint.node = new aiNode();
        joint.node->mName.Set(joint.name);
        joint.node->mTransformation = joint.localMatrix;
    }

    aiNode *root = pScene->mRootNode;

    for (unsigned int jointIndex = 0; jointIndex < joints.size(); ++jointIndex) {
        aiNode *const node = joints[jointIndex].node;
        const int parent = joints[jointIndex].parent;
        aiNode *const parentNode = parent >= 0 ? joints[parent].node : root;

        aiNode **oldChildren = parentNode->mChildren;
        const unsigned int oldCount = parentNode->mNumChildren;
        parentNode->mChildren = new aiNode *[oldCount + 1];
        for (unsigned int childIndex = 0; childIndex < oldCount; ++childIndex) {
            parentNode->mChildren[childIndex] = oldChildren[childIndex];
        }
        parentNode->mChildren[oldCount] = node;
        parentNode->mNumChildren = oldCount + 1;
        node->mParent = parentNode;
        delete[] oldChildren;
    }
}

// Turns the BLENDINDEX/BLENDWEIGHT vertex arrays into aiBone bindings. The
// offset matrix is the inverse of the joint's global base pose. Blend indices
// address the global IQM vertex table, so each mesh's first_vertex offset is
// subtracted to obtain the mesh-local vertex id used by aiBone.
void BuildBoneBindings(aiScene *pScene, const std::vector<IQMJointData> &joints,
        const unsigned char *blendIndices, const unsigned char *blendWeights,
        const unsigned int *meshFirstVertex, unsigned int blendComponents,
        size_t fileSize, unsigned int numVertices, const unsigned char *fileBase) {
    if (joints.empty() || !blendIndices || !blendWeights || blendComponents == 0) {
        return;
    }

    const uint64_t blendBytes = uint64_t(numVertices) * blendComponents;
    const uint64_t indicesOffset = uint64_t(blendIndices - fileBase);
    const uint64_t weightsOffset = uint64_t(blendWeights - fileBase);
    if (blendBytes > fileSize || indicesOffset > fileSize || weightsOffset > fileSize ||
            blendBytes > fileSize - indicesOffset || blendBytes > fileSize - weightsOffset) {
        throw DeadlyImportError("IQM-file has blend vertex arrays referencing data outside of the file.");
    }
    struct WeightEntry {
        unsigned int vertexId;
        float weight;
    };

    for (unsigned int meshIndex = 0; meshIndex < pScene->mNumMeshes; ++meshIndex) {
        aiMesh *mesh = pScene->mMeshes[meshIndex];
        if (!mesh->mNumVertices) {
            continue;
        }

        std::vector<std::vector<WeightEntry>> weightsPerJoint(joints.size());
        const unsigned int firstVertex = meshFirstVertex[meshIndex];
        for (unsigned int vertex = 0; vertex < mesh->mNumVertices; ++vertex) {
            const unsigned int globalVertex = firstVertex + vertex;
            const unsigned char *indices = blendIndices + globalVertex * blendComponents;
            const unsigned char *weights = blendWeights + globalVertex * blendComponents;
            for (unsigned int component = 0; component < blendComponents; ++component) {
                const unsigned int jointIndex = indices[component];
                const float weight = weights[component] / 255.0f;
                if (weight == 0.0f) {
                    continue;
                }
                if (jointIndex >= joints.size()) {
                    throw DeadlyImportError("IQM-file has a blend index outside of the joint table.");
                }
                weightsPerJoint[jointIndex].push_back({ vertex, weight });
            }
        }

        unsigned int numBones = 0;
        for (const std::vector<WeightEntry> &entries : weightsPerJoint) {
            if (!entries.empty()) {
                ++numBones;
            }
        }
        if (numBones == 0) {
            continue;
        }

        mesh->mNumBones = numBones;
        mesh->mBones = new aiBone *[numBones]();
        unsigned int boneSlot = 0;
        for (unsigned int jointIndex = 0; jointIndex < joints.size(); ++jointIndex) {
            const std::vector<WeightEntry> &entries = weightsPerJoint[jointIndex];
            if (entries.empty()) {
                continue;
            }

            aiBone *bone = mesh->mBones[boneSlot++] = new aiBone();
            bone->mName.Set(joints[jointIndex].name);
            bone->mOffsetMatrix = joints[jointIndex].inverseGlobalMatrix;
            bone->mNumWeights = static_cast<unsigned int>(entries.size());
            bone->mWeights = new aiVertexWeight[bone->mNumWeights];
            for (unsigned int weightIndex = 0; weightIndex < bone->mNumWeights; ++weightIndex) {
                bone->mWeights[weightIndex].mVertexId = entries[weightIndex].vertexId;
                bone->mWeights[weightIndex].mWeight = entries[weightIndex].weight;
            }
        }
    }
}

// Decodes the compressed frame table (one unsigned short per enabled pose
// channel) into assimp animation channels. Key times are expressed in ticks so
// that mTicksPerSecond equals the file-declared frame rate: one key per frame,
// at times 0, 1, ..., numFrames-1. No resampling is performed.
void BuildAnimations(aiScene *pScene, const unsigned char *data, const iqmheader &hdr,
        const std::vector<IQMJointData> &joints, const std::vector<IQMPoseData> &poses,
        const std::vector<IQMAnimData> &animations) {
    pScene->mNumAnimations = static_cast<unsigned int>(animations.size());
    pScene->mAnimations = new aiAnimation *[pScene->mNumAnimations]();

    const uint16_t *frameTable = reinterpret_cast<const uint16_t *>(data + hdr.ofs_frames);

    for (unsigned int animIndex = 0; animIndex < animations.size(); ++animIndex) {
        const IQMAnimData &sourceAnim = animations[animIndex];
        aiAnimation *animation = pScene->mAnimations[animIndex] = new aiAnimation();
        animation->mName.Set(sourceAnim.name);
        animation->mTicksPerSecond = sourceAnim.frameRate;
        animation->mDuration = static_cast<double>(sourceAnim.numFrames - 1);

        animation->mNumChannels = static_cast<unsigned int>(joints.size());
        animation->mChannels = new aiNodeAnim *[animation->mNumChannels]();

        for (unsigned int jointIndex = 0; jointIndex < joints.size(); ++jointIndex) {
            aiNodeAnim *channel = animation->mChannels[jointIndex] = new aiNodeAnim();
            channel->mNodeName.Set(joints[jointIndex].name);
            channel->mNumPositionKeys = sourceAnim.numFrames;
            channel->mNumRotationKeys = sourceAnim.numFrames;
            channel->mNumScalingKeys = sourceAnim.numFrames;
            channel->mPositionKeys = new aiVectorKey[sourceAnim.numFrames];
            channel->mRotationKeys = new aiQuatKey[sourceAnim.numFrames];
            channel->mScalingKeys = new aiVectorKey[sourceAnim.numFrames];
        }

        const unsigned int numJoints = static_cast<unsigned int>(joints.size());
        for (unsigned int frameOffset = 0; frameOffset < sourceAnim.numFrames; ++frameOffset) {
            const unsigned int sourceFrame = sourceAnim.firstFrame + frameOffset;
            const uint16_t *frameCursor = frameTable + sourceFrame * hdr.num_framechannels;
            const double keyTime = static_cast<double>(frameOffset);

            for (unsigned int jointIndex = 0; jointIndex < numJoints; ++jointIndex) {
                const IQMPoseData &pose = poses[jointIndex];
                float channelValue[IQM_NUM_CHANNELS];
                for (unsigned int channel = 0; channel < IQM_NUM_CHANNELS; ++channel) {
                    float value = pose.channelOffset[channel];
                    if (pose.mask & (1u << channel)) {
                        uint16_t raw;
                        memcpy(&raw, frameCursor, sizeof(raw));
                        AI_SWAP2(raw);
                        frameCursor++;
                        value += raw * pose.channelScale[channel];
                    }
                    channelValue[channel] = value;
                }

                aiNodeAnim *channel = animation->mChannels[jointIndex];
                channel->mPositionKeys[frameOffset].mTime = keyTime;
                channel->mPositionKeys[frameOffset].mValue =
                        aiVector3D(channelValue[0], channelValue[1], channelValue[2]);

                aiQuaternion rotation(channelValue[6], channelValue[3], channelValue[4], channelValue[5]);
                rotation.Normalize();
                channel->mRotationKeys[frameOffset].mTime = keyTime;
                channel->mRotationKeys[frameOffset].mValue = rotation;

                channel->mScalingKeys[frameOffset].mTime = keyTime;
                channel->mScalingKeys[frameOffset].mValue =
                        aiVector3D(channelValue[7], channelValue[8], channelValue[9]);
            }
        }
    }
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Builds the joint hierarchy as scene nodes, the per-mesh bone bindings and
// the animation channels/keyframes declared by the file.
void IQMImporter::ImportSkeletonAndAnimations(const std::string &file, aiScene *pScene,
        const unsigned char *data, const iqmheader &hdr) {
    const auto fail = [&file](const char *message) {
        throw DeadlyImportError("IQM-file ", file, " ", message);
    };

    const auto rangeInFile = [fileSize = size_t(hdr.filesize)](uint64_t offset, uint64_t bytes) {
        return bytes <= fileSize && offset <= fileSize - bytes;
    };

    if (hdr.num_joints > 0 && !rangeInFile(hdr.ofs_joints, uint64_t(hdr.num_joints) * IQM_JOINT_V2_SIZE)) {
        fail("references joints outside of the file");
    }
    if (hdr.num_poses > 0 && !rangeInFile(hdr.ofs_poses, uint64_t(hdr.num_poses) * IQM_POSE_V2_SIZE)) {
        fail("references poses outside of the file");
    }
    if (hdr.num_anims > 0 && !rangeInFile(hdr.ofs_anims, uint64_t(hdr.num_anims) * IQM_ANIM_SIZE)) {
        fail("references animations outside of the file");
    }
    const uint64_t frameBytes = uint64_t(hdr.num_frames) * hdr.num_framechannels * sizeof(uint16_t);
    if (frameBytes > 0 && !rangeInFile(hdr.ofs_frames, frameBytes)) {
        fail("references frame data outside of the file");
    }

    const char *textBlock = hdr.num_text ? reinterpret_cast<const char *>(data + hdr.ofs_text) : "";
    const auto readString = [&](unsigned int offset) -> std::string {
        if (offset >= hdr.num_text) {
            fail("references a name outside of the text block");
        }
        return std::string(textBlock + offset);
    };

    if (hdr.num_poses != 0 && hdr.num_poses != hdr.num_joints) {
        fail("does not provide one pose per joint");
    }
    if ((hdr.num_anims != 0 || hdr.num_frames != 0) && hdr.num_poses != hdr.num_joints) {
        fail("contains animation data but no matching joint poses");
    }

    std::vector<IQMJointData> joints;
    joints.reserve(hdr.num_joints);

    for (unsigned int jointIndex = 0; jointIndex < hdr.num_joints; ++jointIndex) {
        iqmjoint joint;
        memcpy(&joint, data + hdr.ofs_joints + jointIndex * IQM_JOINT_V2_SIZE, sizeof(joint));
        SwapPod(joint.name);
        AI_SWAP4(joint.parent);
        for (float &component : joint.translate) {
            (void)component;
            AI_SWAP4(component);
        }
        for (float &component : joint.rotate) {
            (void)component;
            AI_SWAP4(component);
        }
        for (float &component : joint.scale) {
            (void)component;
            AI_SWAP4(component);
        }

        if (joint.parent >= int(hdr.num_joints)) {
            fail("has a joint with an invalid parent index");
        }

        IQMJointData parsed;
        parsed.name = readString(joint.name);
        parsed.parent = joint.parent;
        parsed.translation = aiVector3D(joint.translate[0], joint.translate[1], joint.translate[2]);
        parsed.rotation = aiQuaternion(joint.rotate[3], joint.rotate[0], joint.rotate[1], joint.rotate[2]);
        parsed.rotation.Normalize();
        parsed.scale = aiVector3D(joint.scale[0], joint.scale[1], joint.scale[2]);
        parsed.node = nullptr;
        parsed.localMatrix = ComposeLocalMatrix(parsed.translation, parsed.rotation, parsed.scale);
        joints.push_back(parsed);
    }

    // The base pose is given in parent-local space; accumulate the global
    // transforms and their inverses, the latter being the bone offset matrices.
    for (unsigned int jointIndex = 0; jointIndex < hdr.num_joints; ++jointIndex) {
        IQMJointData &joint = joints[jointIndex];
        if (joint.parent >= 0) {
            joint.globalMatrix = joints[joint.parent].globalMatrix * joint.localMatrix;
        } else {
            joint.globalMatrix = joint.localMatrix;
        }
        joint.inverseGlobalMatrix = joint.globalMatrix;
        joint.inverseGlobalMatrix.Inverse();
        // The inverse of an affine TRS matrix is affine as well, so its fourth
        // row is exactly (0, 0, 0, 1). Pin it to remove the floating point drift
        // produced by the generic matrix inversion (keeps glTF/skinning consumers
        // such as the preview pipeline happy).
        joint.inverseGlobalMatrix.d1 = 0.0f;
        joint.inverseGlobalMatrix.d2 = 0.0f;
        joint.inverseGlobalMatrix.d3 = 0.0f;
        joint.inverseGlobalMatrix.d4 = 1.0f;
    }

    std::vector<IQMPoseData> poses;
    poses.reserve(hdr.num_poses);
    for (unsigned int poseIndex = 0; poseIndex < hdr.num_poses; ++poseIndex) {
        iqmpose pose;
        memcpy(&pose, data + hdr.ofs_poses + poseIndex * IQM_POSE_V2_SIZE, sizeof(pose));
        AI_SWAP4(pose.parent);
        AI_SWAP4(pose.mask);
        for (float &component : pose.channeloffset) {
            (void)component;
            AI_SWAP4(component);
        }
        for (float &component : pose.channelscale) {
            (void)component;
            AI_SWAP4(component);
        }

        if (pose.parent >= int(hdr.num_joints)) {
            fail("has a pose with an invalid parent index");
        }
        if (pose.mask & ~IQM_CHANNEL_MASK_ALL) {
            fail("uses an unsupported animation channel");
        }

        IQMPoseData parsed;
        parsed.parent = pose.parent;
        parsed.mask = pose.mask;
        memcpy(parsed.channelOffset, pose.channeloffset, sizeof(parsed.channelOffset));
        memcpy(parsed.channelScale, pose.channelscale, sizeof(parsed.channelScale));
        poses.push_back(parsed);
    }

    std::vector<IQMAnimData> animations;
    animations.reserve(hdr.num_anims);
    for (unsigned int animIndex = 0; animIndex < hdr.num_anims; ++animIndex) {
        iqmanim anim;
        memcpy(&anim, data + hdr.ofs_anims + animIndex * IQM_ANIM_SIZE, sizeof(anim));
        SwapPod(anim.name);
        SwapPod(anim.first_frame);
        SwapPod(anim.num_frames);
        AI_SWAP4(anim.framerate);
        SwapPod(anim.flags);

        if (anim.framerate <= 0.0f) {
            fail("declares an animation with a non-positive frame rate");
        }
        const uint64_t lastFrame = uint64_t(anim.first_frame) + anim.num_frames;
        if (anim.num_frames == 0 || lastFrame > hdr.num_frames) {
            fail("references animation frames outside of the frame table");
        }

        IQMAnimData parsed;
        parsed.name = readString(anim.name);
        parsed.firstFrame = anim.first_frame;
        parsed.numFrames = anim.num_frames;
        parsed.frameRate = anim.framerate;
        parsed.flags = anim.flags;
        animations.push_back(parsed);
    }

    // Cross-check the frame table against the pose masks: the file must contain
    // exactly one unsigned short per enabled channel per frame.
    if (hdr.num_frames > 0) {
        unsigned int channelsPerFrame = 0;
        for (const IQMPoseData &pose : poses) {
            channelsPerFrame += PopCount(pose.mask);
        }
        if (hdr.num_framechannels != channelsPerFrame) {
            fail("has a frame channel count that does not match the pose masks");
        }
    }

    BuildSkeletonNodes(pScene, joints);
    if (!animations.empty()) {
        BuildAnimations(pScene, data, hdr, joints, poses, animations);
    }

    // Hand the parsed base poses back to the main import routine so it can
    // attach bone bindings once the blend vertex arrays have been read.
    mJoints = std::move(joints);
}

// ------------------------------------------------------------------------------------------------
//  Model 3D import implementation
void IQMImporter::InternReadFile(const std::string &file, aiScene *pScene, IOSystem *pIOHandler) {
    // Read file into memory
    std::unique_ptr<IOStream> pStream(pIOHandler->Open(file, "rb"));
    if (!pStream) {
        throw DeadlyImportError("Failed to open file ", file, ".");
    }

    // Get the file-size and validate it, throwing an exception when fails
    const size_t fileSize = pStream->FileSize();
    if (fileSize < sizeof( iqmheader )) {
        throw DeadlyImportError("IQM-file ", file, " is too small.");
    }
    std::vector<unsigned char> buffer(fileSize);
    unsigned char *data = buffer.data();
    ai_assert(data != nullptr);
    if (fileSize != pStream->Read(data, 1, fileSize)) {
        throw DeadlyImportError("Failed to read the file ", file, ".");
    }

    // Get the header
    iqmheader &hdr = reinterpret_cast<iqmheader&>( *data );
    swap_block( &hdr.version, sizeof( iqmheader ) - sizeof( iqmheader::magic ) );

    // Extra check for header
    if (memcmp(data, IQM_MAGIC, sizeof( IQM_MAGIC ) )
     || hdr.version != IQM_VERSION
     || hdr.filesize != fileSize) {
        throw DeadlyImportError("Bad binary header in file ", file, ".");
    }

    // Reject headers whose offset/count fields reference data outside of the
    // file, otherwise the parsing below reads out of bounds (CWE-125). The
    // offset is validated (and required to be suitably aligned for the pointer
    // casts below) even when count is zero, since the parsing code still forms
    // pointers from these offsets before checking the count.
    auto region_in_bounds = [fileSize](uint64_t offset, uint64_t count, uint64_t elemSize, uint64_t align) {
        if (align != 0 && (offset % align) != 0) {
            return false;
        }
        if (offset > fileSize) {
            return false;
        }
        if (count == 0) {
            return true;
        }
        if (elemSize == 0) {
            return false;
        }
        return count <= (fileSize - offset) / elemSize;
    };

    if (!region_in_bounds(hdr.ofs_text, hdr.num_text, 1, alignof(char))
     || !region_in_bounds(hdr.ofs_meshes, hdr.num_meshes, sizeof(iqmmesh), alignof(iqmmesh))
     || !region_in_bounds(hdr.ofs_vertexarrays, hdr.num_vertexarrays, sizeof(iqmvertexarray), alignof(iqmvertexarray))
     || !region_in_bounds(hdr.ofs_triangles, hdr.num_triangles, sizeof(iqmtriangle), alignof(iqmtriangle))) {
        throw DeadlyImportError("IQM-file ", file, " references data outside of the file.");
    }

    // The text block holds the null-terminated material/joint names read below;
    // require it to be terminated so those reads cannot run past its end.
    if (hdr.num_text != 0 && data[hdr.ofs_text + hdr.num_text - 1] != 0) {
        throw DeadlyImportError("IQM-file ", file, " has a non-terminated text block.");
    }

    ASSIMP_LOG_DEBUG("IQM: loading ", file);

    // Every offset/count below is taken straight from the (attacker-controlled)
    // header, so make sure each table and the text blob lie fully inside the file
    // before any of them is dereferenced. Arithmetic is done in 64-bit to avoid
    // wrapping a 32-bit product.
    const auto rangeInFile = [fileSize](uint64_t offset, uint64_t bytes) {
        // Written so neither side of the comparison can wrap, even if a caller
        // passes a byte count that saturated on overflow.
        return bytes <= fileSize && offset <= fileSize - bytes;
    };
    if (!rangeInFile(hdr.ofs_meshes, (uint64_t)hdr.num_meshes * sizeof(iqmmesh))
             || !rangeInFile(hdr.ofs_vertexarrays, (uint64_t)hdr.num_vertexarrays * sizeof(iqmvertexarray))
             || !rangeInFile(hdr.ofs_triangles, (uint64_t)hdr.num_triangles * sizeof(iqmtriangle))
             || !rangeInFile(hdr.ofs_text, hdr.num_text)) {
        throw DeadlyImportError("Corrupt header offsets in IQM file ", file, ".");
    }
    // A material name is read with strlen from the text blob, so it must be
    // NULL-terminated inside that blob.
    if (hdr.num_text && data[(size_t)hdr.ofs_text + hdr.num_text - 1] != 0) {
        throw DeadlyImportError("Unterminated text section in IQM file ", file, ".");
    }

    // create the root node
    pScene->mRootNode = new aiNode( "<IQMRoot>" );
    // Now rotate the whole scene 90 degrees around the x axis to convert to internal coordinate system
    pScene->mRootNode->mTransformation = aiMatrix4x4(
            1.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, -1.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 1.f);
    pScene->mRootNode->mNumMeshes = hdr.num_meshes;
    pScene->mRootNode->mMeshes = new unsigned int[hdr.num_meshes];
    std::iota( pScene->mRootNode->mMeshes, pScene->mRootNode->mMeshes + pScene->mRootNode->mNumMeshes, 0 );

    mScene = pScene;

    // Allocate output storage
    pScene->mNumMeshes = 0;
    pScene->mMeshes = new aiMesh *[hdr.num_meshes](); // Set arrays to zero to ensue proper destruction if an exception is raised

    pScene->mNumMaterials = 0;
    pScene->mMaterials = new aiMaterial *[hdr.num_meshes]();

    // Blend vertex arrays are file-global; remember them while the meshes are
    // parsed so the bone bindings can be built afterwards.
    const unsigned char *blendIndices = nullptr;
    const unsigned char *blendWeights = nullptr;
    unsigned int blendComponents = 0;
    std::vector<unsigned int> meshFirstVertex(hdr.num_meshes);

    // Swap vertex arrays beforehand...
    for( auto array = reinterpret_cast<iqmvertexarray*>( data + hdr.ofs_vertexarrays ), end = array + hdr.num_vertexarrays; array != end; ++array ) {
        swap_block( &array->type, sizeof( iqmvertexarray ) );
    }

    // Build the joint hierarchy, base pose nodes and animation channels. Bone
    // bindings are attached below once the blend vertex arrays have been read.
    ImportSkeletonAndAnimations(file, pScene, data, hdr);

    // Read all surfaces from the file
    for( auto imesh = reinterpret_cast<iqmmesh*>( data + hdr.ofs_meshes ), end_ = imesh + hdr.num_meshes; imesh != end_; ++imesh ) {
        swap_block( &imesh->name, sizeof( iqmmesh ) );
        if (imesh->material >= hdr.num_text
         || imesh->first_triangle > hdr.num_triangles
         || imesh->num_triangles > hdr.num_triangles - imesh->first_triangle
         || imesh->first_vertex > hdr.num_vertexes
         || imesh->num_vertexes > hdr.num_vertexes - imesh->first_vertex) {
            throw DeadlyImportError("IQM-file ", file, " has a mesh referencing data outside of the file.");
        }
        // Allocate output mesh & material
        auto mesh = pScene->mMeshes[pScene->mNumMeshes++] = new aiMesh();
        mesh->mMaterialIndex = pScene->mNumMaterials;
        meshFirstVertex[pScene->mNumMeshes - 1] = imesh->first_vertex;
        auto mat = pScene->mMaterials[pScene->mNumMaterials++] = new aiMaterial();
        {
            auto text = reinterpret_cast<char*>( data + hdr.ofs_text );
            aiString name( text + imesh->material );
            mat->AddProperty( &name, AI_MATKEY_NAME );
            mat->AddProperty( &name, AI_MATKEY_TEXTURE_DIFFUSE(0) );
        }

        // Fill mesh information
        mesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;
        mesh->mNumFaces = 0;
        mesh->mFaces = new aiFace[imesh->num_triangles];

        // Fill in all triangles
        for( auto tri = reinterpret_cast<iqmtriangle*>( data + hdr.ofs_triangles ) + imesh->first_triangle, end = tri + imesh->num_triangles; tri != end; ++tri ) {
            swap_block( tri->vertex, sizeof( tri->vertex ) );
            // Each triangle index must reference a vertex owned by this mesh;
            // otherwise the first_vertex subtraction below underflows and emits
            // out-of-range face indices (CWE-125/CWE-191).
            const uint32_t vertexEnd = imesh->first_vertex + imesh->num_vertexes;
            if (tri->vertex[0] < imesh->first_vertex || tri->vertex[0] >= vertexEnd ||
                tri->vertex[1] < imesh->first_vertex || tri->vertex[1] >= vertexEnd ||
                tri->vertex[2] < imesh->first_vertex || tri->vertex[2] >= vertexEnd) {
                throw DeadlyImportError("IQM-file ", file, " has a triangle referencing vertices outside of its mesh.");
            }
            auto& face = mesh->mFaces[mesh->mNumFaces++];
            face.mNumIndices = 3;
            face.mIndices = new unsigned int[3]{ tri->vertex[0] - imesh->first_vertex,
                                                 tri->vertex[2] - imesh->first_vertex,
                                                 tri->vertex[1] - imesh->first_vertex };
        }

        // Fill in all vertices
        for( auto array = reinterpret_cast<const iqmvertexarray*>( data + hdr.ofs_vertexarrays ), end__ = array + hdr.num_vertexarrays; array != end__; ++array ) {
            const unsigned int nVerts = imesh->num_vertexes;
            const unsigned int step = array->size;

            uint32_t elemBytes = 0;
            switch ( array->format )
            {
            case IQM_BYTE: case IQM_UBYTE: elemBytes = 1; break;
            case IQM_SHORT: case IQM_USHORT: case IQM_HALF: elemBytes = 2; break;
            case IQM_INT: case IQM_UINT: case IQM_FLOAT: elemBytes = 4; break;
            case IQM_DOUBLE: elemBytes = 8; break;
            default: break;
            }
            if ( elemBytes != 0
             && !region_in_bounds( array->offset, static_cast<uint64_t>( imesh->first_vertex + nVerts ) * step, elemBytes, elemBytes ) ) {
                throw DeadlyImportError("IQM-file ", file, " has a vertex array referencing data outside of the file.");
            }

            switch ( array->type )
            {
            case IQM_POSITION:
                if( array->format == IQM_FLOAT && step >= 3 ){
                    mesh->mNumVertices = nVerts;
                    auto v = mesh->mVertices = new aiVector3D[nVerts];
                    for( auto f = reinterpret_cast<const float*>( data + array->offset ) + imesh->first_vertex * step,
                            end = f + nVerts * step; f != end; f += step, ++v ) {
                        *v = { AI_BE( f[0] ),
                               AI_BE( f[1] ),
                               AI_BE( f[2] ) };
                    }
                }
                break;
            case IQM_TEXCOORD:
                if( array->format == IQM_FLOAT && step >= 2) {
                    auto v = mesh->mTextureCoords[0] = new aiVector3D[nVerts];
                    mesh->mNumUVComponents[0] = 2;
                    for( auto f = reinterpret_cast<const float*>( data + array->offset ) + imesh->first_vertex * step,
                            end = f + nVerts * step; f != end; f += step, ++v ) {
                        *v = { AI_BE( f[0] ),
                           1 - AI_BE( f[1] ), 0 };
                    }
                }
                break;
            case IQM_NORMAL:
                if (array->format == IQM_FLOAT && step >= 3) {
                    auto v = mesh->mNormals = new aiVector3D[nVerts];
                    for( auto f = reinterpret_cast<const float*>( data + array->offset ) + imesh->first_vertex * step,
                            end = f + nVerts * step; f != end; f += step, ++v ) {
                        *v = { AI_BE( f[0] ),
                               AI_BE( f[1] ),
                               AI_BE( f[2] ) };
                    }
                }
                break;
            case IQM_COLOR:
                if (array->format == IQM_UBYTE && step >= 3) {
                    auto v = mesh->mColors[0] = new aiColor4D[nVerts];
                    for( auto f = ( data + array->offset ) + imesh->first_vertex * step,
                            end = f + nVerts * step; f != end; f += step, ++v ) {
                        *v = { ( f[0] ) / 255.f,
                               ( f[1] ) / 255.f,
                               ( f[2] ) / 255.f,
                               step == 3? 1 : ( f[3] ) / 255.f };
                    }
                } else if (array->format == IQM_FLOAT && step >= 3) {
                    auto v = mesh->mColors[0] = new aiColor4D[nVerts];
                    for( auto f = reinterpret_cast<const float*>( data + array->offset ) + imesh->first_vertex * step,
                            end = f + nVerts * step; f != end; f += step, ++v ) {
                        *v = { AI_BE( f[0] ),
                               AI_BE( f[1] ),
                               AI_BE( f[2] ),
                               step == 3? 1 : AI_BE( f[3] ) };
                    }
                }
                break;
            case IQM_TANGENT:
#if 0
                if (array->format == IQM_FLOAT && step >= 3) {
                    auto v = mesh->mTangents = new aiVector3D[nVerts];
                    for( auto f = reinterpret_cast<const float*>( data + array->offset ) + imesh->first_vertex * step,
                            end = f + nVerts * step; f != end; f += step, ++v ) {
                        *v = { AI_BE( f[0] ),
                               AI_BE( f[1] ),
                               AI_BE( f[2] ) };
                    }
                }
#endif
                break;
            case IQM_CUSTOM:
                break; // these attributes are not relevant.
            case IQM_BLENDINDEXES:
                if (array->format == IQM_UBYTE) {
                    if (blendComponents == 0) {
                        blendComponents = step;
                    } else if (blendComponents != step) {
                        throw DeadlyImportError("IQM-file ", file, " has mismatched blend index/weight component counts.");
                    }
                    blendIndices = data + array->offset;
                }
                break;
            case IQM_BLENDWEIGHTS:
                if (array->format == IQM_UBYTE) {
                    if (blendComponents == 0) {
                        blendComponents = step;
                    } else if (blendComponents != step) {
                        throw DeadlyImportError("IQM-file ", file, " has mismatched blend index/weight component counts.");
                    }
                    blendWeights = data + array->offset;
                }
                break;

            default:
                break;
            }
        }
    }

    BuildBoneBindings(pScene, mJoints, blendIndices, blendWeights, meshFirstVertex.data(),
            blendComponents, fileSize, hdr.num_vertexes, data);
}

// ------------------------------------------------------------------------------------------------

} // Namespace Assimp

#endif // !! ASSIMP_BUILD_NO_IQM_IMPORTER
