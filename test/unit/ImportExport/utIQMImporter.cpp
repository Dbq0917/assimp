/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2026, assimp team

All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

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
---------------------------------------------------------------------------
*/

#include "UnitTestPCH.h"

#include <assimp/matrix4x4.h>
#include <assimp/postprocess.h>
#include <assimp/quaternion.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include <functional>
#include <string>

using namespace Assimp;

TEST(utIQMImporter, importMrFixit) {
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(ASSIMP_TEST_MODELS_DIR "/IQM/mrfixit.iqm", 0);
    ASSERT_NE(nullptr, scene);
}

TEST(utIQMImporter, importMrFixitSkeleton) {
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(ASSIMP_TEST_MODELS_DIR "/IQM/mrfixit.iqm", aiProcess_ValidateDataStructure);
    ASSERT_NE(nullptr, scene);

    // The file declares 75 joints; each becomes a scene node below the IQM root.
    const aiNode *root = scene->mRootNode;
    ASSERT_NE(nullptr, root);

    const aiNode *findNode = nullptr;
    const aiNode *hips = nullptr;
    std::function<void(const aiNode *)> findSpine = [&](const aiNode *node) {
        if (std::string(node->mName.C_Str()) == "Spine1") {
            findNode = node;
        }
        if (std::string(node->mName.C_Str()) == "Hips") {
            hips = node;
        }
        for (unsigned int i = 0; i < node->mNumChildren; ++i) {
            findSpine(node->mChildren[i]);
        }
    };
    findSpine(root);
    ASSERT_NE(nullptr, findNode);
    ASSERT_NE(nullptr, hips);

    // Names must be preserved verbatim and the file's parent indices honoured.
    EXPECT_EQ(std::string("Spine1"), std::string(findNode->mName.C_Str()));
    ASSERT_NE(nullptr, findNode->mParent);
    EXPECT_EQ(std::string("Hips"), std::string(findNode->mParent->mName.C_Str()));

    // Both meshes are skinned and keep their material references.
    ASSERT_EQ(2u, scene->mNumMeshes);
    EXPECT_GT(scene->mMeshes[0]->mNumBones, 0u);
    EXPECT_GT(scene->mMeshes[1]->mNumBones, 0u);
    ASSERT_EQ(2u, scene->mNumMaterials);
    for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex) {
        const aiMesh *mesh = scene->mMeshes[meshIndex];
        for (unsigned int boneIndex = 0; boneIndex < mesh->mNumBones; ++boneIndex) {
            const aiBone *bone = mesh->mBones[boneIndex];
            EXPECT_FALSE(bone->mName.Empty());
            EXPECT_GT(bone->mNumWeights, 0u);
        }
    }
}

TEST(utIQMImporter, importMrFixitAnimation) {
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(ASSIMP_TEST_MODELS_DIR "/IQM/mrfixit.iqm", 0);
    ASSERT_NE(nullptr, scene);

    ASSERT_EQ(1u, scene->mNumAnimations);
    const aiAnimation *animation = scene->mAnimations[0];
    ASSERT_NE(nullptr, animation);

    // The single "idle" clip declares 24 fps and 101 frames. Tick time follows
    // the file declaration, so ticks per second is 24 and duration is 100.
    EXPECT_EQ(std::string("idle"), std::string(animation->mName.C_Str()));
    EXPECT_DOUBLE_EQ(24.0, animation->mTicksPerSecond);
    EXPECT_DOUBLE_EQ(100.0, animation->mDuration);
    ASSERT_EQ(75u, animation->mNumChannels);

    const aiNodeAnim *spineChannel = nullptr;
    for (unsigned int channelIndex = 0; channelIndex < animation->mNumChannels; ++channelIndex) {
        const aiNodeAnim *channel = animation->mChannels[channelIndex];
        ASSERT_NE(nullptr, channel);
        // One key per source frame, no resampling.
        EXPECT_EQ(101u, channel->mNumPositionKeys);
        EXPECT_EQ(101u, channel->mNumRotationKeys);
        EXPECT_EQ(101u, channel->mNumScalingKeys);
        for (unsigned int frame = 0; frame < channel->mNumPositionKeys; ++frame) {
            EXPECT_DOUBLE_EQ(double(frame), channel->mPositionKeys[frame].mTime);
            EXPECT_DOUBLE_EQ(double(frame), channel->mRotationKeys[frame].mTime);
            EXPECT_DOUBLE_EQ(double(frame), channel->mScalingKeys[frame].mTime);
        }
        if (std::string(channel->mNodeName.C_Str()) == "Spine1") {
            spineChannel = channel;
        }
    }
    ASSERT_NE(nullptr, spineChannel);

    // Frame 0 of Spine1 is the rest pose, reconstructed from the compressed
    // pose channels. The values are the reference numbers produced by decoding
    // the file with the documented IQM formula
    // (value = channeloffset + frame * channelscale).
    const aiVector3D &translation = spineChannel->mPositionKeys[0].mValue;
    EXPECT_FLOAT_EQ(0.0f, translation.x);
    EXPECT_FLOAT_EQ(0.73367f, translation.y);
    EXPECT_FLOAT_EQ(-0.212907f, translation.z);

    const aiQuaternion &rotation = spineChannel->mRotationKeys[0].mValue;
    EXPECT_NEAR(0.0f, rotation.x, 1e-6f);
    EXPECT_NEAR(0.0f, rotation.z, 1e-6f);
    EXPECT_NEAR(-1.0f, rotation.w, 1e-6f);

    // Recomposing a keyframe must reproduce the expected local transformation
    // matrix; compare against the independently decoded reference matrix.
    const aiMatrix4x4 frame0(spineChannel->mScalingKeys[0].mValue,
            spineChannel->mRotationKeys[0].mValue,
            spineChannel->mPositionKeys[0].mValue);
    EXPECT_NEAR(1.0f, frame0.a1, 1e-6f);
    EXPECT_NEAR(0.0f, frame0.a2, 1e-6f);
    EXPECT_NEAR(0.0f, frame0.a4, 1e-6f);

    const aiMatrix4x4 frame75(spineChannel->mScalingKeys[75].mValue,
            spineChannel->mRotationKeys[75].mValue,
            spineChannel->mPositionKeys[75].mValue);
    EXPECT_NEAR(0.9993117f, frame75.a1, 1e-6f);
    EXPECT_NEAR(0.0370961f, frame75.a3, 1e-6f);
}

TEST(utIQMImporter, importMrFixitMaterials) {
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(ASSIMP_TEST_MODELS_DIR "/IQM/mrfixit.iqm", 0);
    ASSERT_NE(nullptr, scene);

    // The animation preview must not render as a grey model: the two declared
    // material names and texture references are imported alongside the skeleton.
    ASSERT_EQ(2u, scene->mNumMaterials);
    aiString firstName;
    ASSERT_EQ(aiReturn_SUCCESS, scene->mMaterials[0]->Get(AI_MATKEY_NAME, firstName));
    EXPECT_EQ(std::string("Body.jpg"), std::string(firstName.C_Str()));
    aiString secondName;
    ASSERT_EQ(aiReturn_SUCCESS, scene->mMaterials[1]->Get(AI_MATKEY_NAME, secondName));
    EXPECT_EQ(std::string("Head.jpg"), std::string(secondName.C_Str()));
}

TEST(utIQMImporter, importInvalidMeshOffset) {
    // Regression test: a crafted .iqm whose header ofs_meshes points at the end
    // of the file used to make the loader read the iqmmesh table out of bounds.
    // The importer must reject the file instead of reading past the buffer.
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(ASSIMP_TEST_MODELS_DIR "/IQM/invalid/InvalidMeshOffset.iqm", 0);
    ASSERT_EQ(nullptr, scene);
}
