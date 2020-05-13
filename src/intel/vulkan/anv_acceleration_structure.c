/*
 * Copyright © 2020 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "anv_private.h"

static uint64_t
anv_bvh_max_size(VkAccelerationStructureTypeKHR type, uint64_t leaf_count)
{
   uint64_t max_size_B = ALIGN(GENX(RT_BVH_length) * 4, 256);

   /* Assuming a 2-ary tree in which no node has only one child, the number of
    * internal nodes is always going to be leaf_count - 1 (this is an easy
    * inductive proof).  While we can do somewhat better with the 6-ary trees
    * that make up our BVH, Embree doesn't guarantee that all internal nodes
    * are full and so it's a bit harder to calculate exactly.  Assuming no
    * internal node is left with exactly one leaf (that would be silly), we
    * can use the 2-ary tree node count as an upper bound.  Due to
    * restrictions on the BVH data structure, we always have at least one
    * internal node.
    */
   uint64_t internal_node_count = MAX2(2, leaf_count) - 1;
   max_size_B += internal_node_count * GENX(RT_BVH_INTERNAL_NODE_length) * 4;

   switch (type) {
   case VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR:
      max_size_B += leaf_count * GENX(RT_BVH_INSTANCE_LEAF_length) * 4;
      break;

   case VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR:
      max_size_B += leaf_count * GENX(RT_BVH_QUAD_LEAF_length) * 4;
      break;

   case VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR:
      unreachable("VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR is not allowed "
                  "in vkGetAccelerationStructureBuildSizesKHR");

   default:
      unreachable("Invalid VkAccelerationStructureTypeKHR");
   }

   return max_size_B;
}

void
anv_GetAccelerationStructureBuildSizesKHR(
    VkDevice                                    device,
    VkAccelerationStructureBuildTypeKHR         buildType,
    const VkAccelerationStructureBuildGeometryInfoKHR* pBuildInfo,
    const uint32_t*                             pMaxPrimitiveCounts,
    VkAccelerationStructureBuildSizesInfoKHR*   pSizeInfo)
{
   assert(pSizeInfo->sType ==
          VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR);

   uint64_t max_prim_count = 0;
   for (uint32_t i = 0; i < pBuildInfo->geometryCount; i++)
      max_prim_count += pMaxPrimitiveCounts[i];

   pSizeInfo->accelerationStructureSize =
      anv_bvh_max_size(pBuildInfo->type, max_prim_count);

   uint64_t cpu_build_scratch_size = 0; /* TODO */
   uint64_t cpu_update_scratch_size = cpu_build_scratch_size;

   uint64_t gpu_build_scratch_size = 0; /* TODO */
   uint64_t gpu_update_scratch_size = gpu_build_scratch_size;

   switch (buildType) {
   case VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR:
      pSizeInfo->buildScratchSize = cpu_build_scratch_size;
      pSizeInfo->updateScratchSize = cpu_update_scratch_size;
      break;

   case VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR:
      pSizeInfo->buildScratchSize = gpu_build_scratch_size;
      pSizeInfo->updateScratchSize = gpu_update_scratch_size;
      break;

   case VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_OR_DEVICE_KHR:
      pSizeInfo->buildScratchSize = MAX2(cpu_build_scratch_size,
                                         gpu_build_scratch_size);
      pSizeInfo->updateScratchSize = MAX2(cpu_update_scratch_size,
                                          gpu_update_scratch_size);
      break;

   default:
      unreachable("Invalid acceleration structure build type");
   }
}

VkResult
anv_CreateAccelerationStructureKHR(
    VkDevice                                    _device,
    const VkAccelerationStructureCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkAccelerationStructureKHR*                 pAccelerationStructure)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   struct anv_acceleration_structure *accel;

   accel = vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*accel), 8,
                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (accel == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   vk_object_base_init(&device->vk, &accel->base,
                       VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR);

   accel->size = pCreateInfo->size;
   accel->address = anv_address_add(buffer->address, pCreateInfo->offset);

   *pAccelerationStructure = anv_acceleration_structure_to_handle(accel);

   return VK_SUCCESS;
}

void
anv_DestroyAccelerationStructureKHR(
    VkDevice                                    _device,
    VkAccelerationStructureKHR                  accelerationStructure,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_acceleration_structure, accel, accelerationStructure);

   if (!accel)
      return;

   vk_object_base_finish(&accel->base);
   vk_free2(&device->vk.alloc, pAllocator, accel);
}

VkDeviceAddress
anv_GetAccelerationStructureDeviceAddressKHR(
    VkDevice                                    device,
    const VkAccelerationStructureDeviceAddressInfoKHR* pInfo)
{
   ANV_FROM_HANDLE(anv_acceleration_structure, accel,
                   pInfo->accelerationStructure);

   assert(!anv_address_is_null(accel->address));
   assert(accel->address.bo->flags & EXEC_OBJECT_PINNED);

   return anv_address_physical(accel->address);
}

void
anv_GetDeviceAccelerationStructureCompatibilityKHR(
    VkDevice                                    device,
    const VkAccelerationStructureVersionInfoKHR* pVersionInfo,
    VkAccelerationStructureCompatibilityKHR*    pCompatibility)
{
   unreachable("Unimplemented");
}

VkResult
anv_BuildAccelerationStructuresKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      deferredOperation,
    uint32_t                                    infoCount,
    const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
    const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos)
{
   unreachable("Unimplemented");
   return vk_error(VK_ERROR_FEATURE_NOT_PRESENT);
}

VkResult
anv_CopyAccelerationStructureKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      deferredOperation,
    const VkCopyAccelerationStructureInfoKHR*   pInfo)
{
   unreachable("Unimplemented");
   return vk_error(VK_ERROR_FEATURE_NOT_PRESENT);
}

VkResult
anv_CopyAccelerationStructureToMemoryKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      deferredOperation,
    const VkCopyAccelerationStructureToMemoryInfoKHR* pInfo)
{
   unreachable("Unimplemented");
   return vk_error(VK_ERROR_FEATURE_NOT_PRESENT);
}

VkResult
anv_CopyMemoryToAccelerationStructureKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      deferredOperation,
    const VkCopyMemoryToAccelerationStructureInfoKHR* pInfo)
{
   unreachable("Unimplemented");
   return vk_error(VK_ERROR_FEATURE_NOT_PRESENT);
}

VkResult
anv_WriteAccelerationStructuresPropertiesKHR(
    VkDevice                                    device,
    uint32_t                                    accelerationStructureCount,
    const VkAccelerationStructureKHR*           pAccelerationStructures,
    VkQueryType                                 queryType,
    size_t                                      dataSize,
    void*                                       pData,
    size_t                                      stride)
{
   unreachable("Unimplemented");
   return vk_error(VK_ERROR_FEATURE_NOT_PRESENT);
}

void
anv_CmdBuildAccelerationStructuresKHR(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    infoCount,
    const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
    const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos)
{
   unreachable("Unimplemented");
}

void
anv_CmdBuildAccelerationStructuresIndirectKHR(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    infoCount,
    const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
    const VkDeviceAddress*                      pIndirectDeviceAddresses,
    const uint32_t*                             pIndirectStrides,
    const uint32_t* const*                      ppMaxPrimitiveCounts)
{
   unreachable("Unimplemented");
}

void
anv_CmdCopyAccelerationStructureKHR(
    VkCommandBuffer                             commandBuffer,
    const VkCopyAccelerationStructureInfoKHR*   pInfo)
{
   unreachable("Unimplemented");
}

void
anv_CmdCopyAccelerationStructureToMemoryKHR(
    VkCommandBuffer                             commandBuffer,
    const VkCopyAccelerationStructureToMemoryInfoKHR* pInfo)
{
   unreachable("Unimplemented");
}

void
anv_CmdCopyMemoryToAccelerationStructureKHR(
    VkCommandBuffer                             commandBuffer,
    const VkCopyMemoryToAccelerationStructureInfoKHR* pInfo)
{
   unreachable("Unimplemented");
}

void
anv_CmdWriteAccelerationStructuresPropertiesKHR(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    accelerationStructureCount,
    const VkAccelerationStructureKHR*           pAccelerationStructures,
    VkQueryType                                 queryType,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery)
{
   unreachable("Unimplemented");
}
