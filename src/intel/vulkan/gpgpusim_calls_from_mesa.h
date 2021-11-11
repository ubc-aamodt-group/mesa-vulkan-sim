#ifndef GPGPUSIM_CALLS_FROM_MESA_H
#define GPGPUSIM_CALLS_FROM_MESA_H

#include "vulkan/vulkan_core.h"

//extern void gpgpusim_init();
extern void gpgpusim_setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos);
extern void gpgpusim_setGeometries(const VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount);
extern void gpgpusim_addTreelets(VkAccelerationStructureKHR accelerationStructure);
extern void gpgpusim_testTraversal(struct anv_bvh_node* root);
extern uint32_t gpgpusim_registerShader(char * shaderPath, uint32_t shader_type);

extern void gpgpusim_vkCmdTraceRaysKHR(
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr);

extern void gpgpusim_setDescriptorSet(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type);

// CPP externs
extern void gpgpusim_addTreelets_cpp(VkAccelerationStructureKHR accelerationStructure);
extern uint32_t gpgpusim_registerShader_cpp(char * shaderPath, uint32_t shader_type);

extern void gpgpusim_vkCmdTraceRaysKHR_cpp(
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr);

extern void gpgpusim_setDescriptorSet_cpp(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type);
#endif /* GPGPUSIM_CALLS_FROM_MESA_H */
