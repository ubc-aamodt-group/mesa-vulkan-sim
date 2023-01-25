#include "nir/nir.h"
#include "lvp_private.h"
#include "vk_pipeline.h"
#include "vk_pipeline_cache.h"


static VkResult
lvp_ray_tracing_pipeline_create(
    VkDevice                                    _device,
    struct vk_pipeline_cache *                  cache,
    const VkRayTracingPipelineCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipeline)
{
   return VK_SUCCESS;
}


VkResult
lvp_CreateRayTracingPipelinesKHR(
    VkDevice                                    _device,
    VkDeferredOperationKHR                      deferredOperation,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    createInfoCount,
    const VkRayTracingPipelineCreateInfoKHR*    pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines)
{
   printf("LVP: Creating ray tracing pipeline...\n");
   LVP_FROM_HANDLE(vk_pipeline_cache, pipeline_cache, pipelineCache);

   // Assume only 1 pipeline currently
   if (createInfoCount > 1) {
      unreachable("Unimplemented");
   }

   VkResult result = lvp_ray_tracing_pipeline_create(_device, pipeline_cache,
                                                     &pCreateInfos[0],
                                                     pAllocator, &pPipelines[0]);

   return result;
}
