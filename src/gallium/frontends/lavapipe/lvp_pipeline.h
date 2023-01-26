#include "nir/nir.h"
#include "lvp_private.h"
#include "vk_pipeline.h"
#include "vk_render_pass.h"


static VkResult
lvp_ray_tracing_pipeline_init(struct lvp_pipeline *pipeline,
                           struct lvp_device *device,
                           struct lvp_pipeline_cache *cache,
                           const VkRayTracingPipelineCreateInfoKHR *pCreateInfo);

nir_shader *
vsim_shader_spirv_to_nir(struct lvp_device *device, const VkPipelineShaderStageCreateInfo *sinfo);