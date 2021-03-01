#ifndef GPGPUSIM_CALLS_H
#define GPGPUSIM_CALLS_H

//extern void gpgpusim_init();
extern void gpgpusim_setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos);
extern void gpgpusim_setGeometries(const VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount);
extern void gpgpusim_testTraversal(struct anv_bvh_node* root);

#endif /* GPGPUSIM_CALLS_H */
