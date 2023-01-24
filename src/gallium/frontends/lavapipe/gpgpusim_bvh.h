#ifndef ANV_PUBLIC_H
#define ANV_PUBLIC_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <assert.h>
#include <stdint.h>
#include "vk_buffer.h"


struct anv_bo {
   uint32_t gem_handle;

   uint32_t refcount;

   /* Index into the current validation list.  This is used by the
    * validation list building alrogithm to track which buffers are already
    * in the validation list so that we can ensure uniqueness.
    */
   uint32_t index;

   /* Index for use with util_sparse_array_free_list */
   uint32_t free_index;

   /* Last known offset.  This value is provided by the kernel when we
    * execbuf and is used as the presumed offset for the next bunch of
    * relocations.
    */
   uint64_t offset;

   /** Size of the buffer not including implicit aux */
   uint64_t size;

   /* Map for internally mapped BOs.
    *
    * If ANV_BO_WRAPPER is set in flags, map points to the wrapped BO.
    */
   void *map;

   /** Size of the implicit CCS range at the end of the buffer
    *
    * On Gen12, CCS data is always a direct 1/256 scale-down.  A single 64K
    * page of main surface data maps to a 256B chunk of CCS data and that
    * mapping is provided on TGL-LP by the AUX table which maps virtual memory
    * addresses in the main surface to virtual memory addresses for CCS data.
    *
    * Because we can't change these maps around easily and because Vulkan
    * allows two VkImages to be bound to overlapping memory regions (as long
    * as the app is careful), it's not feasible to make this mapping part of
    * the image.  (On Gen11 and earlier, the mapping was provided via
    * RENDER_SURFACE_STATE so each image had its own main -> CCS mapping.)
    * Instead, we attach the CCS data directly to the buffer object and setup
    * the AUX table mapping at BO creation time.
    *
    * This field is for internal tracking use by the BO allocator only and
    * should not be touched by other parts of the code.  If something wants to
    * know if a BO has implicit CCS data, it should instead look at the
    * has_implicit_ccs boolean below.
    *
    * This data is not included in maps of this buffer.
    */
   uint32_t _ccs_size;

   /** Flags to pass to the kernel through drm_i915_exec_object2::flags */
   uint32_t flags;

   /** True if this BO may be shared with other processes */
   bool is_external:1;

   /** True if this BO is a wrapper
    *
    * When set to true, none of the fields in this BO are meaningful except
    * for anv_bo::is_wrapper and anv_bo::map which points to the actual BO.
    * See also anv_bo_unwrap().  Wrapper BOs are not allowed when use_softpin
    * is set in the physical device.
    */
   bool is_wrapper:1;

   /** See also ANV_BO_ALLOC_FIXED_ADDRESS */
   bool has_fixed_address:1;

   /** True if this BO wraps a host pointer */
   bool from_host_ptr:1;

   /** See also ANV_BO_ALLOC_CLIENT_VISIBLE_ADDRESS */
   bool has_client_visible_address:1;

   /** True if this BO has implicit CCS data attached to it */
   bool has_implicit_ccs:1;
};

struct anv_address {
   struct anv_bo *bo;
   uint64_t offset;
};

struct anv_batch {
   const VkAllocationCallbacks *                alloc;

   /**
    * Sum of all the anv_batch_bo sizes allocated for this command buffer.
    * Used to increase allocation size for long command buffers.
    */
   size_t                                       total_batch_size;

   struct anv_address                           start_addr;

   void *                                       start;
   void *                                       end;
   void *                                       next;

   struct anv_reloc_list *                      relocs;

   /* This callback is called (with the associated user data) in the event
    * that the batch runs out of space.
    */
   VkResult (*extend_cb)(struct anv_batch *, uint32_t, void *);
   void *                                       user_data;

   /**
    * Current error status of the command buffer. Used to track inconsistent
    * or incomplete command buffer states that are the consequence of run-time
    * errors such as out of memory scenarios. We want to track this in the
    * batch because the command buffer object is not visible to some parts
    * of the driver.
    */
   VkResult                                     status;
};

struct anv_buffer {
   struct vk_buffer vk;

   /* Set when bound */
   struct anv_address address;
};


static inline uint64_t
intel_canonical_address(uint64_t v)
{
   /* From the Broadwell PRM Vol. 2a, MI_LOAD_REGISTER_MEM::MemoryAddress:
    *
    *    "This field specifies the address of the memory location where the
    *    register value specified in the DWord above will read from. The
    *    address specifies the DWord location of the data. Range =
    *    GraphicsVirtualAddress[63:2] for a DWord register GraphicsAddress
    *    [63:48] are ignored by the HW and assumed to be in correct
    *    canonical form [63:48] == [47]."
    */
   const int shift = 63 - 47;
   return (int64_t)(v << shift) >> shift;
}

static inline struct anv_address
anv_address_add(struct anv_address addr, uint64_t offset)
{
   addr.offset += offset;
   return addr;
}

static inline void *
anv_address_map(struct anv_address addr)
{
   if (addr.bo == NULL)
      return NULL;

   if (addr.bo->map == NULL)
      return NULL;

   return (uint8_t*)addr.bo->map + addr.offset;
}

#define ANV_NULL_ADDRESS ((struct anv_address) { NULL, 0 })

static inline struct anv_address
anv_address_from_u64(uint64_t addr_u64)
{
   assert(addr_u64 == intel_canonical_address(addr_u64));
   return (struct anv_address) {
      .bo = NULL,
      .offset = addr_u64,
   };
}

static inline bool
anv_address_is_null(struct anv_address addr)
{
   return addr.bo == NULL && addr.offset == 0;
}


static inline uint64_t
anv_address_physical(struct anv_address addr)
{
   uint64_t address = (addr.bo ? addr.bo->offset : 0ull) + addr.offset;
   return intel_canonical_address(address);
}


static inline VkResult
lvp_batch_set_error(struct anv_batch *batch, VkResult error)
{
   assert(error != VK_SUCCESS);
   if (batch->status == VK_SUCCESS)
      batch->status = error;
   return batch->status;
}


static inline bool
lvp_batch_has_error(struct anv_batch *batch)
{
   return batch->status != VK_SUCCESS;
}


VkResult anv_reloc_list_add_bo(struct anv_reloc_list *list,
                               const VkAllocationCallbacks *alloc,
                               struct anv_bo *target_bo);


struct GENERAL_SBT {
    uint64_t General;
};

struct TRIANGLES_SBT {
   uint64_t ClosestHit;
   uint64_t AnyHit;
};

struct PROCEDURAL_SBT {
   uint64_t ClosestHit;
   uint64_t Intersection;
};

#endif /* ANV_PUBLIC_H */