#pragma once

#ifdef SPCRAFT_USE_CUDA

#include <cuda_runtime.h>

#include "core/CooMatrix.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Coordinate matrix stored in CUDA device memory as an array of entries.
 * Link SPCraft for float/double/bool and signed (IT, OT) widths
 * (32, 32), (32, 64), or (64, 64).
 * Owns cudaMalloc storage or borrows device entries through the pointer constructor.
 * Clone() copies device storage; ToHost() downloads; View() borrows device storage.
 * Allocation and transfer failures throw; destruction ignores CUDA errors.
 */
SP_MAT_TEMP_DEFAULT
class cuCooMatrix
{
 public:
  /*****************************************************************************
   *                                Data Members                               *
   ****************************************************************************/
  using Entry = TupleEntry<IT, NT>;
  Entry* entries = nullptr;  //!< Coordinate entries, size nnz, in device memory.
  OT nnz = 0;
  IT m = 0;
  IT n = 0;
  bool memowned = true;

  /*****************************************************************************
   *                                Constructors                               *
   ****************************************************************************/
  cuCooMatrix() noexcept;

  //! Borrow externally managed device entries.
  cuCooMatrix(Entry* entries_, OT count, IT rows, IT columns);

  //! Upload a host COO matrix into owned device storage.
  cuCooMatrix(const SPCOO& host);

  cuCooMatrix(const cuCooMatrix&) = delete;

  cuCooMatrix& operator=(const cuCooMatrix&) = delete;

  cuCooMatrix(cuCooMatrix&& rhs) noexcept;

  cuCooMatrix& operator=(cuCooMatrix&& rhs) noexcept;

  ~cuCooMatrix();

  /*****************************************************************************
   *                              Member Functions                             *
   ****************************************************************************/
  //! Allocate zeroed replacement storage, preserving this object on failure.
  void Allocate(OT count, IT rows, IT columns);

  SPND cuCooMatrix Clone() const;

  //! Release ownership without freeing; use on moved-from objects or views.
  void Reset();

  //! Download into newly allocated host storage.
  SPND SPCOO ToHost() const;

  //! Borrow device entries; the view cannot outlive the device allocation.
  SPND SPCOO View() const;

 private:
  /*****************************************************************************
   *                               Static Members                              *
   ****************************************************************************/
  SPND static Entry* SafeAllocate(OT count, IT rows, IT columns);

  static void SafeDelete(bool owned, Entry* entries);
};

}  // namespace spcraft

#endif  // SPCRAFT_USE_CUDA
