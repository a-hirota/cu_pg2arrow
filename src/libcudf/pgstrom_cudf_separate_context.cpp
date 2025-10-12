/*
 * cuDF wrapper with separate CUDA context
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include "pgstrom_cudf.h"

static thread_local char error_buffer[512];
static CUcontext cudf_context = nullptr;
static bool context_initialized = false;

struct pgstrom_cudf_table_handle {
	std::unique_ptr<cudf::table> table;
	cudf::io::table_with_metadata metadata;
	
	pgstrom_cudf_table_handle(cudf::io::table_with_metadata&& result)
		: metadata(std::move(result))
	{
		table = std::move(metadata.tbl);
	}
};

// Initialize separate CUDA context for cuDF
static bool initialize_cudf_context() {
	if (context_initialized) return true;
	
	CUresult cu_err;
	
	// Get current context (PG-Strom's)
	CUcontext original_ctx = nullptr;
	cuCtxGetCurrent(&original_ctx);
	
	fprintf(stderr, "DEBUG: Original CUDA context: %p\n", original_ctx);
	
	// Create a new context for cuDF on device 0
	CUdevice device;
	cu_err = cuDeviceGet(&device, 0);
	if (cu_err != CUDA_SUCCESS) {
		fprintf(stderr, "ERROR: cuDeviceGet failed: %d\n", cu_err);
		return false;
	}
	
	// Create new context with default flags
	cu_err = cuCtxCreate(&cudf_context, CU_CTX_SCHED_AUTO, device);
	if (cu_err != CUDA_SUCCESS) {
		fprintf(stderr, "ERROR: cuCtxCreate failed: %d\n", cu_err);
		return false;
	}
	
	fprintf(stderr, "DEBUG: Created new CUDA context for cuDF: %p\n", cudf_context);
	
	// Initialize RMM in this new context
	try {
		static rmm::mr::cuda_memory_resource cuda_mr;
		rmm::mr::set_current_device_resource(&cuda_mr);
		fprintf(stderr, "DEBUG: RMM initialized in new context\n");
	} catch (const std::exception& e) {
		fprintf(stderr, "ERROR: RMM initialization failed: %s\n", e.what());
		cuCtxDestroy(cudf_context);
		return false;
	}
	
	// Disable hardware decompression
	setenv("LIBCUDF_HW_DECOMPRESSION", "OFF", 1);
	
	// Restore original context
	if (original_ctx) {
		cuCtxSetCurrent(original_ctx);
	} else {
		cuCtxSetCurrent(nullptr);
	}
	
	context_initialized = true;
	return true;
}

extern "C" pgstrom_cudf_table_handle *
pgstrom_cudf_read_parquet(const char *filename,
						  int row_group_index,
						  const int *column_indices,
						  int num_columns,
						  const char **error_message)
{
	*error_message = nullptr;
	error_buffer[0] = '\0';
	
	// Initialize cuDF context if needed
	if (!initialize_cudf_context()) {
		snprintf(error_buffer, sizeof(error_buffer), "Failed to initialize cuDF context");
		*error_message = error_buffer;
		return nullptr;
	}
	
	// Save current context (PG-Strom's)
	CUcontext original_ctx = nullptr;
	cuCtxGetCurrent(&original_ctx);
	
	// Switch to cuDF context
	CUresult cu_err = cuCtxSetCurrent(cudf_context);
	if (cu_err != CUDA_SUCCESS) {
		snprintf(error_buffer, sizeof(error_buffer), "Failed to switch to cuDF context: %d", cu_err);
		*error_message = error_buffer;
		return nullptr;
	}
	
	pgstrom_cudf_table_handle *handle = nullptr;
	
	try {
		// Read Parquet in cuDF's context
		auto source = cudf::io::source_info(filename);
		std::vector<std::vector<cudf::size_type>> row_groups = {{row_group_index}};
		auto options = cudf::io::parquet_reader_options::builder(source)
			.row_groups(row_groups)
			.build();
		
		auto result = cudf::io::read_parquet(options);
		handle = new pgstrom_cudf_table_handle(std::move(result));
		
		fprintf(stderr, "DEBUG: cuDF read succeeded in separate context\n");
	}
	catch (const std::exception &e) {
		snprintf(error_buffer, sizeof(error_buffer), "cuDF error: %s", e.what());
		*error_message = error_buffer;
	}
	
	// Restore original context
	cuCtxSetCurrent(original_ctx);
	
	return handle;
}

extern "C" int64_t
pgstrom_cudf_get_num_rows(pgstrom_cudf_table_handle *handle)
{
	return handle->table->num_rows();
}

extern "C" int
pgstrom_cudf_get_num_columns(pgstrom_cudf_table_handle *handle)
{
	return handle->table->num_columns();
}

extern "C" int
pgstrom_cudf_get_column(pgstrom_cudf_table_handle *handle,
						int column_index,
						pgstrom_cudf_column *column_out,
						const char **error_message)
{
	// Save current context
	CUcontext original_ctx = nullptr;
	cuCtxGetCurrent(&original_ctx);
	
	// Switch to cuDF context
	cuCtxSetCurrent(cudf_context);
	
	int result = -1;
	
	try {
		auto col_view = handle->table->get_column(column_index).view();
		
		column_out->null_mask = const_cast<void*>(static_cast<const void*>(col_view.null_mask()));
		column_out->null_mask_size = cudf::bitmask_allocation_size_bytes(col_view.size());
		column_out->num_rows = col_view.size();
		column_out->null_count = col_view.null_count();
		
		if (col_view.type().id() == cudf::type_id::STRING) {
			auto strings_view = cudf::strings_column_view(col_view);
			auto chars_ptr = strings_view.chars_begin(cudf::get_default_stream());
			column_out->data = const_cast<void*>(static_cast<const void*>(chars_ptr));
			column_out->data_size = strings_view.chars_size(cudf::get_default_stream());
			
			auto offsets_col = col_view.child(0);
			column_out->offsets = const_cast<void*>(static_cast<const void*>(offsets_col.head()));
			column_out->offsets_size = offsets_col.size() * sizeof(int32_t);
		} else {
			column_out->data = const_cast<void*>(static_cast<const void*>(col_view.head()));
			column_out->data_size = col_view.size() * cudf::size_of(col_view.type());
			column_out->offsets = nullptr;
			column_out->offsets_size = 0;
		}
		
		result = 0;
	}
	catch (const std::exception &e) {
		snprintf(error_buffer, sizeof(error_buffer), "cuDF get_column error: %s", e.what());
		*error_message = error_buffer;
	}
	
	// Restore original context
	cuCtxSetCurrent(original_ctx);
	
	return result;
}

extern "C" void
pgstrom_cudf_free_table(pgstrom_cudf_table_handle *handle)
{
	// Save current context
	CUcontext original_ctx = nullptr;
	cuCtxGetCurrent(&original_ctx);
	
	// Switch to cuDF context for cleanup
	cuCtxSetCurrent(cudf_context);
	
	delete handle;
	
	// Restore original context
	cuCtxSetCurrent(original_ctx);
}
